# app.py — Vale Mastering Assistant (Console-first UI)
import os, json, re, shutil, uuid, base64, io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# matplotlib/librosa for thumbnails (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa, librosa.display
import soundfile as sf

from utils import session_tmp_path
from dsp import (
    analyze_audio,
    render_adaptive_from_plans,
    derive_section_plans_from_single,
    detect_resonances_simple,
)
from ai import llm_plan
from diagnostics import validate_plan
from corrective import llm_corrective_cleanup, apply_corrective_eq
from chat_ui import render_chat, add_chat

# -------------------------- Compare strip helpers --------------------------
def _ok_warn_pill(label, ok=True):
    cls = "cc-pill" + ("" if ok else " warn")
    return f'<span class="{cls}">{label}</span>'

def _tp_ok(tp):  # true-peak ok?
    try:
        return float(tp) <= -1.0
    except:
        return True

def ensure_print_store():
    st.session_state.setdefault("prints", [])  # list of dicts per variation

def record_print(var_idx: int, path: str, analysis: dict):
    ensure_print_store()
    st.session_state["prints"].append({
        "idx": var_idx,
        "path": path,
        "lufs": analysis.get("lufs_integrated"),
        "tp": analysis.get("true_peak_dbfs_est"),
    })

def render_compare_strip():
    ensure_print_store()
    if not st.session_state["prints"]:
        return None

    cards = []
    for p in st.session_state["prints"]:
        lufs = p["lufs"]; tp = p["tp"]
        pill_lufs = _ok_warn_pill(f"LUFS {lufs:.2f}", ok=(-14.5 <= lufs <= -9.0) if lufs is not None else True)
        pill_tp   = _ok_warn_pill(f"TP {tp:.2f} dB", ok=_tp_ok(tp) if tp is not None else True)
        cards.append(f"""
          <div class="compare-card">
            <div class="cc-title"><span>Variation {p['idx']}</span><span style="opacity:.6;">master</span></div>
            <div class="cc-kpis">{pill_lufs}{pill_tp}</div>
            <div class="cc-audio">[AUDIO_{p['idx']}]</div>
          </div>
        """)
    return f'<div class="compare-strip">{"".join(cards)}</div>'

# -------------------------- Status ticker helpers --------------------------
def _init_status():
    st.session_state.setdefault("status_steps", [])   # list[str]
    st.session_state.setdefault("status_now", "")     # current str

def status_set(text: str):
    _init_status()
    st.session_state["status_now"] = text

def status_push(step: str):
    _init_status()
    st.session_state["status_steps"].append(step)
    st.session_state["status_steps"] = st.session_state["status_steps"][-12:]  # keep last ~12

def status_clear():
    st.session_state["status_steps"] = []
    st.session_state["status_now"] = ""

def render_status_bar():
    _init_status()
    crumbs = " • ".join(st.session_state["status_steps"])
    html = f"""
    <div class="vale-status-wrap">
      <div class="vale-status">
        <span class="vale-dot"></span>
        <span class="vale-breadcrumb">{st.session_state['status_now'] or 'Ready'}</span>
        <span style="opacity:.45;margin-left:8px;">{crumbs}</span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -------------------------- Chat KPI + thumbs helpers --------------------------
def _fmt_db(x):
    try: return f"{x:.2f} dB"
    except: return "--"

def _fmt_num(x):
    try: return f"{x:.2f}"
    except: return "--"

def kpi_strip_html(*, lufs=None, tp=None, crest=None, lra=None):
    cls_lufs = "ok" if (lufs is not None and -14.5 <= lufs <= -9.0) else "warn"
    cls_tp   = "ok" if (tp   is not None and tp <= -1.0) else "warn"
    parts = ['<div class="kpi-row">']
    if lufs is not None:
        parts.append(f'<div class="kpi {cls_lufs}"><div class="lab">Integrated LUFS</div><div class="val">{_fmt_num(lufs)}<span class="pill">target −14…−9</span></div></div>')
    if tp is not None:
        parts.append(f'<div class="kpi {cls_tp}"><div class="lab">True Peak</div><div class="val">{_fmt_db(tp)}<span class="pill">≤ −1.0</span></div></div>')
    if crest is not None:
        parts.append(f'<div class="kpi ok"><div class="lab">Crest / DR</div><div class="val">{_fmt_num(crest)}</div></div>')
    if lra is not None:
        parts.append(f'<div class="kpi ok"><div class="lab">Loudness Range</div><div class="val">{_fmt_num(lra)} LU</div></div>')
    parts.append('</div>')
    return "".join(parts)

def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def waveform_thumb_b64(path: str, w=260, h=72) -> str:
    y, sr = sf.read(path, always_2d=False)
    y = y[:,0] if np.ndim(y)==2 else y
    if len(y) > sr*180:
        step = int(len(y)/(sr*180))
        y = y[::max(1,step)]
    y = y / (np.max(np.abs(y)) + 1e-12)
    fig = plt.figure(figsize=(w/96, h/96), dpi=96); fig.patch.set_alpha(0)
    ax = plt.axes([0,0,1,1], facecolor="none"); ax.plot(y, linewidth=1.2); ax.set_axis_off()
    return _png_b64(fig)

def spectrum_thumb_b64(path: str, w=260, h=72) -> str:
    y, sr = sf.read(path, always_2d=False)
    y = y[:,0] if np.ndim(y)==2 else y
    Y = np.abs(np.fft.rfft(y)) + 1e-8
    f = np.fft.rfftfreq(len(y), 1/sr)
    f_log = np.geomspace(20, min(sr/2, 20000), 400)
    Yi = np.interp(f_log, f, Y); Yi = 20*np.log10(Yi/np.max(Yi))
    fig = plt.figure(figsize=(w/96, h/96), dpi=96); fig.patch.set_alpha(0)
    ax = plt.axes([0,0,1,1], facecolor="none"); ax.plot(Yi, linewidth=1.2); ax.set_axis_off()
    return _png_b64(fig)

def post_analysis_dashboard_to_chat(infile_path: str, analysis: dict):
    try:
        wf = waveform_thumb_b64(infile_path)
        sp = spectrum_thumb_b64(infile_path)
    except Exception:
        wf = sp = None

    lufs = analysis.get("lufs_integrated")
    tp   = analysis.get("true_peak_dbfs_est")
    crest = analysis.get("crest_factor")
    lra   = analysis.get("loudness_range")

    chips = kpi_strip_html(lufs=lufs, tp=tp, crest=crest, lra=lra)
    thumbs = ""
    if wf and sp:
        thumbs = (
            f"<span class='thumb'><img src='data:image/png;base64,{wf}' width='260'></span>"
            f"<span class='thumb'><img src='data:image/png;base64,{sp}' width='260'></span>"
        )
    add_chat("assistant", "quick read on your file—numbers first, pictures second.<br>"+chips+thumbs, html=True)

# -------------------------- Altair theme --------------------------
def _vale_altair_theme():
    return {
        "config": {
            "background": "transparent",
            "axis": {
                "labelColor": "#C9D2E0",
                "titleColor": "#C9D2E0",
                "gridColor": "#27303A",
                "domainColor": "#2B3038",
            },
            "view": {"stroke": "transparent"},
            "line": {"strokeWidth": 2, "color": "#5EA2FF"},
            "point": {"filled": True, "size": 60, "color": "#8B7CFF"},
        }
    }
alt.themes.register("vale_dark", _vale_altair_theme)
alt.themes.enable("vale_dark")

def tonal_png_data_url(bands_pct_8: dict) -> str | None:
    try:
        BAND_ORDER  = ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]
        vals = [max(1e-12, float(bands_pct_8.get(b, 0.0))) for b in BAND_ORDER]
        total = sum(vals) or 1.0
        shares = [v/total for v in vals]
        gmean = float(np.exp(np.mean(np.log(shares))))
        dev_db = [20.0*np.log10(s/gmean) for s in shares]
        x_labels = ["Sub","Lo Bass","Hi Bass","Lo Mids","Mids","Hi Mids","Highs","Air"]
        fig, ax = plt.subplots(figsize=(6.0, 2.2), dpi=180)
        ax.plot(range(len(dev_db)), dev_db, marker="o")
        ax.set_xticks(range(len(x_labels))); ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylim(-12, 12); ax.grid(True, alpha=0.25); ax.set_ylabel("Δ vs band avg (dB)", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", transparent=True); plt.close(fig)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return None

# -------------------------- assets & page --------------------------
def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_PATH   = "imgs/2.png"
AVATAR_PATH = "imgs/shadow.png"
LOGO_B64    = img_to_base64(LOGO_PATH)
AVATAR_B64  = img_to_base64(AVATAR_PATH)

st.set_page_config(page_title="Vale Mastering Assistant", page_icon=LOGO_PATH, layout="wide")

# -------------------------- global CSS --------------------------
st.markdown("""
<style>
:root{
  --bg:#0E1116; --panel:#15181D; --panel-hi:#1A1D22;
  --ink:#E9EDF5; --ink-dim:#9BA7B6; --hair:#2B3038;
  --vale:#5EA2FF; --vale-ghost:rgba(94,162,255,.12);
  --purple:#8B7CFF; --danger:#FF5C7A; --radius:14px;
}
#MainMenu, header, footer {display:none !important;}
[data-testid="stAppViewContainer"] > .main{padding-top:0 !important;}
main .block-container{padding-top:0 !important;}
html, body{background:var(--bg) !important;}
.block-container{background:transparent !important;}
/* navbar */
.vale-nav{position:sticky;top:0;z-index:1000;display:flex;justify-content:center;align-items:center;gap:12px;padding:10px 0 12px;background:rgba(14,17,22,.72);backdrop-filter:blur(8px);border-bottom:1px solid #161A20;}
.vale-nav .pill{width:44px;height:44px;border-radius:50%;background:radial-gradient(120% 120% at 20% 20%, rgba(94,162,255,.28), rgba(139,124,255,.18));border:1px solid #2D3340;box-shadow:0 0 22px rgba(94,162,255,.28);}
.vale-shell{width:min(1200px,70vw);margin:12px auto 0;}
.vale-card{background:linear-gradient(180deg,var(--panel) 0%,#12151A 120%);border:1px solid var(--hair);border-radius:var(--radius);box-shadow:0 18px 40px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.03);padding:16px;}
.stTextInput input,.stTextArea textarea{background:#0F1318;color:var(--ink);border:1px solid var(--hair);border-radius:10px;}
.stButton>button{background:#10151B;color:var(--ink);border:1px solid var(--hair);border-radius:10px;padding:8px 12px;}
.stButton>button:hover{border-color:var(--vale);box-shadow:0 0 0 3px var(--vale-ghost);}
[data-testid="stSlider"] [role="slider"]{background:var(--vale) !important;}
/* status bar */
.vale-status-wrap{position:sticky;top:64px;z-index:999;backdrop-filter:saturate(120%) blur(8px);background:linear-gradient(180deg,#0F1319 0%, #0C1016 120%);border-bottom:1px solid #242A33;}
.vale-status{max-width:70vw;margin:0 auto;padding:8px 12px;color:#DCE6F6;font-size:12px;letter-spacing:.25px;display:flex;align-items:center;gap:8px;white-space:nowrap;overflow:hidden;}
.vale-dot{width:6px;height:6px;border-radius:50%;background:#5EA2FF;box-shadow:0 0 10px rgba(94,162,255,.7);}
.vale-breadcrumb{overflow:hidden;text-overflow:ellipsis;}
/* KPI chips */
.kpi-row{display:flex;flex-wrap:wrap;gap:10px;margin:.35rem 0;}
.kpi{min-width:116px;padding:10px 12px;border-radius:12px;background:linear-gradient(180deg,#171B21 0%,#12161C 120%);border:1px solid #2B3038;box-shadow:0 12px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03);}
.kpi .lab{font-size:11px;letter-spacing:.25px;color:#9BA7B6;text-transform:uppercase;}
.kpi .val{font-size:18px;font-weight:750;color:#E9EDF5;line-height:1.2}
.kpi.ok{box-shadow:inset 0 0 0 1px rgba(94,162,255,.10),0 10px 22px rgba(94,162,255,.10);}
.kpi.warn{box-shadow:inset 0 0 0 1px rgba(255,92,122,.10),0 10px 22px rgba(255,92,122,.08);}
.kpi .pill{display:inline-block;margin-left:6px;padding:2px 6px;border-radius:999px;font-size:10px;letter-spacing:.2px;color:#BFD7FF;background:rgba(94,162,255,.14);border:1px solid #2D3340;}
/* compare strip */
.compare-strip{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 6px;}
.compare-card{width:280px;padding:10px;border-radius:12px;background:linear-gradient(180deg,#171B21 0%,#12161C 120%);border:1px solid #2B3038;box-shadow:0 10px 26px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.03);}
.cc-title{font-size:12px;color:#AAB6C6;margin-bottom:6px;display:flex;justify-content:space-between;}
.cc-kpis{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0;}
.cc-pill{font-size:11px;color:#E9EDF5;padding:3px 7px;border-radius:999px;border:1px solid #2D3340;background:rgba(94,162,255,.12);}
.cc-pill.warn{background:rgba(255,92,122,.12);color:#FFD3DA;border-color:rgba(255,92,122,.24);}
.cc-audio{margin-top:6px;}
</style>
""", unsafe_allow_html=True)

# Navbar + status
st.markdown(f"""<div class="vale-nav"><img src="data:image/png;base64,{LOGO_B64}" alt="Vale" height="54"></div>""", unsafe_allow_html=True)
render_status_bar()

# Central shell (70% width)
st.markdown('<div class="vale-shell">', unsafe_allow_html=True)

# -------------------------- session state --------------------------
st.session_state.setdefault("chat", [])
st.session_state.setdefault("base_dir", session_tmp_path())
base_dir = st.session_state["base_dir"]
llm_model = st.session_state.get("llm_model", "gpt-4o-mini")

# -------------------------- small helpers --------------------------
def _safe_name(name: str) -> str:
    base_name = os.path.basename(name or "upload")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)

BAND_ORDER  = ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]
BAND_LABELS = {
    "sub":"Sub (20–60 Hz)","low_bass":"Low Bass (60–120 Hz)","high_bass":"High Bass (120–250 Hz)",
    "low_mids":"Low Mids (250–500 Hz)","mids":"Mids (500 Hz–3.5 kHz)","high_mids":"High Mids (3.5–8 kHz)",
    "highs":"Highs (8–10 kHz)","air":"Air (10–20 kHz)",
}

def compute_deviation_db_from_bands(bands_pct_8: dict) -> pd.DataFrame:
    vals = [max(1e-12, float(bands_pct_8.get(b, 0.0))) for b in BAND_ORDER]
    total = sum(vals) or 1.0
    shares = [v / total for v in vals]
    gmean = float(np.exp(np.mean(np.log(shares))))
    dev_db = [20.0 * np.log10(s / gmean) for s in shares]
    return pd.DataFrame({"BandKey": BAND_ORDER,"Band": [BAND_LABELS[b] for b in BAND_ORDER],
                         "Deviation (dB)": dev_db,"Share (%)": [v * 100.0 for v in shares]})

def tonal_chart(df: pd.DataFrame):
    alt.themes.register("vale_light_cli", lambda: {
        "config": {
            "background": "#ffffff",
            "axis": {"labelColor": "#2A3542", "titleColor": "#2A3542", "gridColor": "#E6EAF1", "domainColor": "#DDE2EA"},
            "view": {"stroke": "transparent"}, "line": {"strokeWidth": 2}, "point": {"filled": True, "size": 60},
        }
    })
    alt.themes.enable("vale_light_cli")
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Band:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Deviation (dB):Q", scale=alt.Scale(domain=[-12, 12]), title="Balance vs band average (dB)"),
        tooltip=["Band", alt.Tooltip("Deviation (dB):Q", format=".2f"), alt.Tooltip("Share (%):Q", format=".1f")],
    ).properties(height=240)

# -------------------------- Vale Console (chat first) --------------------------
chat_box = st.container()
render_chat(chat_box, state_key="chat", height=420, avatar_img_b64=AVATAR_B64)

# Action row under chat
act_col1, act_col2, _ = st.columns([0.22, 0.30, 0.48])
with act_col1:
    analyze_click = st.button("Analyze")
with act_col2:
    gen_click = st.button("Render 3 Variations")

# -------------------------- Session Tools --------------------------
with st.expander("Session Tools", expanded=True):
    st.markdown('<div class="vale-card">', unsafe_allow_html=True)
    upload_box = st.file_uploader(
        "Upload premaster (WAV/AIFF/FLAC — no limiter, ~−6 dBFS headroom)",
        type=["wav", "aiff", "aif", "flac"]
    )

    reference_txt = st.text_input("Reference (optional)", placeholder="Return of the Jaded – Soma")
    reference_weight = st.slider("Reference weight", 0.0, 1.0, 0.9 if reference_txt else 0.0, 0.1)

    c1, c2 = st.columns(2)
    with c1:
        prompt_txt = st.text_area("Intent (vibe/goals)", placeholder="dark, preserve dynamics, tape weight; like Soma")
        tone = st.slider("Tone: Dark ↔ Bright", -1.0, 1.0, 0.0, 0.1)
        dynamics = st.slider("Dynamics: Preserve ↔ Loud", -1.0, 1.0, 0.0, 0.1)
    with c2:
        stereo = st.slider("Stereo: Narrow ↔ Wide", -1.0, 1.0, 0.0, 0.1)
        character = st.slider("Character: Clean ↔ Tape", -1.0, 1.0, 0.5, 0.1)

    intent = {"tone": float(tone), "dynamics": float(dynamics), "stereo": float(stereo), "character": float(character)}

    preclean = st.checkbox("Pre-clean (auto corrective EQ) before mastering", value=True)
    post_notch = st.checkbox("Add up to 3 safety notches before render", value=True)

    llm_model = st.text_input("OpenAI model", value=llm_model, help="e.g. gpt-4o-mini")
    st.session_state["llm_model"] = llm_model
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- file persistence --------------------------
if upload_box is None and "in_path" not in st.session_state:
    add_chat("assistant", "upload a premaster to begin—i’ll analyze it and suggest musical moves.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if upload_box is not None:
    if ("uploaded_name" not in st.session_state) or (upload_box.name != st.session_state["uploaded_name"]):
        st.session_state["uploaded_name"] = _safe_name(upload_box.name)
        in_path = os.path.join(base_dir, st.session_state["uploaded_name"])
        try:
            upload_box.seek(0)
            with open(in_path, "wb") as f:
                shutil.copyfileobj(upload_box, f, length=1 * 1024 * 1024)
            upload_box.seek(0)
        except Exception as e:
            add_chat("assistant", "i couldn’t write your file to disk.")
            st.exception(e)
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()
        st.session_state["in_path"] = in_path
        st.session_state.pop("analysis", None)

in_path = st.session_state["in_path"]

# -------------------------- ANALYZE --------------------------
if analyze_click or "analysis" not in st.session_state:
    status_set("Analyzing…"); status_push("Analyze")
    try:
        st.session_state["analysis"] = analyze_audio(in_path)
        a = st.session_state["analysis"]
        status_set("Analysis ready ✓"); status_push("Analysis ✓")
        add_chat("assistant", "alright—got a handle on it. here’s what i’m seeing:")
        post_analysis_dashboard_to_chat(in_path, a)

        # Tonal snapshot into chat
        bands = a.get("bands_pct_8") or {}
        if bands and sum(float(bands.get(k, 0.0)) for k in BAND_ORDER) > 1e-9:
            data_url = tonal_png_data_url(bands)
            if data_url:
                add_chat(
                    "assistant",
                    "here’s the tonal snapshot—think of it like an EQ curve: above 0 dB is comparatively boosted, below is reduced.",
                    attachments=[{"type":"img","src":data_url,"alt":"Tonal balance (8-band)"}]
                )
    except Exception as e:
        add_chat("assistant", "analysis had a wobble—try again in a sec?")
        st.exception(e)

# -------------------------- Pre-clean corrective EQ (optional) --------------------------
master_input_path = in_path
if preclean and "analysis" in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    try:
        corr_eq8, corr_notches, corr_msg = llm_corrective_cleanup(
            api_key=api_key, analysis=st.session_state["analysis"], user_prompt=prompt_txt, model=llm_model
        )
        if corr_eq8:
            add_chat("assistant", "i can pre-clean the premaster—little mud control, little harshness relief. doing that now…")
            corrected_path = os.path.join(os.path.dirname(in_path), "premaster_corrected.wav")
            try:
                apply_corrective_eq(in_path, corrected_path, corr_eq8, corr_notches)
                master_input_path = corrected_path
                status_set("Analyzing…"); status_push("Analyze")
                st.session_state["analysis"] = analyze_audio(corrected_path)
                status_set("Analysis ready ✓"); status_push("Analysis ✓")
                add_chat("assistant", "pre-clean done. i re-analyzed the corrected premaster.")
                st.audio(corrected_path)
            except Exception as e:
                add_chat("assistant", "corrective render tripped—continuing without pre-clean.")
                st.exception(e)
        elif corr_msg:
            add_chat("assistant", f"skipping pre-clean: {corr_msg}")
    except Exception as e:
        add_chat("assistant", "the corrective planner stumbled—skipping cleanup.")
        st.exception(e)

# -------------------------- Generate adaptive masters (3 variations) --------------------------
variant_notes = [
    "Variant A – slightly darker, preserve transients, keep hats smooth.",
    "Variant B – neutral mid focus, tighten lows a touch, gentle presence.",
    "Variant C – a hair brighter, careful with harshness, avoid pumping.",
]

def get_sectioned_plans(variant_idx: int, seed: str, api_key: str):
    status_set("Rendering variations…"); status_push("Render")
    note = variant_notes[variant_idx]
    v_prompt = (st.session_state.get("vibe_prompt") or "").strip()
    v_prompt = f"{v_prompt}\n{note}\nVariation seed: {seed}. Make choices distinct from other variations; different EQ emphases, thresholds, and saturation rationale."

    plan, msg = llm_plan(
        analysis=st.session_state["analysis"],
        intent={"tone": float(st.session_state.get("tone", 0.0)),
                "dynamics": float(st.session_state.get("dynamics", 0.0)),
                "stereo": float(st.session_state.get("stereo", 0.0)),
                "character": float(st.session_state.get("character", 0.5))},
        user_prompt=v_prompt,
        model=st.session_state.get("llm_model", "gpt-4o-mini"),
        reference_txt=st.session_state.get("reference_txt", ""),
        reference_weight=float(st.session_state.get("reference_weight", 0.0)),
        api_key=api_key,
    )
    if not plan:
        return None, msg

    ok, errs = validate_plan(plan)
    if not ok:
        if isinstance(plan, dict) and "targets" in plan:
            v_plan, d_plan = derive_section_plans_from_single(plan)
            return {"verse": v_plan, "drop": d_plan}, f"validated with warnings: {errs}"
        return None, f"plan failed validation: {errs}"

    if "verse" in plan and "drop" in plan:
        return plan, "ok"
    else:
        v_plan, d_plan = derive_section_plans_from_single(plan)
        return {"verse": v_plan, "drop": d_plan}, "derived"

# Persist a few inputs for the planner (so get_sectioned_plans can read them)
st.session_state["reference_txt"]    = reference_txt
st.session_state["reference_weight"] = reference_weight
st.session_state["vibe_prompt"]      = prompt_txt
st.session_state["tone"]             = tone
st.session_state["dynamics"]         = dynamics
st.session_state["stereo"]           = stereo
st.session_state["character"]        = character

if gen_click and "analysis" in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    detected_notches = []
    try:
        if post_notch:
            detected_notches = detect_resonances_simple(master_input_path, max_notches=3, min_prom_db=3.0) or []
            if detected_notches:
                add_chat("assistant", f"i found a few narrow resonances—i’ll notch them gently: {detected_notches}")
            else:
                add_chat("assistant", "no strong resonances jumped out—going clean.")
    except Exception as e:
        add_chat("assistant", "resonance detection flaked—continuing without notches.")
        st.exception(e)
        detected_notches = []

    for i in range(3):
        st.markdown(f"### Variation {i+1}")
        try:
            status_set(f"Rendering V{i+1}…"); status_push(f"V{i+1}")
            seed = f"{uuid.uuid4().hex[:8]}-{i+1}"
            sectioned, status = get_sectioned_plans(i, seed, api_key)
            if not sectioned:
                add_chat("assistant", f"planning hiccup on variation {i+1}.")
                continue

            with st.expander(f"AI Plan – Variation {i+1} ({status})"):
                st.code(json.dumps(sectioned, indent=2))

            out_path = os.path.join(base_dir, f"master_ai_adaptive_v{i+1}.wav")
            render_adaptive_from_plans(
                master_input_path, out_path,
                sectioned["verse"], sectioned["drop"],
                notches=detected_notches
            )

            add_chat("assistant", f"variation {i+1} is ready—verse/drop tailored to your vibe.")
            status_set(f"V{i+1} printed ✓"); status_push(f"V{i+1} ✓")

            # Analyze the printed master & record for compare strip
            try:
                status_set("Analyzing…"); status_push("Analyze")
                a_out = analyze_audio(out_path)
                record_print(i+1, out_path, a_out)
                add_chat("assistant", f"variation {i+1} — quick check on the print:")
                post_analysis_dashboard_to_chat(out_path, a_out)
                status_set("Analysis ready ✓"); status_push("Analysis ✓")
            except Exception as e:
                st.exception(e)

            # Playback + download
            st.audio(out_path)
            with open(out_path, "rb") as f:
                st.download_button(f"Download Adaptive V{i+1}", f.read(), file_name=f"master_ai_adaptive_v{i+1}.wav")

        except Exception as e:
            add_chat("assistant", f"render failed on variation {i+1}.")
            st.exception(e)

    # === Compare strip (after the loop) ===
    cmp_html = render_compare_strip()
    if cmp_html:
        st.markdown(cmp_html, unsafe_allow_html=True)
        ensure_print_store()
        cols = st.columns(len(st.session_state["prints"]))
        for c, p in zip(cols, st.session_state["prints"]):
            with c:
                st.audio(p["path"])

    status_set("All variations complete ✓"); status_push("Done ✓")

# close shell
st.markdown('</div>', unsafe_allow_html=True)
