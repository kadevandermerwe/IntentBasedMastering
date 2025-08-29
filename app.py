# app.py — Vale Mastering Assistant (Console-first UI)
import os, json, re, shutil, uuid, base64
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import io, base64
import matplotlib.pyplot as plt


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

# ---------- assets ----------
def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_PATH   = "imgs/2.png"
AVATAR_PATH = "imgs/shadow.png"
LOGO_B64    = img_to_base64(LOGO_PATH)
AVATAR_B64  = img_to_base64(AVATAR_PATH)

# ---------- page ----------
st.set_page_config(page_title="Vale Mastering Assistant", page_icon=LOGO_PATH, layout="wide")

# Kill Streamlit chrome/padding
st.markdown("""
<style>
header[data-testid="stHeader"], div[data-testid="stToolbar"], #MainMenu, footer { display:none !important; }
[data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"]>.main, main .block-container { padding-top:0 !important; }
.stApp { margin-top:0 !important; }
html, body { margin:0 !important; padding:0 !important; background:#FAF9F6; }

/* Nav */
.vale-nav { position:fixed; top:0; left:0; right:0; z-index:999; display:flex; justify-content:center; background:#fff;
  border-bottom:1px solid rgba(0,0,0,.08); padding:10px 0; }
.vale-shell { width:70vw; margin:84px auto 24px; }

/* Typography + controls */
:root{
  --panel:#FFFFFF; --ink:#2F3640; --ink-dim:#6B7280; --border:#E6EAF1; --accent:#5EA2FF; --accent-ghost:rgba(94,162,255,.10);
  --mono:ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", "Courier New", monospace; --radius:4px;
}
html, body { color:var(--ink); font-family:var(--mono); }
.stTextInput input, .stTextArea textarea { background:#FBFCFE; border:1px solid var(--border); border-radius:var(--radius); }
.stTextInput input:focus, .stTextArea textarea:focus { outline:2px solid var(--accent-ghost); border-color:var(--accent); box-shadow:none; }
.stButton>button { background:#fff; border:1px solid var(--border); border-radius:var(--radius); padding:8px 12px; }
.stButton>button:hover { border-color:var(--accent); box-shadow:0 0 0 4px var(--accent-ghost); }

/* Cards */
.vale-card { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:12px; }

/* Action row */
.vale-actions { display:flex; gap:8px; align-items:center; }
</style>
""", unsafe_allow_html=True)

# Navbar (full width)
st.markdown(f"""
<div class="vale-nav">
  <img src="data:image/png;base64,{LOGO_B64}" alt="Vale" height="54">
</div>
""", unsafe_allow_html=True)

# Central shell (70% width)
st.markdown('<div class="vale-shell">', unsafe_allow_html=True)

# ---------- session state ----------
st.session_state.setdefault("chat", [])
st.session_state.setdefault("base_dir", session_tmp_path())

base_dir = st.session_state["base_dir"]
llm_model = st.session_state.get("llm_model", "gpt-4o-mini")

# ---------- helper ----------
def _safe_name(name: str) -> str:
    base_name = os.path.basename(name or "upload")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)

BAND_ORDER  = ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]
BAND_LABELS = {
    "sub":       "Sub (20–60 Hz)",
    "low_bass":  "Low Bass (60–120 Hz)",
    "high_bass": "High Bass (120–250 Hz)",
    "low_mids":  "Low Mids (250–500 Hz)",
    "mids":      "Mids (500 Hz–3.5 kHz)",
    "high_mids": "High Mids (3.5–8 kHz)",
    "highs":     "Highs (8–10 kHz)",
    "air":       "Air (10–20 kHz)",
}

def compute_deviation_db_from_bands(bands_pct_8: dict) -> pd.DataFrame:
    vals = [max(1e-12, float(bands_pct_8.get(b, 0.0))) for b in BAND_ORDER]
    total = sum(vals) or 1.0
    shares = [v / total for v in vals]
    gmean = float(np.exp(np.mean(np.log(shares))))
    dev_db = [20.0 * np.log10(s / gmean) for s in shares]
    return pd.DataFrame({
        "BandKey": BAND_ORDER,
        "Band": [BAND_LABELS[b] for b in BAND_ORDER],
        "Deviation (dB)": dev_db,
        "Share (%)": [v * 100.0 for v in shares],
    })

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
        y=alt.Y("Deviation (dB):Q", scale=alt.Scale(domain=[-12, 12]),
                title="Balance vs band average (dB)"),
        tooltip=["Band", alt.Tooltip("Deviation (dB):Q", format=".2f"), alt.Tooltip("Share (%):Q", format=".1f")],
    ).properties(height=240)

# ---------- Vale Console (chat first) ----------
chat_box = st.container()
render_chat(chat_box, state_key="chat", height=420, avatar_img_b64=AVATAR_B64)

# Action row under chat
act_col1, act_col2, spacer = st.columns([0.22, 0.30, 0.48])
with act_col1:
    analyze_click = st.button("Analyze")
with act_col2:
    gen_click = st.button("Render 3 Variations")

# ---------- Session Tools (drawer) ----------
with st.expander("Session Tools", expanded=True):
    st.markdown('<div class="vale-card">', unsafe_allow_html=True)
    upload_box = st.file_uploader(
        "Upload premaster (WAV/AIFF/FLAC — no limiter, ~−6 dBFS headroom)",
        type=["wav", "aiff", "aif", "flac"]
    )

    # Reference + intent
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

    # Options
    preclean = st.checkbox("Pre-clean (auto corrective EQ) before mastering", value=True)
    post_notch = st.checkbox("Add up to 3 safety notches before render", value=True)

    # Small model input if you want to tweak
    llm_model = st.text_input("OpenAI model", value=llm_model, help="e.g. gpt-4o-mini")
    st.session_state["llm_model"] = llm_model
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- file persistence ----------
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

# ---------- ANALYZE ----------
if analyze_click or "analysis" not in st.session_state:
    try:
        st.session_state["analysis"] = analyze_audio(in_path)
        a = st.session_state["analysis"]
        add_chat("assistant",
                 f"analyzed your track—lufs: **{a['lufs_integrated']:.2f}**, "
                 f"true peak (est.): **{a['true_peak_dbfs_est']:.2f} dBFS**. "
                 f"the balance leans bass-forward; here’s the curve below.")
    except Exception as e:
        add_chat("assistant", "analysis had a wobble—try again in a sec?")
        st.exception(e)

# Tonal visual (under chat)
# Tonal visual (now injected into the chat as an attachment)
if "analysis" in st.session_state:
    bands = st.session_state["analysis"].get("bands_pct_8") or {}
    if bands and sum(float(bands.get(k, 0.0)) for k in ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]) > 1e-9:
        data_url = tonal_png_data_url(bands)
        if data_url:
            from chat_ui import add_chat
            add_chat(
                "assistant",
                "here’s the tonal snapshot—think of it like an EQ curve: above 0 dB is comparatively boosted, below is reduced.",
                attachments=[{"type":"img","src":data_url,"alt":"Tonal balance (8-band)"}]
            )


# ---------- Pre-clean corrective EQ (optional) ----------
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
                # Re-analyze the corrected file
                st.session_state["analysis"] = analyze_audio(corrected_path)
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

# ---------- Generate adaptive masters (3 variations) ----------
variant_notes = [
    "Variant A – slightly darker, preserve transients, keep hats smooth.",
    "Variant B – neutral mid focus, tighten lows a touch, gentle presence.",
    "Variant C – a hair brighter, careful with harshness, avoid pumping.",
]

def get_sectioned_plans(variant_idx: int, seed: str, api_key: str):
    note = variant_notes[variant_idx]
    v_prompt = (st.session_state.get("vibe_prompt") or "").strip()
    # use UI prompt but ensure uniqueness per variant
    v_prompt = f"{(v_prompt or '').strip()}\n{note}\nVariation seed: {seed}. Make choices distinct from other variations; different EQ emphases, thresholds, and saturation rationale."

    plan, msg = llm_plan(
        analysis=st.session_state["analysis"],
        intent={"tone": float(st.session_state.get("tone", 0.0)),
                "dynamics": float(st.session_state.get("dynamics", 0.0)),
                "stereo": float(st.session_state.get("stereo", 0.0)),
                "character": float(st.session_state.get("character", 0.5))},
        user_prompt=v_prompt,
        model=llm_model,
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

# Persist a few inputs for the planner
st.session_state["reference_txt"]   = reference_txt
st.session_state["reference_weight"] = reference_weight
st.session_state["vibe_prompt"]     = prompt_txt
st.session_state["tone"]            = tone
st.session_state["dynamics"]        = dynamics
st.session_state["stereo"]          = stereo
st.session_state["character"]       = character

if gen_click and "analysis" in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    # Optional: detect safety notches AFTER corrective cleanup / BEFORE mastering
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
            add_chat(
                "assistant",
                f"variation {i+1} is ready—verse/drop tailored to your vibe.",
                attachments=[{
                    "type": "html",
                    "html": f"""
                    <div style='border:1px solid #e6eaf0;border-radius:6px;padding:8px'>
                    <div style='font-size:12px;color:#6b7280;margin-bottom:6px'>Result • V{i+1}</div>
                    <div style='display:flex;gap:12px;flex-wrap:wrap;font-size:13px'>
                        <span>LUFS target: {sectioned['verse'].get('targets',{}).get('lufs_i','—')}</span>
                        <span>TP: {sectioned['verse'].get('targets',{}).get('true_peak_db','—')} dBFS</span>
                    </div>
                    </div>
                    """
                }]
            )

        except Exception as e:
            add_chat("assistant", f"render failed on variation {i+1}.")
            st.exception(e)

def tonal_png_data_url(bands_pct_8: dict) -> str | None:
    """Make a small EQ-like line chart PNG and return a data URL for embedding in chat."""
    try:
        BAND_ORDER  = ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]
        vals = [max(1e-12, float(bands_pct_8.get(b, 0.0))) for b in BAND_ORDER]
        total = sum(vals) or 1.0
        shares = [v/total for v in vals]
        gmean = float(np.exp(np.mean(np.log(shares))))
        dev_db = [20.0*np.log10(s/gmean) for s in shares]

        # pretty labels
        x_labels = ["Sub", "Lo Bass", "Hi Bass", "Lo Mids", "Mids", "Hi Mids", "Highs", "Air"]

        fig, ax = plt.subplots(figsize=(6.0, 2.2), dpi=180)
        ax.plot(range(len(dev_db)), dev_db, marker="o")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylim(-12, 12)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Δ vs band avg (dB)", fontsize=8)
        for spine in ("top","right"):
            ax.spines[spine].set_visible(False)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return None


# close shell
st.markdown('</div>', unsafe_allow_html=True)
