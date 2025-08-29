# app.py — Vale Mastering Assistant (DAW-style UI + Vale Chat)
import os, json, re, shutil, uuid
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit.components.v1 import html as components_html
import random, re


from utils import session_tmp_path
from dsp import (
    analyze_audio,
    render_adaptive_from_plans,
    derive_section_plans_from_single,
    detect_resonances_simple,
    BAND_NAMES,  # ensure this matches your dsp.BAND_NAMES
)
from ai import llm_plan
from diagnostics import validate_plan
from corrective import llm_corrective_cleanup, apply_corrective_eq

# ---------------- Page / Theme ----------------
st.set_page_config(page_title="Vale Mastering Assistant", page_icon="🎛️", layout="wide")
# after: import altair as alt
def _vale_altair_theme():
    return {
        "config": {
            "background": "transparent",
            "axis": {
                "labelColor": "#2A3542",
                "titleColor": "#2A3542",
                "gridColor": "#E6EAF1",
                "domainColor": "#DDE2EA",
            },
            "view": {"stroke": "transparent"},
            "line": {"strokeWidth": 2},
            "point": {"filled": True, "size": 60},
        }
    }
alt.themes.register("vale_light_cli", _vale_altair_theme)
alt.themes.enable("vale_light_cli")

# Inject DAW-like CSS (matte, flat, minimal)
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, header, footer { display:none !important; }

/* ---------- Palette (light terminal) ---------- */
:root{
  --bg: #F7F8FA;
  --panel: #FFFFFF;
  --ink: #2F3640;          /* readable but not heavy */
  --ink-dim: #6B7280;      /* secondary */
  --border: #E6EAF1;       /* very light stroke */
  --accent: #5EA2FF;       /* soft muted blue */
  --accent-ghost: rgba(94,162,255,0.10);
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", "Courier New", monospace;
  --radius: 4px;           /* hard corners */
}

/* Page canvas */
html, body, .block-container {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: var(--mono) !important;
  
}

body {margin: auto;
  padding: 10px;
  max-width: 90%;}

/* Avatar */
.vale-header { display:flex; align-items:center; gap:10px; margin-bottom:8px; }

.vale-nav {display: block; position absolute; top: 0px;}

.vale-avatar {
  position: relative;
  width: 40px; height: 40px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background:
    radial-gradient(120% 120% at 20% 20%, rgba(94,162,255,0.18), transparent 55%),
    linear-gradient(180deg, rgba(94,162,255,0.10), rgba(94,162,255,0.04));
  display: grid; place-items: center;
  font-family: var(--mono);
  font-weight: 700;
  color: var(--accent);
  letter-spacing: .6px;
}

/* subtle pulse ring (optional) */
.vale-avatar::after{
  content:"";
  position:absolute; inset:-3px;
  border-radius: inherit;
  box-shadow: 0 0 0 0 rgba(94,162,255,0.35);
  animation: valePulse 2.4s ease-out infinite;
}
@keyframes valePulse{
  0%   { box-shadow: 0 0 0 0 rgba(94,162,255,0.25); }
  60%  { box-shadow: 0 0 0 7px rgba(94,162,255,0.00); }
  100% { box-shadow: 0 0 0 0 rgba(94,162,255,0.00); }
}

/* small subtitle under heading */
.vale-sub { font-size: 11px; color: var(--ink-dim); margin-top: 2px; }


/* Headings */
h1,h2,h3 { color: var(--ink); margin: 0 0 6px 0; letter-spacing:.2px; }
h1 { font-size: 24px !important; }
h2 { font-size: 16px !important; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
  background:#FBFCFE;
  border:1px solid var(--border);
  border-radius: var(--radius);
  color:var(--ink);
}
.stTextInput input:focus, .stTextArea textarea:focus {
  outline: 2px solid var(--accent-ghost);
  border-color: var(--accent);
  box-shadow:none;
}

/* Buttons */
.stButton>button {
  background: #FFFFFF;
  border:1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px 12px;
  color: var(--ink);
}
.stButton>button:hover { border-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-ghost); }

/* ---------- Chat panel ---------- */
.vale-chat-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px;
}

#vale-chatbox {
  height: 60vh;             /* scrollable area height */
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: #FFFFFF;
  padding: 10px;
}

.vale-msg {
  border: 1px dashed var(--border);
  border-radius: 3px;       /* even harder corners inside */
  padding: 8px 10px;
  margin-bottom: 8px;
  background: #FFFFFF;
}

.vale-msg.assistant {
  border-color: rgba(94,162,255,0.4);
  background: linear-gradient(0deg, rgba(94,162,255,0.07), rgba(94,162,255,0.07));
}

.vale-role {
  font-size: 11px;
  color: var(--ink-dim);
  margin-bottom: 4px;
  letter-spacing: .3px;
}

.vale-input-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 6px;
  margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)



# Top bar (plugin-like)
st.markdown(
    "<div class='vale-nav'>"
    "<div style='display:flex; gap:10px; align-items:center;'>"
    "<div style='font-size:40px; font-weight:600;'>Vale</div>"
    "<div style='opacity:.65'>Mastering Engineer</div>"
    "</div>"
    "<div style='opacity:.6'>You create ideas, we make them real.</div>"
    "</div>",
    unsafe_allow_html=True
)

def _esc(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def init_chat_state():
    if "chat" not in st.session_state:
        st.session_state["chat"] = [
            {"role":"assistant","text":"Hey — drop a premaster and describe the vibe. I’ll adapt as we go."}
        ]

def add_chat(role: str, text: str):
    st.session_state["chat"].append({"role": role, "text": text})

def _autoscroll_script(div_id: str):
    # Inject this right after the messages so it pins to that container
    components_html(f"""
    <div></div>
    <script>
    (function(){{
      const ID = '{div_id}';
      function scrollBox(){{
        try {{
          const box = window.parent.document.getElementById(ID);
          if (box) box.scrollTop = box.scrollHeight;
        }} catch(e) {{}}
      }}
      // initial + retries
      scrollBox(); setTimeout(scrollBox, 60); setTimeout(scrollBox, 200);
      // observe new messages
      try {{
        const box = window.parent.document.getElementById(ID);
        if (box){{
          const obs = new MutationObserver(scrollBox);
          obs.observe(box, {{childList:true, subtree:true}});
        }}
      }} catch(e) {{}}
    }})();
    </script>
    """, height=0)

def render_chatbox(container, div_id: str = "vale-chatbox"):
    # Make sure chat history exists
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
        add_chat("assistant", vale_say("drop me a premaster when you’re ready—i’ll give it a quick once-over."))

    with container:
        # Header
        st.markdown("""
        <div class='vale-chat-panel'>
          <div class='vale-header'>
            <div class='vale-avatar'>V</div>
            <div>
              <h3 style="margin:0; font-weight:700;">Vale · Console</h3>
              <div class='vale-sub'>always on your team</div>
            </div>
          </div>
        """, unsafe_allow_html=True)

        # Messages area
        st.markdown(f"<div id='{div_id}'>", unsafe_allow_html=True)
        for m in st.session_state["chat"]:
            role = (m.get("role") or "assistant").lower()
            cls = "assistant" if role != "user" else "user"
            st.markdown(
                f"<div class='vale-msg {cls}'>"
                f"<div class='vale-role'>{role.upper()}</div>"
                f"<div>{_esc(m.get('text',''))}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-scroll hooks *inside the same container*
        _autoscroll_script(div_id)

        # Input row
        st.markdown("<div class='vale-input-row'>", unsafe_allow_html=True)
        user_txt = st.text_input("Message", key="vale_chat_input", label_visibility="collapsed",
                                 placeholder="Type to Vale…")
        send = st.button("Send")
        st.markdown("</div></div>", unsafe_allow_html=True)  # close vale-chat-panel

        if send and (user_txt or "").strip():
            add_chat("user", user_txt.strip())
            add_chat("assistant", vale_say("nice—i’ll fold that into the next pass."))
            st.experimental_rerun()



       


VALE_OPENERS = [
  "so…", "okay—", "alright,", "cool—", "nice—", "gotcha.", "heads up—",
]
VALE_HEDGES = [
  "kinda", "a touch", "a hair", "slightly", "pretty", "fairly",
]
def _condense_numbers(txt:str)->str:
    # turn things like "(~+2.3 dB)" into "+2 dB" for vibe
    return re.sub(r"([+\-]?\d+(\.\d+)?)\s*dB", lambda m: f"{round(float(m.group(1)))} dB", txt)

def vale_say(message: str) -> str:
    opener = random.choice(VALE_OPENERS)
    msg = message.strip()
    # remove stiff lead-ins
    msg = re.sub(r'^(analysis|result|note)[:\- ]+', '', msg, flags=re.I)
    msg = _condense_numbers(msg)
    return f"{opener} {msg[0].lower()}{msg[1:] if len(msg)>1 else ''}"




# ---------------- Session helpers (chat + base dir) ----------------
if "chat" not in st.session_state:
    st.session_state["chat"] = []  # list of dicts {role: 'user'|'assistant', 'text': str}

if "base_dir" not in st.session_state:
    st.session_state["base_dir"] = session_tmp_path()
base = st.session_state["base_dir"]


# ---------------- Utility ----------------
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
    """Turn 8-band % into EQ-like deviation curve centered at 0 dB."""
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
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Band:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Deviation (dB):Q", scale=alt.Scale(domain=[-12, 12]),
                title="Balance vs band average (dB)"),
        tooltip=["Band",
                 alt.Tooltip("Deviation (dB):Q", format=".2f"),
                 alt.Tooltip("Share (%):Q", format=".1f")],
    ).properties(height=240)

# ---------------- LAYOUT: two panes ----------------
left, right = st.columns([0.92, 1.48], gap="medium")

# ===== Left: Controls panel =====
with left:
    st.markdown("<h2>Session</h2>", unsafe_allow_html=True)
    upload_box = st.file_uploader(
        "Upload premaster (WAV/AIFF/FLAC — no limiter, ~−6 dBFS headroom)",
        type=["wav", "aiff", "aif", "flac"],
        label_visibility="collapsed",
    )
    st.markdown("<div class='vale-divider'></div>", unsafe_allow_html=True)

    st.markdown("<h2>Reference</h2>", unsafe_allow_html=True)
    reference_txt = st.text_input("", placeholder="Reference (e.g. Return of the Jaded – Soma)")
    reference_weight = st.slider("Reference weight", 0.0, 1.0, 0.9 if reference_txt else 0.0, 0.1)
    st.markdown("<div class='vale-divider'></div>", unsafe_allow_html=True)

    st.markdown("<h2>Intent</h2>", unsafe_allow_html=True)
    prompt_txt = st.text_area("", placeholder="Describe the vibe (e.g., dark, preserve dynamics, tape weight).")
    c1, c2 = st.columns(2)
    with c1:
        tone = st.slider("Tone: Dark ↔ Bright", -1.0, 1.0, 0.0, 0.1)
        dynamics = st.slider("Dynamics: Preserve ↔ Loud", -1.0, 1.0, 0.0, 0.1)
    with c2:
        stereo = st.slider("Stereo: Narrow ↔ Wide", -1.0, 1.0, 0.0, 0.1)
        character = st.slider("Character: Clean ↔ Tape", -1.0, 1.0, 0.5, 0.1)

    intent = {
        "tone": float(tone), "dynamics": float(dynamics),
        "stereo": float(stereo), "character": float(character)
    }

    st.markdown("<div class='vale-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2>Options</h2>", unsafe_allow_html=True)
    preclean = st.checkbox("Pre-clean (auto corrective EQ) before mastering", value=True)
    post_notch = st.checkbox("Add up to 3 safety notches before render", value=True)

    st.markdown("<div class='vale-divider'></div>", unsafe_allow_html=True)
    cA, cB, cC = st.columns(3)
    with cA:
        analyze_click = st.button("Analyze", use_container_width=True)
    with cB:
        gen_click = st.button("Generate 3 Adaptive Masters", use_container_width=True)
    with cC:
        if st.button("Reset", use_container_width=True):
            for k in ("uploaded_name", "analysis", "in_path", "base_dir", "chat"):
                st.session_state.pop(k, None)
            st.experimental_rerun()

# ===== Handle upload/write =====
if upload_box is None and "in_path" not in st.session_state:
    with right:
        add_chat("assistant", "Upload a premaster to begin. I’ll analyze it and suggest musical moves.")
    st.stop()

if upload_box is not None:
    if ("uploaded_name" not in st.session_state) or (upload_box.name != st.session_state["uploaded_name"]):
        st.session_state["uploaded_name"] = _safe_name(upload_box.name)
        in_path = os.path.join(base, st.session_state["uploaded_name"])
        try:
            upload_box.seek(0)
            with open(in_path, "wb") as f:
                shutil.copyfileobj(upload_box, f, length=1 * 1024 * 1024)  # 1MB chunks
            upload_box.seek(0)
        except Exception as e:
            with right:
                add_chat("assistant", "I couldn't write your file to disk.")
               
                st.exception(e)
            st.stop()
        st.session_state["in_path"] = in_path
        st.session_state.pop("analysis", None)

# Always use the path from session hereafter
in_path = st.session_state["in_path"]

# ===== Right: Vale Chat & Visuals =====
with right:
    st.markdown("<h2>Vale</h2>", unsafe_allow_html=True)

             # anchor
    render_chatbox(container) 

    # ANALYZE
    if analyze_click or "analysis" not in st.session_state:
        try:
            st.session_state["analysis"] = analyze_audio(in_path)
            a = st.session_state["analysis"]
            add_chat("assistant",
                     f"Analyzed your track.\n"
                     f"- Loudness (integrated LUFS): **{a['lufs_integrated']:.2f}**\n"
                     f"- True peak (est.): **{a['true_peak_dbfs_est']:.2f} dBFS**\n"
                     f"- Tonal balance is bass-forward. I’ll show you the curve below.")
        except Exception as e:
            add_chat("assistant", "Analysis failed.")
            st.exception(e)
            st.stop()

    # Tonal balance visualizer
    analysis = st.session_state["analysis"]
    bands = analysis.get("bands_pct_8") or {}
    if bands and sum(float(bands.get(k, 0.0)) for k in BAND_ORDER) > 1e-9:
        df = compute_deviation_db_from_bands(bands)
        st.altair_chart(tonal_chart(df), use_container_width=True)
        st.caption("Above 0 dB = comparatively boosted; below 0 dB = comparatively reduced. "
                   "This visualizes balance (like an EQ curve), not absolute EQ settings.")
    else:
        st.caption("No 8-band data available for the visualizer.")

# ---------------- Pre-clean corrective EQ (optional) ----------------
master_input_path = in_path
if preclean:
    with right:
        api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        corr_eq8, corr_notches, corr_msg = llm_corrective_cleanup(
            api_key=api_key,
            analysis=analysis,
            user_prompt=prompt_txt,
            model=st.sidebar.text_input("OpenAI model", value="gpt-4o-mini", key="model_hidden")  # stable key
        )
        if corr_eq8:
            add_chat("assistant", "I can pre-clean the premaster: subtle mud control / harshness relief. Rendering…")
            
            corrected_path = os.path.join(os.path.dirname(in_path), "premaster_corrected.wav")
            try:
                apply_corrective_eq(in_path, corrected_path, corr_eq8, corr_notches)
                master_input_path = corrected_path
                # Re-analyze the corrected file so the mastering plan “hears” it
                analysis = analyze_audio(corrected_path)
                st.session_state["analysis"] = analysis
                add_chat("assistant", "Pre-clean complete. I re-analyzed the corrected premaster.")
          
                # Optional preview
                st.audio(corrected_path)
            except Exception as e:
                add_chat("assistant", "Corrective render failed; continuing without pre-clean.")
              
                st.exception(e)
        else:
            if corr_msg:
                add_chat("assistant", f"ℹ️ Skipping pre-clean: {corr_msg}")
               

# ---------------- Generate adaptive masters (3 variations) ----------------
variant_notes = [
    "Variant A – slightly darker, preserve transients, keep hats smooth.",
    "Variant B – neutral mid focus, tighten lows a touch, gentle presence.",
    "Variant C – a hair brighter, careful with harshness, avoid pumping.",
]

def get_sectioned_plans(variant_idx: int, seed: str, api_key: str):
    """Ask the LLM for one plan (may be single or sectioned); normalize to verse/drop."""
    note = variant_notes[variant_idx]
    v_prompt = (prompt_txt or "").strip()
    v_prompt = f"{v_prompt}\n{note}\nVariation seed: {seed}. Make choices distinct from other variations; different EQ emphases, thresholds, and saturation rationale."

    plan, msg = llm_plan(
        analysis=analysis,
        intent=intent,
        user_prompt=v_prompt,
        model=st.session_state.get("model_hidden", "gpt-4o-mini"),
        reference_txt=reference_txt,
        reference_weight=reference_weight,
        api_key=api_key,
    )
    if not plan:
        return None, msg

    ok, errs = validate_plan(plan)
    if not ok:
        # Try to normalize single → sectioned anyway; still report validation errors
        if isinstance(plan, dict) and "targets" in plan:
            v_plan, d_plan = derive_section_plans_from_single(plan)
            return {"verse": v_plan, "drop": d_plan}, f"validated with warnings: {errs}"
        return None, f"plan failed validation: {errs}"

    if "verse" in plan and "drop" in plan:
        return plan, "ok"
    else:
        v_plan, d_plan = derive_section_plans_from_single(plan)
        return {"verse": v_plan, "drop": d_plan}, "derived"

if gen_click:
    with right:
        api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        # Optional: detect up to 3 safety notches AFTER corrective cleanup and BEFORE mastering
        detected_notches = []
        if post_notch:
            try:
                detected_notches = detect_resonances_simple(master_input_path, max_notches=3, min_prom_db=3.0)
                if detected_notches:
                    add_chat("assistant", f"I found a few narrow resonances. I’ll notch them subtly: {detected_notches}")
                else:
                    add_chat("assistant", "No strong resonances detected — proceeding clean.")
            except Exception:
                add_chat("assistant", "Resonance detection had an issue; continuing without notches.")
     
        for i in range(3):
            st.markdown(f"<h2>Variation {i+1}</h2>", unsafe_allow_html=True)
            try:
                seed = f"{uuid.uuid4().hex[:8]}-{i+1}"
                sectioned, status = get_sectioned_plans(i, seed, api_key)
                if not sectioned:
                    add_chat("assistant", f"LLM plan unavailable for Variation {i+1}.")
                 
                    continue

                with st.expander(f"AI Plan – Variation {i+1} ({status})"):
                    st.code(json.dumps(sectioned, indent=2))

                out_path = os.path.join(base, f"master_ai_adaptive_v{i+1}.wav")
                render_adaptive_from_plans(
                    master_input_path,
                    out_path,
                    sectioned["verse"],
                    sectioned["drop"],
                    notches=detected_notches
                )
                add_chat("assistant", f"🎧 Variation {i+1} is ready. I adjusted verse/drop separately to fit the vibe.")
           
                st.audio(out_path)
                with open(out_path, "rb") as f:
                    st.download_button(
                        f"Download Adaptive V{i+1}",
                        f.read(),
                        file_name=f"master_ai_adaptive_v{i+1}.wav"
                    )
            except Exception as e:
                add_chat("assistant", f"Render failed for Variation {i+1}.")
            
                st.exception(e)

# ---------------- Debug (collapsible) ----------------
with st.expander("Debug"):
    st.write({
        "in_path_exists": os.path.exists(in_path),
        "in_path": in_path,
        "base_dir": base,
        "reference_txt": reference_txt,
        "reference_weight": reference_weight,
        "chat_len": len(st.session_state["chat"]),
    })
