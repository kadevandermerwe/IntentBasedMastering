# app.py
import os, json
import streamlit as st
from utils import session_tmp_path
from dsp import analyze_audio, render_variant, render_adaptive_from_plans
from ai import llm_plan

# --- in app.py (UI controls) ---
reference_txt = st.text_input("Reference track/artist (optional)", placeholder="e.g., Return of the Jaded – Soma")
reference_weight = st.slider("Reference weight", 0.0, 1.0, 0.0 if not reference_txt else 0.9, 0.1)


st.set_page_config(page_title="Vale Mastering Assistant", page_icon="🎛️", layout="centered")
st.title("🎛️ Vale Mastering Assistant — AI + Adaptive")

# ---- Sidebar: LLM settings
use_llm   = st.sidebar.checkbox("Use OpenAI LLM planner (required)", value=True, disabled=True)
llm_model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")
st.sidebar.caption("Set OPENAI_API_KEY in Streamlit Cloud → Settings → Secrets.")

# ---- Intent controls
prompt_txt = st.text_area("Intent / Reference (e.g. “dark, preserve dynamics, tape; like Soma by Return of the Jaded”.)")
col1, col2 = st.columns(2)
with col1:
    tone = st.slider("Tone: Dark ↔ Bright", -1.0, 1.0, 0.0, 0.1)
    dynamics = st.slider("Dynamics: Preserve ↔ Loud", -1.0, 1.0, 0.0, 0.1)
with col2:
    stereo = st.slider("Stereo: Narrow ↔ Wide", -1.0, 1.0, 0.0, 0.1)
    character = st.slider("Character: Clean ↔ Tape", -1.0, 1.0, 0.5, 0.1)

intent = {"tone": float(tone), "dynamics": float(dynamics), "stereo": float(stereo), "character": float(character)}
st.caption(f"Resolved intent → tone {tone:+.2f} • dynamics {dynamics:+.2f} • stereo {stereo:+.2f} • character {character:+.2f}")

# ---- Adaptive toggle
adaptive = st.sidebar.checkbox("Adaptive per-section rendering (detect drops/verses)", value=True)

# ---- Upload
uploaded = st.file_uploader("Upload premaster (WAV/AIFF/FLAC — no limiter, ~−6 dBFS headroom)", type=["wav","aiff","aif","flac"])

# ---- Persist upload in session
if uploaded is not None:
    if ("uploaded_name" not in st.session_state
        or uploaded.name != st.session_state.get("uploaded_name")
        or "uploaded_bytes" not in st.session_state):
        st.session_state["uploaded_name"] = uploaded.name
        st.session_state["uploaded_bytes"] = uploaded.read()
        for k in ("analysis", "in_path"):
            st.session_state.pop(k, None)

if "uploaded_bytes" not in st.session_state:
    st.info("Upload a premaster to begin.")
    st.stop()

# ---- Materialize to /tmp per session
base = session_tmp_path()
in_path = os.path.join(base, st.session_state["uploaded_name"].replace(" ", "_"))
with open(in_path, "wb") as f:
    f.write(st.session_state["uploaded_bytes"])
st.session_state["in_path"] = in_path

# ---- Controls
cols = st.columns(3)
with cols[0]:
    analyze_click = st.button("Analyze file")
with cols[1]:
    gen_click = st.button("Generate Master(s)")
with cols[2]:
    if st.button("Reset file"):
        for k in ("uploaded_name","uploaded_bytes","analysis","in_path"):
            st.session_state.pop(k, None)
        st.experimental_rerun()

# ---- Analysis (run once or on demand)
if analyze_click or "analysis" not in st.session_state:
    try:
        st.session_state["analysis"] = analyze_audio(in_path)
    except Exception as e:
        st.error("❌ Analysis failed.")
        st.exception(e)
        st.stop()

analysis = st.session_state["analysis"]
st.subheader("Analysis")
st.json(analysis)
st.subheader("Tonal balance (8 bands)")
st.json(analysis.get("bands_pct_8", {}))

# ---- LLM Plan (required)
api_key = st.secrets.get("OPENAI_API_KEY", "")
plan = None; plan_msg = None
plan, msg = llm_plan(analysis, intent, prompt_txt, llm_model, reference_txt, reference_weight)
if not plan:
    
    st.error(f"LLM plan unavailable. {plan_msg}  \nSet your OPENAI_API_KEY and try again.")
    st.stop()

st.subheader("AI Plan")
st.caption(
    f"LLM plan → LUFS {plan['targets']['lufs_i']:.1f}, TP {plan['targets']['true_peak_db']:.1f} | "
    f"Ref bias: {reference_weight:.2f} | EQ8: sub {plan['eq8']['sub']:+.1f} … air {plan['eq8']['air']:+.1f}"
)

st.code(json.dumps(plan, indent=2))

# ---- Generate masters
if gen_click:
    try:
        # If sectioned, render adaptive; else render single chain
        if "verse" in plan and "drop" in plan:
            out_ad = os.path.join(base, "master_ai_adaptive.wav")
            render_adaptive_from_plans(in_path, out_ad, plan["verse"], plan["drop"])
            st.markdown("### 🎯 AI Adaptive (verse/drop)")
            st.audio(out_ad)
            with open(out_ad, "rb") as f:
                st.download_button("Download AI Adaptive", f.read(), file_name="master_ai_adaptive.wav")
        else:
            out_ai = os.path.join(base, "master_ai_full.wav")
            from dsp import build_fg_from_plan, render_variant
            fg = build_fg_from_plan(plan)
            render_variant(in_path, out_ai, fg)
            st.markdown("### 🧩 AI Master (Full)")
            st.audio(out_ai)
            with open(out_ai, "rb") as f:
                st.download_button("Download AI Master (Full)", f.read(), file_name="master_ai_full.wav")
    except Exception as e:
        st.error("Render failed.")
        st.exception(e)

with st.expander("Debug"):
    st.write({
        "in_path_exists": os.path.exists(in_path),
        "in_path": in_path,
        "bytes_len": len(st.session_state.get("uploaded_bytes", b""))
    })
