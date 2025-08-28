# app.py
import os, json
import streamlit as st

from utils import session_tmp_path
from dsp import (
    analyze_audio,
    render_variant,
    render_adaptive_from_plans,
    build_fg_from_plan,
    derive_section_plans_from_single,
)
from ai import llm_plan
from corrective import llm_corrective_eq8, apply_corrective_eq
from diagnostics import validate_plan  # start using diagnostics!
from corrective import llm_corrective_cleanup, apply_corrective_eq

# --- Page ---
st.set_page_config(page_title="Vale Mastering Assistant", page_icon="🎛️", layout="centered")
st.title("🎛️ Vale Mastering Assistant — AI + Adaptive")

# --- Reference controls (used by the LLM) ---
reference_txt = st.text_input(
    "Reference track/artist (optional)",
    placeholder="e.g., Return of the Jaded – Soma",
)
reference_weight = st.slider(
    "Reference weight",
    0.0, 1.0,
    0.9 if reference_txt else 0.0, 0.1,
)

# --- Sidebar: LLM settings ---
use_llm   = st.sidebar.checkbox("Use OpenAI LLM planner (required)", value=True, disabled=True)
llm_model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")
st.sidebar.caption("Set OPENAI_API_KEY in Streamlit Cloud → Settings → Secrets.")

# --- Intent controls ---
prompt_txt = st.text_area(
    "Intent / Reference (e.g. “dark, preserve dynamics, tape; like Soma by Return of the Jaded”.)"
)
col1, col2 = st.columns(2)
with col1:
    tone = st.slider("Tone: Dark ↔ Bright", -1.0, 1.0, 0.0, 0.1)
    dynamics = st.slider("Dynamics: Preserve ↔ Loud", -1.0, 1.0, 0.0, 0.1)
with col2:
    stereo = st.slider("Stereo: Narrow ↔ Wide", -1.0, 1.0, 0.0, 0.1)
    character = st.slider("Character: Clean ↔ Tape", -1.0, 1.0, 0.5, 0.1)

intent = {
    "tone": float(tone),
    "dynamics": float(dynamics),
    "stereo": float(stereo),
    "character": float(character),
}
st.caption(
    f"Resolved intent → tone {tone:+.2f} • dynamics {dynamics:+.2f} • "
    f"stereo {stereo:+.2f} • character {character:+.2f}"
)

# --- Adaptive toggle ---
adaptive = st.sidebar.checkbox("Adaptive per-section rendering (detect drops/verses)", value=True)

preclean = st.sidebar.checkbox("Pre-master corrective cleanup (EQ)", value=True)

# --- Upload ---
uploaded = st.file_uploader(
    "Upload premaster (WAV/AIFF/FLAC — no limiter, ~−6 dBFS headroom)",
    type=["wav","aiff","aif","flac"]
)

# --- Persist upload in session ---
if uploaded is not None:
    if (
        "uploaded_name" not in st.session_state
        or uploaded.name != st.session_state.get("uploaded_name")
        or "uploaded_bytes" not in st.session_state
    ):
        st.session_state["uploaded_name"] = uploaded.name
        st.session_state["uploaded_bytes"] = uploaded.read()
        for k in ("analysis", "in_path"):
            st.session_state.pop(k, None)

if "uploaded_bytes" not in st.session_state:
    st.info("Upload a premaster to begin.")
    st.stop()

# --- Materialize to /tmp per session ---
base = session_tmp_path()
in_path = os.path.join(base, st.session_state["uploaded_name"].replace(" ", "_"))
with open(in_path, "wb") as f:
    f.write(st.session_state["uploaded_bytes"])
st.session_state["in_path"] = in_path

# --- Controls ---
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

# --- Analysis (run once or on demand) ---
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

#---Pre-clean Corrective eq--#
master_input_path = in_path
if preclean:
    api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    corr_eq8, corr_notches, corr_msg = llm_corrective_cleanup(
        api_key=api_key,
        analysis=analysis,
        user_prompt=prompt_txt,
        model=llm_model,
    )
    if corr_eq8:
        st.subheader("Corrective EQ (pre-master)")
        st.json({"eq8": corr_eq8, "notches": corr_notches or []})
        corrected_path = os.path.join(os.path.dirname(in_path), "premaster_corrected.wav")
        try:
            apply_corrective_eq(in_path, corrected_path, corr_eq8, corr_notches)
            st.audio(corrected_path)
            # Re-analyze corrected file
            analysis = analyze_audio(corrected_path)
            st.caption("Re-analysis after corrective cleanup")
            st.json(analysis)
            master_input_path = corrected_path
        except Exception as e:
            st.error("Corrective render failed.")
            st.exception(e)
    else:
        st.warning(f"No corrective plan: {corr_msg}")

# --- LLM Plan (required) ---
api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

# (Temporary diagnostics – remove later if you want)
st.sidebar.write(f"🧪 Secrets has key: {'yes' if st.secrets.get('OPENAI_API_KEY') else 'no'}")
st.sidebar.write(f"🧪 Env has key: {'yes' if os.environ.get('OPENAI_API_KEY') else 'no'}")
st.sidebar.write(f"🧪 Using key length: {len(api_key) if api_key else 0}")

# BUT: pass the updated 'analysis' and use master_input_path later for rendering
plan, msg = llm_plan(
    analysis=analysis,
    intent=intent,
    user_prompt=prompt_txt,
    model=llm_model,
    reference_txt=reference_txt,
    reference_weight=reference_weight,
    api_key=api_key,
)

# Normalize to verse/drop immediately (so adaptive can use it)
verse_plan = drop_plan = None
if plan and isinstance(plan, dict):
    if "verse" in plan and "drop" in plan:
        verse_plan, drop_plan = plan["verse"], plan["drop"]
    else:
        verse_plan, drop_plan = derive_section_plans_from_single(plan)

if not plan:
    st.error(f"LLM plan unavailable. {msg}\nSet your OPENAI_API_KEY and try again.")
else:
    st.success("LLM plan OK")
    st.subheader("AI Plan")
    st.code(json.dumps(plan, indent=2))


ok, errs = validate_plan(plan) if plan else (False, ["no plan"])
if not ok:
    st.error("LLM plan failed validation:")
    for e in errs:
        st.write(f"• {e}")
    st.stop()
    
# --- Generate masters ---
if gen_click:
    try:
        # Always provide a full-pass AI master (use the single plan if present,
        # otherwise use verse_plan as the "full" chain).
        full_plan = plan if (plan and "targets" in plan) else verse_plan
        if full_plan:
            out_ai = os.path.join(base, "master_ai_full.wav")
            fg = build_fg_from_plan(full_plan)
            render_variant(master_input_path, out_ai, fg)
            st.markdown("### 🧩 AI Master (Full)")
            st.audio(out_ai)
            with open(out_ai, "rb") as f:
                st.download_button("Download AI Master (Full)", f.read(), file_name="master_ai_full.wav")

        # Adaptive per-section master (only if checkbox is on and both plans exist)
        if adaptive and verse_plan and drop_plan:
            out_ad = os.path.join(base, "master_ai_adaptive.wav")
            render_adaptive_from_plans(master_input_path, out_ad, verse_plan, drop_plan)
            st.markdown("### 🎯 AI Adaptive (verse/drop)")
            st.audio(out_ad)
            with open(out_ad, "rb") as f:
                st.download_button("Download AI Adaptive", f.read(), file_name="master_ai_adaptive.wav")
        elif adaptive and not (verse_plan and drop_plan):
            st.caption("Adaptive was requested but plans were missing; skipped adaptive render.")

    except Exception as e:
        st.error("Render failed.")
        st.exception(e)

# --- Debug ---
with st.expander("Debug"):
    st.write({
        "in_path_exists": os.path.exists(in_path),
        "in_path": in_path,
        "bytes_len": len(st.session_state.get("uploaded_bytes", b"")),
        "adaptive": adaptive,
        "have_plan": bool(plan),
        "have_verse_drop": bool(verse_plan and drop_plan),
        "reference_txt": reference_txt,
        "reference_weight": reference_weight,
    })
