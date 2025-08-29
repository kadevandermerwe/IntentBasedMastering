# app.py
import os, json, re, shutil, uuid
import pandas as pd
import altair as alt
import streamlit as st

from utils import session_tmp_path
from dsp import (
    analyze_audio,
    render_adaptive_from_plans,
    derive_section_plans_from_single,
    detect_resonances_simple,
    BAND_NAMES,  # for ordered tonal-balance plotting
)
from ai import llm_plan
from diagnostics import validate_plan
from corrective import llm_corrective_cleanup, apply_corrective_eq

# ---------------- Page ----------------
st.set_page_config(page_title="Vale Mastering Assistant", page_icon="🎛️", layout="centered")
st.title("🎛️ Vale Mastering Assistant — AI Adaptive (3 Variations)")

# ---------------- Reference controls (used by the LLM) ----------------
reference_txt = st.text_input(
    "Reference track/artist (optional)",
    placeholder="e.g., Return of the Jaded – Soma",
)
reference_weight = st.slider(
    "Reference weight",
    0.0, 1.0,
    0.9 if reference_txt else 0.0, 0.1,
    help="How strongly the AI should bias toward the reference’s typical tonal/dynamic vibe.",
)

# ---------------- Sidebar: LLM settings ----------------
st.sidebar.checkbox("Use OpenAI LLM planner (required)", value=True, disabled=True)
llm_model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")
st.sidebar.caption("Set OPENAI_API_KEY in Streamlit Cloud → Settings → Secrets.")

# ---------------- Toggles ----------------
adaptive = st.sidebar.checkbox("Adaptive per-section rendering (verse/drop)", value=True, disabled=True)
preclean = st.sidebar.checkbox("Pre-master corrective cleanup (EQ)", value=True)
post_notch = st.sidebar.checkbox("Add up to 3 safety notches before render", value=True)

# ---------------- Intent controls ----------------
prompt_txt = st.text_area(
    "Intent (describe the vibe / goals)",
    placeholder="e.g. dark, preserve dynamics, tape weight; keep hats smooth; like Soma by Return of the Jaded",
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

# ---------------- Upload ----------------
def _safe_name(name: str) -> str:
    base = os.path.basename(name or "upload")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)

uploaded = st.file_uploader(
    "Upload premaster (WAV/AIFF/FLAC — no limiter, ~−6 dBFS headroom)",
    type=["wav", "aiff", "aif", "flac"]
)

if uploaded is None:
    st.info("Upload a premaster to begin.")
    st.stop()

# Make sure we have a stable per-session base dir (persist in session_state)
if "base_dir" not in st.session_state:
    st.session_state["base_dir"] = session_tmp_path()
base = st.session_state["base_dir"]

# Only (re)write if a new file arrives
if ("uploaded_name" not in st.session_state) or (uploaded.name != st.session_state["uploaded_name"]):
    st.session_state["uploaded_name"] = _safe_name(uploaded.name)

    in_path = os.path.join(base, st.session_state["uploaded_name"])
    try:
        uploaded.seek(0)  # make sure pointer is at start
        with open(in_path, "wb") as f:
            # stream to disk to avoid giant reads; ~1MB chunks
            shutil.copyfileobj(uploaded, f, length=1 * 1024 * 1024)
        uploaded.seek(0)
    except Exception as e:
        st.error("❌ Failed writing the uploaded file to /tmp.")
        st.exception(e)
        st.stop()

    # Clear stale analysis and remember the new path
    st.session_state.pop("analysis", None)
    st.session_state["in_path"] = in_path

# Use the path from session for the rest of the app
in_path = st.session_state["in_path"]

# ---------------- Controls ----------------
cols = st.columns(3)
with cols[0]:
    analyze_click = st.button("Analyze file")
with cols[1]:
    gen_click = st.button("Generate Adaptive (3 Variations)")
with cols[2]:
    if st.button("Reset file"):
        for k in ("uploaded_name", "analysis", "in_path"):
            st.session_state.pop(k, None)
        # keep base_dir so temp files remain accessible this session
        st.experimental_rerun()

# ---------------- Analysis (run once or on demand) ----------------
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

# ---- Tonal balance line graph (8-band)
# ---- Tonal balance (EQ-style, readable)
import numpy as np
import pandas as pd
import altair as alt

# Low → high, with Hz labels you can tweak
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

bands = analysis.get("bands_pct_8", {})
st.subheader("Tonal balance (8 bands)")

if bands:
    # Pull band shares in fixed order
    raw_vals = [max(1e-12, float(bands.get(b, 0.0))) for b in BAND_ORDER]

    # Convert percent → proportion and normalize
    total = sum(raw_vals) or 1.0
    shares = [v / total for v in raw_vals]

    # Center like an EQ: deviation from geometric mean (so 0 dB = average band energy)
    gmean = float(np.exp(np.mean(np.log(shares))))
    dev_db = [20.0 * np.log10(s / gmean) for s in shares]

    df = pd.DataFrame({
        "BandKey": BAND_ORDER,
        "Band": [BAND_LABELS[b] for b in BAND_ORDER],
        "Deviation (dB)": dev_db,
        "Share (%)": [v * 100.0 for v in shares],
    })

    # Line chart with points, ±12 dB window feels like an EQ range
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Band:N", sort=None, title=None),
        y=alt.Y("Deviation (dB):Q",
                scale=alt.Scale(domain=[-12, 12]),
                title="Balance vs band average (dB)"),
        tooltip=["Band", alt.Tooltip("Deviation (dB):Q", format=".2f"), alt.Tooltip("Share (%):Q", format=".1f")]
    ).properties(height=240)

    st.altair_chart(chart, use_container_width=True)
    st.caption("Above 0 dB = comparatively boosted band; below 0 dB = comparatively reduced. "
               "Computed from relative band energy (not absolute EQ).")
else:
    st.info("No 8-band analysis available.")


# ---------------- Pre-clean corrective EQ (optional) ----------------
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
            # Re-analyze corrected file and switch input for mastering
            analysis = analyze_audio(corrected_path)
            st.caption("Re-analysis after corrective cleanup")
            st.json(analysis)
            master_input_path = corrected_path
        except Exception as e:
            st.error("Corrective render failed.")
            st.exception(e)
    else:
        st.warning(f"No corrective plan: {corr_msg}")

# ---------------- LLM key diagnostics (temporary) ----------------
api_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
st.sidebar.write(f"🧪 Secrets has key: {'yes' if st.secrets.get('OPENAI_API_KEY') else 'no'}")
st.sidebar.write(f"🧪 Env has key: {'yes' if os.environ.get('OPENAI_API_KEY') else 'no'}")
st.sidebar.write(f"🧪 Using key length: {len(api_key) if api_key else 0}")

# ---------------- Variation directives ----------------
# Each variation gets a distinct seed + note to strongly encourage different plans
variant_notes = [
    "Variant A – slightly darker, preserve transients, keep hats smooth.",
    "Variant B – neutral mid focus, tighten lows a touch, gentle presence.",
    "Variant C – a hair brighter, careful with harshness, avoid pumping.",
]

def get_sectioned_plans(variant_idx: int, seed: str):
    """
    Ask the LLM for one plan (may be single or sectioned); normalize to verse/drop.
    Adds a per-variation seed to encourage distinct results.
    """
    note = variant_notes[variant_idx]
    v_prompt = (prompt_txt or "").strip()
    v_prompt = f"{v_prompt}\n{note}\nVariation seed: {seed}. Make choices distinct from other variations; different EQ emphasis, thresholds, and saturation rationale."

    plan, msg = llm_plan(
        analysis=analysis,                 # use latest analysis (post-corrective if applied)
        intent=intent,
        user_prompt=v_prompt,
        model=llm_model,
        reference_txt=reference_txt,
        reference_weight=reference_weight,
        api_key=api_key,
    )
    if not plan:
        return None, msg

    # Validate + normalize
    ok, errs = validate_plan(plan)
    if not ok:
        # Try to normalize single → sectioned anyway; still report validation errors
        if isinstance(plan, dict) and "targets" in plan:
            v_plan, d_plan = derive_section_plans_from_single(plan)
            return {"verse": v_plan, "drop": d_plan}, f"plan validated with warnings: {errs}"
        return None, f"plan failed validation: {errs}"

    if "verse" in plan and "drop" in plan:
        return plan, "ok"
    else:
        v_plan, d_plan = derive_section_plans_from_single(plan)
        return {"verse": v_plan, "drop": d_plan}, "derived"

# ---------------- Generate adaptive masters (3 variations) ----------------
if gen_click:
    # Optional: detect up to 3 safety notches AFTER corrective cleanup and BEFORE mastering
    detected_notches = []
    if post_notch:
        try:
            detected_notches = detect_resonances_simple(master_input_path, max_notches=3, min_prom_db=3.0)
            if detected_notches:
                st.markdown("**Safety notches (auto):**")
                st.json(detected_notches)
        except Exception as e:
            st.warning("Resonance detection failed; continuing without notches.")
            st.exception(e)
            detected_notches = []

    # Ensure we have a base dir in scope during render (fixes NameError)
    base = st.session_state.get("base_dir", session_tmp_path())

    for i in range(3):
        st.markdown(f"## Variation {i+1}")
        try:
            seed = f"{uuid.uuid4().hex[:8]}-{i+1}"
            sectioned, status = get_sectioned_plans(i, seed)
            if not sectioned:
                st.error(f"LLM plan unavailable for Variation {i+1}.")
                continue

            # Show the plan for this variation
            with st.expander(f"AI Plan – Variation {i+1} ({status})"):
                st.code(json.dumps(sectioned, indent=2))

            # Always adaptive (verse/drop)
            out_path = os.path.join(base, f"master_ai_adaptive_v{i+1}.wav")
            render_adaptive_from_plans(
                master_input_path,           # use corrected premaster if available
                out_path,
                sectioned["verse"],
                sectioned["drop"],
                notches=detected_notches     # pass safety notches to renderer
            )

            st.audio(out_path)
            with open(out_path, "rb") as f:
                st.download_button(
                    f"Download Adaptive V{i+1}",
                    f.read(),
                    file_name=f"master_ai_adaptive_v{i+1}.wav"
                )

        except Exception as e:
            st.error(f"Render failed for Variation {i+1}.")
            st.exception(e)

# ---------------- Debug ----------------
with st.expander("Debug"):
    st.write({
        "in_path_exists": os.path.exists(in_path),
        "in_path": in_path,
        "base_dir": st.session_state.get("base_dir"),
        "adaptive_only": True,
        "reference_txt": reference_txt,
        "reference_weight": reference_weight,
    })
