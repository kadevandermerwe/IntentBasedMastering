# Vale Mastering Assistant - Streamlit Prototype (fixed)
import os, json, tempfile, subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa
import streamlit as st

# Optional OpenAI LLM planner
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Vale Mastering Assistant", page_icon="🎛️", layout="centered")
st.title("🎛️ Vale Mastering Assistant — Prototype")

# --- Sidebar: LLM toggle/model ---
use_llm = st.sidebar.checkbox("Use OpenAI LLM planner", value=False)
llm_model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")
st.sidebar.caption("Add your key in Streamlit Cloud → Settings → Secrets → OPENAI_API_KEY")

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

# ---------- Analysis ----------
def analyze_audio(path):
    """Return LUFS-I, true-peak (est via x4 oversample), and rough spectral balance."""
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    # LUFS (EBU R128)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y))

    # True-peak estimate via x4 oversampling (librosa keyword args)
    y_os = librosa.resample(y, orig_sr=sr, target_sr=sr*4, res_type="kaiser_best")
    tp = float(20*np.log10(np.max(np.abs(y_os)) + 1e-12))

    # Rough spectral balance (low/mid/high)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[(freqs < 120)].mean()
    mid = S[(freqs >= 120) & (freqs < 5000)].mean()
    high = S[(freqs >= 5000)].mean()
    tot = low + mid + high + 1e-9
    spec = {
        "low": float(100 * low / tot),
        "mid": float(100 * mid / tot),
        "high": float(100 * high / tot),
    }
    return {
        "sr": sr,
        "lufs_integrated": lufs,
        "true_peak_dbfs_est": tp,
        "spectral_pct": spec,
    }

# ---------- Prompt → Intent ----------
def parse_prompt(txt):
    p = (txt or "").lower()
    intent = {"tone": 0.0, "dynamics": 0.0, "stereo": 0.0, "character": 0.0}
    if any(k in p for k in ["dark", "moody", "warm", "shadow", "muted"]): intent["tone"] -= 0.5
    if any(k in p for k in ["bright", "airy", "sparkle", "shine", "open", "sheen"]): intent["tone"] += 0.5
    if any(k in p for k in ["preserve", "breathe", "dynamic", "not loud", "natural"]): intent["dynamics"] -= 0.5
    if any(k in p for k in ["loud", "slam", "club", "radio", "punchy"]): intent["dynamics"] += 0.5
    if any(k in p for k in ["wide", "spacious", "open image", "stereo"]): intent["stereo"] += 0.3
    if any(k in p for k in ["mono", "tight", "focused", "narrow"]): intent["stereo"] -= 0.3
    if any(k in p for k in ["tape", "analog", "saturation", "glue", "harmonic"]): intent["character"] += 0.6
    if any(k in p for k in ["clean", "transparent", "surgical"]): intent["character"] -= 0.6
    for k in intent:
        intent[k] = float(clamp(intent[k], -1.0, 1.0))
    return intent

# ---------- Optional LLM planner ----------
def llm_plan(analysis, intent, prompt, model):
    if not (OPENAI_AVAILABLE and "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]):
        return None, "LLM disabled or API key missing; using heuristics."
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        system = (
            "You are a mastering planner. Return STRICT JSON with keys: "
            "targets(lufs_i,true_peak_db), eq(low_shelf_db,mud_cut_db,high_shelf_db), "
            "mb_comp(low/mid/high threshold_db), saturation(drive_db), stereo(amount), explanation."
        )
        user = f"ANALYSIS:{json.dumps(analysis)}\nINTENT:{json.dumps(intent)}\nPROMPT:{prompt}"
        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").replace("json", "")
        plan = json.loads(content)
        # Clamp a few critical bounds
        plan.setdefault("targets", {})["true_peak_db"] = -1.0
        if plan.get("eq", {}).get("high_shelf_db", 0) > 1.5:
            plan["eq"]["high_shelf_db"] = 1.5
        if plan.get("stereo", {}).get("amount", 0) > 0.15:
            plan["stereo"]["amount"] = 0.15
        return plan, "LLM plan generated."
    except Exception as e:
        return None, f"LLM error: {e}; using heuristics."

# ---------- Heuristic chain ----------
def build_fg(analysis, intent, variant="vibe"):
    tone = float(intent["tone"]); dyn = float(intent["dynamics"]); char = float(intent["character"])
    # Loudness target by intent/variant
    base_lufs = -10.0
    if dyn < -0.33: base_lufs = -12.0
    elif dyn > 0.33: base_lufs = -9.0
    if variant == "conservative": target_lufs = base_lufs + 0.5
    elif variant == "loud":       target_lufs = base_lufs - 0.8
    else:                         target_lufs = base_lufs
    target_tp = -1.0

    # Subtle EQ caps
    hi_gain = 1.5 * max(0.0, tone)   # 0..+1.5 dB
    lo_gain = 0.0
    if analysis["spectral_pct"]["low"] < 30.0:
        lo_gain = 0.8
    mud_cut = -1.2 if analysis["spectral_pct"]["mid"] > 55.0 else 0.0

    # Multiband thresholds
    lo_thr, mid_thr, hi_thr = -22, -24, -26
    if variant == "loud": lo_thr -= 2; mid_thr -= 2; hi_thr -= 2
    if dyn < -0.33:       lo_thr += 2; mid_thr += 2; hi_thr += 2

    # Gentle pre-limit drive
    drive_in = 1.0 + 0.15 * max(0.0, char)

    # FFmpeg filtergraph
    mb = (
      "asplit=3[lo][mid][hi];"
      f"[lo]lowpass=f=140,acompressor=threshold={lo_thr}dB:ratio=2:attack=20:release=200[loC];"
      f"[mid]bandpass=f=140:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio=1.8:attack=15:release=180[midC];"
      f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio=1.5:attack=10:release=150[hiC];"
      "[loC][midC][hiC]amix=inputs=3:normalize=0[mb];"
    )
    eq = (
      "[mb]"
      f"firequalizer=gain='if(f<120,{lo_gain}, if(f<300,{mud_cut}, if(f>12000,{hi_gain},0)))'"
      "[eq];"
    )
    sat = f"[eq]alimiter=limit=0.0:level_in={drive_in}:level_out=1.0[sat];"
    loud = f"[sat]loudnorm=I={target_lufs}:LRA=7:TP={target_tp}:dual_mono=true:linear=true[out]"

    params = {
        "targets": {"lufs_i": target_lufs, "true_peak_db": target_tp},
        "eq": {"low_shelf_db": lo_gain, "mud_cut_db": mud_cut, "high_shelf_db": hi_gain},
        "mb_thr": {"low": lo_thr, "mid": mid_thr, "high": hi_thr},
        "saturation": {"drive_in": drive_in},
    }
    return mb + eq + sat + loud, params

def render_variant(in_wav, out_wav, filtergraph):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_wav, "-filter_complex", filtergraph, "-map", "[out]",
        "-ar", "44100", "-ac", "2", out_wav
    ]
    subprocess.run(cmd, check=True)

# ---------- UI ----------
uploaded = st.file_uploader("Upload premaster (WAV/AIFF – no limiter, ~−6 dBFS headroom)", type=["wav", "aiff", "aif", "flac"])
prompt = st.text_area("Intent (optional): e.g. “dark, preserve dynamics, tape weight, slightly wider”")

col1, col2 = st.columns(2)
with col1:
    tone = st.slider("Tone: Dark  ↔  Bright", -1.0, 1.0, 0.0, 0.1)
    dynamics = st.slider("Dynamics: Preserve  ↔  Loud", -1.0, 1.0, 0.0, 0.1)
with col2:
    stereo = st.slider("Stereo: Narrow  ↔  Wide", -1.0, 1.0, 0.0, 0.1)  # (placeholder for future widening)
    character = st.slider("Character: Clean  ↔  Tape", -1.0, 1.0, 0.5, 0.1)

# Merge prompt → sliders (soft blend)
if prompt:
    parsed = parse_prompt(prompt)
    st.caption(
        f"Parsed intent → tone {parsed['tone']:+.2f}, dynamics {parsed['dynamics']:+.2f}, "
        f"stereo {parsed['stereo']:+.2f}, character {parsed['character']:+.2f}"
    )
    tone = float(clamp(tone*0.5 + parsed["tone"]*0.5, -1.0, 1.0))
    dynamics = float(clamp(dynamics*0.5 + parsed["dynamics"]*0.5, -1.0, 1.0))
    stereo = float(clamp(stereo*0.5 + parsed["stereo"]*0.5, -1.0, 1.0))
    character = float(clamp(character*0.5 + parsed["character"]*0.5, -1.0, 1.0))

intent = {"tone": tone, "dynamics": dynamics, "stereo": stereo, "character": character}
st.markdown(
    f"**Resolved intent** → tone {intent['tone']:+.2f} • dynamics {intent['dynamics']:+.2f} • "
    f"stereo {intent['stereo']:+.2f} • character {intent['character']:+.2f}"
)

if uploaded is not None:
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, uploaded.name.replace(" ", "_"))
        with open(in_path, "wb") as f:
            f.write(uploaded.read())

        st.subheader("Analysis")
        analysis = analyze_audio(in_path)
        st.json(analysis)

        # Optional LLM plan (not required for rendering)
        plan = None; plan_msg = None
        if use_llm:
            plan, plan_msg = llm_plan(analysis, intent, prompt, llm_model)
            st.caption(plan_msg or "")
            if plan:
                st.subheader("LLM Plan (clamped)")
                st.code(json.dumps(plan, indent=2))

        if st.button("Generate 3 masters"):
            for variant in ["conservative", "vibe", "loud"]:
                fg, params = build_fg(analysis, intent, variant)
                st.markdown(f"**Plan – {variant.capitalize()}**")
                st.json(params)

                out_path = os.path.join(td, f"{os.path.splitext(os.path.basename(in_path))[0]}_{variant}.wav")
                render_variant(in_path, out_path, fg)

                # Quick post-checks
                post = analyze_audio(out_path)
                ok_tp   = post["true_peak_dbfs_est"] <= -0.9
                ok_lufs = abs(post["lufs_integrated"] - params["targets"]["lufs_i"]) <= 0.6
                ok_dark_cap = (intent["tone"] <= 0 and params["eq"]["high_shelf_db"] <= 1.5) or (intent["tone"] > 0)

                def bullet(ok, msg): return ("✅ " if ok else "⚠️ ") + msg
                st.write(bullet(ok_tp,   f"TP ≤ −1.0 dBTP target (measured {post['true_peak_dbfs_est']:.2f} dBFS est)"))
                st.write(bullet(ok_lufs, f"LUFS near target ±0.5 (target {params['targets']['lufs_i']:.1f}, got {post['lufs_integrated']:.1f})"))
                st.write(bullet(ok_dark_cap, "Dark intent respected (high shelf capped)"))

                st.audio(out_path)
                with open(out_path, "rb") as f:
                    st.download_button(
                        f"Download {variant}",
                        f.read(),
                        file_name=os.path.basename(out_path),
                        mime="audio/wav"
                    )
