# Vale Mastering Assistant - Streamlit Prototype (Adaptive, Stable)
import os, json, tempfile, subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa
import streamlit as st

# ---------------- UI setup ----------------
st.set_page_config(page_title="Vale Mastering Assistant", page_icon="🎛️", layout="centered")
st.title("🎛️ Vale Mastering Assistant — Prototype")

# ---------- Helpers ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------- Section detection (energy-based) ----------
def detect_sections(path, frame_ms=200, hop_ms=100, energy_sigma=1.0, min_len_s=4.0):
    """Return (drop_spans, total_duration_sec) using short-term RMS thresholding."""
    y, sr = librosa.load(path, sr=None, mono=True)
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop, n_fft=frame)
    thr = rms.mean() + energy_sigma * rms.std()
    mask = rms > thr

    spans, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = t[i]
        if not on and start is not None:
            end = t[i]
            if end - start >= min_len_s:
                spans.append((float(start), float(end)))
            start = None
    if start is not None:
        end = float(t[-1])
        if end - start >= min_len_s:
            spans.append((float(start), float(end)))
    return spans, float(len(y) / sr)

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
    y_os = librosa.resample(y, orig_sr=sr, target_sr=sr * 4, res_type="kaiser_best")
    tp = float(20 * np.log10(np.max(np.abs(y_os)) + 1e-12))

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

# ---------- FFmpeg renderer ----------
def render_variant(in_wav, out_wav, filtergraph):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_wav, "-filter_complex", filtergraph, "-map", "[out]",
        "-ar", "44100", "-ac", "2", out_wav
    ]
    subprocess.run(cmd, check=True)

# ---------- Heuristic chain (updated targets, safer) ----------
def build_fg(analysis, intent, variant="vibe"):
    tone = float(intent["tone"]); dyn = float(intent["dynamics"]); char = float(intent["character"])

    # Slightly more conservative loudness so drops breathe
    if variant == "conservative":
        target_lufs = -12.0
    elif variant == "vibe":
        target_lufs = -11.5
    else:  # "loud"
        target_lufs = -10.5
    target_tp = -1.2  # safer TP; no extra limiter stage

    # Subtle EQ caps
    hi_gain = 1.5 * max(0.0, tone)   # 0..+1.5 dB
    lo_gain = 0.0 if analysis["spectral_pct"]["low"] >= 30.0 else 0.8
    mud_cut = -1.2 if analysis["spectral_pct"]["mid"] > 55.0 else 0.0

    # Multiband thresholds (gentle), lows a bit tighter
    lo_thr, mid_thr, hi_thr = -20, -24, -26
    # If user really wants loud, sneak thresholds down a hair
    if variant == "loud":
        lo_thr -= 2; mid_thr -= 2; hi_thr -= 2
    if dyn < -0.33:  # preserve dynamics
        lo_thr += 2; mid_thr += 2; hi_thr += 2

    # Character pre-gain (no limiter here)
    drive_in = 1.0 + 0.15 * max(0.0, char)

    # Filtergraph
    mb = (
      "asplit=3[lo][mid][hi];"
      f"[lo]lowpass=f=120,acompressor=threshold={lo_thr}dB:ratio=2.5:attack=6:release=80[loC];"
      f"[mid]bandpass=f=120:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio=1.7:attack=15:release=180[midC];"
      f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio=1.4:attack=12:release=150[hiC];"
      "[loC][midC][hiC]amix=inputs=3[mb];"
    )
    eq = (
      "[mb]"
      f"firequalizer=gain='if(lt(f,120),{lo_gain},"
      f" if(lt(f,300),{mud_cut},"
      f"  if(gt(f,12000),{hi_gain},0)))'"
      "[eq];"
    )
    # Pre-gain saturation feel
    sat = f"[eq]volume={drive_in}[sat];"
    # Single-stage loudness/TP
    loud = f"[sat]loudnorm=I={target_lufs}:LRA=7:TP={target_tp}:dual_mono=true:linear=true[out]"

    params = {
        "targets": {"lufs_i": target_lufs, "true_peak_db": target_tp},
        "eq": {"low_shelf_db": lo_gain, "mud_cut_db": mud_cut, "high_shelf_db": hi_gain},
        "mb_thr": {"low": lo_thr, "mid": mid_thr, "high": hi_thr},
        "saturation": {"pre_gain": drive_in},
    }
    return mb + eq + sat + loud, params

# ---------- Adaptive per-section chain ----------
def build_fg_adaptive(analysis, intent, *, is_drop: bool):
    tone = float(intent["tone"]); dyn = float(intent["dynamics"]); char = float(intent["character"])

    target_lufs = -11.0 if is_drop else -12.5
    target_tp = -1.2

    hi_gain = 1.2 * max(0.0, tone)
    lo_gain = 0.0 if analysis["spectral_pct"]["low"] >= 30.0 else 0.8
    mud_cut = -1.2 if analysis["spectral_pct"]["mid"] > 55.0 else 0.0

    lo_thr, mid_thr, hi_thr = (-18, -23, -25) if is_drop else (-20, -24, -26)
    lo_ratio = 3.0 if is_drop else 2.3

    mb = (
      "asplit=3[lo][mid][hi];"
      f"[lo]lowpass=f=120,acompressor=threshold={lo_thr}dB:ratio={lo_ratio}:attack=6:release=80[loC];"
      f"[mid]bandpass=f=120:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio=1.7:attack=15:release=180[midC];"
      f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio=1.4:attack=12:release=150[hiC];"
      "[loC][midC][hiC]amix=inputs=3[mb];"
    )
    eq = (
      "[mb]"
      f"firequalizer=gain='if(lt(f,120),{lo_gain},"
      f" if(lt(f,300),{mud_cut},"
      f"  if(gt(f,12000),{hi_gain},0)))'"
      "[eq];"
    )
    drive_in = 1.0 + 0.15 * max(0.0, char)
    sat = f"[eq]volume={drive_in}[sat];"
    loud = f"[sat]loudnorm=I={target_lufs}:LRA=7:TP={target_tp}:dual_mono=true:linear=true[out]"

    params = {
      "targets": {"lufs_i": target_lufs, "true_peak_db": target_tp},
      "eq": {"low_shelf_db": lo_gain, "mud_cut_db": mud_cut, "high_shelf_db": hi_gain},
      "mb_thr": {"low": lo_thr, "mid": mid_thr, "high": hi_thr},
      "saturation": {"pre_gain": drive_in},
      "section": "drop" if is_drop else "verse",
    }
    return mb + eq + sat + loud, params

def render_adaptive(in_wav, out_wav, analysis, intent):
    """Detect drops, process each segment with tailored FG, concat."""
    drops, total_dur = detect_sections(in_wav)
    segments = []
    cursor = 0.0
    idx = 0

    def cut_and_process(t0, t1, is_drop, idx):
        fg, params = build_fg_adaptive(analysis, intent, is_drop=is_drop)
        seg_out = os.path.join(os.path.dirname(out_wav), f"seg_{idx:03d}.wav")
        # Chain trim → adaptive FG (which outputs [out])
        cmd = [
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", in_wav,
            "-filter_complex", f"atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS,{fg}",
            "-map","[out]","-ar","44100","-ac","2", seg_out
        ]
        subprocess.run(cmd, check=True)
        return seg_out, params

    for (d0, d1) in drops:
        if d0 > cursor + 0.05:
            seg, _ = cut_and_process(cursor, d0, False, idx); segments.append(seg); idx += 1
        seg, _ = cut_and_process(d0, d1, True, idx); segments.append(seg); idx += 1
        cursor = d1
    if cursor < total_dur - 0.05:
        seg, _ = cut_and_process(cursor, total_dur, False, idx); segments.append(seg); idx += 1

    # Concat segments
    list_path = out_wav + ".txt"
    with open(list_path, "w") as f:
        for s in segments:
            f.write(f"file '{s}'\n")
    subprocess.run(
        ["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",list_path,"-c","copy",out_wav],
        check=True
    )

# ---------------- Sidebar controls ----------------
adaptive = st.sidebar.checkbox("Adaptive mode (detect drops & process per-section)", value=True)

# (Optional) LLM planner toggle
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

use_llm = st.sidebar.checkbox("Use OpenAI LLM planner", value=False)
llm_model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")
st.sidebar.caption("Add your key in Streamlit Cloud → Settings → Secrets → OPENAI_API_KEY")

def llm_plan(analysis, intent, prompt, model):
    if not (OPENAI_AVAILABLE and "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]):
        return None, "LLM disabled or API key missing; using heuristics."
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        system = ("You are a mastering planner. Return STRICT JSON with keys: "
                  "targets(lufs_i,true_peak_db), eq(low_shelf_db,mud_cut_db,high_shelf_db), "
                  "mb_comp(low/mid/high threshold_db), saturation(drive_db), stereo(amount), explanation.")
        user = f"ANALYSIS:{json.dumps(analysis)}\nINTENT:{json.dumps(intent)}\nPROMPT:{prompt}"
        resp = client.chat.completions.create(
            model=model, temperature=0.3,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").replace("json", "")
        plan = json.loads(content)
        plan.setdefault("targets", {})["true_peak_db"] = -1.2
        if plan.get("eq", {}).get("high_shelf_db", 0) > 1.5: plan["eq"]["high_shelf_db"] = 1.5
        if plan.get("stereo", {}).get("amount", 0) > 0.15:   plan["stereo"]["amount"] = 0.15
        return plan, "LLM plan generated."
    except Exception as e:
        return None, f"LLM error: {e}; using heuristics."

# ---------------- Main UI ----------------
uploaded = st.file_uploader("Upload premaster (WAV/AIFF — no limiter, ~−6 dBFS headroom)", type=["wav","aiff","aif","flac"])
prompt = st.text_area("Intent (optional): e.g. “dark, preserve dynamics, tape weight, slightly wider”")

col1, col2 = st.columns(2)
with col1:
    tone = st.slider("Tone: Dark  ↔  Bright", -1.0, 1.0, 0.0, 0.1)
    dynamics = st.slider("Dynamics: Preserve  ↔  Loud", -1.0, 1.0, 0.0, 0.1)
with col2:
    stereo = st.slider("Stereo: Narrow  ↔  Wide", -1.0, 1.0, 0.0, 0.1)  # placeholder for future widening
    character = st.slider("Character: Clean  ↔  Tape", -1.0, 1.0, 0.5, 0.1)

# Cache uploaded audio in session so reruns don't wipe it
# --- Upload & analysis (robust) ---

# 1) Cache the upload only when a new file arrives
if uploaded is not None:
    if (
        "uploaded_name" not in st.session_state
        or uploaded.name != st.session_state.get("uploaded_name")
        or "uploaded_bytes" not in st.session_state
    ):
        st.session_state["uploaded_name"] = uploaded.name
        st.session_state["uploaded_bytes"] = uploaded.read()
        # clear stale analysis if a new file comes in
        for k in ("analysis", "in_path"):
            st.session_state.pop(k, None)

# 2) Create a persistent temp dir for this session (so files persist across reruns)
if "tmpdir" not in st.session_state:
    st.session_state["tmpdir"] = tempfile.TemporaryDirectory()

# 3) Materialize the uploaded bytes to a file path we can reuse safely
if "uploaded_bytes" in st.session_state:
    in_path = os.path.join(
        st.session_state["tmpdir"].name,
        st.session_state["uploaded_name"].replace(" ", "_")
    )
    # write bytes every run so the file always exists before analysis/render
    with open(in_path, "wb") as f:
        f.write(st.session_state["uploaded_bytes"])
    st.session_state["in_path"] = in_path

    # 4) Controls: analyze + reset
    cols = st.columns(2)
    with cols[0]:
        analyze_click = st.button("Analyze file")
    with cols[1]:
        if st.button("Reset file"):
            # clean only file-related state (keep tmpdir)
            for k in ("uploaded_name","uploaded_bytes","analysis","in_path"):
                st.session_state.pop(k, None)
            st.experimental_rerun()

    # 5) Run analysis on click (or if not yet computed) with visible errors
    if analyze_click or "analysis" not in st.session_state:
        try:
            st.session_state["analysis"] = analyze_audio(st.session_state["in_path"])
        except Exception as e:
            st.error("❌ Analysis failed.")
            st.exception(e)  # shows full traceback in the app
            st.stop()

    # 6) Show analysis
    analysis = st.session_state["analysis"]
    st.subheader("Analysis")
    st.json(analysis)

else:
    st.info("Upload a premaster to begin.")
    st.stop()
