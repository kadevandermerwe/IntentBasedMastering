# dsp.py
import os, numpy as np, soundfile as sf
from utils import clamp, safe_run

# ---------- Tonal bands (Hz) ----------
BAND_NAMES = ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]
BAND_EDGES = [20, 80, 120, 250, 500, 3500, 8000, 10000, 20000]
# Bands: [20-80), [80-120), [120-250), [250-500), [500-3500), [3500-8000), [8000-10000), [10000-20000]

def _band_mask(freqs: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    return (freqs >= f_lo) & (freqs < f_hi)

def analyze_bands_8(y: np.ndarray, sr: int) -> dict:
    """Return 8-band energy % distribution (sum≈100)."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_means = []
    for i in range(len(BAND_EDGES) - 1):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i+1]
        mask = _band_mask(freqs, lo, hi)
        val = float(S[mask, :].mean() if np.any(mask) else 0.0)
        band_means.append(val)
    total = sum(band_means) + 1e-9
    pct = [100.0 * v / total for v in band_means]
    return {BAND_NAMES[i]: pct[i] for i in range(8)}

def analyze_audio(path: str) -> dict:
    """LUFS-I, true-peak est (x4 oversample), 3-band legacy %, and 8-band %."""
    import pyloudnorm as pyln
    import librosa

    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y))

    # True-peak estimate via 4x oversampling
    y_os = librosa.resample(y, orig_sr=sr, target_sr=sr*4, res_type="kaiser_best")
    tp = float(20*np.log10(np.max(np.abs(y_os)) + 1e-12))

    # Legacy coarse split (for backward visibility)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[(freqs < 120)].mean()
    mid = S[(freqs >= 120) & (freqs < 5000)].mean()
    high = S[(freqs >= 5000)].mean()
    tot = low + mid + high + 1e-9
    spec_3 = {"low": float(100*low/tot), "mid": float(100*mid/tot), "high": float(100*high/tot)}

    spec_8 = analyze_bands_8(y, sr)

    return {
        "sr": sr,
        "lufs_integrated": lufs,
        "true_peak_dbfs_est": tp,
        "spectral_pct": spec_3,
        "bands_pct_8": spec_8,
    }

def detect_sections(path: str, frame_ms=200, hop_ms=100, energy_sigma=1.0, min_len_s=4.0):
    """Return (drop_spans, total_duration_sec) by RMS thresholding (energy > mean+σ*std)."""
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    frame = int(sr*frame_ms/1000); hop = int(sr*hop_ms/1000)
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
    return spans, float(len(y)/sr)

def firequalizer_from_eq8(eq8: dict) -> str:
    """Build a piecewise constant firequalizer gain expression from 8 band gains (dB)."""
    gains = [float(eq8.get(name, 0.0)) for name in BAND_NAMES]  # 8 values
    # clamp ±2 dB for safety
    gains = [max(-2.0, min(2.0, g)) for g in gains]
    edges = BAND_EDGES[:]  # 9 values

    expr = ""
    for i in range(8):
        cond = f"lt(f,{edges[i+1]})"
        gi = f"{gains[i]:.3f}"
        if i == 0:
            expr = f"if({cond},{gi},__NEXT__)"
        elif i < 7:
            expr = expr.replace("__NEXT__", f"if({cond},{gi},__NEXT__)")
        else:
            expr = expr.replace("__NEXT__", gi)

    return f"firequalizer=gain='{expr}'"

def build_fg_from_plan(plan: dict) -> str:
    """
    Translate an LLM plan → FFmpeg filtergraph (MB comp -> EQ (eq8 preferred) -> loudnorm).
    No heuristics. If plan misses eq/eq8, EQ stage is neutral.
    """
    tgt = plan.get("targets", {}) or {}
    mb  = plan.get("mb_comp", {}) or {}
    sat = plan.get("saturation", {}) or {}
    eq8 = plan.get("eq8")
    eq3 = plan.get("eq")

    tgt_lufs = float(tgt.get("lufs_i", -11.5))
    tgt_tp   = float(tgt.get("true_peak_db", -1.2))

    lo_thr = float(mb.get("low_thr_db",  -20))
    mid_thr= float(mb.get("mid_thr_db",  -24))
    hi_thr = float(mb.get("high_thr_db", -26))

    drive_db = float(sat.get("drive_db", 0.0))
    drive_in = 1.0 + 0.1 * max(0.0, min(3.0, drive_db))

    mb_chain = (
      "asplit=3[lo][mid][hi];"
      f"[lo]lowpass=f=120,acompressor=threshold={lo_thr}dB:ratio=2.0:attack=10:release=180[loC];"
      f"[mid]bandpass=f=120:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio=1.6:attack=16:release=220[midC];"
      f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio=1.3:attack=12:release=180[hiC];"
      "[loC][midC][hiC]amix=inputs=3[mb];"
    )

    if isinstance(eq8, dict) and all(k in eq8 for k in BAND_NAMES):
        eq_block = f"[mb]{firequalizer_from_eq8(eq8)}[eq];"
    elif isinstance(eq3, dict):
        lo_gain = float(eq3.get("low_shelf_db", 0.0))
        mud_cut = float(eq3.get("mud_cut_db", 0.0))
        hi_gain = float(eq3.get("high_shelf_db", 0.0))
        eq_block = (
          "[mb]"
          f"firequalizer=gain='if(lt(f,120),{lo_gain:.3f}, if(lt(f,300),{mud_cut:.3f}, if(gt(f,12000),{hi_gain:.3f},0)))'"
          "[eq];"
        )
    else:
        eq_block = "[mb]anull[eq];"

    sat_block  = f"[eq]volume={drive_in:.3f}[sat];"
    loud_block = f"[sat]loudnorm=I={tgt_lufs}:LRA=9:TP={tgt_tp}:dual_mono=true:linear=true[out]"
    return mb_chain + eq_block + sat_block + loud_block

def render_variant(in_wav: str, out_wav: str, filtergraph: str):
    cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", in_wav, "-filter_complex", filtergraph, "-map", "[out]",
        "-ar", "44100", "-ac", "2", out_wav
    ]
    safe_run(cmd)

def render_adaptive_from_plans(in_wav: str, out_wav: str, verse_plan: dict, drop_plan: dict):
    """Detect drops, process each segment with verse/drop plans, concat."""
    drops, total_dur = detect_sections(in_wav)
    segments, cursor, idx = [], 0.0, 0

    def cut_and_process(t0, t1, plan, i):
        fg = build_fg_from_plan(plan)
        seg_out = os.path.join(os.path.dirname(out_wav), f"seg_{i:03d}.wav")
        cmd = [
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", in_wav,
            "-filter_complex", f"atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS,{fg}",
            "-map","[out]","-ar","44100","-ac","2", seg_out
        ]
        safe_run(cmd); return seg_out

    for (d0, d1) in drops:
        if d0 > cursor + 0.05:
            segments.append(cut_and_process(cursor, d0, verse_plan, idx)); idx += 1
        segments.append(cut_and_process(d0, d1, drop_plan, idx)); idx += 1
        cursor = d1
    if cursor < total_dur - 0.05:
        segments.append(cut_and_process(cursor, total_dur, verse_plan, idx)); idx += 1

    list_path = out_wav + ".txt"
    with open(list_path, "w") as f:
        for s in segments:
            f.write(f"file '{s}'\n")
    safe_run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",list_path,"-c","copy",out_wav])
