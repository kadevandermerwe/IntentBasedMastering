# dsp.py
import os, numpy as np, soundfile as sf
from utils import clamp, safe_run

# ---------- Tonal bands (Hz) ----------
BAND_NAMES = ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]
BAND_EDGES = [20, 80, 120, 250, 500, 3500, 8000, 10000, 20000]

# Bands: [20-80), [80-120), [120-250), [250-500), [500-3500), [3500-8000), [8000-10000), [10000-20000]

def _clamp_db(x, lo=-2.0, hi=2.0):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(lo, min(hi, x))

def _eq8_firequalizer_expr(eq8: dict) -> str:
    """
    Piecewise gain curve for 8 bands using firequalizer's expression.
    Band edges (Hz):
      sub: <80
      low_bass: 80–120
      high_bass: 120–250
      low_mids: 250–500
      mids: 500–3500
      high_mids: 3500–8000
      highs: 8000–10000
      air: >10000
    """
    # safe gains for all bands (floats, within ±2.0 ideally)
    gains = {name: float(eq8.get(name, 0.0)) for name in BAND_NAMES}

    # nested ifs using lt()/gt() so it's compatible with ffmpeg expr
    # NOTE: Keep commas and spaces exact; firequalizer is picky.
    expr = (
        "if(lt(f,80),{sub},"
        " if(lt(f,120),{low_bass},"
        "  if(lt(f,250),{high_bass},"
        "   if(lt(f,500),{low_mids},"
        "    if(lt(f,3500),{mids},"
        "     if(lt(f,8000),{high_mids},"
        "      if(lt(f,10000),{highs},{air})))))))"
    ).format(**gains)

    # return full firequalizer filter segment (no trailing semicolon)
    return f"firequalizer=gain='{expr}'"


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

def detect_sections(path, frame_ms=200, hop_ms=100, energy_sigma=1.0, min_len_s=4.0):
    # Load audio with soundfile (safer than librosa.load)
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono

    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)

    # Short-term energy
    rms = []
    for i in range(0, len(y) - frame, hop):
        seg = y[i:i+frame]
        rms.append(np.sqrt(np.mean(seg**2)))
    rms = np.array(rms)

    t = np.arange(len(rms)) * hop / sr
    thr = rms.mean() + energy_sigma * rms.std()

    spans, start = [], None
    for i, on in enumerate(rms > thr):
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
    Translate a single LLM plan dict → FFmpeg filtergraph (string).
    Expects keys:
      - targets: lufs_i, true_peak_db
      - mb_comp: low_thr_db, mid_thr_db, high_thr_db
      - saturation: drive_db
      - eq8 (preferred) or eq (legacy)
    """
    # ----- targets -----
    tgt = plan.get("targets", {}) or {}
    tgt_lufs = float(tgt.get("lufs_i", -11.5))
    tgt_tp   = float(tgt.get("true_peak_db", -1.2))

    # ----- thresholds (define them! this fixed your NameError) -----
    mb = plan.get("mb_comp", {}) or {}
    lo_thr  = float(mb.get("low_thr_db",  -20))
    mid_thr = float(mb.get("mid_thr_db",  -24))
    hi_thr  = float(mb.get("high_thr_db", -26))

    # ratios tuned gentle; you can tweak if you like
    lo_ratio, mid_ratio, hi_ratio = 2.0, 1.6, 1.3

    # ----- saturation pre-gain -----
    sat = plan.get("saturation", {}) or {}
    drive_db = float(sat.get("drive_db", 1.0))
    drive_in = 1.0 + 0.1 * max(0.0, drive_db)   # convert “dB-ish” into linear-ish gain

    # ----- EQ (prefer 8-band eq8; fallback to legacy 3-band eq) -----
    eq8 = plan.get("eq8")
    eq_expr = None
    if isinstance(eq8, dict):
        eq_expr = _eq8_firequalizer_expr(eq8)
    else:
        # legacy 3-band support (kept for safety)
        eq3 = plan.get("eq", {}) or {}
        low_shelf  = float(eq3.get("low_shelf_db",  0.0))
        mud_cut    = float(eq3.get("mud_cut_db",    0.0))
        high_shelf = float(eq3.get("high_shelf_db", 0.0))
        eq_expr = (
            "firequalizer=gain='"
            f"if(lt(f,120),{low_shelf}, "
            f" if(lt(f,300),{mud_cut}, "
            f"  if(gt(f,12000),{high_shelf},0)))'"
        )

    # ----- multiband split → comp → sum -----
    # NOTE: we removed amix normalize flag; Fire up/downstream manages level.
    mb_chain = (
        "asplit=3[lo][mid][hi];"
        f"[lo]lowpass=f=120,acompressor=threshold={lo_thr}dB:ratio={lo_ratio}:attack=10:release=180[loC];"
        f"[mid]bandpass=f=120:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio={mid_ratio}:attack=16:release=220[midC];"
        f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio={hi_ratio}:attack=12:release=180[hiC];"
        "[loC][midC][hiC]amix=inputs=3[mb]"
    )

    # ----- chain it all -----
    # [mb] -> firequalizer -> volume (sat) -> loudnorm -> [out]
    fg = (
        mb_chain + ";" +
        "[mb]" + eq_expr + "[eq];" +
        f"[eq]volume={drive_in}[sat];" +
        f"[sat]loudnorm=I={tgt_lufs}:LRA=9:TP={tgt_tp}:dual_mono=true:linear=true[out]"
    )
    return fg


def render_variant(in_wav: str, out_wav: str, filtergraph: str):
    cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", in_wav, "-filter_complex", filtergraph, "-map", "[out]",
        "-ar", "44100", "-ac", "2", out_wav
    ]
    safe_run(cmd)

# in dsp.py — add this near your other builders

from schema import BAND_NAMES  # make sure dsp.py imports BAND_NAMES

def _eq8_expression(eq8: dict) -> str:
    """
    Returns a firequalizer gain expression for the 8 named bands in BAND_NAMES.
    The expression is step-wise per band threshold; safe for ffmpeg eval.
    """
    # default 0.0 dB for any missing band
    g = {k: float(eq8.get(k, 0.0)) for k in BAND_NAMES}
    # bands (edges): <80, <120, <250, <500, <3500, <8000, <10000, else
    return (
        "if(lt(f,80),{sub},"
        " if(lt(f,120),{low_bass},"
        "  if(lt(f,250),{high_bass},"
        "   if(lt(f,500),{low_mids},"
        "    if(lt(f,3500),{mids},"
        "     if(lt(f,8000),{high_mids},"
        "      if(lt(f,10000),{highs},{air})"
        "     )"
        "    )"
        "   )"
        "  )"
        " )"
        ")"
    ).format(**g)
    
def build_fg_eq8_only(eq8: dict) -> str:
    """
    Pure 8-band tone (firequalizer) → [out]
    No loudness or limiting. Intended for the corrective pre-pass.
    """
    expr = _eq8_expression(eq8)
    return f"firequalizer=gain='{expr}'[out]"

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

def derive_section_plans_from_single(base_plan: dict):
    """
    If LLM returns a single plan, derive gentler verse & firmer drop
    without changing tonality (keeps eq8).
    """
    import copy
    verse = copy.deepcopy(base_plan)
    drop  = copy.deepcopy(base_plan)

    # Verse a bit quieter & gentler thresholds
    verse["targets"]["lufs_i"] = float(max(-14.0, min(-9.0, verse["targets"].get("lufs_i", -11.5) - 1.0)))
    verse["targets"]["true_peak_db"] = float(min(-1.0, verse["targets"].get("true_peak_db", -1.2)))
    for k, delta in (("low_thr_db", +2), ("mid_thr_db", +2), ("high_thr_db", +2)):
        verse.setdefault("mb_comp", {})[k] = float(max(-30.0, min(-18.0, verse["mb_comp"].get(k, -24) + delta)))

    # Drop a touch louder & firmer low band
    drop["targets"]["lufs_i"] = float(max(-14.0, min(-9.0, drop["targets"].get("lufs_i", -11.5) + 0.7)))
    drop["targets"]["true_peak_db"] = float(min(-1.0, drop["targets"].get("true_peak_db", -1.2)))
    drop.setdefault("mb_comp", {})["low_thr_db"] = float(max(-30.0, min(-18.0, drop["mb_comp"].get("low_thr_db", -20) - 2)))

    return verse, drop

def _notch_chain(label_in: str, notches: list) -> tuple[str, str]:
    """
    Build up to MAX_NOTCHES parametric 'equalizer' notches in series.
    Returns (filter_snippet, last_label).
    - Uses FFmpeg equalizer (parametric): equalizer=f=<Hz>:t=q:w=<Q>:g=<dB>
    """
    cur = label_in
    parts = []
    for i, n in enumerate((notches or [])[:MAX_NOTCHES]):
        try:
            f = float(n.get("freq_hz", 1000.0))
            g = float(n.get("gain_db", -3.0))
            q = float(n.get("q", 8.0))
        except Exception:
            continue

        # clamp to safe limits
        f = clamp(f, *NOTCH_LIMITS["freq_hz"])
        g = clamp(g, *NOTCH_LIMITS["gain_db"])
        q = clamp(q, *NOTCH_LIMITS["q"])

        nxt = f"n{i}"
        parts.append(f"[{cur}]equalizer=f={f}:t=q:w={q}:g={g}[{nxt}]")
        cur = nxt

    return (";".join(parts) + (";" if parts else "")), cur

def build_fg_eq8_plus_notches(eq8: dict, notches: list) -> str:
    """
    8-band firequalizer followed by up to 3 parametric notches → [out].
    No loudness/limiting — use in the corrective cleanup stage only.
    """
    expr = _eq8_expression(eq8)

    # 8-band first → [pre]
    chain = f"firequalizer=gain='{expr}'[pre];"

    # optional notches in series
    notch_snip, last_lbl = _notch_chain("pre", notches or [])
    chain += notch_snip

    final = last_lbl if notch_snip else "pre"
    # finalize to a named pad [out]
    chain += f"[{final}]anull[out]"

    return chain

