# schema.py
# Central contract for band names, ranges, and safety limits used across the app.

from __future__ import annotations

# Canonical 8-band labels (order matters)
BAND_NAMES = [
    "sub",        # ~20–80 Hz
    "low_bass",   # ~81–120 Hz
    "high_bass",  # ~121–250 Hz
    "low_mids",   # ~251–500 Hz
    "mids",       # ~500–3500 Hz
    "high_mids",  # ~3.5k–8k Hz
    "highs",      # ~8k–10k Hz
    "air",        # ~10k–20k Hz
]

# Approximate analysis ranges per band (for reference & future plots/tools)
BAND_RANGES_HZ = {
    "sub": (20, 80),
    "low_bass": (81, 120),
    "high_bass": (121, 250),
    "low_mids": (251, 500),
    "mids": (500, 3500),
    "high_mids": (3500, 8000),
    "highs": (8000, 10000),
    "air": (10000, 20000),
}

# Safety/guardrail limits used by the LLM clamping & diagnostics
LIMITS = {
    "lufs_i": (-14.0, -9.0),
    "true_peak_db": (-60.0, -1.0),  # we enforce ≤ -1.0
    "eq8_db": (-2.0, 2.0),
    "mb_thr_db": (-30.0, -18.0),
    "saturation_drive_db": (0.0, 3.0),
    "stereo_amount": (-0.2, 0.2),
}

# Expected top-level keys for a single plan (not sectioned)
PLAN_KEYS_SINGLE = {
    "targets": ["lufs_i", "true_peak_db"],
    "eq8": BAND_NAMES,  # dict with each band key present
    "mb_comp": ["low_thr_db", "mid_thr_db", "high_thr_db"],
    "saturation": ["drive_db"],
    "stereo": ["amount"],
    # "explanation": string (optional for rendering; present for UX)
}

# schema.py (append or merge)
MAX_NOTCHES = 3
NOTCH_LIMITS = {
    "freq_hz": (60.0, 16000.0),   # stay out of infrasonic & dog-whistle territory
    "gain_db": (-9.0, -1.0),      # cuts only for corrective (no boosts)
    "q":       (4.0, 16.0),       # reasonably narrow
}

