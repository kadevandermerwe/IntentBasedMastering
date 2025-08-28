# corrective.py
import json, os
import streamlit as st
from typing import Dict, Tuple, List

from utils import clamp
from schema import BAND_NAMES, LIMITS, MAX_NOTCHES, NOTCH_LIMITS
from ai_client import make_client
from dsp import build_fg_eq8_only, build_fg_eq8_plus_notches, render_variant

def _clamp_eq8(eq8: Dict[str, float]) -> Dict[str, float]:
    lo, hi = LIMITS["eq8_db"]
    return {b: float(clamp(eq8.get(b, 0.0), lo, hi)) for b in BAND_NAMES}

def _clamp_notches(notches: List[dict]) -> List[dict]:
    clean = []
    for n in (notches or [])[:MAX_NOTCHES]:
        try:
            f = float(n.get("freq_hz", 1000.0))
            g = float(n.get("gain_db", -3.0))
            q = float(n.get("q", 8.0))
        except Exception:
            continue
        f = clamp(f, *NOTCH_LIMITS["freq_hz"])
        g = clamp(g, *NOTCH_LIMITS["gain_db"])
        q = clamp(q, *NOTCH_LIMITS["q"])
        clean.append({"freq_hz": f, "gain_db": g, "q": q})
    return clean

def llm_corrective_cleanup(
    api_key: str | None,
    analysis: dict,
    user_prompt: str,
    model: str,
) -> Tuple[Dict[str, float] | None, List[dict] | None, str]:
    """
    Returns (eq8 dict or None, notches list or None, message).
    """
    try:
        client = make_client(api_key)
    except Exception as e:
        return None, None, f"OpenAI client init failed: {e}"

    system = """
You are a mastering engineer doing a PRE-MASTER corrective cleanup.
Goal: gently reduce BOOMINESS, MUD, BOXINESS and HARSHNESS before final mastering.
Do NOT do tonal styling here—that's for mastering. Keep moves subtle.

Output STRICT JSON ONLY:
{
  "corrective_eq8": {
    "sub": float, "low_bass": float, "high_bass": float, "low_mids": float,
    "mids": float, "high_mids": float, "highs": float, "air": float
  },
  "notches": [
    {"freq_hz": float, "gain_db": float, "q": float},
    {"freq_hz": float, "gain_db": float, "q": float},
    {"freq_hz": float, "gain_db": float, "q": float}
  ],
  "notes": "one short sentence"
}

Rules:
- 8-band moves are in dB within [-2.0, +2.0]; prefer CUTS for mud/boxiness/harshness.
- 'notches' are optional (0-3 entries). Use ONLY for obvious narrow problems (resonances, harshness).
- Notch constraints: freq 60–16000 Hz, gain -9…-1 dB (cuts only), Q 4–16.
- This is NOT loudness, compression, saturation, or creative tone.
""".strip()

    user = f"ANALYSIS:{json.dumps(analysis)}\nUSER_PROMPT:{user_prompt or ''}"

    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.15,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}]
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        return None, None, f"LLM corrective request failed: {e}"

    # Extract JSON
    try:
        if content.startswith("```"):
            content = content.strip().split("\n",1)[-1]
            if content.endswith("```"): content = content[:-3].strip()
            if content.lower().startswith("json"): content = content[4:].strip()
        start, end = content.find("{"), content.rfind("}")
        data = json.loads(content[start:end+1])
    except Exception as e:
        return None, None, f"Bad JSON from corrective LLM: {e}"

    eq8 = data.get("corrective_eq8")
    notches = data.get("notches", [])
    if not isinstance(eq8, dict):
        return None, None, "Corrective JSON missing 'corrective_eq8'."

    return _clamp_eq8(eq8), _clamp_notches(notches), data.get("notes", "corrective plan")

def apply_corrective_eq(
    in_path: str,
    out_path: str,
    eq8: Dict[str, float],
    notches: List[dict] | None = None,
) -> None:
    """
    Render a corrected premaster using the 8-band curve + up to 3 optional notches.
    No loudness/limiting here. Keeps headroom.
    """
    if notches:
        fg = build_fg_eq8_plus_notches(eq8, notches)
    else:
        fg = build_fg_eq8_only(eq8)
    render_variant(in_path, out_path, fg)
