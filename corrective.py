# corrective.py
import json
import os
import streamlit as st
from typing import Dict, Tuple

from utils import clamp
from schema import BAND_NAMES, LIMITS
from ai_client import make_client
from dsp import build_fg_eq8_only, render_variant, analyze_audio  # you'll add build_fg_eq8_only (below)

def _clamp_eq8(eq8: Dict[str, float]) -> Dict[str, float]:
    lo, hi = LIMITS["eq8_db"]
    return {b: float(clamp(eq8.get(b, 0.0), lo, hi)) for b in BAND_NAMES}

def llm_corrective_eq8(
    api_key: str | None,
    analysis: dict,
    user_prompt: str,
    model: str,
) -> Tuple[Dict[str, float] | None, str]:
    """
    Returns (eq8 dict or None, message).
    eq8 uses the same 8 bands you already standardized.
    """
    try:
        client = make_client(api_key)
    except Exception as e:
        return None, f"OpenAI client init failed: {e}"

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
  "notes": "one short sentence"
}

Rules:
- Band gains are in dB, each in [-2.0, +2.0].
- Prefer CUTS for mud/boxiness/harshness. Avoid boosting lows/highs unless analysis is clearly too thin.
- This is not loudness or saturation; ONLY gentle balance cleanup.
""".strip()

    user = f"ANALYSIS:{json.dumps(analysis)}\nUSER_PROMPT:{user_prompt or ''}"

    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}]
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        return None, f"LLM corrective request failed: {e}"

    # Extract JSON robustly
    try:
        if content.startswith("```"):
            content = content.strip().split("\n",1)[-1]
            if content.endswith("```"): content = content[:-3].strip()
            if content.lower().startswith("json"): content = content[4:].strip()
        start, end = content.find("{"), content.rfind("}")
        data = json.loads(content[start:end+1])
    except Exception as e:
        return None, f"Bad JSON from corrective LLM: {e}"

    eq8 = data.get("corrective_eq8")
    if not isinstance(eq8, dict):
        return None, "Corrective JSON missing 'corrective_eq8'."

    return _clamp_eq8(eq8), data.get("notes", "corrective plan")
    

def apply_corrective_eq(
    in_path: str,
    out_path: str,
    eq8: Dict[str, float],
) -> None:
    """
    Render a corrected premaster using the 8-band firequalizer curve only.
    No loudness/limiting here. Keeps headroom.
    """
    fg = build_fg_eq8_only(eq8)  # returns a [out] chain that just EQs
    render_variant(in_path, out_path, fg)
