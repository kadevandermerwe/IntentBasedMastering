# ai.py
import json
import streamlit as st
from openai import OpenAI

# Pull clamp; if not present, define a safe fallback
try:
    from utils import clamp
except Exception:
    def clamp(v, lo, hi):
        try:
            v = float(v)
        except Exception:
            v = lo
        return max(lo, min(hi, v))

# Pull band names; if missing, default to our 8-band layout
try:
    from dsp import BAND_NAMES  # expected: ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]
except Exception:
    BAND_NAMES = ["sub","low_bass","high_bass","low_mids","mids","high_mids","highs","air"]


def _extract_json_block(text: str) -> dict | None:
    """Extract the first {...} JSON block, stripping code fences if present."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        # ```json ... ``` or ``` ... ```
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[:-3].strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(s[start:end+1])
    except Exception:
        return None


def _clamp_plan(p: dict) -> dict:
    """Clamp and normalize plan dict. Supports eq8 (preferred) or legacy eq (3-band)."""
    if not isinstance(p, dict):
        p = {}

    tgt  = p.get("targets")    or {}
    mb   = p.get("mb_comp")    or {}
    sat  = p.get("saturation") or {}
    ster = p.get("stereo")     or {}
    eq8  = p.get("eq8")
    eq3  = p.get("eq")

    # Targets
    tgt["lufs_i"] = clamp(tgt.get("lufs_i", -11.5), -14.0, -9.0)
    tp = tgt.get("true_peak_db", -1.2)
    try:
        tp = float(tp)
    except Exception:
        tp = -1.2
    tgt["true_peak_db"] = tp if tp <= -1.0 else -1.2

    # Multiband thresholds
    mb["low_thr_db"]  = clamp(mb.get("low_thr_db",  -20), -30, -18)
    mb["mid_thr_db"]  = clamp(mb.get("mid_thr_db",  -24), -30, -18)
    mb["high_thr_db"] = clamp(mb.get("high_thr_db", -26), -30, -18)

    # Saturation & stereo
    sat["drive_db"] = clamp(sat.get("drive_db", 1.0), 0.0, 3.0)
    ster["amount"]  = clamp(ster.get("amount", 0.0), -0.2, 0.2)

    # EQ: prefer 8-band
    if isinstance(eq8, dict):
        p["eq8"] = {k: float(clamp(eq8.get(k, 0.0), -2.0, 2.0)) for k in BAND_NAMES}
        p.pop("eq", None)
    elif isinstance(eq3, dict):
        eq3["low_shelf_db"]  = clamp(eq3.get("low_shelf_db",  0.0), -2.0, 2.0)
        eq3["mud_cut_db"]    = clamp(eq3.get("mud_cut_db",    0.0), -2.0, 2.0)
        eq3["high_shelf_db"] = clamp(eq3.get("high_shelf_db", 0.0), -2.0, 2.0)
        p["eq"] = {**eq3}
    # else: no EQ block → leave as-is (neutral)

    p["targets"], p["mb_comp"], p["saturation"], p["stereo"] = tgt, mb, sat, ster
    # Keep explanation if present
    if "explanation" in p and not isinstance(p["explanation"], str):
        p["explanation"] = str(p["explanation"])
    return p


def llm_plan(analysis, intent, user_prompt, model, reference_txt: str = "", reference_weight: float = 0.0):
    """Ask the LLM for a mastering plan. May return sectioned {'verse','drop'} or a single plan dict."""
    # Check key safely
    if "OPENAI_API_KEY" not in st.secrets or not st.secrets["OPENAI_API_KEY"]:
        return None, "LLM disabled or missing key."

    # Create client *inside* function
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Clamp weight to [0,1]
    try:
        ref_w = float(reference_weight)
    except Exception:
        ref_w = 0.0
    ref_w = clamp(ref_w, 0.0, 1.0)

    # System prompt
    system = f"""
You are an expert mastering engineer.
Blend three signals of evidence:
1) ANALYSIS (numbers from THIS premaster),
2) INTENT (user sliders + text),
3) REFERENCE (named track/artist), with bias weight = {ref_w:.2f} in [0,1].

BEHAVIOR:
- If reference weight ~0.0, treat REFERENCE as color only; prioritize ANALYSIS + INTENT.
- If reference weight ~1.0, strongly bias tonal/level targets toward the REFERENCE’s typical style norms, but never violate guardrails or the actual ANALYSIS.
- If REFERENCE is empty, behave as if weight=0.0.

OUTPUT STRICT JSON ONLY with keys:
{{
  "targets": {{ "lufs_i": float, "true_peak_db": float }},
  "eq8": {{
    "sub": float, "low_bass": float, "high_bass": float, "low_mids": float,
    "mids": float, "high_mids": float, "highs": float, "air": float
  }},
  "mb_comp": {{ "low_thr_db": float, "mid_thr_db": float, "high_thr_db": float }},
  "saturation": {{ "drive_db": float }},
  "stereo": {{ "amount": float }},
  "explanation": string
}}

CONSTRAINTS:
- LUFS in [-14, -9]; true_peak ≤ -1.0 (prefer -1.2)
- EQ band moves in ±2.0 dB each
- Multiband thresholds in [-30, -18] dB
- Saturation drive in [0, 3] dB
- Stereo amount in [-0.2, 0.2]

NOTES:
- If REFERENCE is a club/techno/house cue and weight is high, bias toward tighter lows, modest top, and LUFS consistent with that genre—but still fit THIS mix’s ANALYSIS and the user INTENT.
- Keep explanation to one sentence. Return VALID JSON ONLY.
""".strip()

    user = (
        f"ANALYSIS:{json.dumps(analysis)}\n"
        f"INTENT:{json.dumps(intent)}\n"
        f"USER_PROMPT:{user_prompt or ''}\n"
        f"REFERENCE:{reference_txt or ''}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.25,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
    except Exception as e:
        return None, f"LLM error: {e}"

    raw = _extract_json_block(content)
    if raw is None:
        return None, "LLM returned no JSON."

    # Sectioned or single
    if isinstance(raw, dict) and "verse" in raw and "drop" in raw:
        try:
            return {
                "verse": _clamp_plan(raw["verse"]),
                "drop":  _clamp_plan(raw["drop"]),
            }, "LLM plan generated (sectioned)."
        except Exception as e:
            return None, f"Plan parse error (sectioned): {e}"
    else:
        try:
            return _clamp_plan(raw), "LLM plan generated (single)."
        except Exception as e:
            return None, f"Plan parse error (single): {e}"
