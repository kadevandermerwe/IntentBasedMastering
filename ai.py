# ai.py
import os, json
from typing import Tuple, Any, Dict
from utils import clamp
from dsp import BAND_NAMES

# --- tiny helpers -------------------------------------------------------------

def _strip_proxies_env() -> None:
    # Some hosted envs set HTTP(S)_PROXY which breaks older httpx paths via OpenAI
    for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
        os.environ.pop(k, None)

def _get_api_key(passed_key: str | None = None) -> str:
    """Prefer explicitly-passed key. Otherwise pull from Streamlit secrets."""
    if passed_key and str(passed_key).strip():
        return str(passed_key).strip()
    try:
        import streamlit as st  # local import to avoid circular import at module load
        return (st.secrets.get("OPENAI_API_KEY", "") or "").strip()
    except Exception:
        return ""

def _extract_json_block(text: str) -> Dict[str, Any]:
    """Extract first {...} block from LLM text (strips code fences if present)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[:-3].strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("No JSON object found in model response.")
    return json.loads(s[a:b+1])

# --- main entry ---------------------------------------------------------------

def llm_plan(
    analysis: Dict[str, Any],
    intent: Dict[str, float],
    user_prompt: str,
    model: str,
    reference_txt: str = "",
    reference_weight: float = 0.0,
    api_key: str | None = None,
) -> Tuple[Dict[str, Any] | None, str]:
    """
    Intent + analysis + (optional) reference → STRICT JSON plan (single or sectioned).
    Returns (plan_dict_or_None, message).
    """
    key = _get_api_key(api_key)
    if not key:
        return None, "OPENAI_API_KEY missing or empty (pass api_key or set in Streamlit Secrets)."

    _strip_proxies_env()

    # lazy import OpenAI inside function (avoids import at app boot)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
    except Exception as e:
        return None, f"OpenAI client init failed: {e}"

    # clamp ref weight
    try:
        ref_w = float(reference_weight)
    except Exception:
        ref_w = 0.0
    ref_w = max(0.0, min(1.0, ref_w))

    system = f"""
You are an expert mastering engineer.
Blend three signals of evidence:
1) ANALYSIS (numbers from THIS premaster),
2) INTENT (user sliders + text),
3) REFERENCE (named track/artist), with bias weight = {ref_w:.2f} in [0,1].

BEHAVIOR:
- If reference weight ~0.0, treat REFERENCE as color only; prioritize ANALYSIS + INTENT.
- If reference weight ~1.0, bias tonal/level targets toward the REFERENCE’s typical style norms, but never violate guardrails or the actual ANALYSIS.
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

Keep explanation to one sentence. Return VALID JSON ONLY.
""".strip()

    user = (
        f"ANALYSIS:{json.dumps(analysis)}\n"
        f"INTENT:{json.dumps(intent)}\n"
        f"USER_PROMPT:{user_prompt or ''}\n"
        f"REFERENCE:{reference_txt or ''}"
    )

    # call LLM
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.25,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
        )
        content = resp.choices[0].message.content or ""
        raw = _extract_json_block(content)
    except Exception as e:
        return None, f"LLM request/parse failed: {e}"

    # clamp + normalize
    def _clamp_plan(p: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(p or {})
        tgt  = dict(p.get("targets")    or {})
        mb   = dict(p.get("mb_comp")    or {})
        sat  = dict(p.get("saturation") or {})
        ster = dict(p.get("stereo")     or {})
        eq8  = p.get("eq8")
        eq3  = p.get("eq")

        # targets
        tgt["lufs_i"]      = float(clamp(tgt.get("lufs_i", -11.5), -14.0, -9.0))
        tp                 = float(tgt.get("true_peak_db", -1.2))
        tgt["true_peak_db"] = tp if tp <= -1.0 else -1.2

        # multiband thresholds
        mb["low_thr_db"]   = float(clamp(mb.get("low_thr_db",  -20), -30, -18))
        mb["mid_thr_db"]   = float(clamp(mb.get("mid_thr_db",  -24), -30, -18))
        mb["high_thr_db"]  = float(clamp(mb.get("high_thr_db", -26), -30, -18))

        # sat & stereo
        sat["drive_db"]    = float(clamp(sat.get("drive_db", 1.0), 0.0, 3.0))
        ster["amount"]     = float(clamp(ster.get("amount", 0.0), -0.2, 0.2))

        # preferred 8-band EQ
        if isinstance(eq8, dict):
            p["eq8"] = {name: float(clamp(eq8.get(name, 0.0), -2.0, 2.0)) for name in BAND_NAMES}
            p.pop("eq", None)
        elif isinstance(eq3, dict):
            p["eq"] = {
                "low_shelf_db":  float(clamp(eq3.get("low_shelf_db",  0.0), -2.0, 2.0)),
                "mud_cut_db":    float(clamp(eq3.get("mud_cut_db",    0.0), -2.0, 2.0)),
                "high_shelf_db": float(clamp(eq3.get("high_shelf_db", 0.0), -2.0, 2.0)),
            }

        p["targets"], p["mb_comp"], p["saturation"], p["stereo"] = tgt, mb, sat, ster
        return p

    if "verse" in raw and "drop" in raw:
        return {
            "verse": _clamp_plan(raw["verse"]),
            "drop":  _clamp_plan(raw["drop"]),
        }, "LLM plan generated (sectioned)."

    return _clamp_plan(raw), "LLM plan generated (single)."
