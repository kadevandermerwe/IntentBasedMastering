# ai.py
import os, json
import streamlit as st
from utils import clamp
from schema import BAND_NAMES
from ai_client import make_client 


# ---------------- helpers ----------------

def _strip_proxies_env():
    """Avoid httpx/proxies bugs in some hosted envs."""
    for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
        if os.environ.get(k):
            os.environ.pop(k, None)

def _resolve_api_key(passed_key: str | None) -> str | None:
    """Priority: function arg > Streamlit secrets > environment variable."""
    if passed_key and passed_key.strip():
        return passed_key.strip()
    key = st.secrets.get("OPENAI_API_KEY", "")
    if key:
        return key.strip()
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key.strip()
    return None

# ---------------- main LLM function ----------------

def llm_plan(
    analysis: dict,
    intent: dict,
    user_prompt: str,
    model: str,
    reference_txt: str = "",
    reference_weight: float = 0.0,
    api_key: str | None = None,
):
    """
    Intent + analysis + optional reference → STRICT JSON plan (single or sectioned).
    Returns: (plan_dict_or_None, message_str)
    """
    # Resolve key & print loud diagnostics
    key = _resolve_api_key(api_key)
    st.sidebar.write(f"🔑 OPENAI key loaded: {'yes' if key else 'no'}  (len={len(key) if key else 0})")

    if not key:
        return None, "LLM disabled or missing key."

    # Clean proxy env first to avoid 'proxies' kwarg errors in http stack
    _strip_proxies_env()

    # Lazy import to avoid module init failures before we know we have a key
    try:
        from openai import OpenAI
    except Exception as e:
        return None, f"OpenAI SDK import failed: {e}"

    # Create client *after* sanitizing env
    try:
        client = make_client(api_key)   # instead of OpenAI(api_key=key)
    except Exception as e:
        return None, f"Failed to init OpenAI client: {e}"

    # Clamp weight to [0,1]
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

NOTES:
- If REFERENCE is club/techno/house and weight is high, bias toward tighter lows, modest top, and genre-typical LUFS—still fitting THIS mix’s ANALYSIS and user INTENT.
- Keep explanation to one sentence. Return VALID JSON ONLY.
""".strip()

    user = (
        f"ANALYSIS:{json.dumps(analysis)}\n"
        f"INTENT:{json.dumps(intent)}\n"
        f"USER_PROMPT:{user_prompt or ''}\n"
        f"REFERENCE:{reference_txt or ''}"
    )

    # Call the model
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.25,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":user}
            ],
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        return None, f"LLM request failed: {e}"

    # Robust JSON extraction
    try:
        if content.startswith("```"):
            content = content.strip().split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3].strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            return None, "LLM returned no JSON."
        raw = json.loads(content[start:end+1])
    except Exception as e:
        return None, f"Bad JSON from LLM: {e}"

    # Clamp & normalize
    def _clamp_plan(p: dict) -> dict:
        p = dict(p or {})
        tgt  = dict(p.get("targets")    or {})
        mb   = dict(p.get("mb_comp")    or {})
        sat  = dict(p.get("saturation") or {})
        ster = dict(p.get("stereo")     or {})
        eq8  = p.get("eq8")
        eq3  = p.get("eq")

        tgt["lufs_i"] = float(clamp(tgt.get("lufs_i", -11.5), -14.0, -9.0))
        tp = float(tgt.get("true_peak_db", -1.2))
        tgt["true_peak_db"] = tp if tp <= -1.0 else -1.2

        mb["low_thr_db"]  = float(clamp(mb.get("low_thr_db",  -20), -30, -18))
        mb["mid_thr_db"]  = float(clamp(mb.get("mid_thr_db",  -24), -30, -18))
        mb["high_thr_db"] = float(clamp(mb.get("high_thr_db", -26), -30, -18))

        sat["drive_db"] = float(clamp(sat.get("drive_db", 1.0), 0.0, 3.0))
        ster["amount"]  = float(clamp(ster.get("amount", 0.0), -0.2, 0.2))

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
        return {"verse": _clamp_plan(raw["verse"]), "drop": _clamp_plan(raw["drop"])}, "LLM plan generated (sectioned)."
    else:
        return _clamp_plan(raw), "LLM plan generated (single)."
