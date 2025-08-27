# ai.py
import json
from utils import clamp
from dsp import BAND_NAMES
import streamlit as st

# --- in ai.py (or wherever llm_plan lives) ---
def llm_plan(analysis, intent, user_prompt, model, reference_txt="", reference_weight=0.0):
    if not ("OPENAI_API_KEY" in st.secrets["OPENAI_API_KEY"]):
        return None, "LLM disabled or missing key."

    # Clamp weight to [0,1]
    try:
        ref_w = float(reference_weight)
    except:
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
- If reference weight ~1.0, strongly bias tonal/level targets toward the REFERENCE’s *typical* style norms, but never violate guardrails or the actual ANALYSIS (don’t distort).
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
- Multiband thresholds in [-30, -18] dB (ratios implied)
- Saturation drive in [0, 3] dB
- Stereo amount in [-0.2, 0.2]

NOTES:
- If REFERENCE is a club/techno/house cue and weight is high, bias toward tighter lows, modest top, and LUFS consistent with that genre—but still fit THIS mix’s ANALYSIS and the user INTENT.
- Keep explanation to one sentence. Return VALID JSON ONLY.
""".strip()

    user = f"ANALYSIS:{json.dumps(analysis)}\nINTENT:{json.dumps(intent)}\nUSER_PROMPT:{user_prompt or ''}\nREFERENCE:{reference_txt or ''}"

    # ... call OpenAI as you already do, parse JSON, clamp values, return plan ...

    resp = client.chat.completions.create(
        model=model,
        temperature=0.25,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}]
    )
    content = resp.choices[0].message.content.strip()

    # Strip fences if any
    if content.startswith("```"):
        content = content.strip().split("\n",1)[-1]
        if content.endswith("```"):
            content = content[:-3].strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return None, "LLM returned no JSON."

    raw = json.loads(content[start:end+1])

    # ---- clamp helper
    def clamp_plan(p: dict) -> dict:
        p = dict(p)
        tgt  = p.get("targets", {}) or {}
        mb   = p.get("mb_comp", {}) or {}
        sat  = p.get("saturation", {}) or {}
        ster = p.get("stereo", {}) or {}
        eq8  = p.get("eq8")
        eq3  = p.get("eq")

        # targets
        tgt["lufs_i"]      = clamp(tgt.get("lufs_i", -11.5), -14.0, -9.0)
        tp = tgt.get("true_peak_db", -1.2)
        tgt["true_peak_db"] = tp if float(tp) <= -1.0 else -1.2

        # multiband thresholds
        mb["low_thr_db"]  = clamp(mb.get("low_thr_db",  -20), -30, -18)
        mb["mid_thr_db"]  = clamp(mb.get("mid_thr_db",  -24), -30, -18)
        mb["high_thr_db"] = clamp(mb.get("high_thr_db", -26), -30, -18)

        # saturation & stereo
        sat["drive_db"]   = clamp(sat.get("drive_db", 1.0), 0.0, 3.0)
        ster["amount"]    = clamp(ster.get("amount", 0.0), -0.2, 0.2)

        # eq8 (preferred)
        if isinstance(eq8, dict):
            p["eq8"] = {k: float(clamp(eq8.get(k, 0.0), -2.0, 2.0)) for k in BAND_NAMES}
            p.pop("eq", None)
        elif isinstance(eq3, dict):
            # legacy 3-band
            eq3["low_shelf_db"]  = clamp(eq3.get("low_shelf_db", 0.0), -2.0, 2.0)
            eq3["mud_cut_db"]    = clamp(eq3.get("mud_cut_db", 0.0), -2.0, 2.0)
            eq3["high_shelf_db"] = clamp(eq3.get("high_shelf_db",0.0), -2.0, 2.0)
            p["eq"] = {**eq3}
        else:
            # no eq block → neutral
            pass

        p["targets"], p["mb_comp"], p["saturation"], p["stereo"] = tgt, mb, sat, ster
        return p

    # Sectioned or single
    if "verse" in raw and "drop" in raw:
        return {
            "verse": clamp_plan(raw["verse"]),
            "drop":  clamp_plan(raw["drop"])
        }, "LLM plan generated (sectioned)."
    else:
        return clamp_plan(raw), "LLM plan generated (single)."
