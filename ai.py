# ai.py
import json
from utils import clamp
from dsp import BAND_NAMES

def llm_plan(analysis: dict, intent: dict, prompt_text: str, model: str, api_key: str):
    """
    Ask the LLM for either a sectioned plan {"verse":{...},"drop":{...}}
    or a single plan. Returns (plan_dict, message).
    """
    if not api_key:
        return None, "OpenAI key missing."

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system = """
You are an expert mastering engineer.

Design a MUSICAL mastering plan that blends:
1) ANALYSIS (numbers from this exact premaster),
2) USER INTENT (sliders + text),
3) REFERENCE vibe or artist/track (if provided).

Return STRICT JSON ONLY. Prefer sectioned plans if the track likely has verses and drops.

Schema A (sectioned):
{
  "verse": {
    "targets": {"lufs_i": float, "true_peak_db": float},
    "mb_comp": {"low_thr_db": float, "mid_thr_db": float, "high_thr_db": float},
    "saturation": {"drive_db": float},
    "stereo": {"amount": float},
    "eq8": {
      "sub": float, "low_bass": float, "high_bass": float, "low_mids": float,
      "mids": float, "high_mids": float, "highs": float, "air": float
    },
    "explanation": string
  },
  "drop": { ...same keys... }
}

Schema B (single):
{
  "targets": {...},
  "mb_comp": {...},
  "saturation": {...},
  "stereo": {...},
  "eq8": {...} OR "eq": {"low_shelf_db": float, "mud_cut_db": float, "high_shelf_db": float},
  "explanation": string
}

GUARDRAILS (HARD LIMITS):
- LUFS target ∈ [-14.0, -9.0]
- True peak ≤ -1.0 dBTP (prefer -1.2)
- Multiband thresholds ∈ [-30, -18] dB
- eq8 band gains ∈ [-2.0, +2.0] dB (small, musical moves)
- Stereo amount ∈ [-0.2, +0.2]
- Saturation drive_db ∈ [0.0, 3.0]

Explain in one sentence max (musical intent, not tech).
Return VALID JSON ONLY.
""".strip()

    user = f"ANALYSIS:{json.dumps(analysis)}\nINTENT:{json.dumps(intent)}\nREFERENCE:{prompt_text or ''}"

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
