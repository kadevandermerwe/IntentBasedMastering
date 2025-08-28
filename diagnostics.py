# diagnostics.py
# Validation helpers to ensure LLM plans are well-formed before rendering.

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from schema import BAND_NAMES, PLAN_KEYS_SINGLE, LIMITS

def _in_range(x: float, lo: float, hi: float) -> bool:
    try:
        xv = float(x)
    except Exception:
        return False
    return (lo <= xv <= hi)

def is_sectioned_plan(plan: Dict[str, Any]) -> bool:
    """Heuristic: sectioned if it has both 'verse' and 'drop' dicts."""
    return isinstance(plan, dict) and ("verse" in plan and "drop" in plan)

def _validate_eq8(eq8: Dict[str, Any], errors: List[str]) -> None:
    # All required bands present
    missing = [b for b in BAND_NAMES if b not in eq8]
    if missing:
        errors.append(f"eq8 missing bands: {missing}")
        return
    # All within ±2 dB
    lo, hi = LIMITS["eq8_db"]
    for b in BAND_NAMES:
        v = eq8.get(b)
        if not _in_range(v, lo, hi):
            errors.append(f"eq8[{b}] out of range {lo}..{hi}: {v}")

def _validate_single_plan(plan: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    # Targets
    tgt = plan.get("targets", {})
    if not _in_range(tgt.get("lufs_i"), *LIMITS["lufs_i"]):
        errors.append(f"targets.lufs_i out of range {LIMITS['lufs_i']}: {tgt.get('lufs_i')}")
    tp = tgt.get("true_peak_db")
    if tp is None or float(tp) > -1.0:
        errors.append(f"targets.true_peak_db must be ≤ -1.0: {tp}")

    # EQ8
    eq8 = plan.get("eq8")
    if not isinstance(eq8, dict):
        errors.append("eq8 block missing or not a dict")
    else:
        _validate_eq8(eq8, errors)

    # Multiband thresholds
    mb = plan.get("mb_comp", {})
    lo, hi = LIMITS["mb_thr_db"]
    for k in ("low_thr_db", "mid_thr_db", "high_thr_db"):
        if not _in_range(mb.get(k), lo, hi):
            errors.append(f"mb_comp.{k} out of range {lo}..{hi}: {mb.get(k)}")

    # Saturation & Stereo
    if not _in_range(plan.get("saturation", {}).get("drive_db"), *LIMITS["saturation_drive_db"]):
        errors.append(f"saturation.drive_db out of range {LIMITS['saturation_drive_db']}: {plan.get('saturation',{}).get('drive_db')}")
    if not _in_range(plan.get("stereo", {}).get("amount"), *LIMITS["stereo_amount"]):
        errors.append(f"stereo.amount out of range {LIMITS['stereo_amount']}: {plan.get('stereo',{}).get('amount')}")

    return errors

def validate_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single or sectioned plan. Returns (ok, list_of_errors)."""
    if not isinstance(plan, dict):
        return False, ["Plan is not a dict"]

    if is_sectioned_plan(plan):
        errs: List[str] = []
        for sect in ("verse", "drop"):
            p = plan.get(sect)
            if not isinstance(p, dict):
                errs.append(f"Section '{sect}' missing or not a dict")
                continue
            errs.extend([f"{sect}: {e}" for e in _validate_single_plan(p)])
        return (len(errs) == 0), errs
    else:
        errs = _validate_single_plan(plan)
        return (len(errs) == 0), errs
