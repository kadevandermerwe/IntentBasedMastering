# utils.py
import os, subprocess, uuid
import streamlit as st

def clamp(v, lo, hi):
    try:
        v = float(v)
    except Exception:
        return lo
    return max(lo, min(hi, v))

def safe_run(cmd: list[str]):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        st.error("❌ FFmpeg failed.")
        st.code(" ".join(cmd))
        st.error(e)
        raise

def session_tmp_path() -> str:
    """Stable /tmp folder per Streamlit session; avoids TemporaryDirectory churn."""
    sid = st.session_state.get("_session_id")
    if not sid:
        sid = str(uuid.uuid4())[:8]
        st.session_state["_session_id"] = sid
    base = f"/tmp/vale_{sid}"
    os.makedirs(base, exist_ok=True)
    return base
