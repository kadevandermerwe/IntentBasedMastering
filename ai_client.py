# ai_client.py
# Centralized OpenAI client creation with robust key resolution and proxy cleanup.

from __future__ import annotations
import os
from typing import Optional
from openai import OpenAI

def _strip_proxies_env() -> None:
    """Avoid httpx/proxies issues in some hosted environments."""
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.environ.get(k):
            os.environ.pop(k, None)

def resolve_api_key(passed_key: Optional[str] = None) -> Optional[str]:
    """Priority: function arg > Streamlit secrets (if available) > env var."""
    if passed_key and passed_key.strip():
        return passed_key.strip()

    # Import streamlit lazily so this module can be imported in non-Streamlit contexts
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key.strip()
    except Exception:
        pass

    key = os.environ.get("OPENAI_API_KEY", "")
    return key.strip() if key else None

def make_client(passed_key: Optional[str] = None) -> OpenAI:
    """Create a configured OpenAI client or raise RuntimeError if missing key."""
    _strip_proxies_env()
    key = resolve_api_key(passed_key)
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found (arg/secrets/env).")
    # Construct client (no proxies kwarg so older httpx builds won't choke)
    return OpenAI(api_key=key)
