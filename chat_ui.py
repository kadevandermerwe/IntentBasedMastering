# chat_ui.py
import html as pyhtml
import random
import re
import streamlit as st
from streamlit.components.v1 import html as components_html

# ---------------- utilities ----------------
def _esc(s: str) -> str:
    return pyhtml.escape(s or "")

_VALE_OPENERS = ["so…", "okay—", "alright,", "cool—", "nice—", "gotcha.", "heads up—"]

def _soften_numbers(txt: str) -> str:
    return re.sub(r"([+\-]?\d+(?:\.\d+)?)\s*dB", lambda m: f"{round(float(m.group(1)))} dB", txt)

def vale_say(message: str) -> str:
    opener = random.choice(_VALE_OPENERS)
    msg = (message or "").strip()
    msg = re.sub(r'^(analysis|result|note)[:\- ]+', '', msg, flags=re.I)
    msg = _soften_numbers(msg)
    if not msg:
        msg = "all set."
    return f"{opener} {msg[0].lower()}{msg[1:] if len(msg)>1 else ''}"

# ---------- state API ----------
def add_chat(role: str, text: str, state_key: str = "chat", attachments: list | None = None):
    """
    attachments: optional list of dicts, each like:
      {"type":"img",  "src":"data:image/png;base64,....", "alt":"Tonal curve"}
      {"type":"html", "html":"<div>custom snippet</div>"}
    """
    st.session_state.setdefault(state_key, [])
    st.session_state[state_key].append({
        "role": (role or "assistant"),
        "text": text or "",
        "attachments": attachments or []
    })

# ---------- renderer ----------
def render_chat(
    container,
    state_key: str = "chat",
    height: int = 420,
    avatar_img_b64: str | None = None,
    avatar_letter: str = "V",
):
    st.session_state.setdefault(state_key, [])
    if not st.session_state[state_key]:
        add_chat("assistant", vale_say("upload a premaster when you’re ready—i’ll give it a quick once-over."), state_key)

    msgs = st.session_state[state_key]

    # build rows
    rows_html = []
    for m in msgs:
        role = (m.get("role") or "assistant").lower()
        cls = "assistant" if role != "user" else "user"
        body = _esc(m.get("text",""))
        # attachments
        atts = m.get("attachments", []) or []
        att_html_parts = []
        for a in atts:
            if isinstance(a, dict) and a.get("type") == "img" and a.get("src"):
                alt = _esc(a.get("alt") or "")
                att_html_parts.append(f'<div class="att-img"><img src="{a["src"]}" alt="{alt}"></div>')
            elif isinstance(a, dict) and a.get("type") == "html" and a.get("html"):
                att_html_parts.append(f'<div class="att-html">{a["html"]}</div>')
        att_html = "\n".join(att_html_parts)

        rows_html.append(
            f"""
            <div class="msg {cls}">
              <div class="msg-role">{_esc(role)}</div>
              <div class="msg-body">{body}{att_html}</div>
            </div>
            """
        )
    rows_html = "\n".join(rows_html)

    # avatar
    if avatar_img_b64:
        avatar_inner = f'<img src="data:image/png;base64,{avatar_img_b64}" style="width:100%;height:100%;border-radius:50%;">'
    else:
        avatar_inner = _esc(avatar_letter)

    # html
    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        :root {{
          --ink:#1b1f23; --ink-dim:#5b6570; --panel:#fff; --border:#e6eaf0; --accent:#5EA2FF; --accent-ghost:rgba(94,162,255,.18);
        }}
        * {{ box-sizing:border-box; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial; }}
        body {{ margin:0; background:transparent; }}

        .panel {{ padding:10px; background:transparent; }}
        .hdr {{ display:flex; justify-content:center; align-items:center; margin-bottom:10px; }}
        .avatar {{
          width:44px; height:44px; border-radius:50%; border:1px solid var(--border);
          display:flex; align-items:center; justify-content:center; color:var(--accent); font-weight:800; font-size:14px;
          background:
            radial-gradient(120% 120% at 20% 20%, rgba(94,162,255,0.18), transparent 55%),
            linear-gradient(180deg, rgba(94,162,255,0.10), rgba(94,162,255,0.04));
        }}

        .box {{
          height:{height-90}px; overflow-y:auto; border:1px solid var(--border); border-radius:6px; padding:8px; background:#fff;
        }}
        .msg {{ border:1px dashed var(--border); border-radius:4px; padding:8px 10px; margin:0 0 8px 0; }}
        .msg-role {{ font-size:10px; color:var(--ink-dim); margin-bottom:4px; letter-spacing:.3px; text-transform:uppercase; }}
        .msg-body {{ font-size:13px; color:var(--ink); line-height:1.35; }}

        .att-img {{ margin-top:8px; }}
        .att-img img {{ width:100%; height:auto; border:1px solid var(--border); border-radius:4px; display:block; }}
        .att-html {{ margin-top:8px; }}
      </style>
    </head>
    <body>
      <div class="panel">
        <div class="hdr"><div class="avatar">{avatar_inner}</div></div>
        <div id="vale-box" class="box">{rows_html}</div>
      </div>
      <script>
        const box = document.getElementById('vale-box');
        const scrollDown = () => {{ try {{ box.scrollTop = box.scrollHeight; }} catch(e) {{}} }};
        scrollDown(); setTimeout(scrollDown, 50); setTimeout(scrollDown, 150);
        try {{ new MutationObserver(scrollDown).observe(box, {{childList:true, subtree:true}}); }} catch(e) {{}}
      </script>
    </body>
    </html>
    """

    with container:
        components_html(html, height=height, scrolling=False)

        # input form (outside iframe)
        with st.form(key="vale_chat_form", clear_on_submit=True):
            user_txt = st.text_input("Message", label_visibility="collapsed", placeholder="Type to Vale…")
            sent = st.form_submit_button("Send")
        if sent and user_txt.strip():
            add_chat("user", user_txt.strip(), state_key)
            add_chat("assistant", vale_say("cool—i’ll fold that in next pass."), state_key)
            st.experimental_rerun()
