# chat_ui.py
import html as pyhtml
import random
import re
import streamlit as st
from streamlit.components.v1 import html as components_html

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

# ------- NEW: allow html blocks -------
def add_chat(role: str, text: str | None = None, state_key: str = "chat", html: str | None = None):
    """Append a chat message. Pass either text (escaped) or html (trusted)."""
    st.session_state.setdefault(state_key, [])
    st.session_state[state_key].append({
        "role": role or "assistant",
        "text": text or "",
        "html": html or "",
    })

def render_chat(
    container,
    state_key: str = "chat",
    height: int = 420,
    avatar_img_b64: str | None = None,
):
    st.session_state.setdefault(state_key, [])
    if not st.session_state[state_key]:
        add_chat("assistant", vale_say("upload a premaster when you’re ready—i’ll give it a quick once-over."), state_key)

    msgs = st.session_state[state_key]

    rows = []
    for m in msgs:
        role = (m.get("role") or "assistant").lower()
        cls = "assistant" if role != "user" else "user"
        if m.get("html"):
            body = m["html"]  # trusted (our code builds it)
        else:
            body = _esc(m.get("text", ""))  # escaped

        rows.append(f"""
          <div class="msg {cls}">
            <div class="msg-role">{_esc(role)}</div>
            <div class="msg-body">{body}</div>
          </div>
        """)
    rows_html = "\n".join(rows)

    avatar_inner = f'<img src="data:image/png;base64,{avatar_img_b64}" style="width:100%;height:100%;border-radius:0%;">' if avatar_img_b64 else "V"

    html_doc = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        :root {{
          --ink: #e6e9ee;
          --ink-dim: #93a0b4;
          --panel: #10141b;
          --panel-soft: #0c1016;
          --border: #1b2330;
          --accent: #8ab4ff;
          --accent-ghost: rgba(138,180,255,.15);
        }}
        * {{ box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial; }}
        body {{ margin:0; background:transparent; color:var(--ink); }}

        .panel {{ padding: 10px; background: transparent; }}
        .hdr {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
        .avatar {{ width:40px; height:40px; border-radius:50%; display:grid; place-items:center; color:var(--accent); font-weight:700; }}
        .box {{
          height: {height-90}px;
          overflow-y: auto;
          padding: 6px;
          border: 1px solid var(--border);
          border-radius: 6px;
          background: var(--panel-soft);
          -ms-overflow-style: none; scrollbar-width: none;
        }}
        .box::-webkit-scrollbar {{ display: none; }}

        .msg {{
          background: var(--panel);
          border: 1px dashed var(--border);
          border-radius: 6px;
          padding: 8px 10px;
          margin: 0 0 8px 0;
        }}
        .msg.user {{ background: #0d1219; }}
        .msg-role {{ font-size: 10px; color: var(--ink-dim); margin-bottom: 6px; letter-spacing: .35px; text-transform: uppercase; }}
        .msg-body {{ font-size: 13px; line-height: 1.4; }}
        .chip {{ display:inline-block; padding:2px 6px; border:1px solid var(--border); border-radius:6px; background:#0d1218; color:#c7d2e3; font-size:11px; }}
        .thumb {{ border:1px solid var(--border); border-radius:6px; background:#0a0f15; padding:6px; }}
        .thumb img {{ display:block; width:100%; height:auto; border-radius:4px; }}
      </style>
    </head>
    <body>
      <div class="panel">
        <div class="hdr">
          <div class="avatar">{avatar_inner}</div>
          <div style="flex:1"></div>
        </div>
        <div id="vale-box" class="box">
          {rows_html}
        </div>
      </div>
      <script>
        const box = document.getElementById('vale-box');
        function scrollDown(){{ try{{ box.scrollTop = box.scrollHeight; }}catch(e){{}} }}
        scrollDown(); setTimeout(scrollDown, 50); setTimeout(scrollDown, 150);
        try{{ new MutationObserver(scrollDown).observe(box, {{childList:true, subtree:true}}); }}catch(e){{}}
      </script>
    </body>
    </html>
    """

    with container:
        components_html(html_doc, height=height, scrolling=False)

        with st.form(key="vale_chat_form", clear_on_submit=True):
            user_txt = st.text_input("Message", label_visibility="collapsed", placeholder="Type to Vale…")
            sent = st.form_submit_button("Send")
        if sent and user_txt.strip():
            add_chat("user", user_txt.strip(), state_key)
            add_chat("assistant", vale_say("cool—i’ll fold that in next pass."), state_key)
            st.experimental_rerun()
