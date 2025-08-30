# chat_ui.py — Vale Dark Dashboard chat
import html as pyhtml
import random
import re
import streamlit as st
from streamlit.components.v1 import html as components_html
from string import Template


# ---------- utilities ----------
def _esc(s: str) -> str:
    return pyhtml.escape(s or "")

_OPENERS = ["so…", "okay—", "alright,", "cool—", "nice—", "gotcha.", "heads up—"]

def _soften_numbers(txt: str) -> str:
    # 2.83 dB ➜ 3 dB for a friendlier voice
    return re.sub(r"([+\-]?\d+(?:\.\d+)?)\s*dB", lambda m: f"{round(float(m.group(1)))} dB", txt)

def vale_say(message: str) -> str:
    opener = random.choice(_OPENERS)
    msg = (message or "").strip()
    msg = re.sub(r'^(analysis|result|note)[:\- ]+', '', msg, flags=re.I)
    msg = _soften_numbers(msg)
    if not msg:
        msg = "all set."
    return f"{opener} {msg[0].lower()}{msg[1:] if len(msg)>1 else ''}"

# ---------- public API ----------
def add_chat(role: str, text: str, state_key: str = "chat", *, html: bool = False, attachments=[{"type":"img","src":None,"alt":""}]):
    """Append a chat message. Set html=True to allow inline HTML (e.g., <img> thumbnails)."""
    st.session_state.setdefault(state_key, [])
    st.session_state[state_key].append(
        {"role": (role or "assistant"), "text": text or "", "html": bool(html)}
    )

def render_chat(
    container,
    state_key: str = "chat",
    height: int = 420,
    avatar_img_b64: str | None = None
):
    st.session_state.setdefault(state_key, [])
    if not st.session_state[state_key]:
        add_chat("assistant", vale_say("nice— upload a premaster when you’re ready—i’ll give it a quick once-over."), state_key)

    msgs = st.session_state[state_key]

    # Build message list HTML (no f-strings inside big template)
    rows_html = []
    for m in msgs:
        role = (m.get("role") or "assistant").lower()
        cls = "assistant" if role != "user" else "user"
        rows_html.append(
            "<div class='msg {cls}'>"
            "<div class='msg-role'>{role}</div>"
            "<div class='msg-body'>{text}</div>"
            "</div>".format(cls=cls, role=_esc(role), text=_esc(m.get("text", "")))
        )
    rows_html = "\n".join(rows_html)

    # Avatar
    if avatar_img_b64:
        avatar_inner = (
            "<img src='data:image/png;base64,{b64}' "
            "style='width:100%;height:100%;border-radius:50%;'>"
        ).format(b64=avatar_img_b64)
    else:
        avatar_inner = "V"

    # Use string.Template so JS/CSS braces are untouched
    tpl = Template("""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        :root {
          --ink: #E7EEF8;
          --ink-dim: #A9B6C9;
          --panel: #111823;
          --border: #1E2530;
          --vale: #5EA2FF;
          --violet: #8B7CFF;
        }
        * { box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
        body { margin:0; background:transparent; }
        .hdr { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
        .avatar {
          width:38px;height:38px;border-radius:50%;
          background: radial-gradient(120% 120% at 25% 25%, rgba(94,162,255,.35), rgba(139,124,255,.20));
          border:1px solid #2C3440;
          display:flex;align-items:center;justify-content:center;
          color:#CFE0FF;font-weight:700;
        }
        .box {
          height: ${height_px}px;
          overflow-y: auto;
          padding-right: 4px;
        }
        .box::-webkit-scrollbar { width: 0; height: 0; }
        .msg { border: 1px dashed var(--border); border-radius: 10px; padding: 10px 12px; margin: 0 0 8px 0; }
        .msg.user { background: rgba(139,124,255,.06); }
        .msg.assistant { background: rgba(94,162,255,.06); }
        .msg-role { font-size: 10px; color: var(--ink-dim); margin-bottom: 4px; letter-spacing: .3px; text-transform: uppercase; }
        .msg-body { font-size: 13px; color: var(--ink); line-height: 1.38; }
      </style>
    </head>
    <body>
      <div class="hdr">
        <div class="avatar">${avatar_inner}</div>
      </div>
      <div id="vale-box" class="box">
        ${rows_html}
      </div>
      <script>
        const box = document.getElementById('vale-box');
        function scrollDown(){ try { box.scrollTop = box.scrollHeight; } catch(e){} }
        scrollDown(); setTimeout(scrollDown, 50); setTimeout(scrollDown, 150);
        try { new MutationObserver(scrollDown).observe(box, {childList:true, subtree:true}); } catch(e){}
      </script>
    </body>
    </html>
    """)

    html_doc = tpl.substitute(
        height_px=max(0, int(height) - 12),
        rows_html=rows_html,
        avatar_inner=avatar_inner,
    )

    with container:
        # visually merge iframe + input
        st.markdown("<div class='vale-chat-wrap'>", unsafe_allow_html=True)
        st.markdown("<div class='iframe-band'>", unsafe_allow_html=True)
        components_html(html_doc, height=height, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.form(key="vale_chat_form", clear_on_submit=True):
            user_txt = st.text_input("Message", label_visibility="collapsed", placeholder="Type to Vale…")
            sent = st.form_submit_button("Send")
        st.markdown("</div>", unsafe_allow_html=True)

        if sent and user_txt.strip():
            add_chat("user", user_txt.strip(), state_key)
            add_chat("assistant", vale_say("cool—i’ll fold that in next pass."), state_key)
            st.experimental_rerun()



