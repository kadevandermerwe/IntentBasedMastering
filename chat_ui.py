# chat_ui.py — Vale Dark Dashboard chat
import html as pyhtml
import random
import re
import streamlit as st
from streamlit.components.v1 import html as components_html

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
def add_chat(role: str, text: str, state_key: str = "chat", *, html: bool = False):
    """Append a chat message. Set html=True to allow inline HTML (e.g., <img> thumbnails)."""
    st.session_state.setdefault(state_key, [])
    st.session_state[state_key].append(
        {"role": (role or "assistant"), "text": text or "", "html": bool(html)}
    )

def render_chat(
    container,
    state_key: str = "chat",
    height: int = 420,
    avatar_img_b64: str | None = None,
    avatar_text: str = "V",
):
    """
    Renders a single chat panel with autoscroll inside `container`.
    Uses an iframe so layout is consistent and not affected by Streamlit rerenders.
    """
    st.session_state.setdefault(state_key, [])
    if not st.session_state[state_key]:
        add_chat("assistant", vale_say("upload a premaster when you’re ready—i’ll give it a quick once-over."), state_key)

    msgs = st.session_state[state_key]

    # Build message list
    rows_html = []
    for m in msgs:
        role = (m.get("role") or "assistant").lower()
        cls = "assistant" if role != "user" else "user"
        rows_html.append(
            """
            <div class="msg {cls}">
              <div class="msg-role">{role}</div>
              <div class="msg-body">{text}</div>
            </div>
            """.format(
                cls=cls,
                role=pyhtml.escape(role),
                text=pyhtml.escape(m.get("text", "")),
            )
        )
    rows_html = "\n".join(rows_html)

    # Avatar element: either image or letter
    if avatar_img_b64:
        avatar_inner = '<img src="data:image/png;base64,{b64}" style="width:100%;height:100%;border-radius:0%;">'.format(b64=avatar_img_b64)
    else:
        avatar_inner = pyhtml.escape(avatar_text or "V")

    # Use a plain string with a neutral placeholder for the avatar to avoid f-string brace issues
    html_tpl = """
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        :root {{
          --ink: #1b1f23;
          --ink-dim: #5b6570;
          --panel: #ffffff;
          --border: #e6eaf0;
          --accent: #5EA2FF;
          --accent-ghost: rgba(94,162,255,.18);
        }}
        * {{ box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }}
        body {{ margin:0; background:transparent; }}

        .panel {{ padding: 10px; }}
        .box::-webkit-scrollbar {{ display: none; }}
        .hdr {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
        .avatar {{ width:150px; }}
        .hdr-title {{ margin:0; font-size:16px; font-weight:800; color: var(--ink); }}
        .hdr-sub {{ font-size:12px; color: var(--ink-dim); }}

        .box {{
          height: {scroll_h}px;  /* scrollable area */
          -ms-overflow-style: none; /* IE and Edge */
          scrollbar-width: none; /* Firefox */
          overflow-y: scroll;
          padding: 8px;
          margin-bottom:auto;
        }}
        .msg {{
          border: 1px dashed var(--border); border-radius: 4px;
          padding: 8px 10px; margin: 0 0 8px 0;
        }}
        .msg.user {{ background: transparent; }}
        .msg.assistant {{ background: transparent; }}
        .msg-role {{ font-size: 10px; color: var(--ink-dim); margin-bottom: 4px; letter-spacing: .3px; text-transform: uppercase; }}
        .msg-body {{ font-size: 13px; color: var(--ink); line-height: 1.35; }}
      </style>
    </head>
    <body>
      <div class="panel">
        <div class="hdr">
          <div class="avatar"></div>
          <div class="avatar">@@VALE_AVATAR@@</div>
          <div class="avatar"></div>
        </div>
        <div id="vale-box" class="box">
          {rows}
        </div>
      </div>

      <script>
        const box = document.getElementById('vale-box');
        function scrollDown() {{
          try {{ box.scrollTop = box.scrollHeight; }} catch(e) {{}}
        }}
        scrollDown(); setTimeout(scrollDown, 50); setTimeout(scrollDown, 150);
        try {{
          new MutationObserver(scrollDown).observe(box, {{childList:true, subtree:true}});
        }} catch(e) {{}}
      </script>
    </body>
    </html>
    """.format(
        scroll_h=height - 90,
        rows=rows_html
    )

    # Inject avatar markup safely
    html = html_tpl.replace("@@VALE_AVATAR@@", avatar_inner)

    with container:
        components_html(html, height=height, scrolling=False)

        # Input form (in Streamlit space)
        with st.form(key="vale_chat_form", clear_on_submit=True):
            user_txt = st.text_input("Message", label_visibility="collapsed", placeholder="Type to Vale…")
            sent = st.form_submit_button("Send")

        if sent and user_txt.strip():
            add_chat("user", user_txt.strip(), state_key)
            # simple friendly echo; swap for real tool calls
            add_chat("assistant", vale_say("cool—i’ll fold that in next pass."), state_key)
            st.experimental_rerun()

