# chat_ui.py
import streamlit as st
from streamlit.components.v1 import html as components_html
import html as pyhtml
import random, re
from app import logo_path

def _esc(s: str) -> str:
    return pyhtml.escape(s or "")

_VALE_OPENERS = ["so…", "okay—", "alright,", "cool—", "nice—", "gotcha.", "heads up—"]
def _soften_numbers(txt: str) -> str:
    return re.sub(r"([+\-]?\d+(\.\d+)?)\s*dB", lambda m: f"{round(float(m.group(1)))} dB", txt)

def vale_say(message: str) -> str:
    opener = random.choice(_VALE_OPENERS)
    msg = (message or "").strip()
    msg = re.sub(r'^(analysis|result|note)[:\- ]+', '', msg, flags=re.I)
    msg = _soften_numbers(msg)
    if not msg:
        msg = "all set."
    return f"{opener} {msg[0].lower()}{msg[1:] if len(msg)>1 else ''}"

def add_chat(role: str, text: str, state_key: str = "chat"):
    st.session_state.setdefault(state_key, [])
    st.session_state[state_key].append({"role": role, "text": text})

def render_chat(container, state_key: str = "chat", height: int = 420):
    """Single, container-scoped chat with autoscroll, rendered in one HTML iframe."""
    st.session_state.setdefault(state_key, [])
    if not st.session_state[state_key]:
        add_chat("assistant", vale_say("upload a premaster when you’re ready—i’ll give it a quick once-over."), state_key)

    msgs = st.session_state[state_key]

    # Build the full HTML for the iframe (header + scroll area + messages)
    rows = []
    for m in msgs:
        role = (m.get("role") or "assistant").lower()
        cls = "assistant" if role != "user" else "user"
        rows.append(
            f"""
            <div class="msg {cls}">
              <div class="msg-role">{role}</div>
              <div class="msg-body">{_esc(m.get('text',''))}</div>
            </div>
            """
        )
    rows_html = "\n".join(rows)

    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        :root {{
          --ink: #1b1f23;
          --ink-dim: #5b6570;
          --panel: #ffffff;
          --border: #e6eaf0;
          --accent: #94a2ff;
          --accent-ghost: rgba(148,162,255,.25);
        }}
        * {{ box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"; }}
        body {{ margin:0; background:transparent; }}
        .panel {{
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 10px;
        }}
        .hdr {{
          display:flex; align-items:center; gap:12px; margin-bottom:10px;
        }}
        .avatar {{
          width:38px; height:38px; border-radius:50%;
          background: var(--panel);
          display:flex; align-items:center; justify-content:center;
          color: var(--accent); font-weight:700;
        }}
        .hdr-title {{ margin:0; font-size:20px; font-weight:800; color: var(--ink); }}
        .hdr-sub {{ font-size:12px; color: var(--ink-dim); }}

        .box {{
          height: {height-90}px;  /* scrollable area */
          overflow-y: auto;
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 8px;
          background: #fff;
        }}

        .msg {{ border: 1px dashed var(--border); border-radius: 4px;
                padding: 8px 10px; margin: 0 0 8px 0; }}
        .msg.user {{ background: rgba(148,162,255,.05); }}
        .msg.assistant {{ background: linear-gradient(0deg, rgba(148,162,255,.07), rgba(148,162,255,.07)); }}
        .msg-role {{ font-size: 10px; color: var(--ink-dim); margin-bottom: 4px; letter-spacing: .3px; text-transform: uppercase; }}
        .msg-body {{ font-size: 13px; color: var(--ink); line-height: 1.35; }}
      </style>
    </head>
    <body>
      <div class="panel">
        <div class="hdr">
          <div class="avatar"></div>
        </div>
        <div id="vale-box" class="box">
          {rows_html}
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
    """

    with container:
        components_html(html, height=height, scrolling=False)

        # Input form (in Streamlit space)
        with st.form(key="vale_chat_form", clear_on_submit=True):
            user_txt = st.text_input("Message", label_visibility="collapsed", placeholder="Type to Vale…")
            sent = st.form_submit_button("Send")

        if sent and user_txt.strip():
            add_chat("user", user_txt.strip(), state_key)
            # example quick reply; swap for real logic
            add_chat("assistant", vale_say("cool—i’ll fold that in next pass."), state_key)
            st.experimental_rerun()
