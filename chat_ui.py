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
    height: int = 480,
    avatar_img_b64: str | None = None,
    avatar_letter: str = "V",
):
    """
    Renders a single chat panel with autoscroll inside `container` as an iframe.
    Supports HTML messages (for inline thumbnails).
    """
    st.session_state.setdefault(state_key, [])
    if not st.session_state[state_key]:
        add_chat("assistant",
                 vale_say("upload a premaster when you’re ready—i’ll give it a quick once-over."),
                 state_key)

    msgs = st.session_state[state_key]

    # Build message list
    rows_html = []
    for m in msgs:
        role = (m.get("role") or "assistant").lower()
        cls = "assistant" if role != "user" else "user"
        body = m.get("text", "")
        if m.get("html"):
            # trust minimal HTML (we control it)
            safe = body
        else:
            safe = _esc(body)
        rows_html.append(
            f"""
            <div class="msg {cls}">
              <div class="msg-role">{_esc(role)}</div>
              <div class="msg-body">{safe}</div>
            </div>
            """
        )

    rows_html = "\n".join(rows_html)

    # Avatar element
    if avatar_img_b64:
        avatar_inner = f'<img src="data:image/png;base64,{avatar_img_b64}" class="avatar-img" />'
    else:
        avatar_inner = _esc((avatar_letter or "V")[:1])

    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        :root {{
          --bg: #111317;                /* page (transparent here) */
          --panel: #1A1D22;             /* card background */
          --panel-hi: #20242B;          /* lifted card */
          --ink: #ECEFF4;               /* primary text */
          --ink-dim: #98A2B3;           /* secondary text */
          --border: #2A2F37;            /* hairline */
          --blue: #5EA2FF;              /* main accent (Vale Blue) */
          --blue-soft: rgba(94,162,255,0.14);
          --purple: #8B7CFF;            /* blue-purple (secondary) */
          --red: #FF5C7A;               /* rare warning */
          --radius: 14px;
          --shadow: 0 8px 28px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.02);
        }}
        * {{ box-sizing: border-box; font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }}
        body {{ margin:0; background: transparent; color: var(--ink); }}

        .panel {{
          background: linear-gradient(180deg, #1A1D22 0%, #171A1F 100%);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 12px;
          box-shadow: var(--shadow);
        }}

        .hdr {{
          display:flex; align-items:center; gap:12px; margin-bottom:10px;
        }}
        .avatar {{
          width:40px; height:40px; border-radius:50%;
          background: radial-gradient(120% 120% at 20% 20%, rgba(94,162,255,.25), rgba(139,124,255,.18));
          border: 1px solid #2D3340;
          box-shadow: 0 0 18px rgba(94,162,255,.25);
          display:grid; place-items:center;
          color: var(--blue); font-weight:800; letter-spacing:.3px;
        }}
        .avatar-img {{ width:100%; height:100%; border-radius:50%; display:block; }}

        .title {{ margin:0; font-size:15px; font-weight:700; color:#E6EAF2; }}
        .sub   {{ margin:0; font-size:12px; color:var(--ink-dim); }}

        .box {{
          height: {height-90}px;
          overflow-y: auto;
          border: 1px solid var(--border);
          border-radius: calc(var(--radius) - 6px);
          background: #15181D;
          padding: 10px;
        }}

        .msg {{
          margin: 0 0 10px 0;
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: #15181D;
        }}
        .msg.assistant {{
          background: linear-gradient(180deg, #151A21 0%, #14171C 100%);
          box-shadow: inset 0 0 0 1px rgba(94,162,255,.06), 0 0 0 2px rgba(94,162,255,.05);
        }}
        .msg.user {{
          background: #14171C;
        }}

        .msg-role {{
          font-size: 10px; color: var(--ink-dim); text-transform: uppercase;
          letter-spacing:.25px; margin-bottom:4px;
        }}
        .msg-body {{
          font-size: 13px; line-height: 1.45; color: var(--ink);
        }}

        /* Inline thumbnails inside messages */
        .thumb {{
          display:inline-block; border:1px solid var(--border); border-radius:10px;
          overflow:hidden; background:#0E1116; margin:.25rem .25rem 0 0;
          box-shadow: 0 8px 16px rgba(0,0,0,.35);
        }}
        .thumb img {{ display:block; width:100%; height:auto; }}
      </style>
    </head>
    <body>
      <div class="panel">
        <div class="hdr">
          <div class="avatar">{avatar_inner}</div>
          <div>
            <div class="title">Vale • Console</div>
            <div class="sub">always on your team</div>
          </div>
        </div>
        <div id="vale-box" class="box">
          {rows_html}
        </div>
      </div>
      <script>
        const box = document.getElementById('vale-box');
        function scrollDown(){ try{{ box.scrollTop = box.scrollHeight; }}catch(e){{}} }
        scrollDown(); setTimeout(scrollDown, 50); setTimeout(scrollDown, 150);
        try {{ new MutationObserver(scrollDown).observe(box, {{childList:true, subtree:true}}); }} catch(e) {{}}
      </script>
    </body>
    </html>
    """

    with container:
        components_html(html, height=height, scrolling=False)
        # Input lives outside the iframe
        with st.form(key="vale_chat_form", clear_on_submit=True):
            user_txt = st.text_input("Message", label_visibility="collapsed", placeholder="Type to Vale…")
            sent = st.form_submit_button("Send")
        if sent and user_txt.strip():
            add_chat("user", user_txt.strip(), state_key)
            add_chat("assistant", vale_say("cool—i’ll fold that in next pass."), state_key)
            st.experimental_rerun()
