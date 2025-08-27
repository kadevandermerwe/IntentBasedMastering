# Vale Mastering Assistant — AI + Adaptive Mastering (drop/verse aware, REST LLM, no aifc)
import os, json, subprocess, uuid, requests
import numpy as np
import soundfile as sf
import streamlit as st

# ---------------- UI / page ----------------
st.set_page_config(page_title="Vale Mastering Assistant", page_icon="🎛️", layout="centered")
st.title("🎛️ Vale Mastering Assistant — Prototype (AI + Adaptive)")

# ---------------- Helpers ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe_run(cmd):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        st.error("❌ FFmpeg failed.")
        st.code(" ".join(cmd))
        st.error(e)
        raise

def session_tmp_path():
    sid = st.session_state.get("_session_id")
    if not sid:
        sid = str(uuid.uuid4())[:8]
        st.session_state["_session_id"] = sid
    base = f"/tmp/vale_{sid}"
    os.makedirs(base, exist_ok=True)
    return base

# ---------------- Analysis ----------------
def analyze_audio(path):
    import pyloudnorm as pyln
    import librosa
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y))
    # True-peak estimate via 4x oversampling
    y_os = librosa.resample(y, orig_sr=sr, target_sr=sr*4, res_type="kaiser_best")
    tp = float(20*np.log10(np.max(np.abs(y_os)) + 1e-12))
    # Rough spectral balance
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[(freqs < 120)].mean()
    mid = S[(freqs >= 120) & (freqs < 5000)].mean()
    high = S[(freqs >= 5000)].mean()
    tot = low + mid + high + 1e-9
    spec = {"low": float(100*low/tot), "mid": float(100*mid/tot), "high": float(100*high/tot)}
    return {"sr": sr, "lufs_integrated": lufs, "true_peak_dbfs_est": tp, "spectral_pct": spec}

# ---------------- Section detection (uses soundfile to avoid audioread/aifc) ----------------
def detect_sections(path, frame_ms=200, hop_ms=100, energy_sigma=1.0, min_len_s=4.0):
    """
    Return (drop_spans, total_duration_sec) using short-term RMS thresholding.
    Read audio via soundfile (WAV/AIFF/FLAC) to avoid audioread→aifc on Python 3.13.
    """
    import librosa

    y, sr = sf.read(path)         # <— soundfile backend only
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)

    # librosa for RMS + time mapping (math only, doesn't touch aifc)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop, n_fft=frame)

    thr = rms.mean() + energy_sigma * rms.std()
    mask = rms > thr

    spans, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = t[i]
        if not on and start is not None:
            end = t[i]
            if end - start >= min_len_s:
                spans.append((float(start), float(end)))
            start = None
    if start is not None:
        end = float(t[-1])
        if end - start >= min_len_s:
            spans.append((float(start), float(end)))

    total_dur = float(len(y) / sr)
    return spans, total_dur

# ---------------- Intent parsing ----------------
def parse_prompt(txt):
    p = (txt or "").lower()
    intent = {"tone": 0.0, "dynamics": 0.0, "stereo": 0.0, "character": 0.0}
    if any(k in p for k in ["dark","moody","warm","shadow","muted"]): intent["tone"] -= 0.5
    if any(k in p for k in ["bright","airy","sparkle","shine","open","sheen"]): intent["tone"] += 0.5
    if any(k in p for k in ["preserve","breathe","dynamic","not loud","natural"]): intent["dynamics"] -= 0.5
    if any(k in p for k in ["loud","slam","club","radio","punchy"]): intent["dynamics"] += 0.5
    if any(k in p for k in ["wide","spacious","open image","stereo"]): intent["stereo"] += 0.3
    if any(k in p for k in ["mono","tight","focused","narrow"]): intent["stereo"] -= 0.3
    if any(k in p for k in ["tape","analog","saturation","glue","harmonic"]): intent["character"] += 0.6
    if any(k in p for k in ["clean","transparent","surgical"]): intent["character"] -= 0.6
    for k in intent:
        intent[k] = float(clamp(intent[k], -1.0, 1.0))
    return intent

# ---------------- FFmpeg builders ----------------
def build_fg_from_plan(plan):
    tgt_lufs = float(plan["targets"].get("lufs_i", -11.5))
    tgt_tp   = float(plan["targets"].get("true_peak_db", -1.2))
    lo_thr   = float(plan["mb_comp"].get("low_thr_db", -20))
    mid_thr  = float(plan["mb_comp"].get("mid_thr_db", -24))
    hi_thr   = float(plan["mb_comp"].get("high_thr_db", -26))
    lo_gain  = float(plan["eq"].get("low_shelf_db", 0.0))
    mud_cut  = float(plan["eq"].get("mud_cut_db", 0.0))
    hi_gain  = float(plan["eq"].get("high_shelf_db", 0.0))
    drive_db = float(plan["saturation"].get("drive_db", 1.0))
    drive_in = 1.0 + 0.1 * clamp(drive_db, 0.0, 3.0)
    mb = (
      "asplit=3[lo][mid][hi];"
      f"[lo]lowpass=f=120,acompressor=threshold={lo_thr}dB:ratio=2.0:attack=10:release=180[loC];"
      f"[mid]bandpass=f=120:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio=1.6:attack=16:release=220[midC];"
      f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio=1.3:attack=12:release=180[hiC];"
      "[loC][midC][hiC]amix=inputs=3[mb];"
    )
    eq = (
      "[mb]"
      f"firequalizer=gain='if(lt(f,120),{lo_gain}, if(lt(f,300),{mud_cut}, if(gt(f,12000),{hi_gain},0)))'"
      "[eq];"
    )
    sat  = f"[eq]volume={drive_in}[sat];"
    loud = f"[sat]loudnorm=I={tgt_lufs}:LRA=9:TP={tgt_tp}:dual_mono=true:linear=true[out]"
    return mb + eq + sat + loud

def build_fg_heuristic(analysis, intent, variant="vibe"):
    tone = float(intent["tone"]); dyn = float(intent["dynamics"]); char = float(intent["character"])
    target_lufs = {"conservative": -13.0, "vibe": -12.0, "loud": -10.8}.get(variant, -12.0)
    target_tp   = -1.2
    hi_gain = 1.0 * max(0.0, tone)
    lo_gain = 0.0 if analysis["spectral_pct"]["low"] >= 30.0 else 0.6
    mud_cut = -0.8 if analysis["spectral_pct"]["mid"] > 55.0 else 0.0
    lo_thr, mid_thr, hi_thr = -18, -22, -24
    if variant == "loud": lo_thr -= 1; mid_thr -= 1; hi_thr -= 1
    if dyn < -0.33: lo_thr += 2; mid_thr += 2; hi_thr += 2
    drive_in = 1.0 if variant != "loud" else 1.05
    mb = (
      "asplit=3[lo][mid][hi];"
      f"[lo]lowpass=f=120,acompressor=threshold={lo_thr}dB:ratio=2.0:attack=10:release=180[loC];"
      f"[mid]bandpass=f=120:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio=1.6:attack=16:release=220[midC];"
      f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio=1.3:attack=12:release=180[hiC];"
      "[loC][midC][hiC]amix=inputs=3[mb];"
    )
    eq = (
      "[mb]"
      f"firequalizer=gain='if(lt(f,120),{lo_gain}, if(lt(f,300),{mud_cut}, if(gt(f,12000),{hi_gain},0)))'"
      "[eq];"
    )
    sat = f"[eq]anull[sat];"
    loud = f"[sat]loudnorm=I={target_lufs}:LRA=9:TP={target_tp}:dual_mono=true:linear=true[out]"
    params = {
        "targets":{"lufs_i":target_lufs,"true_peak_db":target_tp},
        "eq":{"low_shelf_db":lo_gain,"mud_cut_db":mud_cut,"high_shelf_db":hi_gain},
        "mb_thr":{"low":lo_thr,"mid":mid_thr,"high":hi_thr},
        "saturation":{"pre_gain":drive_in}
    }
    return mb + eq + sat + loud, params

# ---------------- Renderer ----------------
def render_variant(in_wav, out_wav, filtergraph):
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i",in_wav,"-filter_complex",filtergraph,"-map","[out]","-ar","44100","-ac","2",out_wav]
    safe_run(cmd)

def render_adaptive_from_plans(in_wav, out_wav, verse_plan, drop_plan):
    drops, total_dur = detect_sections(in_wav)
    segments, cursor, idx = [], 0.0, 0
    def cut_and_process(t0, t1, plan, i):
        fg = build_fg_from_plan(plan)
        seg_out = os.path.join(os.path.dirname(out_wav), f"seg_{i:03d}.wav")
        cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i",in_wav,
               "-filter_complex", f"atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS,{fg}",
               "-map","[out]","-ar","44100","-ac","2",seg_out]
        safe_run(cmd); return seg_out
    for (d0, d1) in drops:
        if d0 > cursor + 0.05:
            segments.append(cut_and_process(cursor, d0, verse_plan, idx)); idx += 1
        segments.append(cut_and_process(d0, d1, drop_plan, idx)); idx += 1
        cursor = d1
    if cursor < total_dur - 0.05:
        segments.append(cut_and_process(cursor, total_dur, verse_plan, idx)); idx += 1
    list_path = out_wav + ".txt"
    with open(list_path,"w") as f:
        for s in segments: f.write(f"file '{s}'\n")
    safe_run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",list_path,"-c","copy",out_wav])

# ---------------- LLM planner (REST) ----------------
use_llm   = st.sidebar.checkbox("Use OpenAI LLM planner", value=True)
llm_model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")
st.sidebar.caption("Set OPENAI_API_KEY in Streamlit secrets.")

def llm_plan(analysis, intent, prompt, model):
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None, "LLM disabled or missing key."
    system = """
You are an expert mastering engineer.
Create a MUSICAL mastering plan from:
- ANALYSIS: measured stats for THIS premaster,
- INTENT: user sliders + text,
- REFERENCE: vibe/track/artist if provided (e.g., “Soma by Return of the Jaded”).

Output STRICT JSON ONLY.
Prefer section-specific plans if appropriate.

Schema A (sectioned):
{
  "verse": {
    "targets": {"lufs_i": float, "true_peak_db": float},
    "eq": {"low_shelf_db": float, "mud_cut_db": float, "high_shelf_db": float},
    "mb_comp": {"low_thr_db": float, "mid_thr_db": float, "high_thr_db": float},
    "saturation": {"drive_db": float},
    "stereo": {"amount": float},
    "explanation": string
  },
  "drop": { ... }
}

Schema B (single plan):
{
  "targets": {...}, "eq": {...}, "mb_comp": {...}, "saturation": {...}, "stereo": {...}, "explanation": string
}

Guardrails:
- LUFS in [-14, -9]; true_peak ≤ -1.0 (prefer -1.2)
- EQ moves within ±2 dB (low_shelf_db, mud_cut_db, high_shelf_db)
- Multiband thresholds in [-30, -18] dB
- Saturation drive in [0, 3] dB
- Stereo amount in [-0.2, 0.2]

Return VALID JSON ONLY.
""".strip()
    user = f"ANALYSIS:{json.dumps(analysis)}\nINTENT:{json.dumps(intent)}\nREFERENCE:{prompt or ''}"
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model, "temperature": 0.25,
                "messages":[{"role":"system","content":system},{"role":"user","content":user}]
            }, timeout=45
        )
        if resp.status_code != 200:
            return None, f"LLM HTTP {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n",1)[-1]
            if content.endswith("```"): content = content[:-3].strip()
            if content.lower().startswith("json"): content = content[4:].strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            return None, "LLM returned no JSON."
        plan = json.loads(content[start:end+1])
        return plan, "LLM plan generated."
    except requests.Timeout:
        return None, "LLM timeout."
    except Exception as e:
        return None, f"LLM error: {e}"

def derive_section_plans_from_single(base_plan):
    import copy
    verse = copy.deepcopy(base_plan); drop = copy.deepcopy(base_plan)
    verse["targets"]["lufs_i"] = float(clamp(verse["targets"].get("lufs_i",-11.5)-1.0,-14.0,-9.0))
    verse["targets"]["true_peak_db"] = float(min(-1.0, verse["targets"].get("true_peak_db", -1.2)))
    for k in ["low_thr_db","mid_thr_db","high_thr_db"]:
        verse.setdefault("mb_comp",{}); verse["mb_comp"][k] = float(min(-18.0, base_plan.get("mb_comp",{}).get(k, -24) + 2))
    drop["targets"]["lufs_i"]  = float(clamp(drop["targets"].get("lufs_i",-11.5)+0.7,-14.0,-9.0))
    drop["targets"]["true_peak_db"] = float(min(-1.0, drop["targets"].get("true_peak_db", -1.2)))
    drop.setdefault("mb_comp",{}); drop["mb_comp"]["low_thr_db"]  = float(max(-30.0, base_plan.get("mb_comp",{}).get("low_thr_db", -20) - 2))
    return verse, drop

# ---------------- Sidebar / Controls ----------------
adaptive = st.sidebar.checkbox("Adaptive per-section rendering", value=True)

uploaded = st.file_uploader(
    "Upload premaster (WAV/AIFF/FLAC — no limiter, ~−6 dBFS headroom)",
    type=["wav","aiff","aif","flac"]
)
prompt_txt = st.text_area("Intent / Reference (e.g. “dark, preserve dynamics, tape; like Soma by Return of the Jaded”.)")

col1, col2 = st.columns(2)
with col1:
    tone = st.slider("Tone: Dark ↔ Bright", -1.0, 1.0, 0.0, 0.1)
    dynamics = st.slider("Dynamics: Preserve ↔ Loud", -1.0, 1.0, 0.0, 0.1)
with col2:
    stereo = st.slider("Stereo: Narrow ↔ Wide", -1.0, 1.0, 0.0, 0.1)
    character = st.slider("Character: Clean ↔ Tape", -1.0, 1.0, 0.5, 0.1)

# Blend parsed prompt into sliders
if prompt_txt:
    parsed = parse_prompt(prompt_txt)
    tone = float(clamp(0.5*tone + 0.5*parsed["tone"], -1.0, 1.0))
    dynamics = float(clamp(0.5*dynamics + 0.5*parsed["dynamics"], -1.0, 1.0))
    stereo = float(clamp(0.5*stereo + 0.5*parsed["stereo"], -1.0, 1.0))
    character = float(clamp(0.5*character + 0.5*parsed["character"], -1.0, 1.0))

intent = {"tone": tone, "dynamics": dynamics, "stereo": stereo, "character": character}
st.caption(f"Resolved intent → tone {tone:+.2f} • dynamics {dynamics:+.2f} • stereo {stereo:+.2f} • character {character:+.2f}")

# ---------------- Upload & analysis (robust, stable /tmp) ----------------
if uploaded is not None:
    if ("uploaded_name" not in st.session_state
        or uploaded.name != st.session_state.get("uploaded_name")
        or "uploaded_bytes" not in st.session_state):
        st.session_state["uploaded_name"] = uploaded.name
        st.session_state["uploaded_bytes"] = uploaded.read()
        for k in ("analysis","in_path"):
            st.session_state.pop(k, None)

if "uploaded_bytes" in st.session_state:
    base = session_tmp_path()
    in_path = os.path.join(base, st.session_state["uploaded_name"].replace(" ","_"))
    try:
        with open(in_path, "wb") as f:
            f.write(st.session_state["uploaded_bytes"])
    except Exception as e:
        st.error("❌ Failed writing uploaded file to /tmp.")
        st.exception(e); st.stop()
    st.session_state["in_path"] = in_path

    cols = st.columns(3)
    with cols[0]:
        analyze_click = st.button("Analyze file")
    with cols[1]:
        gen_click = st.button("Generate Master(s)")
    with cols[2]:
        if st.button("Reset file"):
            for k in ("uploaded_name","uploaded_bytes","analysis","in_path"):
                st.session_state.pop(k, None)
            st.experimental_rerun()

    # Analysis
    if analyze_click or "analysis" not in st.session_state:
        try:
            st.session_state["analysis"] = analyze_audio(st.session_state["in_path"])
        except Exception as e:
            st.error("❌ Analysis failed."); st.exception(e); st.stop()

    analysis = st.session_state.get("analysis")
    if analysis:
        st.subheader("Analysis")
        st.json(analysis)

        # AI plan (optional)
        ai_plan, ai_msg = (None, None)
        if use_llm:
            ai_plan, ai_msg = llm_plan(analysis, intent, prompt_txt, llm_model)
            if not ai_plan:
                st.warning(f"LLM plan unavailable → falling back to heuristics. Reason: {ai_msg}")
            else:
                if "verse" in ai_plan and "drop" in ai_plan:
                    st.success("LLM: sectioned plans (verse+drop) received.")
                elif "targets" in ai_plan:
                    st.info("LLM: single plan received → deriving verse/drop.")
                else:
                    st.warning("LLM returned JSON but keys not recognized → fallback to heuristics.")
                st.subheader("AI Plan")
                st.code(json.dumps(ai_plan, indent=2))

        # Generate
        if gen_click:
            # Determine plans
            verse_plan = drop_plan = None
            if ai_plan and "verse" in ai_plan and "drop" in ai_plan:
                verse_plan, drop_plan = ai_plan["verse"], ai_plan["drop"]
            elif ai_plan and "targets" in ai_plan:
                verse_plan, drop_plan = derive_section_plans_from_single(ai_plan)

            # AI full-pass (single chain)
            if ai_plan:
                try:
                    fg_full = build_fg_from_plan(ai_plan if "targets" in ai_plan else verse_plan)
                    out_ai = os.path.join(base, "master_ai_full.wav")
                    render_variant(in_path, out_ai, fg_full)
                    st.markdown("### 🧩 AI Master (Full)")
                    st.audio(out_ai)
                    with open(out_ai,"rb") as f:
                        st.download_button("Download AI Master (Full)", f.read(), file_name="master_ai_full.wav")
                except Exception as e:
                    st.error("AI full-pass render failed."); st.exception(e)

            # Adaptive AI (per-section)
            if adaptive and verse_plan and drop_plan:
                try:
                    out_ad = os.path.join(base, "master_ai_adaptive.wav")
                    render_adaptive_from_plans(in_path, out_ad, verse_plan, drop_plan)
                    post = analyze_audio(out_ad)
                    st.markdown("### 🎯 AI Adaptive (verse/drop)")
                    st.json(post)
                    st.audio(out_ad)
                    with open(out_ad,"rb") as f:
                        st.download_button("Download AI Adaptive", f.read(), file_name="master_ai_adaptive.wav")
                except Exception as e:
                    st.error("Adaptive render failed."); st.exception(e)

            # Heuristic fallbacks (A/B reference)
            try:
                for variant in ["conservative","vibe","loud"]:
                    fg, params = build_fg_heuristic(analysis, intent, variant)
                    out_h = os.path.join(base, f"master_{variant}.wav")
                    render_variant(in_path, out_h, fg)
                    st.markdown(f"### 📊 Heuristic — {variant.capitalize()}")
                    st.json(params)
                    st.audio(out_h)
                    with open(out_h,"rb") as f:
                        st.download_button(f"Download {variant}", f.read(), file_name=f"master_{variant}.wav")
            except Exception as e:
                st.error("Fallback render failed."); st.exception(e)

    # Optional tiny debug expander
    with st.expander("Debug (upload)"):
        st.write({
            "in_path_exists": os.path.exists(st.session_state.get("in_path","")),
            "in_path": st.session_state.get("in_path"),
            "bytes_len": len(st.session_state.get("uploaded_bytes", b""))
        })

else:
    st.info("Upload a premaster to begin.")
