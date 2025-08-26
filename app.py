# Vale Mastering Assistant - Streamlit Prototype
import os, json, tempfile, subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa
import streamlit as st

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Vale Mastering Assistant", page_icon="🎛️", layout="centered")
st.title("🎛️ Vale Mastering Assistant — Prototype")

use_llm = st.sidebar.checkbox("Use OpenAI LLM planner", value=False)
llm_model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")

def clamp(v, lo, hi): return max(lo, min(hi, v))

def analyze_audio(path):
    y, sr = sf.read(path)
    if y.ndim > 1: y = y.mean(axis=1)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y))
    y_os = librosa.resample(y.astype(np.float32), sr, sr*4, res_type="kaiser_best")
    tp = float(20*np.log10(np.max(np.abs(y_os))+1e-12))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[(freqs<120)].mean(); mid = S[(freqs>=120)&(freqs<5000)].mean(); high = S[(freqs>=5000)].mean()
    tot = low+mid+high+1e-9
    return {"sr":sr,"lufs_integrated":lufs,"true_peak_dbfs_est":tp,
            "spectral_pct":{"low":float(100*low/tot),"mid":float(100*mid/tot),"high":float(100*high/tot)}}

def parse_prompt(txt):
    p=(txt or "").lower()
    intent={"tone":0.0,"dynamics":0.0,"stereo":0.0,"character":0.0}
    if any(k in p for k in ["dark","moody","warm"]): intent["tone"]-=0.5
    if any(k in p for k in ["bright","airy","sparkle"]): intent["tone"]+=0.5
    if any(k in p for k in ["preserve","breathe","dynamic","not loud"]): intent["dynamics"]-=0.5
    if any(k in p for k in ["loud","slam","club","radio"]): intent["dynamics"]+=0.5
    if any(k in p for k in ["wide","spacious"]): intent["stereo"]+=0.3
    if any(k in p for k in ["mono","tight","focused"]): intent["stereo"]-=0.3
    if any(k in p for k in ["tape","analog","saturation","glue"]): intent["character"]+=0.6
    if any(k in p for k in ["clean","transparent"]): intent["character"]-=0.6
    for k in intent: intent[k]=float(clamp(intent[k],-1.0,1.0))
    return intent

def llm_plan(analysis, intent, prompt, model):
    if not (OPENAI_AVAILABLE and "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]):
        return None, "LLM disabled or key missing; using heuristics."
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        system = "You are a mastering planner. Return STRICT JSON with keys: targets, eq, mb_comp, saturation, stereo, explanation."
        user = f"ANALYSIS:{json.dumps(analysis)}\\nINTENT:{json.dumps(intent)}\\nPROMPT:{prompt}"
        resp = client.chat.completions.create(model=model,temperature=0.3,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}])
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"): content = content.strip("`").replace("json","")
        plan = json.loads(content)
        plan.setdefault("targets",{})["true_peak_db"]=-1.0
        if plan.get("eq",{}).get("high_shelf_db",0)>1.5: plan["eq"]["high_shelf_db"]=1.5
        if plan.get("stereo",{}).get("amount",0)>0.15: plan["stereo"]["amount"]=0.15
        return plan,"LLM plan generated."
    except Exception as e:
        return None,f"LLM error: {e}; using heuristics."

def build_fg(analysis, intent, variant="vibe"):
    tone=intent["tone"]; dyn=intent["dynamics"]; char=intent["character"]
    base_lufs=-10.0
    if dyn<-0.33: base_lufs=-12.0
    elif dyn>0.33: base_lufs=-9.0
    if variant=="conservative": target_lufs=base_lufs+0.5
    elif variant=="loud": target_lufs=base_lufs-0.8
    else: target_lufs=base_lufs
    target_tp=-1.0
    hi_gain=1.5*max(0.0,tone); lo_gain=0.0
    if analysis["spectral_pct"]["low"]<30.0: lo_gain=0.8
    mud_cut=-1.2 if analysis["spectral_pct"]["mid"]>55.0 else 0.0
    lo_thr,mid_thr,hi_thr=-22,-24,-26
    if variant=="loud": lo_thr-=2; mid_thr-=2; hi_thr-=2
    if dyn<-0.33: lo_thr+=2; mid_thr+=2; hi_thr+=2
    drive_in=1.0+0.15*max(0.0,char)
    mb=(
      "asplit=3[lo][mid][hi];"
      f"[lo]lowpass=f=140,acompressor=threshold={lo_thr}dB:ratio=2:attack=20:release=200[loC];"
      f"[mid]bandpass=f=140:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio=1.8:attack=15:release=180[midC];"
      f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio=1.5:attack=10:release=150[hiC];"
      "[loC][midC][hiC]amix=inputs=3:normalize=0[mb];")
    eq="[mb]firequalizer=gain='if(f<120,{}, if(f<300,{}, if(f>12000,{},0)))'[eq];".format(lo_gain,mud_cut,hi_gain)
    sat=f"[eq]alimiter=limit=0.0:level_in={drive_in}:level_out=1.0[sat];"
    loud=f"[sat]loudnorm=I={target_lufs}:LRA=7:TP={target_tp}:dual_mono=true:linear=true[out]"
    return mb+eq+sat+loud

def render_variant(in_wav,out_wav,fg):
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error","-i",in_wav,"-filter_complex",fg,"-map","[out]","-ar","44100","-ac","2",out_wav]
    subprocess.run(cmd,check=True)

uploaded=st.file_uploader("Upload premaster (WAV/AIFF)",type=["wav","aiff","aif","flac"])
prompt=st.text_area("Intent (optional)")

tone=st.slider("Tone: Dark↔Bright",-1.0,1.0,0.0,0.1)
dynamics=st.slider("Dynamics: Preserve↔Loud",-1.0,1.0,0.0,0.1)
stereo=st.slider("Stereo: Narrow↔Wide",-1.0,1.0,0.0,0.1)
character=st.slider("Character: Clean↔Tape",-1.0,1.0,0.5,0.1)

intent={"tone":tone,"dynamics":dynamics,"stereo":stereo,"character":character}
if prompt:
    parsed=parse_prompt(prompt)
    for k in intent: intent[k]=float(clamp(intent[k]*0.5+parsed[k]*0.5,-1.0,1.0))

if uploaded:
    with tempfile.TemporaryDirectory() as td:
        in_path=os.path.join(td,uploaded.name.replace(" ","_"))
        with open(in_path,"wb") as f: f.write(uploaded.read())
        analysis=analyze_audio(in_path)
        st.subheader("Analysis"); st.json(analysis)

        plan=None; msg=None
        if use_llm:
            plan,msg=llm_plan(analysis,intent,prompt,llm_model)
            st.caption(msg or ""); 
            if plan: st.json(plan)

        if st.button("Generate 3 masters"):
            for variant in ["conservative","vibe","loud"]:
                fg=build_fg(analysis,intent,variant)
                out_path=os.path.join(td,f"{os.path.splitext(uploaded.name)[0]}_{variant}.wav")
                render_variant(in_path,out_path,fg)
                with open(out_path,"rb") as f: st.download_button(f"Download {variant}",f.read(),file_name=os.path.basename(out_path),mime="audio/wav")
