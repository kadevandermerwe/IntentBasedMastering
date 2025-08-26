import json, subprocess, tempfile, os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa

def clamp(v, lo, hi): return max(lo, min(hi, v))

def analyze_audio(path):
    y, sr = sf.read(path)
    if y.ndim > 1: y = y.mean(axis=1)
    duration = len(y) / sr
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y))
    # true peak estimate: oversample x4
    y_os = librosa.resample(y.astype(np.float32), sr, sr*4, res_type="kaiser_best")
    tp = float(20*np.log10(np.max(np.abs(y_os))+1e-12))
    # rough spectral balance
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[(freqs<120)].mean(); mid = S[(freqs>=120)&(freqs<5000)].mean(); high = S[(freqs>=5000)].mean()
    tot = low+mid+high+1e-9
    spec = {"low": float(100*low/tot), "mid": float(100*mid/tot), "high": float(100*high/tot)}
    return {"sr": sr, "duration_sec": duration, "lufs_integrated": lufs, "true_peak_dbfs_est": tp, "spectral_pct": spec}

def build_filtergraph(analysis, intent, variant="vibe"):
    tone=intent["tone"]; dyn=intent["dynamics"]; char=intent["character"]
    base_lufs=-10.0
    if dyn<-0.33: base_lufs=-12.0
    elif dyn>0.33: base_lufs=-9.0
    target_lufs = base_lufs + (0.5 if variant=="conservative" else -0.8 if variant=="loud" else 0.0)
    target_tp=-1.0
    hi_gain=1.5*max(0.0,tone); lo_gain=0.0
    if analysis["spectral_pct"]["low"]<30.0: lo_gain=0.8
    mud_cut=-1.2 if analysis["spectral_pct"]["mid"]>55.0 else 0.0
    lo_thr,mid_thr,hi_thr=-22,-24,-26
    if variant=="loud": lo_thr-=2; mid_thr-=2; hi_thr-=2
    if dyn<-0.33: lo_thr+=2; mid_thr+=2; hi_thr+=2
    drive_in=1.0+0.15*max(0.0,char)
    mb=( "asplit=3[lo][mid][hi];"
         f"[lo]lowpass=f=140,acompressor=threshold={lo_thr}dB:ratio=2:attack=20:release=200[loC];"
         f"[mid]bandpass=f=140:width_type=o:w=3,acompressor=threshold={mid_thr}dB:ratio=1.8:attack=15:release=180[midC];"
         f"[hi]highpass=f=4000,acompressor=threshold={hi_thr}dB:ratio=1.5:attack=10:release=150[hiC];"
         "[loC][midC][hiC]amix=inputs=3:normalize=0[mb];")
    eq=f"[mb]firequalizer=gain='if(f<120,{lo_gain}, if(f<300,{mud_cut}, if(f>12000,{hi_gain},0)))'[eq];"
    sat=f"[eq]alimiter=limit=0.0:level_in={drive_in}:level_out=1.0[sat];"
    loud=f"[sat]loudnorm=I={target_lufs}:LRA=7:TP={target_tp}:dual_mono=true:linear=true[out]"
    return mb+eq+sat+loud, {
        "targets":{"lufs_i":target_lufs,"true_peak_db":target_tp},
        "eq":{"low_shelf_db":lo_gain,"mud_cut_db":mud_cut,"high_shelf_db":hi_gain},
        "mb_thr":{"low":lo_thr,"mid":mid_thr,"high":hi_thr},
        "saturation":{"drive_in":drive_in}
    }

def render_variant(in_wav, out_wav, filtergraph):
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error","-i",in_wav,
         "-filter_complex",filtergraph,"-map","[out]","-ar","44100","-ac","2",out_wav]
    subprocess.run(cmd, check=True)

def codec_sim_truepeak(path_wav):
    aac = path_wav.replace(".wav","_aac.m4a")
    wav2= path_wav.replace(".wav","_aac_decode.wav")
    try:
        subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",path_wav,"-c:a","aac","-b:a","256k",aac], check=True)
        subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",aac, wav2], check=True)
        y, sr = sf.read(wav2); 
        if y.ndim>1: y=y.mean(axis=1)
        y_os = librosa.resample(y.astype(np.float32), sr, sr*4)
        tp = 20*np.log10(np.max(np.abs(y_os))+1e-12)
        return float(tp)
    finally:
        for f in (aac,wav2):
            if os.path.exists(f): os.remove(f)
