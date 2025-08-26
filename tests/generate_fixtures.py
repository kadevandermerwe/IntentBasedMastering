import os, numpy as np, soundfile as sf

OUT="tests/fixtures"
os.makedirs(OUT, exist_ok=True)
sr=44100
dur=30  # seconds

def norm(x): 
    m = np.max(np.abs(x))+1e-12
    return x/m*0.5  # ~ -6 dBFS headroom

# 1) Deep house kick + sub + hats (dynamic)
t=np.linspace(0,dur,int(sr*dur),endpoint=False)
kick_env = (np.sin(2*np.pi*50*t)*np.exp(-t*30)).astype(np.float32)
kick = np.zeros_like(t)
for i in range(0, dur*sr, int(sr*0.5)):  # 120 BPM 4-on-floor
    end=min(i+len(kick_env), len(kick))
    kick[i:end]+=kick_env[:end-i]
sub = 0.2*np.sin(2*np.pi*50*t)
hats = 0.05*np.sign(np.sin(2*np.pi*8*t))  # 8 Hz tick (placeholder)
mix = norm(kick + sub + hats)
sf.write(f"{OUT}/deep_house_fixture.wav", mix, sr)

# 2) Bright sibilant pad (to test don’t over-brighten)
pad = 0.2*np.sin(2*np.pi*220*t) + 0.2*np.sin(2*np.pi*440*t)
sizzle = 0.1*np.random.randn(len(t))
bright = norm(pad + sizzle)
sf.write(f"{OUT}/bright_sizzle_fixture.wav", bright, sr)

# 3) Sparse ambient (breakdown preservation)
ambient = norm(0.2*np.sin(2*np.pi*110*t) * np.exp(-t*0.2))
sf.write(f"{OUT}/ambient_fixture.wav", ambient, sr)
print("Fixtures written to", OUT)
