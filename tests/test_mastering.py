import os, tempfile
from core import analyze_audio, build_filtergraph, render_variant, codec_sim_truepeak

SR_FIX = "tests/fixtures/deep_house_fixture.wav"
BR_FIX = "tests/fixtures/bright_sizzle_fixture.wav"
AM_FIX = "tests/fixtures/ambient_fixture.wav"

def _render_variant_on(fixture, intent, variant):
    a = analyze_audio(fixture)
    fg, params = build_filtergraph(a, intent, variant)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, f"{variant}.wav")
        render_variant(fixture, out, fg)
        a_post = analyze_audio(out)
        tp_aac = codec_sim_truepeak(out)
        return a, a_post, tp_aac, params

def test_true_peak_safety_deep_house():
    intent = {"tone":-0.3,"dynamics":-0.3,"stereo":0.0,"character":0.6}
    a, post, tp_aac, _ = _render_variant_on(SR_FIX, intent, "vibe")
    assert post["true_peak_dbfs_est"] <= -0.9, "True peak not capped before codec"
    assert tp_aac is None or tp_aac <= -0.5, "True peak overs after AAC encode"

def test_loudness_targets_vary_by_variant():
    intent = {"tone":0.0,"dynamics":0.0,"stereo":0.0,"character":0.0}
    _, post_cons, _, _ = _render_variant_on(SR_FIX, intent, "conservative")
    _, post_vibe, _, _ = _render_variant_on(SR_FIX, intent, "vibe")
    _, post_loud, _, _ = _render_variant_on(SR_FIX, intent, "loud")
    assert post_loud["lufs_integrated"] > post_vibe["lufs_integrated"]  # more negative is quieter
    assert post_cons["lufs_integrated"] < post_vibe["lufs_integrated"]

def test_dark_intent_does_not_overbrighten():
    intent = {"tone":-1.0,"dynamics":0.0,"stereo":0.0,"character":0.0}
    a, post, _, _ = _render_variant_on(BR_FIX, intent, "vibe")
    # high band % should not jump > ~2 dB equivalent; we approximate by % delta
    hi_pre = a["spectral_pct"]["high"]; hi_post = post["spectral_pct"]["high"]
    assert hi_post <= hi_pre + 3.0, f"Highs boosted too much for dark intent ({hi_pre:.1f} → {hi_post:.1f}%)"

def test_breakdown_preserves_dynamics_on_ambient():
    intent = {"tone":0.0,"dynamics":-1.0,"stereo":0.0,"character":0.0}
    a, post, _, _ = _render_variant_on(AM_FIX, intent, "vibe")
    # Ambient LUFS should not be pushed close to -9; keep dynamic space
    assert post["lufs_integrated"] <= -10.5, "Ambient got pushed too loud; dynamics not preserved"
