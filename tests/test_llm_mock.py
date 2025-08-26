from core import analyze_audio, build_filtergraph

def fake_plan_dark_preserve(analysis, intent, prompt):
    # emulate an LLM plan you’d accept
    return {
      "targets":{"lufs_i":-11.5,"true_peak_db":-1.0},
      "eq":{"low_shelf_db":0.5,"mud_cut_db":-1.2,"high_shelf_db":0.8},
      "mb_comp":{"low":{"threshold_db":-22},"mid":{"threshold_db":-24},"high":{"threshold_db":-26}},
      "saturation":{"drive_db":1.5},
      "stereo":{"amount":0.08},
      "explanation":"dark intent; gentle air and mud cut"
    }

def test_fake_llm_plan_bounds():
    plan = fake_plan_dark_preserve({}, {}, "")
    assert plan["eq"]["high_shelf_db"] <= 1.5
    assert plan["targets"]["true_peak_db"] == -1.0
