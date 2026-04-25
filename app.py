import gradio as gr
import numpy as np
from scipy import stats

DATASETS = {
    "temperature_climate": {
        "name": "Climate Temperature Study",
        "group_a": [22.1, 23.4, 21.8, 24.2, 22.9, 23.1, 21.5, 24.8, 22.3, 23.7],
        "group_b": [25.3, 26.1, 24.8, 27.2, 25.9, 26.4, 24.5, 27.8, 25.1, 26.9],
        "context": "Comparing temperatures between two climate zones (Celsius)",
        "correct_test": "welch_t",
        "expected_conclusion": "significant"
    },
    "drug_response": {
        "name": "Drug Response Trial",
        "group_a": [45, 52, 48, 51, 47, 50, 46, 53, 49, 44],
        "group_b": [48, 51, 47, 52, 49, 50, 46, 53, 48, 51],
        "context": "Comparing biomarker levels between treatment and control groups",
        "correct_test": "welch_t",
        "expected_conclusion": "not_significant"
    },
    "exam_scores": {
        "name": "Teaching Method Comparison",
        "group_a": [72, 68, 75, 71, 69, 73, 70, 74, 68, 76],
        "group_b": [81, 85, 79, 83, 87, 82, 84, 80, 86, 88],
        "context": "Comparing exam scores between two teaching methods",
        "correct_test": "welch_t",
        "expected_conclusion": "significant"
    },
    "reaction_time": {
        "name": "Reaction Time Study",
        "group_a": [0.231, 0.245, 0.228, 0.251, 0.239],
        "group_b": [0.229, 0.241, 0.235, 0.248, 0.242],
        "context": "Comparing reaction times (seconds) between two groups",
        "correct_test": "welch_t",
        "expected_conclusion": "not_significant"
    }
}

class SciAgentEpisode:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step = 0
        self.dataset_key = None
        self.dataset = None
        self.steps_completed = []
        self.total_reward = 0.0
        self.history = []

episode = SciAgentEpisode()

def reward_step_completion(response):
    return 0.1 if response and len(response.strip()) > 20 else 0.0

def reward_hypothesis_quality(hypothesis):
    keywords = ["hypothesis", "expect", "predict", "difference", "similar", "group", "significant"]
    return min(sum(0.015 for k in keywords if k.lower() in hypothesis.lower()), 0.15)

def reward_test_selection(chosen, correct):
    return 0.2 if chosen == correct else 0.05

def reward_numerical_accuracy(p_guess, actual_p):
    try:
        error = abs(float(p_guess) - actual_p)
        if error < 0.01: return 0.2
        elif error < 0.05: return 0.1
        return 0.05
    except:
        return 0.0

def reward_conclusion_correctness(conclusion, expected):
    c = conclusion.lower()
    if expected == "significant" and ("significant" in c or "reject" in c or "differ" in c):
        return 0.25
    if expected == "not_significant" and ("not significant" in c or "fail to reject" in c or "no difference" in c):
        return 0.25
    return 0.05

def step1_explore(dataset_choice):
    episode.reset()
    episode.dataset_key = dataset_choice
    episode.dataset = DATASETS[dataset_choice]
    episode.step = 1
    d = episode.dataset
    a, b = d["group_a"], d["group_b"]
    r = reward_step_completion(dataset_choice)
    episode.total_reward += r
    episode.steps_completed.append("explore")
    episode.history.append(f"Dataset: {d['name']}")
    return f"""[ STEP 01 :: DATA INITIALIZED ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET   » {d['name']}
CONTEXT   » {d['context']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUP_A   » {a}
  MEAN={round(np.mean(a),3)}  STD={round(np.std(a),3)}  N={len(a)}

GROUP_B   » {b}
  MEAN={round(np.mean(b),3)}  STD={round(np.std(b),3)}  N={len(b)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD    » +{round(r,3)} | TOTAL={round(episode.total_reward,3)}
STATUS    » PROCEED TO STEP 02"""

def step2_hypothesize(hypothesis):
    if episode.step != 1:
        return "[ ERROR ] Complete Step 01 first."
    episode.step = 2
    r = reward_step_completion(hypothesis) + reward_hypothesis_quality(hypothesis)
    episode.total_reward += r
    episode.steps_completed.append("hypothesize")
    episode.history.append(f"Hypothesis: {hypothesis}")
    return f"""[ STEP 02 :: HYPOTHESIS LOGGED ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT     » "{hypothesis}"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY   » ANALYZED
REWARD    » +{round(r,3)} | TOTAL={round(episode.total_reward,3)}
STATUS    » PROCEED TO STEP 03"""

def step3_plan(test_choice):
    if episode.step != 2:
        return "[ ERROR ] Complete Step 02 first."
    episode.step = 3
    correct = episode.dataset["correct_test"]
    r = reward_step_completion(test_choice) + reward_test_selection(test_choice, correct)
    episode.total_reward += r
    episode.steps_completed.append("plan")
    episode.history.append(f"Test: {test_choice}")
    verdict = "OPTIMAL SELECTION" if test_choice == correct else f"SUBOPTIMAL — RECOMMENDED: {correct}"
    return f"""[ STEP 03 :: TEST PROTOCOL SELECTED ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST      » {test_choice.upper()}
VERDICT   » {verdict}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD    » +{round(r,3)} | TOTAL={round(episode.total_reward,3)}
STATUS    » PROCEED TO STEP 04"""

def step4_execute(p_value_input):
    if episode.step != 3:
        return "[ ERROR ] Complete Step 03 first."
    episode.step = 4
    a, b = episode.dataset["group_a"], episode.dataset["group_b"]
    t_stat, actual_p = stats.ttest_ind(a, b, equal_var=False)
    r = reward_step_completion(p_value_input) + reward_numerical_accuracy(p_value_input, actual_p)
    episode.total_reward += r
    episode.steps_completed.append("execute")
    episode.history.append(f"p={p_value_input}, actual={round(actual_p,4)}")
    return f"""[ STEP 04 :: STATISTICAL ANALYSIS COMPLETE ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T-STAT    » {round(t_stat,4)}
P-VALUE   » {round(actual_p,4)}  (actual)
YOUR EST  » {p_value_input}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD    » +{round(r,3)} | TOTAL={round(episode.total_reward,3)}
STATUS    » PROCEED TO STEP 05"""

def step5_conclude(conclusion):
    if episode.step != 4:
        return "[ ERROR ] Complete Step 04 first."
    episode.step = 5
    expected = episode.dataset["expected_conclusion"]
    a, b = episode.dataset["group_a"], episode.dataset["group_b"]
    _, actual_p = stats.ttest_ind(a, b, equal_var=False)
    r = reward_step_completion(conclusion) + reward_conclusion_correctness(conclusion, expected)
    if len(episode.history) >= 3: r += 0.1
    episode.total_reward += r
    episode.steps_completed.append("conclude")
    ground_truth = "SIGNIFICANT DIFFERENCE DETECTED" if actual_p < 0.05 else "NO SIGNIFICANT DIFFERENCE"
    perf = "EXCELLENT" if episode.total_reward > 0.7 else "GOOD" if episode.total_reward > 0.5 else "NEEDS IMPROVEMENT"
    return f"""[ STEP 05 :: EPISODE TERMINATED ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCLUSION » "{conclusion}"
GROUND TRUTH » {ground_truth} (p={round(actual_p,4)})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEPS COMPLETED » {' → '.join(episode.steps_completed)}
FINAL REWARD    » {round(episode.total_reward,3)} / 1.0
PERFORMANCE     » {perf}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESET TO INITIALIZE NEW EPISODE"""

def do_reset():
    episode.reset()
    return "[ SYSTEM RESET ] New episode ready. Return to Step 01."

CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #050510 !important;
    font-family: 'Courier New', monospace !important;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}
h1 { 
    color: #7B61FF !important; 
    font-size: 2.2rem !important;
    letter-spacing: 6px !important;
    text-transform: uppercase !important;
    text-shadow: 0 0 30px #7B61FF88 !important;
    font-family: 'Courier New', monospace !important;
}
h3 {
    color: #00D4FF !important;
    letter-spacing: 3px !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    border-left: 2px solid #7B61FF !important;
    padding-left: 10px !important;
    font-family: 'Courier New', monospace !important;
}
p {
    color: #8888aa !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    font-family: 'Courier New', monospace !important;
}
.block {
    background: #0a0a1a !important;
    border: 1px solid #7B61FF33 !important;
    border-radius: 4px !important;
}
label span {
    color: #00D4FF !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-family: 'Courier New', monospace !important;
}
input, textarea, select {
    background: #050510 !important;
    border: 1px solid #7B61FF55 !important;
    color: #00D4FF !important;
    font-family: 'Courier New', monospace !important;
    border-radius: 2px !important;
}
textarea {
    color: #7B61FF !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
}
button {
    background: transparent !important;
    border: 1px solid #7B61FF !important;
    color: #7B61FF !important;
    font-family: 'Courier New', monospace !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    transition: all 0.2s !important;
}
button:hover {
    background: #7B61FF22 !important;
    box-shadow: 0 0 20px #7B61FF44 !important;
    color: #00D4FF !important;
}
.gap { gap: 12px !important; }
"""

with gr.Blocks(css=CSS, title="SciAgent // RL Environment") as demo:
    gr.Markdown("# SCIAGENT")
    gr.Markdown("REINFORCEMENT LEARNING ENVIRONMENT // SCIENTIFIC HYPOTHESIS TESTING // 5-STEP PROTOCOL")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 01 // EXPLORE")
            dataset_dropdown = gr.Dropdown(
                choices=list(DATASETS.keys()),
                label="SELECT DATASET",
                value="temperature_climate"
            )
            btn1 = gr.Button("INITIALIZE DATASET")
            out1 = gr.Textbox(label="SYSTEM OUTPUT", lines=12)
            btn1.click(step1_explore, inputs=dataset_dropdown, outputs=out1)

        with gr.Column():
            gr.Markdown("### 02 // HYPOTHESIZE")
            hypothesis_input = gr.Textbox(
                label="ENTER HYPOTHESIS",
                placeholder="I expect that Group A and Group B will...",
                lines=3
            )
            btn2 = gr.Button("TRANSMIT HYPOTHESIS")
            out2 = gr.Textbox(label="SYSTEM OUTPUT", lines=7)
            btn2.click(step2_hypothesize, inputs=hypothesis_input, outputs=out2)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 03 // PLAN")
            test_dropdown = gr.Dropdown(
                choices=["welch_t", "student_t", "mann_whitney", "anova"],
                label="SELECT TEST PROTOCOL",
                value="welch_t"
            )
            btn3 = gr.Button("ENGAGE PROTOCOL")
            out3 = gr.Textbox(label="SYSTEM OUTPUT", lines=7)
            btn3.click(step3_plan, inputs=test_dropdown, outputs=out3)

        with gr.Column():
            gr.Markdown("### 04 // EXECUTE")
            pval_input = gr.Textbox(
                label="ENTER P-VALUE",
                placeholder="e.g. 0.032"
            )
            btn4 = gr.Button("RUN ANALYSIS")
            out4 = gr.Textbox(label="SYSTEM OUTPUT", lines=7)
            btn4.click(step4_execute, inputs=pval_input, outputs=out4)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 05 // CONCLUDE")
            conclusion_input = gr.Textbox(
                label="ENTER CONCLUSION",
                placeholder="Based on the results...",
                lines=3
            )
            btn5 = gr.Button("FINALIZE EPISODE")
            out5 = gr.Textbox(label="SYSTEM OUTPUT", lines=10)
            btn5.click(step5_conclude, inputs=conclusion_input, outputs=out5)

        with gr.Column(scale=1):
            gr.Markdown("### SYS // RESET")
            gr.Markdown("Terminate current episode and reinitialize environment.")
            btn_reset = gr.Button("RESET SYSTEM", variant="secondary")
            reset_out = gr.Textbox(label="STATUS")
            btn_reset.click(do_reset, outputs=reset_out)

demo.launch()
