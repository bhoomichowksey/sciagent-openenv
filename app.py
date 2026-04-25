import gradio as gr
import numpy as np
from scipy import stats
import json

# === DATASETS ===
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

# === EPISODE STATE ===
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

    def get_state(self):
        return {
            "step": self.step,
            "dataset": self.dataset_key,
            "steps_completed": self.steps_completed,
            "total_reward": round(self.total_reward, 3)
        }

episode = SciAgentEpisode()

# === REWARD FUNCTIONS ===
def reward_step_completion(step_num, response):
    if response and len(response.strip()) > 20:
        return 0.1
    return 0.0

def reward_hypothesis_quality(hypothesis):
    keywords = ["hypothesis", "expect", "predict", "difference", "similar", "group", "significant"]
    score = sum(0.015 for k in keywords if k.lower() in hypothesis.lower())
    return min(score, 0.15)

def reward_test_selection(chosen_test, correct_test):
    if chosen_test == correct_test:
        return 0.2
    return 0.05

def reward_numerical_accuracy(p_value_guess, actual_p):
    try:
        error = abs(float(p_value_guess) - actual_p)
        if error < 0.01:
            return 0.2
        elif error < 0.05:
            return 0.1
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

def reward_reasoning_coherence(full_history):
    if len(full_history) >= 3:
        return 0.1
    return 0.0

# === STEP HANDLERS ===
def step1_explore(dataset_choice):
    episode.reset()
    episode.dataset_key = dataset_choice
    episode.dataset = DATASETS[dataset_choice]
    episode.step = 1

    d = episode.dataset
    a = d["group_a"]
    b = d["group_b"]

    r = reward_step_completion(1, dataset_choice)
    episode.total_reward += r
    episode.steps_completed.append("explore")
    episode.history.append(f"Selected dataset: {d['name']}")

    output = f"""**Step 1 — Explore Data**

Dataset: {d['name']}
Context: {d['context']}

Group A: {a}
- Mean: {round(np.mean(a), 3)}, Std: {round(np.std(a), 3)}, N={len(a)}

Group B: {b}
- Mean: {round(np.mean(b), 3)}, Std: {round(np.std(b), 3)}, N={len(b)}

Step reward: +{round(r, 3)} | Total: {round(episode.total_reward, 3)}

**Now move to Step 2 → Write your hypothesis**"""
    return output

def step2_hypothesize(hypothesis):
    if episode.step != 1:
        return "Please complete Step 1 first."
    episode.step = 2

    r = reward_step_completion(2, hypothesis) + reward_hypothesis_quality(hypothesis)
    episode.total_reward += r
    episode.steps_completed.append("hypothesize")
    episode.history.append(f"Hypothesis: {hypothesis}")

    output = f"""**Step 2 — Hypothesis Recorded**

Your hypothesis: "{hypothesis}"

Reward for quality hypothesis: +{round(r, 3)} | Total: {round(episode.total_reward, 3)}

**Now move to Step 3 → Choose your statistical test**"""
    return output

def step3_plan(test_choice):
    if episode.step != 2:
        return "Please complete Step 2 first."
    episode.step = 3

    correct = episode.dataset["correct_test"]
    r = reward_step_completion(3, test_choice) + reward_test_selection(test_choice, correct)
    episode.total_reward += r
    episode.steps_completed.append("plan")
    episode.history.append(f"Test chosen: {test_choice}")

    feedback = "Correct test selected!" if test_choice == correct else f"Hint: {correct} is more appropriate here."

    output = f"""**Step 3 — Test Selected**

You chose: {test_choice}
{feedback}

Step reward: +{round(r, 3)} | Total: {round(episode.total_reward, 3)}

**Now move to Step 4 → Run the test and enter the p-value you observe**"""
    return output

def step4_execute(p_value_input):
    if episode.step != 3:
        return "Please complete Step 3 first."
    episode.step = 4

    a = episode.dataset["group_a"]
    b = episode.dataset["group_b"]
    t_stat, actual_p = stats.ttest_ind(a, b, equal_var=False)

    r = reward_step_completion(4, p_value_input) + reward_numerical_accuracy(p_value_input, actual_p)
    episode.total_reward += r
    episode.steps_completed.append("execute")
    episode.history.append(f"p-value entered: {p_value_input}, actual: {round(actual_p, 4)}")

    output = f"""**Step 4 — Test Executed**

Actual Welch t-test result:
- t-statistic: {round(t_stat, 4)}
- p-value: {round(actual_p, 4)}

Your p-value estimate: {p_value_input}
Accuracy reward: +{round(r, 3)} | Total: {round(episode.total_reward, 3)}

**Now move to Step 5 → Write your conclusion**"""
    return output

def step5_conclude(conclusion):
    if episode.step != 4:
        return "Please complete Step 4 first."
    episode.step = 5

    expected = episode.dataset["expected_conclusion"]
    r = (reward_step_completion(5, conclusion) +
         reward_conclusion_correctness(conclusion, expected) +
         reward_reasoning_coherence(episode.history))
    episode.total_reward += r
    episode.steps_completed.append("conclude")
    episode.history.append(f"Conclusion: {conclusion}")

    a = episode.dataset["group_a"]
    b = episode.dataset["group_b"]
    _, actual_p = stats.ttest_ind(a, b, equal_var=False)
    ground_truth = "significant difference" if actual_p < 0.05 else "no significant difference"

    output = f"""**Step 5 — Episode Complete**

Your conclusion: "{conclusion}"
Ground truth: {ground_truth} (p={round(actual_p, 4)})

Final step reward: +{round(r, 3)}

**Episode Summary:**
- Steps completed: {', '.join(episode.steps_completed)}
- Total reward: {round(episode.total_reward, 3)} / 1.0
- Performance: {'Excellent' if episode.total_reward > 0.7 else 'Good' if episode.total_reward > 0.5 else 'Needs improvement'}

Reset to try another dataset!"""
    return output

# === GRADIO UI ===
with gr.Blocks(title="SciAgent — Multi-Step Science RL Environment") as demo:
    gr.Markdown("# SciAgent")
    gr.Markdown("A 5-step reinforcement learning environment for scientific hypothesis testing. Complete all steps in order to maximize your reward score.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 1 — Explore Data")
            dataset_dropdown = gr.Dropdown(
                choices=list(DATASETS.keys()),
                label="Choose a dataset",
                value="temperature_climate"
            )
            btn1 = gr.Button("Load Dataset")
            out1 = gr.Textbox(label="Output", lines=10)
            btn1.click(step1_explore, inputs=dataset_dropdown, outputs=out1)

        with gr.Column():
            gr.Markdown("### Step 2 — Form Hypothesis")
            hypothesis_input = gr.Textbox(label="Write your hypothesis", placeholder="I expect that Group A and Group B will...")
            btn2 = gr.Button("Submit Hypothesis")
            out2 = gr.Textbox(label="Output", lines=6)
            btn2.click(step2_hypothesize, inputs=hypothesis_input, outputs=out2)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 3 — Choose Statistical Test")
            test_dropdown = gr.Dropdown(
                choices=["welch_t", "student_t", "mann_whitney", "anova"],
                label="Select test",
                value="welch_t"
            )
            btn3 = gr.Button("Select Test")
            out3 = gr.Textbox(label="Output", lines=6)
            btn3.click(step3_plan, inputs=test_dropdown, outputs=out3)

        with gr.Column():
            gr.Markdown("### Step 4 — Run Test & Enter p-value")
            pval_input = gr.Textbox(label="Enter p-value you calculated", placeholder="e.g. 0.032")
            btn4 = gr.Button("Submit p-value")
            out4 = gr.Textbox(label="Output", lines=8)
            btn4.click(step4_execute, inputs=pval_input, outputs=out4)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 5 — Write Conclusion")
            conclusion_input = gr.Textbox(label="Write your conclusion", placeholder="Based on the results...")
            btn5 = gr.Button("Submit Conclusion")
            out5 = gr.Textbox(label="Final Results", lines=10)
            btn5.click(step5_conclude, inputs=conclusion_input, outputs=out5)

        with gr.Column():
            gr.Markdown("### Reset")
            gr.Markdown("Click below to start a new episode with a different dataset.")
            btn_reset = gr.Button("Reset Episode", variant="secondary")
            reset_out = gr.Textbox(label="Status")
            def do_reset():
                episode.reset()
                return "Episode reset. Go back to Step 1!"
            btn_reset.click(do_reset, outputs=reset_out)

demo.launch()
