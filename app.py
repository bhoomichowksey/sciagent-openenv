"""
SciAgent — HuggingFace Gradio Space
=====================================
Deploy this as a public Gradio Space on HuggingFace.
SDK: Gradio | Visibility: Public

Upload this file + requirements.txt to your Space.
"""

import json
import os
import sys

import gradio as gr
from scipy import stats

# ── Import environment (works both locally and in Space) ──────────────────
sys.path.insert(0, os.path.dirname(__file__))
try:
    from environment.sciagent_env import SciAgentEnv, DATASETS
except ImportError:
    # Inline fallback for Space deployment without subfolder
    import random
    from scipy import stats as _stats

    DATASETS = [
        {"name": "exam_scores",  "data": {"group_a": [72,85,90,68,75,88,92,70], "group_b": [65,70,78,62,69,74,80,67]}, "question": "Does group A score significantly higher than group B?"},
        {"name": "drug_trial",   "data": {"treatment": [8.2,7.5,9.1,6.8,8.8,7.2,9.5,8.0], "placebo": [7.1,6.8,7.5,6.2,7.8,6.5,7.0,6.9]}, "question": "Does the drug reduce symptoms significantly vs placebo?"},
        {"name": "website_ab",   "data": {"variant_a": [3.2,4.1,2.8,3.9,4.5,3.0,4.2,3.7], "variant_b": [2.1,2.8,1.9,2.5,3.1,2.3,2.7,2.4]}, "question": "Does variant A produce a significantly higher conversion rate?"},
        {"name": "temperature",  "data": {"city_x": [22,24,23,25,21,26,24,23], "city_y": [18,20,19,21,17,22,20,19]}, "question": "Is city X significantly warmer than city Y?"},
        {"name": "crop_yield",   "data": {"fertilizer_a": [45.2,48.1,44.7,50.3,47.8,46.5,49.1,45.9], "fertilizer_b": [40.1,42.5,39.8,43.7,41.2,40.9,42.1,41.5]}, "question": "Does fertilizer A produce significantly higher crop yield?"},
    ]

    class SciAgentEnv:
        def __init__(self, seed=42):
            self.rng = random.Random(seed)
            self.dataset = None
            self.step_count = 0
            self.history = []

        def reset(self, seed=None):
            if seed is not None:
                self.rng = random.Random(seed)
            self.dataset = self.rng.choice(DATASETS)
            self.step_count = 0
            self.history = []
            return self.state(), {}

        def step(self, action_str):
            self.step_count += 1
            try:
                action = json.loads(action_str.strip().lstrip("```json").rstrip("```").strip())
            except:
                return self.state(), 0.0, True, False, {"error": "invalid_json"}
            reward = self._reward(action)
            done = self.step_count >= 3 or action.get("conclusion") is not None
            return self.state(), reward, done, False, {"reward": reward}

        def state(self):
            return {"question": self.dataset["question"], "data": self.dataset["data"], "step": self.step_count, "history": self.history}

        def _reward(self, a):
            s = 0.2 * bool(a.get("hypothesis")) + 0.2 * bool(a.get("statistical_test")) + 0.2 * (a.get("conclusion") is not None)
            if any(t in (a.get("statistical_test") or "").lower() for t in ["t-test","ttest","welch"]):
                s += 0.2
            if a.get("conclusion") is not None:
                d = self.dataset["data"]; keys = list(d.keys())
                try:
                    _, p = _stats.ttest_ind(d[keys[0]], d[keys[1]], equal_var=False)
                    if (p < 0.05) == any(w in str(a.get("conclusion","")).lower() for w in ["true","yes","significant","confirmed"]):
                        s += 0.2
                except: pass
            return min(round(s, 4), 1.0)


# ── Gradio UI ─────────────────────────────────────────────────────────────

def run_demo(dataset_name: str, hypothesis: str, test: str, reasoning: str, conclusion_str: str):
    conclusion_map = {"True (significant)": True, "False (not significant)": False, "Null (need more data)": None}
    conclusion = conclusion_map.get(conclusion_str, None)

    action = json.dumps({
        "hypothesis": hypothesis,
        "statistical_test": test,
        "reasoning": reasoning,
        "conclusion": conclusion,
    })

    env = SciAgentEnv(seed=42)
    env.reset()
    target = next((d for d in DATASETS if d["name"] == dataset_name), DATASETS[0])
    env.dataset = target
    obs, reward, done, _, info = env.step(action)

    d = target["data"]
    keys = list(d.keys())
    _, pval = stats.ttest_ind(d[keys[0]], d[keys[1]], equal_var=False)
    ground_truth = pval < 0.05

    result = f"""## Evaluation Results

**Dataset:** {dataset_name}
**Question:** {target['question']}

---

### Your Action
```json
{json.dumps(json.loads(action), indent=2)}
```

---

### Reward Breakdown
| Component | Points |
|---|---|
| Hypothesis present | {0.2 if hypothesis.strip() else 0.0} / 0.2 |
| Statistical test named | {0.2 if test.strip() else 0.0} / 0.2 |
| Conclusion drawn | {0.2 if conclusion is not None else 0.0} / 0.2 |
| Appropriate test (t-test family) | {0.2 if any(t in test.lower() for t in ['t-test','ttest','welch']) else 0.0} / 0.2 |
| Correct conclusion | {0.2 if conclusion is not None and (pval < 0.05) == (conclusion is True) else 0.0} / 0.2 |
| **Total** | **{reward:.2f} / 1.0** |

---

### Ground Truth (Welch t-test)
- **p-value:** {pval:.4f}
- **Significant at α=0.05:** {'✅ Yes' if ground_truth else '❌ No'}
- **Your conclusion:** {'✅ Correct!' if conclusion is not None and (pval < 0.05) == (conclusion is True) else '❌ Incorrect'}
"""
    return result


def load_dataset_info(dataset_name: str):
    target = next((d for d in DATASETS if d["name"] == dataset_name), DATASETS[0])
    info = f"**Question:** {target['question']}\n\n**Data:**\n```json\n{json.dumps(target['data'], indent=2)}\n```"
    return info


dataset_names = [d["name"] for d in DATASETS]

with gr.Blocks(
    title="SciAgent — RL for Scientific Reasoning",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
    .container { max-width: 860px; margin: 0 auto; }
    .reward-box { background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 12px; }
    """
) as demo:

    gr.Markdown("""
# 🔬 SciAgent — RL Environment for Hypothesis-Driven Science

**Meta PyTorch OpenEnv Hackathon 2026**

SciAgent trains LLMs to act like scientists: form a hypothesis, choose a statistical test, interpret results.
This demo lets you interact with the environment directly and see your reward score.

> 📖 [GitHub](https://github.com/YOUR_USERNAME/sciagent-openenv) | 🤗 [Model](https://huggingface.co/YOUR_USERNAME/sciagent-qwen2.5-3b) | 📓 [Colab](https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Choose a Dataset")
            dataset_dd = gr.Dropdown(choices=dataset_names, value=dataset_names[0], label="Dataset")
            dataset_info = gr.Markdown(load_dataset_info(dataset_names[0]))
            dataset_dd.change(fn=load_dataset_info, inputs=dataset_dd, outputs=dataset_info)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Act as the Agent")
            hypothesis_tb = gr.Textbox(
                label="Hypothesis",
                placeholder="e.g. Group A scores significantly higher than Group B",
                lines=2,
            )
            test_tb = gr.Textbox(
                label="Statistical Test",
                placeholder="e.g. Welch t-test",
            )
            reasoning_tb = gr.Textbox(
                label="Reasoning",
                placeholder="e.g. Two independent groups with unknown variance → Welch t-test",
                lines=3,
            )
            conclusion_dd = gr.Dropdown(
                choices=["True (significant)", "False (not significant)", "Null (need more data)"],
                value="True (significant)",
                label="Conclusion",
            )
            run_btn = gr.Button("▶ Run Agent Step", variant="primary")

    output_md = gr.Markdown("*Click 'Run Agent Step' to see your reward.*")

    run_btn.click(
        fn=run_demo,
        inputs=[dataset_dd, hypothesis_tb, test_tb, reasoning_tb, conclusion_dd],
        outputs=output_md,
    )

    gr.Markdown("""
---
### How SciAgent Works

| Step | Agent Action | Reward Signal |
|---|---|---|
| 1 | Form hypothesis | +0.2 if hypothesis present |
| 2 | Choose statistical test | +0.2 if test named; +0.2 if t-test family |
| 3 | Draw conclusion | +0.2 if conclusion present; +0.2 if statistically correct |

**Training:** GRPO (Group Relative Policy Optimization) on Qwen2.5-3B-Instruct via Unsloth + TRL.
Reward improved from ~0.40 (random) to ~0.72+ after 300 training steps.
    """)

if __name__ == "__main__":
    demo.launch()
