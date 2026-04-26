---
title: SciAgent
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: true
---

# SciAgent — RL Environment for Scientific Reasoning

A **5-step reinforcement learning environment** where an LLM agent learns to conduct rigorous statistical hypothesis testing through trial, feedback, and reward-based improvement.

Built for the **OpenEnv Hackathon Round 2** — Theme: Long-Horizon Planning & Instruction Following.

---

## 🔗 Links
| Resource | Link |
|----------|------|
| 🤗 HuggingFace Space | [bhoomichowksey/sciagent](https://huggingface.co/spaces/bhoomichowksey/sciagent) |
| 📓 Colab Training Notebook | [Open in Colab](https://colab.research.google.com/drive/1PaD5YtRZiUtMTIU5eKi--S0uZ7IkJtJ9?usp=sharing) |
| 📝 Blog Post | [Read the writeup](https://huggingface.co/spaces/bhoomichowksey/sciagent/blob/main/blog.md) |
---

## What is SciAgent?

SciAgent frames scientific reasoning as an RL problem. The agent must complete a full hypothesis testing protocol across 5 sequential steps — it cannot skip steps, must maintain logical consistency, and is scored at every stage.

Each episode = one complete scientific investigation.

---

## The 5-Step Protocol

| Step | Task | Max Reward |
|------|------|--------|
| 01 Explore | Read and interpret dataset statistics | +0.10 |
| 02 Hypothesize | Form a scientific prediction | +0.15 |
| 03 Plan | Select the correct statistical test | +0.20 |
| 04 Execute | Run the test, estimate p-value | +0.20 |
| 05 Conclude | Write an evidence-based conclusion | +0.25 |
| Coherence bonus | Consistent reasoning across all steps | +0.10 |

**Max reward per episode: 1.0**

---

## Datasets

- Climate Temperature Study
- Drug Response Trial
- Teaching Method Comparison
- Reaction Time Study

---

## Reward Model

6 independent reward signals — each targeting a distinct scientific reasoning capability:

- Step completion
- Hypothesis quality (keyword + structure analysis)
- Statistical test selection accuracy
- Numerical p-value accuracy
- Conclusion correctness vs ground truth
- Cross-step reasoning coherence

---

## Training Results

Training was performed using GRPO via TRL + Unsloth on Qwen2.5-7B-Instruct.

### Reward Curve
![Reward Curve](reward_curve.png)

### Loss Curve
![Loss Curve](loss_curve.png)

---

## Training Strategy

- **Base model**: Qwen2.5-7B-Instruct
- **Algorithm**: GRPO via TRL + Unsloth
- **Target**: Reward improvement from ~0.35 → >0.70 after 300 steps
- **Evaluation**: Held-out episodes across all datasets, 3 random seeds

---

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- Gradio
- scipy / numpy
- TRL + Unsloth (training)
