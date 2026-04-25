"""
SciAgent — GRPO Training Script
================================
Run this in a Colab T4/A100 notebook to train Qwen2.5-3B with GRPO
on the SciAgent scientific reasoning environment.

Usage:
    python train_grpo.py

Outputs:
    plots/loss_curve.png
    plots/reward_curve.png
    sciagent-output/  (checkpoint)
"""

import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset

# ── Add repo root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from environment.sciagent_env import SciAgentEnv, DATASETS

os.makedirs("plots", exist_ok=True)
os.makedirs("sciagent-output", exist_ok=True)

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SciAgent, an AI scientist. Given a dataset and a research question:
1. Form a testable hypothesis
2. Choose the most appropriate statistical test
3. Provide step-by-step reasoning
4. State your conclusion (true=significant difference, false=not significant, null=need more info)

ALWAYS respond with valid JSON only — no preamble, no markdown fences:
{
  "hypothesis": "...",
  "statistical_test": "...",
  "reasoning": "...",
  "conclusion": true
}"""


# ── Build dataset ──────────────────────────────────────────────────────────

def build_training_dataset(n: int = 800, seed: int = 42) -> Dataset:
    rng = random.Random(seed)
    pool = DATASETS * (n // len(DATASETS) + 1)
    rng.shuffle(pool)

    examples = []
    for d in pool[:n]:
        prompt = f"Dataset: {json.dumps(d['data'])}\nQuestion: {d['question']}"
        examples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": prompt},
            ],
            "dataset_name": d["name"],
        })
    return Dataset.from_list(examples)


# ── Reward functions ───────────────────────────────────────────────────────

def programmatic_reward(completions, dataset_name=None, **kwargs):
    """Score each completion using the SciAgentEnv reward function."""
    rewards = []
    for completion in completions:
        env = SciAgentEnv(seed=42)
        target_name = dataset_name if dataset_name else DATASETS[0]["name"]
        match = next((d for d in DATASETS if d["name"] == target_name), DATASETS[0])
        env.dataset = match
        _, reward, _, _, _ = env.step(completion)
        rewards.append(float(reward))
    return rewards


def format_reward(completions, **kwargs):
    """Bonus reward for valid JSON format."""
    rewards = []
    for c in completions:
        try:
            cleaned = c.strip().lstrip("```json").rstrip("```").strip()
            obj = json.loads(cleaned)
            required = {"hypothesis", "statistical_test", "reasoning", "conclusion"}
            score = 0.15 if required.issubset(obj.keys()) else 0.05
        except Exception:
            score = 0.0
        rewards.append(score)
    return rewards


# ── Main training ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SciAgent GRPO Training")
    print("=" * 60)

    # 1. Load model
    print("\n[1/5] Loading model with Unsloth...")
    try:
        from unsloth import FastLanguageModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-3B-Instruct",
            max_seq_length=1024,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print("✅ Model loaded (Qwen2.5-3B-Instruct, 4-bit, LoRA r=8)")
    except ImportError:
        print("❌ unsloth not found. Install with: pip install unsloth")
        sys.exit(1)

    # 2. Dataset
    print("\n[2/5] Building training dataset...")
    ds = build_training_dataset(n=800)
    print(f"✅ Dataset: {len(ds)} examples across {len(DATASETS)} scenario types")

    # 3. Verify reward
    print("\n[3/5] Verifying reward functions...")
    test_action = json.dumps({
        "hypothesis": "Group A scores significantly higher than Group B",
        "statistical_test": "Welch t-test",
        "reasoning": "Two independent samples with potentially unequal variances",
        "conclusion": True,
    })
    r = programmatic_reward([test_action], dataset_name="exam_scores")
    f = format_reward([test_action])
    print(f"✅ Programmatic reward: {r[0]:.2f}  |  Format reward: {f[0]:.2f}")

    # 4. GRPO training
    print("\n[4/5] Starting GRPO training...")
    print("Expected time: ~45 min on T4, ~15 min on A100")
    print("Watch the reward column — it should climb from ~0.4 → 0.7+\n")

    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        learning_rate=5e-6,
        max_steps=300,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_new_tokens=256,
        temperature=0.8,
        output_dir="./sciagent-output",
        logging_steps=10,
        save_steps=100,
        report_to="none",
        seed=42,
        optim="adamw_8bit",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[programmatic_reward, format_reward],
        args=config,
        train_dataset=ds,
    )

    trainer.train()
    print("✅ Training complete!")

    # 5. Save plots
    print("\n[5/5] Saving training curves...")
    log = trainer.state.log_history

    # Loss
    loss_steps = [l["step"]  for l in log if "loss"   in l]
    loss_vals  = [l["loss"]  for l in log if "loss"   in l]

    plt.figure(figsize=(9, 4))
    plt.plot(loss_steps, loss_vals, color="#7c3aed", linewidth=2.5)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss",           fontsize=12)
    plt.title("SciAgent — Training Loss (GRPO, Qwen2.5-3B-Instruct)", fontsize=13)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("plots/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved plots/loss_curve.png")

    # Reward
    rew_keys   = [k for k in (log[0].keys() if log else []) if "reward" in k.lower()]
    rew_key    = rew_keys[0] if rew_keys else "reward"
    rew_steps  = [l["step"]   for l in log if rew_key in l]
    rew_vals   = [l[rew_key]  for l in log if rew_key in l]

    baseline = rew_vals[0] if rew_vals else 0.4
    plt.figure(figsize=(9, 4))
    plt.plot(rew_steps, rew_vals, color="#059669", linewidth=2.5, label="GRPO reward")
    plt.axhline(y=baseline, color="#6b7280", linestyle="--", alpha=0.6, label=f"Baseline ({baseline:.2f})")
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Composite Reward", fontsize=12)
    plt.title("SciAgent — Reward Progression (GRPO vs Baseline)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("plots/reward_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved plots/reward_curve.png")

    # Push to HF Hub (optional)
    hf_username = os.environ.get("HF_USERNAME", "YOUR_USERNAME")
    if hf_username != "YOUR_USERNAME":
        print(f"\nPushing model to HuggingFace Hub as {hf_username}/sciagent-qwen2.5-3b ...")
        model.push_to_hub_merged(
            f"{hf_username}/sciagent-qwen2.5-3b",
            tokenizer,
            save_method="merged_16bit",
        )
        print("✅ Model pushed!")
    else:
        print("\nℹ  Set HF_USERNAME env var to auto-push to HuggingFace Hub")

    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print("  1. Download plots/loss_curve.png and plots/reward_curve.png")
    print("  2. Upload both PNGs to GitHub under plots/")
    print("  3. Share this Colab notebook link in your README")
    print("=" * 60)


if __name__ == "__main__":
    main()
