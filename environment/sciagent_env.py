"""
SciAgent RL Environment
=======================
Trains LLMs to conduct hypothesis-driven scientific reasoning.

Observation: dataset (two groups) + research question
Action:      JSON with hypothesis, statistical_test, reasoning, conclusion
Reward:      Composite 0-1 score (structural + statistical correctness)
"""

import json
import random
import numpy as np
from scipy import stats


# ── Built-in datasets ────────────────────────────────────────────────────────

DATASETS = [
    {
        "name": "exam_scores",
        "data": {
            "group_a": [72, 85, 90, 68, 75, 88, 92, 70],
            "group_b": [65, 70, 78, 62, 69, 74, 80, 67],
        },
        "question": "Does group A score significantly higher than group B?",
    },
    {
        "name": "drug_trial",
        "data": {
            "treatment": [8.2, 7.5, 9.1, 6.8, 8.8, 7.2, 9.5, 8.0],
            "placebo":   [7.1, 6.8, 7.5, 6.2, 7.8, 6.5, 7.0, 6.9],
        },
        "question": "Does the drug reduce symptoms significantly compared to placebo?",
    },
    {
        "name": "website_ab",
        "data": {
            "variant_a": [3.2, 4.1, 2.8, 3.9, 4.5, 3.0, 4.2, 3.7],
            "variant_b": [2.1, 2.8, 1.9, 2.5, 3.1, 2.3, 2.7, 2.4],
        },
        "question": "Does variant A produce a significantly higher conversion rate?",
    },
    {
        "name": "temperature",
        "data": {
            "city_x": [22, 24, 23, 25, 21, 26, 24, 23],
            "city_y": [18, 20, 19, 21, 17, 22, 20, 19],
        },
        "question": "Is city X significantly warmer than city Y on average?",
    },
    {
        "name": "crop_yield",
        "data": {
            "fertilizer_a": [45.2, 48.1, 44.7, 50.3, 47.8, 46.5, 49.1, 45.9],
            "fertilizer_b": [40.1, 42.5, 39.8, 43.7, 41.2, 40.9, 42.1, 41.5],
        },
        "question": "Does fertilizer A produce significantly higher crop yield than fertilizer B?",
    },
]


# ── Environment ───────────────────────────────────────────────────────────────

class SciAgentEnv:
    """
    Gym-compatible environment for scientific hypothesis testing.

    Usage
    -----
    env = SciAgentEnv()
    obs, _ = env.reset()

    action = json.dumps({
        "hypothesis": "Group A scores higher than Group B",
        "statistical_test": "Welch t-test",
        "reasoning": "Two independent groups, unknown variance",
        "conclusion": True
    })
    obs, reward, done, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int = 42, max_steps: int = 3):
        self.seed_val  = seed
        self.max_steps = max_steps
        self.rng       = random.Random(seed)

        self.dataset    = None
        self.step_count = 0
        self.history    = []

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.dataset    = self.rng.choice(DATASETS)
        self.step_count = 0
        self.history    = []
        return self.state(), {}

    def step(self, action_str: str):
        self.step_count += 1
        action = self._parse_action(action_str)

        if action is None:
            obs  = self.state()
            done = self.step_count >= self.max_steps
            return obs, 0.0, done, False, {"error": "invalid_json", "step": self.step_count}

        reward = self._programmatic_reward(action)
        self.history.append({"step": self.step_count, "action": action, "reward": reward})

        done = (self.step_count >= self.max_steps) or (action.get("conclusion") is not None)
        return self.state(), reward, done, False, {"reward": reward, "step": self.step_count}

    def state(self) -> dict:
        return {
            "question": self.dataset["question"],
            "data":     self.dataset["data"],
            "step":     self.step_count,
            "history":  self.history,
        }

    def render(self, mode="human"):
        s = self.state()
        print(f"\n=== SciAgentEnv | Step {s['step']} ===")
        print(f"Question : {s['question']}")
        print(f"Data     : {s['data']}")
        if s["history"]:
            last = s["history"][-1]
            print(f"Last reward: {last['reward']:.3f}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _parse_action(self, s: str):
        try:
            cleaned = s.strip()
            for fence in ["```json", "```"]:
                cleaned = cleaned.lstrip(fence).rstrip(fence).strip()
            return json.loads(cleaned)
        except Exception:
            return None

    def _programmatic_reward(self, action: dict) -> float:
        score = 0.0

        has_hyp  = bool(action.get("hypothesis", "").strip())
        has_test = bool(action.get("statistical_test", "").strip())
        has_conc = action.get("conclusion") is not None

        score += 0.2 * has_hyp
        score += 0.2 * has_test
        score += 0.2 * has_conc

        # Appropriate test family?
        test_str = (action.get("statistical_test") or "").lower()
        appropriate = any(
            t in test_str
            for t in ["t-test", "t_test", "ttest", "welch", "student"]
        )
        score += 0.2 * appropriate

        # Statistically correct conclusion?
        if has_conc:
            d    = self.dataset["data"]
            keys = list(d.keys())
            try:
                _, pval      = stats.ttest_ind(d[keys[0]], d[keys[1]], equal_var=False)
                true_sig     = pval < 0.05
                claimed_text = str(action.get("conclusion", "")).lower()
                claimed_sig  = any(
                    w in claimed_text
                    for w in ["true", "yes", "significant", "confirmed", "supports", "1"]
                )
                if true_sig == claimed_sig:
                    score += 0.2
            except Exception:
                pass

        return min(round(score, 4), 1.0)

    def _ground_truth(self) -> dict:
        """Return the correct answer for the current dataset."""
        d    = self.dataset["data"]
        keys = list(d.keys())
        t, p = stats.ttest_ind(d[keys[0]], d[keys[1]], equal_var=False)
        return {"t_statistic": round(t, 4), "p_value": round(p, 4), "significant": p < 0.05}
