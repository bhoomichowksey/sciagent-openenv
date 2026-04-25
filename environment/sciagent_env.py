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
