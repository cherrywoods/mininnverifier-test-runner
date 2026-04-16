# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Scoring functions for test results.

Each scoring function maps a test result to a score in [0, max_points],
with optional bonus points for exceeding expectations.
Signature: (max_points, result, **params) -> float

test.json format:
    "points": 10,
    "scoring": {"function": "binary"}           # default for eval/grad
    "scoring": {"function": "exponential", "k": 5}  # for train
    "scoring": {"function": "proportional"}      # for fuzz

Bonus points:
    "points": 10,
    "bonus_points": 5,
    "scoring": {"function": "exponential", "k": 5, "bonus_threshold": 0.9}

    Base points are awarded up to the bonus_threshold, bonus points beyond it.
    A bonus-only test uses "points": 0 (or omits it) with "bonus_points": 5.
"""

import math


def binary(max_points, result, bonus_points=0, bonus_threshold=None):
    """Full points if passed, else 0. Bonus awarded in full if passed."""
    passed = result.get("passed")
    score = max_points if passed else 0.0
    bonus = float(bonus_points) if passed else 0.0
    return score, bonus


def _exp_curve(x, k):
    """Exponential curve mapping [0, 1] -> [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return (math.exp(k * x) - 1) / (math.exp(k) - 1)


def exponential(max_points, result, k=5, bonus_points=0, bonus_threshold=None):
    """Exponential scoring based on best_test_accuracy.

    Maps accuracy in [0, 1] to [0, max_points] using:
        max_points * (exp(k * accuracy) - 1) / (exp(k) - 1)

    Higher k rewards improvements near 100% more heavily.

    When bonus_points and bonus_threshold are set, base points are awarded
    for accuracy in [0, bonus_threshold] and bonus points for accuracy in
    (bonus_threshold, 1.0].
    """
    accuracy = result.get("best_test_accuracy", 0.0)

    if bonus_points > 0 and bonus_threshold is not None and bonus_threshold < 1.0:
        # Base score: accuracy up to bonus_threshold maps to [0, max_points]
        base_frac = min(accuracy / bonus_threshold, 1.0) if bonus_threshold > 0 else 1.0
        score = max_points * _exp_curve(base_frac, k)
        # Bonus: accuracy beyond bonus_threshold maps to [0, bonus_points]
        if accuracy > bonus_threshold:
            bonus_frac = (accuracy - bonus_threshold) / (1.0 - bonus_threshold)
            bonus = bonus_points * _exp_curve(bonus_frac, k)
        else:
            bonus = 0.0
    else:
        score = max_points * _exp_curve(accuracy, k)
        bonus = 0.0

    return score, bonus


def proportional(max_points, result, bonus_points=0, bonus_threshold=None):
    """Score proportional to the fraction of passed trials.

    When bonus_points and bonus_threshold are set, base points are awarded
    for pass rates up to bonus_threshold, bonus points for rates beyond it.
    """
    n_trials = result.get("n_trials", 0)
    if n_trials == 0:
        return 0.0, 0.0
    n_passed = result.get("n_passed", 0)
    frac = n_passed / n_trials

    if bonus_points > 0 and bonus_threshold is not None and bonus_threshold < 1.0:
        base_frac = min(frac / bonus_threshold, 1.0) if bonus_threshold > 0 else 1.0
        score = max_points * base_frac
        if frac > bonus_threshold:
            bonus_frac = (frac - bonus_threshold) / (1.0 - bonus_threshold)
            bonus = bonus_points * bonus_frac
        else:
            bonus = 0.0
    else:
        score = max_points * frac
        bonus = 0.0

    return score, bonus


SCORING_FUNCTIONS = {
    "binary": binary,
    "exponential": exponential,
    "proportional": proportional,
}

DEFAULT_SCORING = {
    "eval": "binary",
    "grad": "binary",
    "train": "exponential",
    "fuzz_eval": "proportional",
    "fuzz_grad": "proportional",
    "bench_eval": "binary",
    "bench_grad": "binary",
}


def compute_score(config, result):
    """Compute the score for a test result based on its config.

    Returns a dict with keys ``score``, ``max_points``, ``bonus``,
    ``max_bonus``, or *None* if the test has neither points nor bonus_points.
    """
    max_points = config.get("points", 0)
    max_bonus = config.get("bonus_points", 0)
    if not max_points and not max_bonus:
        return None

    scoring_spec = config.get("scoring", {})
    if isinstance(scoring_spec, str):
        func_name = scoring_spec
        params = {}
    else:
        func_name = scoring_spec.get("function", DEFAULT_SCORING.get(config.get("command"), "binary"))
        params = {k: v for k, v in scoring_spec.items() if k != "function"}

    if func_name not in SCORING_FUNCTIONS:
        raise ValueError(
            f"unknown scoring function {func_name!r}, "
            f"expected one of {sorted(SCORING_FUNCTIONS)}"
        )

    score, bonus = SCORING_FUNCTIONS[func_name](
        max_points, result, bonus_points=max_bonus, **params
    )
    return {"score": score, "max_points": max_points, "bonus": bonus, "max_bonus": max_bonus}
