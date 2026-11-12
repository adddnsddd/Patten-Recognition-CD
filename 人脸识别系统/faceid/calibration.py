"""Validation-only threshold scan; detection failures remain in denominators."""
import math

import numpy as np


def choose_threshold(known, unknown, max_far=0.05):
    """Rows: (true_identity, candidate_or_None, distance_or_None)."""
    if not known or not unknown:
        raise ValueError("阈值校准需要已知验证和未知验证两组真实样本。")
    if not 0 <= max_far <= 1:
        raise ValueError("误接受率上限必须在 0 到 1 之间。")
    distances = [float(r[2]) for r in known + unknown
                 if r[1] is not None and r[2] is not None and math.isfinite(float(r[2]))]
    if not distances:
        raise ValueError("验证集未检出任何有效人脸，无法校准。")
    candidates = sorted(set([0.0] + distances))
    curve = []
    for threshold in candidates:
        correct = sum(c == truth and d is not None and d <= threshold for truth, c, d in known)
        accepted = sum(c is not None and d is not None and d <= threshold for _, c, d in unknown)
        curve.append(dict(threshold=threshold, correct=correct, known_total=len(known),
                          far_count=accepted, unknown_total=len(unknown),
                          known_accuracy=correct / len(known), far=accepted / len(unknown)))
    feasible = [r for r in curve if r["far"] <= max_far]
    if feasible:
        best = min(feasible, key=lambda r: (-r["known_accuracy"], r["threshold"]))
    else:
        best = min(curve, key=lambda r: (r["far"], -r["known_accuracy"], r["threshold"]))
    return dict(best, feasible=bool(feasible), curve=curve)
