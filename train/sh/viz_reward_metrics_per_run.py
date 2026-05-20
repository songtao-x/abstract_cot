"""Per-run overview: one PNG per run, a grid subplot of *every* numeric metric
found in that run's reward_metrics.jsonl.

Same x-axis convention as viz_reward_metrics_compare.py: record-index, averaged
in groups of 8 (= gradient_accumulation_steps), faint raw + bold smoothed line.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Friendly titles for known keys; unknown keys fall back to the raw key.
LABELS = {
    "reward/base_reward_mean":      "Base Reward (task accuracy)",
    "reward/valid_fraction":        "Valid Fraction (parseable)",
    "reward/think_with_mean":       "Think: reward WITH plan",
    "reward/think_without_mean":    "Think: reward WITHOUT plan",
    "reward/answer_with_mean":      "Answer: reward WITH plan",
    "reward/answer_without_mean":   "Answer: reward WITHOUT plan",
    "reward/think_delta_mean":      "Think Δ (with − without)",
    "reward/think_delta_reg_mean":  "Think Δ (regularised)",
    "reward/answer_delta_mean":     "Answer Δ",
    "reward/final_delta_mean":      "Final Δ",
    "reward/think_entropy_mean":    "Think entropy",
    "reward/answer_entropy_mean":   "Answer entropy",
    "reward/attn_answer_to_plan":   "Attn answer→plan",
    "lengths/abstract_chars_mean":  "Abstract length (chars)",
    "lengths/think_chars_mean":     "Think length (chars)",
}


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _metric_keys(rows: list[dict]) -> list[str]:
    """All numeric keys (excluding step/bools), in first-seen order."""
    order: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k, v in r.items():
            if k in seen or k == "step":
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            seen.add(k)
            order.append(k)
    return order


def _column(rows: list[dict], key: str):
    xs, ys = [], []
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None:
            continue
        xs.append(i)
        ys.append(v)
    return np.array(xs), np.array(ys, dtype=float)


def _group_avg(x: np.ndarray, y: np.ndarray, group: int):
    n = (len(y) // group) * group
    if n == 0:
        return x[:0], y[:0]
    return x[:n].reshape(-1, group).mean(axis=1), y[:n].reshape(-1, group).mean(axis=1)


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) < w:
        return y.copy()
    return np.convolve(y, np.ones(w) / w, mode="valid")


def _plot_run(name: str, rows: list[dict], out_path: Path, *, group: int, smooth_w: int):
    keys = _metric_keys(rows)
    ncols = 3
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows),
                             constrained_layout=True, squeeze=False)
    fig.suptitle(f"{name} — all reward metrics  (avg over groups of {group})",
                 fontsize=15, fontweight="bold")

    for ax, key in zip(axes.flat, keys):
        x, y = _column(rows, key)
        if len(y):
            x, y = _group_avg(x, y, group)
            ax.plot(x, y, color="tab:blue", alpha=0.20, lw=1)
            if len(y) >= 5:
                w = min(smooth_w, max(2, len(y) // 5))
                sm = _smooth(y, w)
                ax.plot(x[: len(sm)], sm, color="tab:blue", lw=2.0)
        ax.set_title(LABELS.get(key, key), fontsize=10)
        ax.set_xlabel("avg-grouped reward-call step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    ap.add_argument("--runs", nargs="+", required=True, help="run-dir basenames")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--smooth", type=int, default=20)
    args = ap.parse_args()

    out_dir = args.out_dir or args.outputs_root
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.runs:
        path = args.outputs_root / name / "reward_metrics.jsonl"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        rows = _load(path)
        out = out_dir / f"{name}__all_metrics.png"
        _plot_run(name, rows, out, group=args.group, smooth_w=args.smooth)
        print(f"loaded {name}: {len(rows)} records -> wrote {out}")


if __name__ == "__main__":
    main()
