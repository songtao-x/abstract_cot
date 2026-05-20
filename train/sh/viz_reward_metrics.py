"""Visualize a reward_metrics.jsonl trace produced by grpo.py.

Produces two PNGs alongside the source jsonl:
  - reward_metrics_viz.png       (raw per-reward-call points + smoothed line)
  - reward_metrics_viz_avg8.png  (averaged in groups of 8 ≈ per training step)

Usage:
  python viz_reward_metrics.py <path/to/reward_metrics.jsonl> [--group N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METRICS = [
    ("reward/base_reward_mean",        "Base Reward",                 "C0"),
    ("reward/valid_fraction",          "Valid Fraction",              "C1"),
    ("reward/think_delta_mean",        "Think Δ (with − without)",    "C3"),
    ("reward/answer_delta_mean",       "Answer Δ",                    "C4"),
    ("reward/think_entropy_mean",      "Think entropy (nats/tok)",    "C8"),
    ("reward/answer_entropy_mean",     "Answer entropy (nats/tok)",   "C9"),
    ("reward/final_delta_mean",        "Final Δ (think − attn·answer)","C2"),
    ("reward/attn_answer_to_plan",     "Attn answer→plan",            "C5"),
    ("lengths/abstract_chars_mean",    "Abstract length (chars)",     "C6"),
    ("lengths/think_chars_mean",       "Think length (chars)",        "C7"),
]


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _column(rows: list[dict], key: str):
    """Return (record_index, value) pairs, dropping None.

    We use the record index (not the logged step) on the x-axis because some
    runs log from multiple ranks, producing duplicate `step` values that make
    line plots zig-zag. Index is monotonic by file order.
    """
    xs, ys = [], []
    for i, r in enumerate(rows):
        if r.get(key) is None:
            continue
        xs.append(i)
        ys.append(r[key])
    return np.array(xs), np.array(ys, dtype=float)


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) < w:
        return y.copy()
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="valid")


def _group_avg(x: np.ndarray, y: np.ndarray, group: int):
    n = (len(y) // group) * group
    if n == 0:
        return x[:0], y[:0]
    xx = x[:n].reshape(-1, group).mean(axis=1)
    yy = y[:n].reshape(-1, group).mean(axis=1)
    return xx, yy


def _plot(rows: list[dict], out_path: Path, title: str, *, group: int = 1, smooth_w: int = 20):
    fig, axes = plt.subplots(5, 2, figsize=(14, 17), constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for ax, (key, label, color) in zip(axes.flat, METRICS):
        x, y = _column(rows, key)
        if len(y) == 0:
            ax.set_title(f"{label} (no data)")
            ax.set_axis_off()
            continue
        if group > 1:
            x, y = _group_avg(x, y, group)
        ax.scatter(x, y, s=6, alpha=0.25, color=color, label="raw")
        if len(y) >= 5:
            sm = _smooth(y, min(smooth_w, max(2, len(y) // 5)))
            ax.plot(x[: len(sm)], sm, color=color, lw=2, label=f"smoothed")
        ax.set_title(f"{label}")
        ax.set_xlabel("reward-call step" if group == 1 else f"averaged step (group={group})")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _stats(rows: list[dict]) -> str:
    lines = []
    n = len(rows)
    keys = [k for k, _, _ in METRICS]
    lines.append(f"records: {n}")
    if n == 0:
        return "\n".join(lines)
    lines.append(f"step range: {rows[0].get('step')} .. {rows[-1].get('step')}")
    valid_frac = [r.get("reward/valid_fraction", 0.0) for r in rows]
    lines.append(f"mean valid_fraction: {np.mean(valid_frac):.3f}")
    for k in keys:
        _, y = _column(rows, k)
        if len(y) == 0:
            continue
        first = y[: max(1, len(y) // 10)].mean()
        last = y[-max(1, len(y) // 10):].mean()
        lines.append(
            f"  {k:38s} n={len(y):4d} mean={y.mean():+.4f} first10%={first:+.4f} last10%={last:+.4f} Δ={last - first:+.4f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--group", type=int, default=8, help="averaging group size (default 8 = grad_accum_steps)")
    args = ap.parse_args()

    rows = _load(args.jsonl)
    out_dir = args.jsonl.parent
    name = args.jsonl.stem.replace("reward_metrics", "").strip("_") or "run"

    title_root = f"{out_dir.name}: reward metrics"
    _plot(rows, out_dir / "reward_metrics_viz.png", title_root, group=1)
    _plot(
        rows,
        out_dir / f"reward_metrics_viz_avg{args.group}.png",
        f"{title_root} (averaged groups of {args.group})",
        group=args.group,
    )

    report = _stats(rows)
    print(report)
    (out_dir / "reward_metrics_report.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
