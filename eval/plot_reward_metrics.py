"""Plot reward metrics from a JSONL log produced by GRPO training.

Usage:
    python plot_reward_metrics.py path/to/reward_metrics.jsonl
    python plot_reward_metrics.py path/to/reward_metrics.jsonl --window 8

Produces one PNG next to the input file:
    <stem>_metrics.png   one subplot per metric, with raw per-step (faded)
                         and rolling-mean (window W, default 8) overlaid.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_metrics(rows, step_key="step"):
    """Return (steps, {metric_name: np.array of values aligned with steps}).

    Missing values for a step become NaN so the rolling mean can ignore them.
    """
    steps = np.array([r[step_key] for r in rows], dtype=float)
    keys = set()
    for r in rows:
        for k, v in r.items():
            if k == step_key:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    series = {}
    for k in sorted(keys):
        vals = np.full(len(rows), np.nan, dtype=float)
        for i, r in enumerate(rows):
            v = r.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals[i] = v
        series[k] = vals
    return steps, series


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware trailing rolling mean of length len(values)."""
    if window <= 1:
        return values.copy()
    out = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        mask = ~np.isnan(chunk)
        if mask.any():
            out[i] = chunk[mask].mean()
    return out


def plot_grid(steps, series, window, title, out_path: Path):
    names = list(series.keys())
    n = len(names)
    if n == 0:
        print(f"[skip] no metrics to plot for {out_path}")
        return
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.0 * rows), squeeze=False)
    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        vals = series[name]
        smoothed = rolling_mean(vals, window)
        m_raw = ~np.isnan(vals)
        m_smooth = ~np.isnan(smoothed)
        ax.plot(steps[m_raw], vals[m_raw],
                linewidth=0.8, alpha=0.35, color="tab:blue", label="raw")
        ax.plot(steps[m_smooth], smoothed[m_smooth],
                linewidth=1.6, color="tab:red", label=f"avg{window}")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="reward_metrics.jsonl file")
    p.add_argument("--window", type=int, default=8, help="rolling-mean window (default 8)")
    p.add_argument("--metrics", nargs="*", default=None, help="optional subset of metric keys to plot")
    args = p.parse_args()

    rows = load_jsonl(args.path)
    if not rows:
        raise SystemExit(f"no rows in {args.path}")

    steps, series = collect_metrics(rows)
    if args.metrics:
        missing = [m for m in args.metrics if m not in series]
        if missing:
            print(f"[warn] requested metrics not found: {missing}")
        series = {k: v for k, v in series.items() if k in set(args.metrics)}

    out_dir = args.path.parent
    stem = args.path.stem
    out_path = out_dir / f"{stem}_metrics.png"
    plot_grid(steps, series, args.window,
              f"{stem} (raw + rolling mean window={args.window})", out_path)


if __name__ == "__main__":
    main()
