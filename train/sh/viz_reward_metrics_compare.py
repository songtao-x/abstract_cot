"""Overlay reward_metrics.jsonl traces from multiple GRPO runs on shared axes.

Uses record-index x-axis (some runs log from multiple ranks; logged step values
can collide across ranks and produce zig-zag lines). All series are averaged
in groups of 8 (= gradient_accumulation_steps in this project), turning each
point into ≈1 training step.

Writes a single PNG `reward_metrics_compare.png` and a small text report
summarising first/last/Δ per run per metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METRICS = [
    ("reward/base_reward_mean",     "Base Reward (task accuracy)"),
    ("reward/valid_fraction",       "Valid Fraction (parseable)"),
    ("reward/think_delta_mean",     "Think Δ (with − without)"),
    ("reward/answer_delta_mean",    "Answer Δ"),
    ("reward/final_delta_mean",     "Final Δ"),
    ("reward/attn_answer_to_plan",  "Attn answer→plan"),
    ("lengths/abstract_chars_mean", "Abstract length (chars)"),
    ("lengths/think_chars_mean",    "Think length (chars)"),
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
    xs, ys = [], []
    for i, r in enumerate(rows):
        if r.get(key) is None:
            continue
        xs.append(i)
        ys.append(r[key])
    return np.array(xs), np.array(ys, dtype=float)


def _group_avg(x: np.ndarray, y: np.ndarray, group: int):
    n = (len(y) // group) * group
    if n == 0:
        return x[:0], y[:0]
    xx = x[:n].reshape(-1, group).mean(axis=1)
    yy = y[:n].reshape(-1, group).mean(axis=1)
    return xx, yy


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) < w:
        return y.copy()
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="valid")


def _plot(runs: list[tuple[str, list[dict]]], out_path: Path, *, group: int, smooth_w: int):
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    fig.suptitle("GRPO reward-metric ablation comparison (avg over groups of 8 reward calls)",
                 fontsize=15, fontweight="bold")
    palette = plt.get_cmap("tab10")

    for ax, (key, label) in zip(axes.flat, METRICS):
        any_data = False
        for idx, (name, rows) in enumerate(runs):
            x, y = _column(rows, key)
            if len(y) == 0:
                continue
            x, y = _group_avg(x, y, group)
            color = palette(idx)
            ax.plot(x, y, color=color, alpha=0.18, lw=1)
            if len(y) >= 5:
                w = min(smooth_w, max(2, len(y) // 5))
                sm = _smooth(y, w)
                ax.plot(x[: len(sm)], sm, color=color, lw=2.0, label=name)
            any_data = True
        ax.set_title(label)
        ax.set_xlabel("avg-grouped reward-call step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        if any_data:
            ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _slug(key: str) -> str:
    return key.replace("/", "_").replace(" ", "_")


def _plot_separate(runs: list[tuple[str, list[dict]]], out_dir: Path, *, group: int, smooth_w: int) -> list[Path]:
    palette = plt.get_cmap("tab10")
    written = []
    for key, label in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        any_data = False
        for idx, (name, rows) in enumerate(runs):
            x, y = _column(rows, key)
            if len(y) == 0:
                continue
            x, y = _group_avg(x, y, group)
            color = palette(idx)
            ax.plot(x, y, color=color, alpha=0.18, lw=1)
            if len(y) >= 5:
                w = min(smooth_w, max(2, len(y) // 5))
                sm = _smooth(y, w)
                ax.plot(x[: len(sm)], sm, color=color, lw=2.0, label=name)
            any_data = True
        ax.set_title(label)
        ax.set_xlabel("avg-grouped reward-call step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        if any_data:
            ax.legend(loc="best", fontsize=9)
        p = out_dir / f"reward_metrics_compare__{_slug(key)}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    return written


def _stats_table(runs: list[tuple[str, list[dict]]]) -> str:
    lines = []
    width = max(len(n) for n, _ in runs)
    for key, label in METRICS:
        lines.append(f"\n=== {label}  [{key}] ===")
        lines.append(f"  {'run':<{width}}  {'n':>5}  {'first10%':>10}  {'last10%':>10}  {'mean':>10}  {'Δ':>10}")
        for name, rows in runs:
            _, y = _column(rows, key)
            if len(y) == 0:
                lines.append(f"  {name:<{width}}  (no data)")
                continue
            head = y[: max(1, len(y) // 10)].mean()
            tail = y[-max(1, len(y) // 10):].mean()
            lines.append(
                f"  {name:<{width}}  {len(y):>5d}  {head:>+10.4f}  {tail:>+10.4f}  {y.mean():>+10.4f}  {tail - head:>+10.4f}"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run-dir basenames to compare (e.g. grpo_abstract_norm grpo_abstract_base_only ...)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="where to write reward_metrics_compare.png/.txt (default: outputs_root)")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--smooth", type=int, default=20)
    ap.add_argument("--separate", action="store_true",
                    help="also write one PNG per metric (reward_metrics_compare__<metric>.png)")
    args = ap.parse_args()

    runs: list[tuple[str, list[dict]]] = []
    for name in args.runs:
        path = args.outputs_root / name / "reward_metrics.jsonl"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        rows = _load(path)
        # Strip the run prefix for shorter legend labels.
        short = name.replace("grpo_abstract_", "").replace("dapo_abstract_", "")
        runs.append((short, rows))
        print(f"loaded {name}: {len(rows)} records")

    if not runs:
        raise SystemExit("no runs loaded")

    out_dir = args.out_dir or args.outputs_root
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "reward_metrics_compare.png"
    txt = out_dir / "reward_metrics_compare.txt"
    _plot(runs, png, group=args.group, smooth_w=args.smooth)
    print(f"wrote {png}")
    if args.separate:
        for p in _plot_separate(runs, out_dir, group=args.group, smooth_w=args.smooth):
            print(f"wrote {p}")
    txt.write_text(_stats_table(runs) + "\n")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()
