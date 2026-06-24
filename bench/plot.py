"""Plot benchmark results: acceptance length and speedup by method and K."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(csv_path: Path):
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def grouped_bar(ax, rows, value_key, title, ylabel):
    methods = ["sequential", "chain", "tree"]
    ks = sorted({r["k"] for r in rows if r["method"] in methods}, key=lambda x: int(x))
    by = defaultdict(dict)
    for r in rows:
        if r["method"] in methods:
            by[r["method"]][r["k"]] = float(r[value_key])

    width = 0.25
    x = range(len(ks))
    for i, m in enumerate(methods):
        vals = [by[m].get(k, 0.0) for k in ks]
        ax.bar([xi + i * width for xi in x], vals, width, label=m)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()


def main(args):
    rows = load(Path(args.csv))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grouped_bar(axes[0], rows, "acceptance_length", "Acceptance length", "tokens / step")
    grouped_bar(axes[1], rows, "speedup_vs_vanilla", "Throughput speedup vs vanilla", "x")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot benchmark CSV.")
    p.add_argument("--csv", default="results/benchmark.csv")
    p.add_argument("--out", default="results/benchmark.png")
    main(p.parse_args())
