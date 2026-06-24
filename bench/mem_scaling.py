"""Measure peak training memory vs. number of sequence-partitioning segments.

Trains one long synthetic example through the drafter at several segment counts
and records peak CUDA memory. More segments instantiate fewer prediction slots
per forward, so peak attention/logits memory drops — this is what lets a long
context fit a small GPU. Writes ``results/mem_scaling.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from pe.config import DrafterConfig, TargetConfig
from pe.drafter import ParallelDrafter
from pe.partition import mtp_backward
from pe.target import load_target_heads


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required for memory measurement")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tcfg = TargetConfig(model_name=args.target, device=args.device, dtype=args.dtype)
    heads = load_target_heads(tcfg)
    dcfg = DrafterConfig(num_layers=args.num_layers, max_depth=args.max_depth)
    drafter = ParallelDrafter.from_target(heads, dcfg).to(args.device, dtype)
    drafter.train()

    torch.manual_seed(0)
    ids = torch.randint(0, heads.vocab_size, (args.seq_len,))
    feats = torch.randn(args.seq_len, heads.feature_dim, dtype=torch.float16)

    rows = []
    for s in args.segments:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        drafter.zero_grad(set_to_none=True)
        try:
            loss = mtp_backward(drafter, ids, feats, num_segments=s)
            peak = torch.cuda.max_memory_allocated() / 2**20
            rows.append({"segments": s, "peak_mib": round(peak, 1), "loss": round(loss, 4)})
            print(f"segments={s:<3} peak={peak:8.1f} MiB  loss={loss:.4f}")
        except torch.OutOfMemoryError:
            rows.append({"segments": s, "peak_mib": "OOM", "loss": "-"})
            print(f"segments={s:<3} OOM")

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "mem_scaling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["segments", "peak_mib", "loss"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'mem_scaling.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Peak memory vs sequence-partitioning segments.")
    p.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--segments", type=int, nargs="+", default=[1, 2, 3, 6])
    p.add_argument("--results-dir", default="results")
    run(p.parse_args())
