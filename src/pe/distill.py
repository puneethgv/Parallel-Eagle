"""Self-distillation feature generation.

The drafter is rewarded at decode time for predicting the target's *argmax*, but
features cached over human-written responses (``pe.features``) carry the human
token as the label — a different objective. This module closes that gap: for each
prompt it runs the target's **own** greedy generation, then caches fused features
over ``[prompt | target response]``. The next-token label in the response region
is therefore the target's argmax by construction, so per-depth cross-entropy
directly optimizes acceptance length.

Output is the same shard layout as :mod:`pe.features` (so :class:`FeatureDataset`
and :mod:`pe.train` read it unchanged), with one addition: each example records
``prompt_len`` so training can mask the prompt region — where the label is the
template/user text, not a target argmax — out of the loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import DistillConfig, TargetConfig
from .decode.baselines import vanilla_generate_cached
from .target import TargetModel, dump_target_heads


def _example_to_prompt_ids(example: dict, tokenizer, max_len: int) -> torch.Tensor | None:
    """Convert a dataset row to the *prompt only*, with a generation prompt appended.

    Mirrors :func:`pe.features._example_to_ids` but keeps only the user turn(s) and
    sets ``add_generation_prompt=True`` so the target continues as the assistant.
    """
    ids = None
    has_template = getattr(tokenizer, "chat_template", None)
    if "messages" in example and has_template:
        msgs = [m for m in example["messages"] if m.get("role") != "assistant"]
        if not msgs:
            return None
        res = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        ids = res["input_ids"] if hasattr(res, "keys") else res
    elif "instruction" in example and has_template:
        user = example["instruction"]
        if example.get("input"):
            user += "\n" + example["input"]
        msgs = [{"role": "user", "content": user}]
        res = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        ids = res["input_ids"] if hasattr(res, "keys") else res
    elif "prompt" in example:
        ids = tokenizer(example["prompt"]).input_ids
    if not ids or len(ids) < 4:
        return None
    return torch.tensor(ids[:max_len], dtype=torch.long)


@torch.no_grad()
def distill_example(
    target: TargetModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Generate the target's greedy continuation of ``prompt_ids`` and featurize the
    full ``[prompt | response]`` sequence.

    Returns ``(full_ids, features_fp16_cpu, prompt_len)`` or ``None`` if the target
    produced no new tokens (e.g. immediate EOS), which carries no training signal.
    """
    prompt = [int(t) for t in prompt_ids.tolist()]
    res = vanilla_generate_cached(target, prompt, max_new_tokens, eos_token_id=eos_token_id)
    if not res.output_ids:
        return None
    full = prompt + res.output_ids
    ids_dev = torch.tensor(full, device=target.device).unsqueeze(0)
    feats = target.forward(ids_dev).fused[0].to(torch.float16).cpu()
    return torch.tensor(full, dtype=torch.long), feats, len(prompt)


def build_distill_dataset(dcfg: DistillConfig, tcfg: TargetConfig) -> Path:
    from datasets import load_dataset

    target = TargetModel(tcfg)
    tokenizer = target.tokenizer
    if tokenizer is None:
        raise RuntimeError("A tokenizer is required for self-distillation.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    ds = load_dataset(dcfg.dataset, split=dcfg.split, streaming=True)
    out_dir = Path(dcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_target_heads(target.model, out_dir / "heads.pt")

    shards: list[str] = []
    buf: list[dict] = []
    n_done = 0

    def flush():
        nonlocal buf
        if not buf:
            return
        name = f"shard_{len(shards):05d}.pt"
        torch.save(buf, out_dir / name)
        shards.append(name)
        buf = []

    for example in ds:
        if n_done >= dcfg.max_examples:
            break
        prompt_ids = _example_to_prompt_ids(example, tokenizer, dcfg.max_prompt_len)
        if prompt_ids is None:
            continue
        out = distill_example(target, prompt_ids, dcfg.max_new_tokens, eos_token_id)
        if out is None:
            continue
        full_ids, feats, prompt_len = out
        buf.append({"input_ids": full_ids, "features": feats, "prompt_len": prompt_len})
        n_done += 1
        if len(buf) >= dcfg.shard_size:
            flush()
        if n_done % 50 == 0:
            print(f"distilled {n_done}/{dcfg.max_examples}")
    flush()

    manifest = {
        "shards": shards,
        "num_examples": n_done,
        "feature_dim": target.feature_dim,
        "feature_layers": list(target.feature_layers),
        "target": tcfg.model_name,
        "self_distilled": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {n_done} self-distilled examples across {len(shards)} shards to {out_dir}")
    return out_dir / "manifest.json"


def _parse_args() -> tuple[DistillConfig, TargetConfig]:
    d, t = DistillConfig(), TargetConfig()
    p = argparse.ArgumentParser(description="Generate self-distilled drafter training data.")
    p.add_argument("--target", default=t.model_name)
    p.add_argument("--dataset", default=d.dataset)
    p.add_argument("--split", default=d.split)
    p.add_argument("--max-examples", type=int, default=d.max_examples)
    p.add_argument("--max-prompt-len", type=int, default=d.max_prompt_len)
    p.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--shard-size", type=int, default=d.shard_size)
    p.add_argument("--out-dir", default=str(d.out_dir))
    p.add_argument("--device", default=t.device)
    a = p.parse_args()
    dcfg = DistillConfig(
        out_dir=Path(a.out_dir),
        dataset=a.dataset,
        split=a.split,
        max_examples=a.max_examples,
        max_prompt_len=a.max_prompt_len,
        max_new_tokens=a.max_new_tokens,
        shard_size=a.shard_size,
    )
    tcfg = TargetConfig(model_name=a.target, device=a.device)
    return dcfg, tcfg


if __name__ == "__main__":
    dcfg, tcfg = _parse_args()
    build_distill_dataset(dcfg, tcfg)
