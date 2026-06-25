"""End-to-end quickstart on a toy model — runs in seconds on CPU, no downloads.

Builds a tiny random target, trains the parallel drafter for a few steps on
features extracted from it, then decodes with the KV-cache speculative loop and
checks the output is identical to plain greedy decoding. This verifies the whole
pipeline (features -> train -> tree drafting -> lossless verification) without a
GPU or any model download. For real acceptance/speedup numbers use a trained
drafter on a real target (see the README's 7B int4 path).
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from pe.config import DrafterConfig, TargetConfig
from pe.decode.baselines import vanilla_generate
from pe.drafter import ParallelDrafter
from pe.partition import mtp_backward
from pe.serve import generate_speculative_cached
from pe.target import TargetModel


def main() -> None:
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, vocab_size=128, max_position_embeddings=256,
    )
    target = TargetModel(TargetConfig(model_name="toy", device="cpu", dtype="float32"),
                         model=LlamaForCausalLM(config).eval(), tokenizer=None)
    drafter = ParallelDrafter.from_target(target, DrafterConfig(num_layers=2, max_depth=5))

    print("training the drafter (a few steps on toy features)...")
    drafter.train()
    opt = torch.optim.AdamW(drafter.trainable_parameters(), lr=1e-3)
    for _ in range(30):
        ids = torch.randint(0, target.vocab_size, (32,))
        feats = target.forward(ids.unsqueeze(0)).fused[0]
        opt.zero_grad(set_to_none=True)
        loss = mtp_backward(drafter, ids, feats)
        opt.step()
    drafter.eval()
    print(f"  final toy loss: {loss:.3f}")

    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    ref = vanilla_generate(target, prompt, 32).output_ids
    res = generate_speculative_cached(target, drafter, prompt, k=5, mode="tree",
                                      max_new_tokens=32, tree_top_k=3, tree_max_nodes=15)
    print(f"  decoded {res.num_generated} tokens | acceptance length {res.acceptance_length:.2f} "
          f"| target calls/token {res.target_calls_per_token:.2f}")
    print(f"  lossless vs vanilla greedy: {res.output_ids == ref}")
    assert res.output_ids == ref, "quickstart losslessness check failed"
    print("pipeline OK.")


if __name__ == "__main__":
    main()
