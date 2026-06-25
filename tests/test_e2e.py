"""End-to-end smoke test: the whole pipeline on the toy model, CPU, in seconds.

Exercises target features -> drafter -> training step -> KV-cache decode, and
asserts the trained drafter still decodes losslessly. Catches integration breaks
that unit tests miss.
"""

import torch

from pe.config import DrafterConfig
from pe.decode.baselines import vanilla_generate
from pe.drafter import ParallelDrafter
from pe.partition import mtp_backward
from pe.serve import generate_speculative_cached


def test_train_then_decode_is_lossless(tiny_target):
    torch.manual_seed(0)
    drafter = ParallelDrafter.from_target(tiny_target, DrafterConfig(num_layers=2, max_depth=4))
    drafter.train()
    opt = torch.optim.AdamW(drafter.trainable_parameters(), lr=1e-3)

    # A few real training steps on features from the (tiny) target.
    for _ in range(5):
        ids = torch.randint(0, tiny_target.vocab_size, (24,))
        feats = tiny_target.forward(ids.unsqueeze(0)).fused[0]
        opt.zero_grad(set_to_none=True)
        loss = mtp_backward(drafter, ids, feats)
        opt.step()
        assert loss > 0

    drafter.eval()
    prompt = [1, 2, 3, 4, 5, 6]
    ref = vanilla_generate(tiny_target, prompt, 16).output_ids
    for mode in ("chain", "tree"):
        res = generate_speculative_cached(
            tiny_target, drafter, prompt, k=4, mode=mode, max_new_tokens=16,
            tree_top_k=3, tree_max_nodes=12,
        )
        assert res.output_ids == ref, mode
        assert res.num_generated == 16
