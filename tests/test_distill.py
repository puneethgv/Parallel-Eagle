"""Self-distillation produces labels equal to the target's argmax.

The whole point of self-distillation is that, over the generated (response)
region, the next-token label *is* the target's greedy choice — exactly what
decode-time acceptance measures. These tests pin that property and the
prompt-region loss masking, on the tiny CPU target (no tokenizer / download).
"""

import torch

from pe.config import DrafterConfig
from pe.distill import distill_example
from pe.drafter import ParallelDrafter
from pe.partition import _total_valid, _total_valid_after, mtp_backward


def test_response_labels_equal_target_argmax(tiny_target):
    prompt = torch.tensor([1, 2, 3, 4, 5, 6])
    out = distill_example(tiny_target, prompt, max_new_tokens=12, eos_token_id=None)
    assert out is not None
    full_ids, feats, prompt_len = out

    assert prompt_len == len(prompt)
    assert full_ids[:prompt_len].tolist() == prompt.tolist()
    assert feats.shape == (full_ids.shape[0], tiny_target.feature_dim)
    assert full_ids.shape[0] > prompt_len  # some response was generated

    # Every response token must be the target's argmax given the preceding context.
    for i in range(prompt_len, full_ids.shape[0]):
        logits = tiny_target.forward(full_ids[:i].unsqueeze(0)).logits[0, -1]
        assert int(logits.argmax()) == int(full_ids[i]), f"position {i} is not the target argmax"


def test_prompt_masking_counts_only_response_slots():
    n, k, prompt_len = 20, 4, 7
    full = _total_valid(n, k)
    masked = _total_valid_after(n, k, prompt_len)
    assert masked < full
    # No slot whose label lands in the prompt region should be counted.
    expected = sum(1 for i in range(n) for d in range(k) if prompt_len <= i + 1 + d < n)
    assert masked == expected


def test_mtp_backward_accepts_prompt_len(tiny_target):
    torch.manual_seed(0)
    drafter = ParallelDrafter.from_target(
        tiny_target, DrafterConfig(num_layers=2, max_depth=4)
    ).train()
    ids = torch.randint(0, tiny_target.vocab_size, (24,))
    feats = torch.randn(24, tiny_target.feature_dim)

    loss = mtp_backward(drafter, ids, feats, num_segments=1, prompt_len=8)
    assert loss > 0
    assert any(p.grad is not None for p in drafter.trainable_parameters())
