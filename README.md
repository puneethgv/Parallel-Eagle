# parallel_eagle

A from-scratch **speculative decoder** whose draft model proposes many future tokens in a
**single forward pass** and verifies a **branching tree** of candidates against the target in
one pass — accelerating autoregressive generation while producing output that is provably
identical to plain decoding.

The project is built to run on a single 8 GB consumer GPU. It implements the full pipeline:
a frozen-target feature extractor, a feature-conditioned parallel drafter, a memory-scalable
training recipe, lossless verification, and a benchmark harness that measures real
wall-clock throughput against several baselines.

## The idea in one paragraph

Autoregressive decoding emits one token per target forward pass and is memory-bandwidth
bound. *Speculative decoding* fixes this by having a cheap draft model propose several next
tokens that the target verifies in a single pass; correct tokens are accepted "for free."
This drafter is **feature-conditioned** — it consumes the target's own intermediate hidden
states rather than raw tokens — so a tiny network can draft accurately. Crucially, it
predicts all `K` tokens in **one parallel pass** instead of `K` sequential passes: the
unknown future positions are filled with a single learnable *shared hidden state* and a
learnable *mask-token embedding*, and positional structure is left to rotary attention. On
top of that, drafting produces a **dynamic tree** of candidates (not a single chain), so one
early mistake no longer throws away the whole draft.

## Why it's interesting

- **Parallel drafting** removes the per-token latency of sequential draft generation.
- **Tree drafting + tree-attention verification** lifts the accepted-tokens-per-step well
  above a single chain, at the cost of one bounded extra-wide target pass.
- **Memory-scalable training** (amortized attention-mask construction + within-sequence
  gradient accumulation) makes long-context drafter training fit in a small memory budget.
- Everything is **lossless**: greedy outputs are token-identical to plain decoding, and
  sampled outputs match the target distribution.

## Architecture

```
            target (frozen)                         drafter (trained)
   prompt ──► N decoder layers ──► fused hidden ──► concat(token emb, proj(features))
                  │  (early/mid/late)                          │
                  └─ LM head (tied, frozen) ◄── decoder stack ◄─┘
                                                  ▲
            depths 2..K filled with a shared hidden state + mask-token embedding
                                                  │
                              one pass ──► logits at K depths ──► draft tree
                                                  │
                target verifies the whole tree in one pass ──► accept longest valid path
```

## Repository layout

```
src/pe/
  config.py     # configuration dataclasses
  target.py     # frozen target: fused hidden states + masked verification forward
  features.py   # offline feature extraction to disk shards
  drafter.py    # parallel multi-token drafter
  masks.py      # amortized training mask + tree attention mask
  partition.py  # sequence partitioning for within-sequence gradient accumulation
  train.py      # drafter training loop
  decode/
    verify.py     # lossless acceptance (greedy + speculative sampling)
    baselines.py  # vanilla AR; independent-draft SD; sequential feature-chain drafter
    chain.py      # parallel chain drafting
    tree.py       # parallel dynamic tree drafting + tree verification
  serve.py      # generation entrypoint
bench/          # benchmark sweep + plotting
tests/          # CPU correctness tests (losslessness, mask equality)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"     # add CUDA torch per your platform
```

## Quickstart

```bash
make features   # cache the frozen target's hidden states over the training data
make train      # train the parallel drafter on the cached features
make bench      # measure acceptance length + tokens/sec across strategies
make test       # CPU correctness tests
```

## Results

Benchmarked on a single 8 GB GPU. (Populated by `bench/run_bench.py` → `results/`.)

| Strategy | Accept. length | Tokens/sec | Speedup vs vanilla | Lossless |
|---|---|---|---|---|
| Vanilla autoregressive | 1.00 | _tbd_ | 1.00× | — |
| Independent-draft SD | _tbd_ | _tbd_ | _tbd_ | ✓ |
| Sequential feature-chain | _tbd_ | _tbd_ | _tbd_ | ✓ |
| Parallel chain | _tbd_ | _tbd_ | _tbd_ | ✓ |
| **Parallel tree** | _tbd_ | _tbd_ | _tbd_ | ✓ |

## License

MIT — see [LICENSE](LICENSE).
