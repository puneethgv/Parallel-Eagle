"""Feature-conditioned parallel-drafting speculative decoder with tree verification.

The package provides:
- a frozen target-model wrapper that exposes fused multi-layer hidden states
  (:mod:`pe.target`);
- an offline feature-extraction pipeline (:mod:`pe.features`);
- a parallel multi-token drafter that proposes K tokens in a single forward pass
  (:mod:`pe.drafter`);
- a memory-scalable training recipe (:mod:`pe.masks`, :mod:`pe.partition`,
  :mod:`pe.train`);
- lossless decode strategies, including dynamic tree drafting (:mod:`pe.decode`).
"""

__version__ = "0.1.0"
