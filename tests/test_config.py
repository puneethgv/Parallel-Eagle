"""Smoke tests: the package imports and configs carry sane defaults.

These run on CPU with no model download, so CI stays fast.
"""

import pe
from pe.config import DecodeConfig, DrafterConfig, TargetConfig, TrainConfig


def test_package_imports():
    assert pe.__version__


def test_target_feature_layers_distinct():
    cfg = TargetConfig()
    assert len(set(cfg.feature_layers)) == len(cfg.feature_layers)


def test_drafter_dims_are_deferred_to_target():
    # Dimensions that must match the target are unset until build time.
    cfg = DrafterConfig()
    assert cfg.hidden_size is None
    assert cfg.vocab_size is None
    assert cfg.num_layers >= 1
    assert cfg.max_depth >= 1


def test_decode_defaults():
    cfg = DecodeConfig()
    assert cfg.tree_top_k >= 1
    assert cfg.tree_max_nodes >= cfg.k_infer
    assert 0.0 <= cfg.temperature


def test_train_segments_default_disabled():
    assert TrainConfig().num_segments == 1
