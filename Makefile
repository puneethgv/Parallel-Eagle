.PHONY: install quickstart features distill train bench test lint format

install:
	pip install -e ".[train,dev]"

# Full pipeline on a toy model in seconds, CPU, no downloads (smoke check).
quickstart:
	python scripts/quickstart.py

# Offline: run the frozen target over training data, cache fused hidden states.
features:
	python -m pe.features

# Self-distillation: cache features over the target's OWN greedy generations, so
# the training label equals the target's argmax (directly optimizes acceptance).
distill:
	python -m pe.distill

# Train the parallel drafter on cached features.
train:
	python -m pe.train

# Benchmark all decode strategies; writes CSV + plots.
bench:
	python bench/run_bench.py

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .
