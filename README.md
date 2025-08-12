# Market Economic Capital – Market Risk (Sprint 1)

This project estimates market-risk Economic Capital for a multi-asset portfolio. **Sprint 1** sets up the repo, data pipeline, validation, returns, MLflow tracking, and CI.

## Quick start
```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# build dataset
python -m market_ec.pipeline.build_dataset --config configs/base.yaml