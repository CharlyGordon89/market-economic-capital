.PHONY: setup run-all test lint type itest mlflow docker-build docker-run

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

run-all:
\tmarket-ec build --config configs/base.yaml && \\
\tmarket-ec calibrate --config configs/base.yaml && \\
\tmarket-ec simulate --config configs/base.yaml && \\
\tmarket-ec stress --config configs/base.yaml && \\
\tmarket-ec backtest --config configs/base.yaml && \\
\tmarket-ec dashboard --config configs/base.yaml

test:
\tpytest -q

lint:
\truff check .

type:
\tmypy src

itest:
\tALLOW_NET_TESTS=1 pytest -q -m integration

mlflow:
\tmlflow ui

docker-build:
\tdocker build -t market-ec:latest .

docker-run:
\tdocker run --rm -it -v $$(pwd):/app market-ec:latest market-ec run --config configs/base.yaml
