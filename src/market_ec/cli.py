from __future__ import annotations
import argparse
import sys
from typing import List

from market_ec.pipeline.build_dataset import build_dataset
from market_ec.pipeline.calibrate_models import run_calibration
from market_ec.pipeline.simulate_portfolio import run_simulation

# These modules expose a script-style main() that parses sys.argv itself.
from market_ec.pipeline import run_stress as _run_stress_mod
from market_ec.pipeline import run_backtests as _run_backtests_mod
from market_ec.dashboard import generate_dashboard as _dashboard_mod


def _call_with_argv(func, argv: List[str]) -> None:
    """
    Temporarily replace sys.argv so script-style mains (which call argparse on sys.argv)
    see only the arguments we pass (e.g., ["--config", "..."]) and not the CLI subcommand.
    """
    orig = sys.argv[:]
    try:
        sys.argv = [orig[0]] + argv
        func()
    finally:
        sys.argv = orig


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="market-ec", description="Market EC CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build", help="Build dataset (prices + returns)")
    p1.add_argument("--config", required=True)

    p2 = sub.add_parser("calibrate", help="Calibrate GBM/OU/GARCH and copula")
    p2.add_argument("--config", required=True)

    p3 = sub.add_parser("simulate", help="Simulate 1Y portfolio and compute VaR/ES")
    p3.add_argument("--config", required=True)

    p4 = sub.add_parser("stress", help="Run historical/parametric stress tests")
    p4.add_argument("--config", required=True)

    p5 = sub.add_parser("backtest", help="Run 1d VaR backtests (Kupiec/Christoffersen)")
    p5.add_argument("--config", required=True)

    p6 = sub.add_parser("dashboard", help="Generate Plotly HTML dashboard")
    p6.add_argument("--config", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "build":
        build_dataset(args.config)
    elif args.cmd == "calibrate":
        run_calibration(args.config)
    elif args.cmd == "simulate":
        run_simulation(args.config)
    elif args.cmd == "stress":
        _call_with_argv(_run_stress_mod.main, ["--config", args.config])
    elif args.cmd == "backtest":
        _call_with_argv(_run_backtests_mod.main, ["--config", args.config])
    elif args.cmd == "dashboard":
        _call_with_argv(_dashboard_mod.main, ["--config", args.config])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
