import pandas as pd
import pytest

from execution_cost.paired_analysis import (
    paired_strategy_comparison,
)


def make_trials() -> pd.DataFrame:
    rows = []

    twap_shortfalls = [10.0, 12.0, 14.0]
    pov_shortfalls = [8.0, 11.0, 10.0]
    vwap_shortfalls = [9.0, 13.0, 13.0]

    for seed in range(3):
        rows.extend(
            [
                {
                    "regime": "normal",
                    "seed": seed,
                    "strategy": "TWAP",
                    "implementation_shortfall_bps": (
                        twap_shortfalls[seed]
                    ),
                    "completion_rate": 1.0,
                },
                {
                    "regime": "normal",
                    "seed": seed,
                    "strategy": "POV",
                    "implementation_shortfall_bps": (
                        pov_shortfalls[seed]
                    ),
                    "completion_rate": 0.9,
                },
                {
                    "regime": "normal",
                    "seed": seed,
                    "strategy": "VWAP",
                    "implementation_shortfall_bps": (
                        vwap_shortfalls[seed]
                    ),
                    "completion_rate": 1.0,
                },
            ]
        )

    return pd.DataFrame(rows)


def test_paired_comparison_calculates_improvement() -> None:
    comparison = paired_strategy_comparison(make_trials())

    pov = comparison.loc[
        comparison["comparison"] == "POV"
    ].iloc[0]

    # TWAP - POV differences are 2, 1 and 4 bps.
    assert pov["mean_shortfall_improvement_bps"] == pytest.approx(
        7 / 3
    )

    assert pov["pairs"] == 3
    assert pov["win_rate"] == pytest.approx(1.0)


def test_paired_comparison_reports_completion_tradeoff() -> None:
    comparison = paired_strategy_comparison(make_trials())

    pov = comparison.loc[
        comparison["comparison"] == "POV"
    ].iloc[0]

    assert pov["mean_completion_delta"] == pytest.approx(-0.1)


def test_paired_comparison_produces_confidence_interval() -> None:
    comparison = paired_strategy_comparison(make_trials())

    pov = comparison.loc[
        comparison["comparison"] == "POV"
    ].iloc[0]

    assert pov["ci95_lower_bps"] < (
        pov["mean_shortfall_improvement_bps"]
    )

    assert pov["ci95_upper_bps"] > (
        pov["mean_shortfall_improvement_bps"]
    )


def test_paired_comparison_rejects_missing_baseline() -> None:
    trials = make_trials()
    trials = trials[trials["strategy"] != "TWAP"]

    with pytest.raises(ValueError, match="Baseline"):
        paired_strategy_comparison(trials)