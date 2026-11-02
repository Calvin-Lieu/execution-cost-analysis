import pandas as pd

from execution_cost.market import (
    MarketGeneratorConfig,
    MarketRegime,
)
from execution_cost.monte_carlo import (
    MonteCarloConfig,
    aggregate_monte_carlo,
    run_monte_carlo,
)
from execution_cost.simulator import ExecutionConfig


def make_trials() -> pd.DataFrame:
    market_config = MarketGeneratorConfig(
        start_time=pd.Timestamp(
            "2026-01-01 10:00:00",
            tz="UTC",
        ),
        intervals=6,
        interval_minutes=5,
        total_market_volume=100_000,
        base_depth=10_000,
        seed=100,
    )

    simulation_config = MonteCarloConfig(
        runs_per_regime=2,
        order_quantity=1_000,
        pov_participation_rate=0.10,
        regimes=(
            MarketRegime.NORMAL,
            MarketRegime.HIGH_VOLATILITY,
        ),
    )

    return run_monte_carlo(
        market_config=market_config,
        simulation_config=simulation_config,
        execution_config=ExecutionConfig(
            max_participation_rate=0.20,
            impact_coefficient=0.50,
            fee_per_unit=0.001,
        ),
    )


def test_monte_carlo_produces_one_row_per_trial_strategy() -> None:
    trials = make_trials()

    # 2 regimes × 2 seeds × 3 strategies
    assert len(trials) == 12
    assert set(trials["strategy"]) == {"TWAP", "VWAP", "POV"}
    assert set(trials["regime"]) == {
        "normal",
        "high_volatility",
    }


def test_monte_carlo_is_reproducible() -> None:
    first = make_trials()
    second = make_trials()

    pd.testing.assert_frame_equal(first, second)


def test_aggregation_produces_one_row_per_regime_strategy() -> None:
    trials = make_trials()
    summary = aggregate_monte_carlo(trials)

    # 2 regimes × 3 strategies
    assert len(summary) == 6

    assert {
        "mean_shortfall_bps",
        "shortfall_std_bps",
        "mean_completion_rate",
    }.issubset(summary.columns)