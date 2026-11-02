import pandas as pd
import pytest

from execution_cost.experiments import run_strategy_comparison
from execution_cost.models import MarketSnapshot, ParentOrder, Side
from execution_cost.simulator import ExecutionConfig


START = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")


def make_parent(quantity: float = 1_000) -> ParentOrder:
    return ParentOrder(
        order_id="comparison-order",
        side=Side.BUY,
        quantity=quantity,
        start_time=START,
        end_time=START + pd.Timedelta(minutes=15),
    )


def make_snapshot(
    minute: int,
    market_volume: float = 1_000,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=START + pd.Timedelta(minutes=minute),
        bid_price=99.99,
        ask_price=100.01,
        market_volume=market_volume,
        bid_depth=10_000,
        ask_depth=10_000,
        volatility=0.001,
    )


def make_market() -> list[MarketSnapshot]:
    return [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
        make_snapshot(15),
    ]


def test_comparison_runs_all_three_strategies() -> None:
    results = run_strategy_comparison(
        parent=make_parent(),
        snapshots=make_market(),
        expected_volume_weights=[1, 2, 2, 1],
        pov_participation_rate=0.50,
        execution_config=ExecutionConfig(
            max_participation_rate=1.0,
            impact_coefficient=0.0,
            fee_per_unit=0.0,
        ),
    )

    assert set(results) == {"TWAP", "VWAP", "POV"}


def test_comparison_uses_same_parent_order() -> None:
    parent = make_parent()

    results = run_strategy_comparison(
        parent=parent,
        snapshots=make_market(),
        expected_volume_weights=[1, 2, 2, 1],
        pov_participation_rate=0.50,
        execution_config=ExecutionConfig(
            max_participation_rate=1.0,
            impact_coefficient=0.0,
            fee_per_unit=0.0,
        ),
    )

    for result in results.values():
        assert result.parent_order == parent


def test_strategies_never_overfill_parent_order() -> None:
    parent = make_parent()

    results = run_strategy_comparison(
        parent=parent,
        snapshots=make_market(),
        expected_volume_weights=[1, 2, 2, 1],
        pov_participation_rate=0.50,
        execution_config=ExecutionConfig(
            max_participation_rate=1.0,
            impact_coefficient=0.0,
            fee_per_unit=0.0,
        ),
    )

    for result in results.values():
        assert result.filled_quantity <= parent.quantity


def test_strategies_complete_when_liquidity_is_sufficient() -> None:
    parent = make_parent()

    results = run_strategy_comparison(
        parent=parent,
        snapshots=make_market(),
        expected_volume_weights=[1, 2, 2, 1],
        pov_participation_rate=0.50,
        execution_config=ExecutionConfig(
            max_participation_rate=1.0,
            impact_coefficient=0.0,
            fee_per_unit=0.0,
        ),
    )

    for result in results.values():
        assert result.filled_quantity == pytest.approx(parent.quantity)
        assert result.completion_rate == pytest.approx(1.0)