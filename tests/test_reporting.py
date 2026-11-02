import math

import pandas as pd
import pytest

from execution_cost.experiments import run_strategy_comparison
from execution_cost.models import MarketSnapshot, ParentOrder, Side
from execution_cost.reporting import results_to_frame
from execution_cost.simulator import ExecutionConfig


START = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")


def make_parent() -> ParentOrder:
    return ParentOrder(
        order_id="report-order",
        side=Side.BUY,
        quantity=1_000,
        start_time=START,
        end_time=START + pd.Timedelta(minutes=15),
    )


def make_snapshot(minute: int, mid: float = 100.0) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=START + pd.Timedelta(minutes=minute),
        bid_price=mid - 0.01,
        ask_price=mid + 0.01,
        market_volume=1_000,
        bid_depth=10_000,
        ask_depth=10_000,
        volatility=0.001,
    )


def make_results():
    parent = make_parent()

    snapshots = [
        make_snapshot(0, mid=100.0),
        make_snapshot(5, mid=100.1),
        make_snapshot(10, mid=100.2),
        make_snapshot(15, mid=100.3),
    ]

    results = run_strategy_comparison(
        parent=parent,
        snapshots=snapshots,
        expected_volume_weights=[1, 2, 2, 1],
        pov_participation_rate=0.50,
        execution_config=ExecutionConfig(
            max_participation_rate=1.0,
            impact_coefficient=0.0,
            fee_per_unit=0.001,
        ),
    )

    return results, snapshots

def test_results_frame_contains_one_row_per_strategy() -> None:
    results, snapshots = make_results()

    frame = results_to_frame(results, snapshots)

    assert set(frame["strategy"]) == {"TWAP", "VWAP", "POV"}
    assert len(frame) == 3


def test_results_frame_contains_valid_metrics() -> None:
    results, snapshots = make_results()

    frame = results_to_frame(results, snapshots)

    assert frame["execution_vwap"].notna().all()
    assert frame["implementation_shortfall_bps"].notna().all()
    assert frame["completion_rate"].between(0, 1).all()


def test_results_frame_rejects_empty_market() -> None:
    results, _ = make_results()

    with pytest.raises(ValueError, match="At least one"):
        results_to_frame(results, [])