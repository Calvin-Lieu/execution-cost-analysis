import math

import pandas as pd
import pytest

from execution_cost.models import (
    ChildOrder,
    MarketSnapshot,
    ParentOrder,
    Side,
)
from execution_cost.simulator import (
    ExecutionConfig,
    available_quantity,
    simulate_fill,
    temporary_impact,
)


START = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
END = pd.Timestamp("2026-01-01 11:00:00", tz="UTC")


def make_snapshot(
    market_volume: float = 10_000,
    bid_depth: float = 5_000,
    ask_depth: float = 5_000,
    volatility: float = 0.001,
    minute: int = 0,
    volume: float | None = None,
) -> MarketSnapshot:
    # "volume" is a convenient alias used by the schedule tests.
    if volume is not None:
        market_volume = volume

    return MarketSnapshot(
        timestamp=START + pd.Timedelta(minutes=minute),
        bid_price=99.99,
        ask_price=100.01,
        market_volume=market_volume,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        volatility=volatility,
    )


def make_parent(
    side: Side = Side.BUY,
    quantity: float = 100.0,
) -> ParentOrder:
    return ParentOrder(
        order_id="order-1",
        side=side,
        quantity=quantity,
        start_time=START,
        end_time=START + pd.Timedelta(minutes=15),
    )


def make_child(quantity: float = 1_000) -> ChildOrder:
    return ChildOrder(
        parent_order_id="order-1",
        timestamp=START,
        quantity=quantity,
    )


def test_buy_uses_ask_plus_impact() -> None:
    snapshot = make_snapshot()
    config = ExecutionConfig()
    child = make_child(1_000)

    fill = simulate_fill(
        make_parent(Side.BUY),
        child,
        snapshot,
        config,
    )

    assert fill is not None

    expected_impact = (
        0.50
        * 100.0
        * 0.001
        * math.sqrt(1_000 / 10_000)
    )

    assert fill.price == pytest.approx(
        snapshot.ask_price + expected_impact
    )


def test_sell_uses_bid_minus_impact() -> None:
    snapshot = make_snapshot()
    config = ExecutionConfig()
    child = make_child(1_000)

    fill = simulate_fill(
        make_parent(Side.SELL),
        child,
        snapshot,
        config,
    )

    assert fill is not None

    expected_impact = (
        0.50
        * 100.0
        * 0.001
        * math.sqrt(1_000 / 10_000)
    )

    assert fill.price == pytest.approx(
        snapshot.bid_price - expected_impact
    )


def test_fill_is_limited_by_participation() -> None:
    snapshot = make_snapshot(
        market_volume=10_000,
        ask_depth=5_000,
    )

    config = ExecutionConfig(max_participation_rate=0.10)

    fill = simulate_fill(
        make_parent(),
        make_child(5_000),
        snapshot,
        config,
    )

    assert fill is not None
    assert fill.quantity == pytest.approx(1_000)


def test_fill_is_limited_by_opposite_depth() -> None:
    snapshot = make_snapshot(
        market_volume=10_000,
        ask_depth=800,
    )

    fill = simulate_fill(
        make_parent(Side.BUY),
        make_child(5_000),
        snapshot,
        ExecutionConfig(max_participation_rate=0.20),
    )

    assert fill is not None
    assert fill.quantity == pytest.approx(800)


def test_zero_market_volume_produces_no_fill() -> None:
    snapshot = make_snapshot(market_volume=0)

    fill = simulate_fill(
        make_parent(),
        make_child(),
        snapshot,
        ExecutionConfig(),
    )

    assert fill is None


def test_larger_participation_has_greater_impact() -> None:
    snapshot = make_snapshot(
        market_volume=10_000,
        ask_depth=10_000,
    )

    config = ExecutionConfig(max_participation_rate=1.0)

    small_impact = temporary_impact(
        100,
        snapshot,
        config,
    )

    large_impact = temporary_impact(
        1_000,
        snapshot,
        config,
    )

    assert large_impact > small_impact


def test_fee_is_calculated_from_filled_quantity() -> None:
    snapshot = make_snapshot()

    config = ExecutionConfig(
        max_participation_rate=1.0,
        fee_per_unit=0.002,
    )

    fill = simulate_fill(
        make_parent(),
        make_child(500),
        snapshot,
        config,
    )

    assert fill is not None
    assert fill.fee == pytest.approx(1.0)


def test_wrong_parent_order_is_rejected() -> None:
    snapshot = make_snapshot()

    child = ChildOrder(
        parent_order_id="different-order",
        timestamp=START,
        quantity=100,
    )

    with pytest.raises(ValueError, match="different parent"):
        simulate_fill(
            make_parent(),
            child,
            snapshot,
            ExecutionConfig(),
        )

from execution_cost.strategies import twap_schedule
from execution_cost.simulator import ExecutionConfig, simulate_schedule


def test_simulate_twap_schedule_fills_entire_order() -> None:
    parent = make_parent(quantity=1_000)

    snapshots = [
        make_snapshot(minute=0, volume=1_000),
        make_snapshot(minute=5, volume=1_000),
        make_snapshot(minute=10, volume=1_000),
        make_snapshot(minute=15, volume=1_000),
    ]

    children = twap_schedule(parent, snapshots)

    result = simulate_schedule(
        parent=parent,
        child_orders=children,
        snapshots=snapshots,
        config=ExecutionConfig(
            max_participation_rate=1.0,
            impact_coefficient=0.0,
            fee_per_unit=0.001,
        ),
    )

    assert result.filled_quantity == pytest.approx(1_000)
    assert result.unfilled_quantity == pytest.approx(0)
    assert result.completion_rate == pytest.approx(1.0)
    assert len(result.fills) == 4


def test_simulate_twap_schedule_can_be_partially_filled() -> None:
    parent = make_parent(quantity=1_000)

    snapshots = [
        make_snapshot(minute=0, volume=1_000),
        make_snapshot(minute=5, volume=1_000),
        make_snapshot(minute=10, volume=1_000),
        make_snapshot(minute=15, volume=1_000),
    ]

    children = twap_schedule(parent, snapshots)

    result = simulate_schedule(
        parent=parent,
        child_orders=children,
        snapshots=snapshots,
        config=ExecutionConfig(
            max_participation_rate=0.10,
            impact_coefficient=0.0,
            fee_per_unit=0.0,
        ),
    )

    # Each interval requests 250, but only 10% of 1,000 is available.
    assert result.filled_quantity == pytest.approx(400)
    assert result.unfilled_quantity == pytest.approx(600)
    assert result.completion_rate == pytest.approx(0.4)


def test_simulate_schedule_rejects_missing_snapshot() -> None:
    parent = make_parent(quantity=1_000)
    scheduling_snapshots = [
        make_snapshot(minute=0, volume=1_000),
        make_snapshot(minute=5, volume=1_000),
    ]

    children = twap_schedule(parent, scheduling_snapshots)

    with pytest.raises(ValueError, match="No market snapshot"):
        simulate_schedule(
            parent=parent,
            child_orders=children,
            snapshots=[scheduling_snapshots[0]],
        )


def test_simulate_schedule_rejects_duplicate_snapshots() -> None:
    parent = make_parent(quantity=1_000)
    snapshot = make_snapshot(minute=0, volume=1_000)
    children = twap_schedule(parent, [snapshot])

    with pytest.raises(ValueError, match="unique"):
        simulate_schedule(
            parent=parent,
            child_orders=children,
            snapshots=[snapshot, snapshot],
        )