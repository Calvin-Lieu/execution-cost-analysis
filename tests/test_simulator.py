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
    *,
    market_volume: float = 10_000,
    bid_depth: float = 5_000,
    ask_depth: float = 5_000,
    volatility: float = 0.001,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=START,
        bid_price=99.99,
        ask_price=100.01,
        market_volume=market_volume,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        volatility=volatility,
    )


def make_parent(side: Side = Side.BUY) -> ParentOrder:
    return ParentOrder(
        order_id="order-1",
        side=side,
        quantity=10_000,
        start_time=START,
        end_time=END,
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