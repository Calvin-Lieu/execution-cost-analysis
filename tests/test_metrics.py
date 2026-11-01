import pandas as pd
import pytest

from execution_cost.metrics import (
    execution_vwap,
    implementation_shortfall,
    implementation_shortfall_bps,
    market_twap,
    market_vwap,
    midpoint,
    quoted_spread,
    quoted_spread_bps,
    signed_cost_bps,
)
from execution_cost.models import (
    ExecutionResult,
    Fill,
    MarketSnapshot,
    ParentOrder,
    Side,
)


START = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
END = pd.Timestamp("2026-01-01 11:00:00", tz="UTC")


def make_snapshot(
    mid: float,
    volume: float,
    minute: int = 0,
    bid_depth: float = 1_000,
    ask_depth: float = 1_000,
    volatility: float = 0.01,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=START + pd.Timedelta(minutes=minute),
        bid_price=mid - 0.01,
        ask_price=mid + 0.01,
        market_volume=volume,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        volatility=volatility,
    )


def make_buy_order(quantity: float = 1_000) -> ParentOrder:
    return ParentOrder(
        order_id="order-1",
        side=Side.BUY,
        quantity=quantity,
        start_time=START,
        end_time=END,
    )


def test_midpoint_and_spread() -> None:
    snapshot = make_snapshot(100.0, 1_000)

    assert midpoint(snapshot) == pytest.approx(100.0)
    assert quoted_spread(snapshot) == pytest.approx(0.02)
    assert quoted_spread_bps(snapshot) == pytest.approx(2.0)


def test_market_twap_weights_intervals_equally() -> None:
    snapshots = [
        make_snapshot(100.0, 100, minute=0),
        make_snapshot(101.0, 300, minute=1),
    ]

    assert market_twap(snapshots) == pytest.approx(100.50)


def test_market_vwap_weights_by_volume() -> None:
    snapshots = [
        make_snapshot(100.0, 100, minute=0),
        make_snapshot(101.0, 300, minute=1),
    ]

    assert market_vwap(snapshots) == pytest.approx(100.75)


def test_market_vwap_rejects_zero_volume() -> None:
    snapshots = [
        make_snapshot(100.0, 0, minute=0),
        make_snapshot(101.0, 0, minute=1),
    ]

    with pytest.raises(ValueError, match="zero total volume"):
        market_vwap(snapshots)


def test_execution_vwap() -> None:
    order = make_buy_order()

    fills = (
        Fill("order-1", START, 300, 100.0),
        Fill("order-1", START + pd.Timedelta(minutes=1), 200, 101.0),
    )

    result = ExecutionResult(order, fills)

    assert execution_vwap(result) == pytest.approx(100.40)


def test_buying_above_benchmark_has_positive_cost() -> None:
    cost = signed_cost_bps(100.04, 100.00, Side.BUY)
    assert cost == pytest.approx(4.0)


def test_selling_below_benchmark_has_positive_cost() -> None:
    cost = signed_cost_bps(99.96, 100.00, Side.SELL)
    assert cost == pytest.approx(4.0)


def test_partial_fill_implementation_shortfall() -> None:
    order = make_buy_order(quantity=1_000)

    fills = (
        Fill(
            parent_order_id="order-1",
            timestamp=START,
            quantity=600,
            price=100.02,
            fee=3.0,
        ),
    )

    result = ExecutionResult(order, fills)

    assert implementation_shortfall(
        result,
        arrival_price=100.0,
        final_price=100.10,
    ) == pytest.approx(55.0)

    assert implementation_shortfall_bps(
        result,
        arrival_price=100.0,
        final_price=100.10,
    ) == pytest.approx(5.5)