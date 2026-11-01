import pandas as pd
import pytest

from execution_cost.models import (
    ExecutionResult,
    Fill,
    MarketSnapshot,
    ParentOrder,
    Side,
)


START = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
END = pd.Timestamp("2026-01-01 11:00:00", tz="UTC")


def make_parent_order() -> ParentOrder:
    return ParentOrder(
        order_id="order-1",
        side=Side.BUY,
        quantity=1_000,
        start_time=START,
        end_time=END,
    )


def test_side_uses_signed_direction() -> None:
    assert Side.BUY.value == 1
    assert Side.SELL.value == -1


def test_valid_market_snapshot() -> None:
    snapshot = MarketSnapshot(
        timestamp=START,
        bid_price=99.99,
        ask_price=100.01,
        market_volume=10_000,
        bid_depth=2_000,
        ask_depth=1_800,
        volatility=0.01,
    )

    assert snapshot.bid_price == 99.99
    assert snapshot.ask_price == 100.01


def test_crossed_market_is_rejected() -> None:
    with pytest.raises(ValueError, match="ask price"):
        MarketSnapshot(
            timestamp=START,
            bid_price=100.01,
            ask_price=99.99,
            market_volume=10_000,
            bid_depth=2_000,
            ask_depth=1_800,
            volatility=0.01,
        )


def test_negative_market_volume_is_rejected() -> None:
    with pytest.raises(ValueError, match="volume"):
        MarketSnapshot(
            timestamp=START,
            bid_price=99.99,
            ask_price=100.01,
            market_volume=-1,
            bid_depth=2_000,
            ask_depth=1_800,
            volatility=0.01,
        )


def test_parent_order_requires_positive_quantity() -> None:
    with pytest.raises(ValueError, match="quantity"):
        ParentOrder(
            order_id="order-1",
            side=Side.BUY,
            quantity=0,
            start_time=START,
            end_time=END,
        )


def test_parent_order_requires_valid_time_window() -> None:
    with pytest.raises(ValueError, match="end_time"):
        ParentOrder(
            order_id="order-1",
            side=Side.BUY,
            quantity=1_000,
            start_time=START,
            end_time=START,
        )


def test_execution_result_calculates_quantities_and_fees() -> None:
    order = make_parent_order()

    fills = (
        Fill("order-1", START, 300, 100.01, fee=1.50),
        Fill("order-1", START + pd.Timedelta(minutes=10), 200, 100.02, fee=1.00),
    )

    result = ExecutionResult(order, fills)

    assert result.filled_quantity == pytest.approx(500)
    assert result.unfilled_quantity == pytest.approx(500)
    assert result.total_fees == pytest.approx(2.50)
    assert result.completion_rate == pytest.approx(0.5)


def test_execution_result_rejects_overfill() -> None:
    order = make_parent_order()

    fills = (
        Fill("order-1", START, 800, 100.01),
        Fill("order-1", START + pd.Timedelta(minutes=10), 300, 100.02),
    )

    with pytest.raises(ValueError, match="exceed"):
        ExecutionResult(order, fills)


def test_execution_result_rejects_wrong_parent() -> None:
    order = make_parent_order()
    fills = (Fill("different-order", START, 100, 100.01),)

    with pytest.raises(ValueError, match="different parent"):
        ExecutionResult(order, fills)


def test_execution_result_rejects_fill_outside_window() -> None:
    order = make_parent_order()
    fills = (
        Fill(
            "order-1",
            END + pd.Timedelta(minutes=1),
            100,
            100.01,
        ),
    )

    with pytest.raises(ValueError, match="outside"):
        ExecutionResult(order, fills)