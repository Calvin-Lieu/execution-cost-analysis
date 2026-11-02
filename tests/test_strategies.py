import pandas as pd
import pytest

from execution_cost.models import MarketSnapshot, ParentOrder, Side
from execution_cost.strategies import twap_schedule, vwap_schedule, pov_schedule

START = pd.Timestamp("2026-01-05 10:00:00")


def make_snapshot(
    minute: int,
    volume: float = 1_000,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=START + pd.Timedelta(minutes=minute),
        bid_price=99.99,
        ask_price=100.01,
        market_volume=volume,
        bid_depth=10_000,
        ask_depth=10_000,
        volatility=0.01,
    )


def make_parent(quantity: float = 1_000) -> ParentOrder:
    return ParentOrder(
        order_id="order-1",
        side=Side.BUY,
        quantity=quantity,
        start_time=START,
        end_time=START + pd.Timedelta(minutes=15),
    )


def test_twap_divides_quantity_evenly() -> None:
    snapshots = [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
        make_snapshot(15),
    ]

    children = twap_schedule(make_parent(), snapshots)

    assert len(children) == 4
    assert [child.quantity for child in children] == pytest.approx(
        [250, 250, 250, 250]
    )


def test_twap_quantities_sum_to_parent_quantity() -> None:
    snapshots = [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
    ]

    parent = make_parent(quantity=1_000)
    children = twap_schedule(parent, snapshots)

    assert sum(child.quantity for child in children) == pytest.approx(
        parent.quantity
    )


def test_twap_uses_market_snapshot_timestamps() -> None:
    snapshots = [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
    ]

    children = twap_schedule(make_parent(), snapshots)

    assert [child.timestamp for child in children] == [
        snapshot.timestamp for snapshot in snapshots
    ]


def test_twap_ignores_snapshots_outside_order_window() -> None:
    snapshots = [
        make_snapshot(-5),
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(20),
    ]

    children = twap_schedule(make_parent(), snapshots)

    assert len(children) == 2
    assert [child.timestamp for child in children] == [
        START,
        START + pd.Timedelta(minutes=5),
    ]


def test_twap_rejects_missing_market_data() -> None:
    with pytest.raises(ValueError, match="No market snapshots"):
        twap_schedule(make_parent(), [])


def test_twap_rejects_unsorted_snapshots() -> None:
    snapshots = [
        make_snapshot(5),
        make_snapshot(0),
    ]

    with pytest.raises(ValueError, match="sorted"):
        twap_schedule(make_parent(), snapshots)


def test_twap_rejects_duplicate_timestamps() -> None:
    snapshots = [
        make_snapshot(0),
        make_snapshot(0),
    ]

    with pytest.raises(ValueError, match="unique"):
        twap_schedule(make_parent(), snapshots)

def test_vwap_allocates_using_expected_volume_profile() -> None:
    parent = make_parent(quantity=1_000)

    snapshots = [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
    ]

    children = vwap_schedule(
        parent=parent,
        snapshots=snapshots,
        expected_volume_weights=[1, 2, 1],
    )

    assert [child.quantity for child in children] == pytest.approx(
        [250, 500, 250]
    )


def test_vwap_normalises_volume_weights() -> None:
    parent = make_parent(quantity=1_000)

    snapshots = [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
    ]

    children = vwap_schedule(
        parent=parent,
        snapshots=snapshots,
        expected_volume_weights=[10, 20, 10],
    )

    assert [child.quantity for child in children] == pytest.approx(
        [250, 500, 250]
    )


def test_vwap_quantities_sum_to_parent_quantity() -> None:
    parent = make_parent(quantity=1_000)

    snapshots = [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
    ]

    children = vwap_schedule(
        parent=parent,
        snapshots=snapshots,
        expected_volume_weights=[1, 1, 1],
    )

    assert sum(child.quantity for child in children) == pytest.approx(
        parent.quantity
    )


def test_vwap_rejects_wrong_number_of_weights() -> None:
    parent = make_parent()

    snapshots = [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
    ]

    with pytest.raises(ValueError, match="one weight"):
        vwap_schedule(
            parent=parent,
            snapshots=snapshots,
            expected_volume_weights=[1, 2],
        )


@pytest.mark.parametrize(
    "weights",
    [
        [1, 0, 1],
        [1, -1, 1],
        [1, float("nan"), 1],
        [1, float("inf"), 1],
    ],
)
def test_vwap_rejects_invalid_weights(weights: list[float]) -> None:
    parent = make_parent()

    snapshots = [
        make_snapshot(0),
        make_snapshot(5),
        make_snapshot(10),
    ]

    with pytest.raises(ValueError, match="finite and positive"):
        vwap_schedule(
            parent=parent,
            snapshots=snapshots,
            expected_volume_weights=weights,
        )

def test_pov_targets_percentage_of_market_volume() -> None:
    parent = make_parent(quantity=1_000)

    snapshots = [
        make_snapshot(0, volume=1_000),
        make_snapshot(5, volume=2_000),
        make_snapshot(10, volume=1_000),
    ]

    children = pov_schedule(
        parent=parent,
        snapshots=snapshots,
        participation_rate=0.10,
    )

    assert [child.quantity for child in children] == pytest.approx(
        [100, 200, 100]
    )


def test_pov_never_exceeds_parent_quantity() -> None:
    parent = make_parent(quantity=150)

    snapshots = [
        make_snapshot(0, volume=1_000),
        make_snapshot(5, volume=1_000),
        make_snapshot(10, volume=1_000),
    ]

    children = pov_schedule(
        parent=parent,
        snapshots=snapshots,
        participation_rate=0.10,
    )

    assert [child.quantity for child in children] == pytest.approx(
        [100, 50]
    )

    assert sum(child.quantity for child in children) == pytest.approx(150)


def test_pov_skips_zero_volume_intervals() -> None:
    parent = make_parent(quantity=1_000)

    snapshots = [
        make_snapshot(0, volume=1_000),
        make_snapshot(5, volume=0),
        make_snapshot(10, volume=2_000),
    ]

    children = pov_schedule(
        parent=parent,
        snapshots=snapshots,
        participation_rate=0.10,
    )

    assert [child.quantity for child in children] == pytest.approx(
        [100, 200]
    )

    assert [child.timestamp for child in children] == [
        START,
        START + pd.Timedelta(minutes=10),
    ]


@pytest.mark.parametrize(
    "participation_rate",
    [0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_pov_rejects_invalid_participation_rate(
    participation_rate: float,
) -> None:
    with pytest.raises(ValueError, match="within"):
        pov_schedule(
            parent=make_parent(),
            snapshots=[make_snapshot(0)],
            participation_rate=participation_rate,
        )