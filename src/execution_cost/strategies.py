import math
from collections.abc import Sequence

from execution_cost.models import ChildOrder, MarketSnapshot, ParentOrder


def snapshots_in_order_window(
    parent: ParentOrder,
    snapshots: Sequence[MarketSnapshot],
) -> list[MarketSnapshot]:
    eligible = [
        snapshot
        for snapshot in snapshots
        if parent.start_time <= snapshot.timestamp <= parent.end_time
    ]

    if not eligible:
        raise ValueError("No market snapshots exist within the order window")

    timestamps = [snapshot.timestamp for snapshot in eligible]

    if timestamps != sorted(timestamps):
        raise ValueError("Market snapshots must be sorted by timestamp")

    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Market snapshot timestamps must be unique")

    return eligible


def twap_schedule(
    parent: ParentOrder,
    snapshots: Sequence[MarketSnapshot],
) -> tuple[ChildOrder, ...]:
    eligible = snapshots_in_order_window(parent, snapshots)

    number_of_intervals = len(eligible)
    quantity_per_interval = parent.quantity / number_of_intervals

    children: list[ChildOrder] = []
    allocated_quantity = 0.0

    for index, snapshot in enumerate(eligible):
        # Give the final child any floating-point remainder.
        if index == number_of_intervals - 1:
            quantity = parent.quantity - allocated_quantity
        else:
            quantity = quantity_per_interval

        children.append(
            ChildOrder(
                parent_order_id=parent.order_id,
                timestamp=snapshot.timestamp,
                quantity=quantity,
            )
        )

        allocated_quantity += quantity

    return tuple(children)

def vwap_schedule(
    parent: ParentOrder,
    snapshots: Sequence[MarketSnapshot],
    expected_volume_weights: Sequence[float],
) -> tuple[ChildOrder, ...]:
    eligible = snapshots_in_order_window(parent, snapshots)

    if len(expected_volume_weights) != len(eligible):
        raise ValueError(
            "Volume profile must contain one weight per eligible snapshot"
        )

    if any(
        not math.isfinite(weight) or weight <= 0
        for weight in expected_volume_weights
    ):
        raise ValueError("Volume weights must be finite and positive")

    total_weight = math.fsum(expected_volume_weights)

    children: list[ChildOrder] = []
    allocated_quantity = 0.0

    for index, (snapshot, weight) in enumerate(
        zip(eligible, expected_volume_weights)
    ):
        if index == len(eligible) - 1:
            # Absorb any floating-point remainder.
            quantity = parent.quantity - allocated_quantity
        else:
            quantity = parent.quantity * weight / total_weight

        children.append(
            ChildOrder(
                parent_order_id=parent.order_id,
                timestamp=snapshot.timestamp,
                quantity=quantity,
            )
        )

        allocated_quantity += quantity

    return tuple(children)