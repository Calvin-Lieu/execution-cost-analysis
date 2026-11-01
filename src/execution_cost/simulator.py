from __future__ import annotations

from dataclasses import dataclass
import math

from execution_cost.metrics import midpoint
from execution_cost.models import (
    ChildOrder,
    Fill,
    MarketSnapshot,
    ParentOrder,
    Side,
)

@dataclass(frozen=True)
class ExecutionConfig:
    max_participation_rate: float = 0.10
    impact_coefficient: float = 0.50
    fee_per_unit: float = 0.001

    def __post_init__(self) -> None:
        values = {
            "max_participation_rate": self.max_participation_rate,
            "impact_coefficient": self.impact_coefficient,
            "fee_per_unit": self.fee_per_unit,
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if not 0 < self.max_participation_rate <= 1:
            raise ValueError(
                "max_participation_rate must be between 0 and 1"
            )

        if self.impact_coefficient < 0:
            raise ValueError("impact_coefficient cannot be negative")

        if self.fee_per_unit < 0:
            raise ValueError("fee_per_unit cannot be negative")

def available_quantity(
    side: Side,
    snapshot: MarketSnapshot,
    config: ExecutionConfig,
) -> float:
    volume_capacity = (
        config.max_participation_rate * snapshot.market_volume
    )

    if side is Side.BUY:
        book_capacity = snapshot.ask_depth
    else:
        book_capacity = snapshot.bid_depth

    return min(volume_capacity, book_capacity)

def temporary_impact(
    quantity: float,
    snapshot: MarketSnapshot,
    config: ExecutionConfig,
) -> float:
    if quantity <= 0 or snapshot.market_volume <= 0:
        return 0.0

    participation_rate = quantity / snapshot.market_volume

    return (
        config.impact_coefficient
        * midpoint(snapshot)
        * snapshot.volatility
        * math.sqrt(participation_rate)
    )

def simulate_fill(
    parent_order: ParentOrder,
    child_order: ChildOrder,
    snapshot: MarketSnapshot,
    config: ExecutionConfig,
) -> Fill | None:
    if child_order.parent_order_id != parent_order.order_id:
        raise ValueError("child order belongs to a different parent order")

    if child_order.timestamp != snapshot.timestamp:
        raise ValueError(
            "child order and market snapshot timestamps must match"
        )

    if not (
        parent_order.start_time
        <= child_order.timestamp
        <= parent_order.end_time
    ):
        raise ValueError("child order lies outside the execution window")

    capacity = available_quantity(
        parent_order.side,
        snapshot,
        config,
    )

    fill_quantity = min(child_order.quantity, capacity)

    if fill_quantity <= 0:
        return None

    impact = temporary_impact(
        fill_quantity,
        snapshot,
        config,
    )

    if parent_order.side is Side.BUY:
        fill_price = snapshot.ask_price + impact
    else:
        fill_price = snapshot.bid_price - impact

    if fill_price <= 0:
        raise ValueError("simulated fill price must remain positive")

    fee = fill_quantity * config.fee_per_unit

    return Fill(
        parent_order_id=parent_order.order_id,
        timestamp=child_order.timestamp,
        quantity=fill_quantity,
        price=fill_price,
        fee=fee,
    )

