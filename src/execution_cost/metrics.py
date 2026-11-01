import math
from collections.abc import Sequence

from execution_cost.models import (
    ExecutionResult,
    MarketSnapshot,
    Side,
)

BASIS_POINTS = 10000

def midpoint(snapshot: MarketSnapshot) -> float:
    return (snapshot.bid_price + snapshot.ask_price) / 2.0

def quoted_spread(snapshot: MarketSnapshot) -> float:
    return snapshot.ask_price - snapshot.bid_price

def quoted_spread_bps(snapshot: MarketSnapshot) -> float:
    return BASIS_POINTS * quoted_spread(snapshot)/midpoint(snapshot)

def market_twap(snapshots: Sequence[MarketSnapshot]) -> float:
    if not snapshots:
        raise ValueError("cannot calculate TWAP without market snapshots")

    return math.fsum(midpoint(snapshot) for snapshot in snapshots) / len(snapshots)

def market_vwap(snapshots: Sequence[MarketSnapshot]) -> float:
    if not snapshots:
        raise ValueError("cannot calculate VWAP without market snapshots")

    total_volume = math.fsum(
        snapshot.market_volume for snapshot in snapshots
    )

    if total_volume <= 0:
        raise ValueError("cannot calculate VWAP with zero total volume")

    weighted_value = math.fsum(
        midpoint(snapshot) * snapshot.market_volume
        for snapshot in snapshots
    )

    return weighted_value / total_volume

def execution_vwap(result: ExecutionResult) -> float:
    if not result.fills:
        raise ValueError("cannot calculate execution VWAP without fills")

    executed_value = math.fsum(
        fill.price * fill.quantity
        for fill in result.fills
    )

    return executed_value / result.filled_quantity

def signed_cost_bps(
    execution_price: float,
    benchmark_price: float,
    side: Side,
) -> float:
    if not math.isfinite(execution_price) or execution_price <= 0:
        raise ValueError("execution price must be positive and finite")

    if not math.isfinite(benchmark_price) or benchmark_price <= 0:
        raise ValueError("benchmark price must be positive and finite")

    return (
        BASIS_POINTS
        * side.value
        * (execution_price - benchmark_price)
        / benchmark_price
    )

def implementation_shortfall(
    result: ExecutionResult,
    arrival_price: float,
    final_price: float,
) -> float:
    if not math.isfinite(arrival_price) or arrival_price <= 0:
        raise ValueError("arrival price must be positive and finite")

    if not math.isfinite(final_price) or final_price <= 0:
        raise ValueError("final price must be positive and finite")

    direction = result.parent_order.side.value

    execution_cost = math.fsum(
        direction
        * fill.quantity
        * (fill.price - arrival_price)
        for fill in result.fills
    )

    opportunity_cost = (
        direction
        * result.unfilled_quantity
        * (final_price - arrival_price)
    )

    return execution_cost + opportunity_cost + result.total_fees

def implementation_shortfall_bps(
    result: ExecutionResult,
    arrival_price: float,
    final_price: float,
) -> float:
    shortfall = implementation_shortfall(
        result,
        arrival_price,
        final_price,
    )

    notional = result.parent_order.quantity * arrival_price

    return BASIS_POINTS * shortfall / notional