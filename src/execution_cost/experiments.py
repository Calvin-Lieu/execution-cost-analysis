from collections.abc import Sequence

from execution_cost.models import (
    ExecutionResult,
    MarketSnapshot,
    ParentOrder,
)
from execution_cost.simulator import ExecutionConfig, simulate_schedule
from execution_cost.strategies import (
    pov_schedule,
    twap_schedule,
    vwap_schedule,
)


def run_strategy_comparison(
    parent: ParentOrder,
    snapshots: Sequence[MarketSnapshot],
    expected_volume_weights: Sequence[float],
    pov_participation_rate: float,
    execution_config: ExecutionConfig | None = None,
) -> dict[str, ExecutionResult]:
    """
    Execute TWAP, VWAP and POV against the same market path.

    expected_volume_weights represents information available before execution,
    rather than the realised volume from the supplied market path.
    """
    if execution_config is None:
        execution_config = ExecutionConfig()

    schedules = {
        "TWAP": twap_schedule(
            parent=parent,
            snapshots=snapshots,
        ),
        "VWAP": vwap_schedule(
            parent=parent,
            snapshots=snapshots,
            expected_volume_weights=expected_volume_weights,
        ),
        "POV": pov_schedule(
            parent=parent,
            snapshots=snapshots,
            participation_rate=pov_participation_rate,
        ),
    }

    return {
        strategy_name: simulate_schedule(
            parent=parent,
            child_orders=child_orders,
            snapshots=snapshots,
            config=execution_config,
        )
        for strategy_name, child_orders in schedules.items()
    }