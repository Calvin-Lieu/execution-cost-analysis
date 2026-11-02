from collections.abc import Mapping, Sequence

import pandas as pd

from execution_cost.metrics import (
    execution_vwap,
    implementation_shortfall_bps,
    market_twap,
    market_vwap,
    midpoint,
)
from execution_cost.models import ExecutionResult, MarketSnapshot


def results_to_frame(
    results: Mapping[str, ExecutionResult],
    snapshots: Sequence[MarketSnapshot],
) -> pd.DataFrame:
    if not snapshots:
        raise ValueError("At least one market snapshot is required")

    arrival_price = midpoint(snapshots[0])
    terminal_price = midpoint(snapshots[-1])

    twap_benchmark = market_twap(snapshots)
    vwap_benchmark = market_vwap(snapshots)

    rows: list[dict[str, object]] = []

    for strategy_name, result in results.items():
        if result.fills:
            average_execution_price = execution_vwap(result)
        else:
            average_execution_price = float("nan")

        shortfall_bps = implementation_shortfall_bps(
            result,
            arrival_price,
            terminal_price,
        )

        rows.append(
            {
                "strategy": strategy_name,
                "side": result.parent_order.side.name,
                "order_quantity": result.parent_order.quantity,
                "filled_quantity": result.filled_quantity,
                "unfilled_quantity": result.unfilled_quantity,
                "completion_rate": result.completion_rate,
                "execution_vwap": average_execution_price,
                "arrival_price": arrival_price,
                "market_twap": twap_benchmark,
                "market_vwap": vwap_benchmark,
                "implementation_shortfall_bps": shortfall_bps,
                "fees": sum(fill.fee for fill in result.fills),
                "fill_count": len(result.fills),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("strategy")
        .reset_index(drop=True)
    )