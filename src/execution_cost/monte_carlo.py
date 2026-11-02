from dataclasses import dataclass, replace
import math

import pandas as pd

from execution_cost.experiments import run_strategy_comparison
from execution_cost.market import (
    MarketGeneratorConfig,
    MarketRegime,
    expected_volume_profile,
    generate_market,
)
from execution_cost.models import ParentOrder, Side
from execution_cost.reporting import results_to_frame
from execution_cost.simulator import ExecutionConfig


@dataclass(frozen=True)
class MonteCarloConfig:
    runs_per_regime: int = 50
    order_quantity: float = 100_000
    pov_participation_rate: float = 0.10
    side: Side = Side.BUY
    regimes: tuple[MarketRegime, ...] = tuple(MarketRegime)

    def __post_init__(self) -> None:
        if self.runs_per_regime <= 0:
            raise ValueError("runs_per_regime must be positive")

        if (
            not math.isfinite(self.order_quantity)
            or self.order_quantity <= 0
        ):
            raise ValueError("order_quantity must be positive and finite")

        if (
            not math.isfinite(self.pov_participation_rate)
            or not 0 < self.pov_participation_rate <= 1
        ):
            raise ValueError(
                "pov_participation_rate must be within (0, 1]"
            )

        if not self.regimes:
            raise ValueError("At least one market regime is required")


def run_monte_carlo(
    market_config: MarketGeneratorConfig,
    simulation_config: MonteCarloConfig,
    execution_config: ExecutionConfig | None = None,
) -> pd.DataFrame:
    if execution_config is None:
        execution_config = ExecutionConfig()

    trial_frames: list[pd.DataFrame] = []

    for regime in simulation_config.regimes:
        for run_index in range(simulation_config.runs_per_regime):
            seed = market_config.seed + run_index

            trial_market_config = replace(
                market_config,
                seed=seed,
            )

            snapshots = generate_market(
                config=trial_market_config,
                regime=regime,
            )

            parent = ParentOrder(
                order_id=f"{regime.value}-{seed}",
                side=simulation_config.side,
                quantity=simulation_config.order_quantity,
                start_time=snapshots[0].timestamp,
                end_time=snapshots[-1].timestamp,
            )

            # This is the forecast structural profile, not realised volume.
            expected_weights = expected_volume_profile(
                len(snapshots)
            )

            results = run_strategy_comparison(
                parent=parent,
                snapshots=snapshots,
                expected_volume_weights=expected_weights,
                pov_participation_rate=(
                    simulation_config.pov_participation_rate
                ),
                execution_config=execution_config,
            )

            frame = results_to_frame(results, snapshots)
            frame["regime"] = regime.value
            frame["seed"] = seed

            trial_frames.append(frame)

    return pd.concat(trial_frames, ignore_index=True)


def aggregate_monte_carlo(
    trials: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "strategy",
        "regime",
        "seed",
        "implementation_shortfall_bps",
        "completion_rate",
        "execution_vwap",
        "fees",
    }

    missing_columns = required_columns.difference(trials.columns)

    if missing_columns:
        raise ValueError(
            f"Missing trial columns: {sorted(missing_columns)}"
        )

    return (
        trials.groupby(["regime", "strategy"], as_index=False)
        .agg(
            trials=("seed", "count"),
            mean_shortfall_bps=(
                "implementation_shortfall_bps",
                "mean",
            ),
            shortfall_std_bps=(
                "implementation_shortfall_bps",
                "std",
            ),
            mean_completion_rate=("completion_rate", "mean"),
            mean_execution_vwap=("execution_vwap", "mean"),
            mean_fees=("fees", "mean"),
        )
        .sort_values(["regime", "strategy"])
        .reset_index(drop=True)
    )