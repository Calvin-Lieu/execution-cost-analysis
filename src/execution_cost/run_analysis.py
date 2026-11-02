from pathlib import Path

import pandas as pd

from execution_cost.market import (
    MarketGeneratorConfig,
    MarketRegime,
)
from execution_cost.monte_carlo import (
    MonteCarloConfig,
    aggregate_monte_carlo,
    run_monte_carlo,
)
from execution_cost.plotting import (
    plot_completion_rates,
    plot_mean_shortfall,
    plot_paired_improvements,
    plot_shortfall_distribution,
)
from execution_cost.simulator import ExecutionConfig

from execution_cost.paired_analysis import (
    paired_strategy_comparison,
)


def main() -> None:
    output_directory = Path("results")
    output_directory.mkdir(parents=True, exist_ok=True)

    market_config = MarketGeneratorConfig(
        start_time=pd.Timestamp(
            "2026-01-05 10:00:00",
            tz="Australia/Sydney",
        ),
        intervals=78,
        interval_minutes=5,
        initial_mid_price=100.0,
        total_market_volume=7_800_000,
        base_depth=50_000,
        base_spread_bps=2.0,
        seed=42,
    )

    simulation_config = MonteCarloConfig(
        runs_per_regime=100,
        order_quantity=500_000,
        pov_participation_rate=0.10,
        regimes=tuple(MarketRegime),
    )

    execution_config = ExecutionConfig(
        max_participation_rate=0.20,
        impact_coefficient=0.50,
        fee_per_unit=0.001,
    )

    trials = run_monte_carlo(
        market_config=market_config,
        simulation_config=simulation_config,
        execution_config=execution_config,
    )

    summary = aggregate_monte_carlo(trials)

    paired_comparisons = paired_strategy_comparison(
        trials,
        baseline="TWAP",
    )

    plot_paired_improvements(
        paired_comparisons,
        output_directory / "paired_improvements.png",
    )

    trials.to_csv(
        output_directory / "trials.csv",
        index=False,
    )

    summary.to_csv(
        output_directory / "summary.csv",
        index=False,
    )

    paired_comparisons.to_csv(
        output_directory / "paired_comparisons.csv",
        index=False,
    )

    plot_mean_shortfall(
        summary,
        output_directory / "mean_shortfall.png",
    )

    plot_completion_rates(
        summary,
        output_directory / "completion_rates.png",
    )

    plot_shortfall_distribution(
        trials,
        output_directory / "shortfall_distribution.png",
    )

    print("\nExecution strategy summary:\n")
    print(summary.round(4).to_string(index=False))

    print("\nPaired comparisons against TWAP:\n")
    print(
        paired_comparisons[
            [
                "regime",
                "comparison",
                "mean_shortfall_improvement_bps",
                "ci95_lower_bps",
                "ci95_upper_bps",
                "win_rate",
                "mean_completion_delta",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"\nResults written to: {output_directory.resolve()}")


if __name__ == "__main__":
    main()