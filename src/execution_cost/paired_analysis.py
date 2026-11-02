import math

import pandas as pd
from scipy.stats import t


def paired_strategy_comparison(
    trials: pd.DataFrame,
    baseline: str = "TWAP",
) -> pd.DataFrame:
    required_columns = {
        "regime",
        "seed",
        "strategy",
        "implementation_shortfall_bps",
        "completion_rate",
    }

    missing_columns = required_columns.difference(trials.columns)

    if missing_columns:
        raise ValueError(
            f"Missing trial columns: {sorted(missing_columns)}"
        )

    duplicates = trials.duplicated(
        subset=["regime", "seed", "strategy"]
    )

    if duplicates.any():
        raise ValueError(
            "Each regime, seed and strategy combination must be unique"
        )

    if baseline not in set(trials["strategy"]):
        raise ValueError(f"Baseline strategy {baseline!r} was not found")

    rows: list[dict[str, object]] = []

    for regime, regime_trials in trials.groupby("regime"):
        shortfall = regime_trials.pivot(
            index="seed",
            columns="strategy",
            values="implementation_shortfall_bps",
        )

        completion = regime_trials.pivot(
            index="seed",
            columns="strategy",
            values="completion_rate",
        )

        comparison_strategies = [
            strategy
            for strategy in shortfall.columns
            if strategy != baseline
        ]

        for comparison in comparison_strategies:
            paired = pd.DataFrame(
                {
                    "baseline_shortfall": shortfall[baseline],
                    "comparison_shortfall": shortfall[comparison],
                    "baseline_completion": completion[baseline],
                    "comparison_completion": completion[comparison],
                }
            ).dropna()

            if paired.empty:
                continue

            # Positive means the comparison strategy cost less.
            improvements = (
                paired["baseline_shortfall"]
                - paired["comparison_shortfall"]
            )

            completion_differences = (
                paired["comparison_completion"]
                - paired["baseline_completion"]
            )

            pair_count = len(improvements)
            mean_improvement = improvements.mean()

            if pair_count >= 2:
                standard_deviation = improvements.std(ddof=1)
                standard_error = (
                    standard_deviation / math.sqrt(pair_count)
                )
                critical_value = t.ppf(
                    0.975,
                    df=pair_count - 1,
                )
                margin = critical_value * standard_error
                lower_bound = mean_improvement - margin
                upper_bound = mean_improvement + margin
            else:
                standard_deviation = float("nan")
                standard_error = float("nan")
                lower_bound = float("nan")
                upper_bound = float("nan")

            rows.append(
                {
                    "regime": regime,
                    "baseline": baseline,
                    "comparison": comparison,
                    "pairs": pair_count,
                    "mean_shortfall_improvement_bps": mean_improvement,
                    "improvement_std_bps": standard_deviation,
                    "standard_error_bps": standard_error,
                    "ci95_lower_bps": lower_bound,
                    "ci95_upper_bps": upper_bound,
                    "win_rate": (improvements > 0).mean(),
                    "mean_completion_delta": (
                        completion_differences.mean()
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["regime", "comparison"])
        .reset_index(drop=True)
    )