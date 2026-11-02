from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_mean_shortfall(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    pivot = summary.pivot(
        index="regime",
        columns="strategy",
        values="mean_shortfall_bps",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title("Mean Implementation Shortfall by Market Regime")
    ax.set_xlabel("Market regime")
    ax.set_ylabel("Implementation shortfall (bps)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Strategy")
    ax.tick_params(axis="x", rotation=30)

    figure = ax.get_figure()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_completion_rates(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    pivot = summary.pivot(
        index="regime",
        columns="strategy",
        values="mean_completion_rate",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title("Mean Completion Rate by Market Regime")
    ax.set_xlabel("Market regime")
    ax.set_ylabel("Completion rate")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Strategy")
    ax.tick_params(axis="x", rotation=30)

    figure = ax.get_figure()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_shortfall_distribution(
    trials: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=trials,
        x="regime",
        y="implementation_shortfall_bps",
        hue="strategy",
        ax=axis,
    )

    axis.set_title(
        "Implementation Shortfall Distribution by Market Regime"
    )
    axis.set_xlabel("Market regime")
    axis.set_ylabel("Implementation shortfall (bps)")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

def plot_paired_improvements(
    comparisons: pd.DataFrame,
    output_path: Path,
) -> None:
    preferred_order = [
        "normal",
        "high_volatility",
        "low_liquidity",
        "wide_spread",
        "trending",
    ]

    regimes = [
        regime
        for regime in preferred_order
        if regime in set(comparisons["regime"])
    ]
    strategies = list(comparisons["comparison"].unique())

    x_positions = np.arange(len(regimes))
    offsets = np.linspace(-0.12, 0.12, len(strategies))

    figure, axis = plt.subplots(figsize=(12, 6))

    for offset, strategy in zip(offsets, strategies):
        strategy_results = (
            comparisons[
                comparisons["comparison"] == strategy
            ]
            .set_index("regime")
            .reindex(regimes)
        )

        means = strategy_results[
            "mean_shortfall_improvement_bps"
        ].to_numpy()

        lower_errors = (
            means
            - strategy_results["ci95_lower_bps"].to_numpy()
        )

        upper_errors = (
            strategy_results["ci95_upper_bps"].to_numpy()
            - means
        )

        axis.errorbar(
            x_positions + offset,
            means,
            yerr=np.vstack([lower_errors, upper_errors]),
            fmt="o",
            capsize=5,
            markersize=7,
            label=f"{strategy} vs TWAP",
        )

    axis.axhline(
        0,
        color="black",
        linewidth=1,
        linestyle="--",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(regimes, rotation=30)
    axis.set_title(
        "Paired Shortfall Improvement Relative to TWAP"
    )
    axis.set_xlabel("Market regime")
    axis.set_ylabel(
        "Shortfall improvement (bps, positive is better)"
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)