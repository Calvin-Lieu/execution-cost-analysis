from pathlib import Path

import matplotlib

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