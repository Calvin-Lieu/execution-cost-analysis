from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from execution_cost.market import (
    MarketGeneratorConfig,
    MarketRegime,
    expected_volume_profile,
    generate_market,
)


START = pd.Timestamp("2026-01-01 09:30:00", tz="UTC")


def make_config(**changes) -> MarketGeneratorConfig:
    config = MarketGeneratorConfig(
        start_time=START,
        intervals=20,
        interval_minutes=5,
        initial_mid_price=100.0,
        total_market_volume=200_000,
        base_depth=10_000,
        base_spread_bps=2.0,
        seed=42,
    )

    return replace(config, **changes)


def average_spread_bps(snapshots) -> float:
    values = []

    for snapshot in snapshots:
        mid = (snapshot.bid_price + snapshot.ask_price) / 2
        spread = snapshot.ask_price - snapshot.bid_price
        values.append(10_000 * spread / mid)

    return sum(values) / len(values)


def test_expected_volume_profile_sums_to_one() -> None:
    profile = expected_volume_profile(20)

    assert profile.sum() == pytest.approx(1.0)
    assert np.all(profile > 0)


def test_expected_volume_profile_is_u_shaped() -> None:
    profile = expected_volume_profile(21)
    middle = len(profile) // 2

    assert profile[0] > profile[middle]
    assert profile[-1] > profile[middle]


def test_market_generation_is_reproducible() -> None:
    config = make_config()

    first = generate_market(config)
    second = generate_market(config)

    assert first == second


def test_generated_market_satisfies_invariants() -> None:
    config = make_config()
    snapshots = generate_market(config)

    assert len(snapshots) == config.intervals

    for index, snapshot in enumerate(snapshots):
        assert snapshot.bid_price > 0
        assert snapshot.ask_price >= snapshot.bid_price
        assert snapshot.market_volume >= 0
        assert snapshot.bid_depth >= 0
        assert snapshot.ask_depth >= 0

        expected_time = START + pd.Timedelta(
            minutes=index * config.interval_minutes
        )

        assert snapshot.timestamp == expected_time


def test_low_liquidity_has_less_volume_and_depth() -> None:
    config = make_config()

    normal = generate_market(config, MarketRegime.NORMAL)
    low_liquidity = generate_market(
        config,
        MarketRegime.LOW_LIQUIDITY,
    )

    normal_volume = sum(item.market_volume for item in normal)
    low_volume = sum(item.market_volume for item in low_liquidity)

    normal_depth = sum(
        item.bid_depth + item.ask_depth for item in normal
    )
    low_depth = sum(
        item.bid_depth + item.ask_depth for item in low_liquidity
    )

    assert low_volume < normal_volume
    assert low_depth < normal_depth


def test_low_liquidity_has_wider_spreads() -> None:
    config = make_config()

    normal = generate_market(config, MarketRegime.NORMAL)
    low_liquidity = generate_market(
        config,
        MarketRegime.LOW_LIQUIDITY,
    )

    assert average_spread_bps(
        low_liquidity
    ) > average_spread_bps(normal)


def test_high_volatility_has_larger_volatility_parameter() -> None:
    config = make_config()

    normal = generate_market(config, MarketRegime.NORMAL)
    volatile = generate_market(
        config,
        MarketRegime.HIGH_VOLATILITY,
    )

    assert volatile[0].volatility > normal[0].volatility