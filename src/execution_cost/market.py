from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import pandas as pd

from execution_cost.models import MarketSnapshot


def expected_volume_profile(intervals: int) -> np.ndarray:
    """Return a normalised U-shaped expected volume profile."""

    if intervals <= 0:
        raise ValueError("intervals must be positive")

    positions = np.linspace(-1.0, 1.0, intervals)

    # Higher expected activity near the beginning and end.
    raw_profile = 1.0 + 1.5 * positions**2

    return raw_profile / raw_profile.sum()

class MarketRegime(str, Enum):
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    WIDE_SPREAD = "wide_spread"
    TRENDING = "trending"


@dataclass(frozen=True)
class RegimeParameters:
    interval_volatility: float
    drift: float
    volume_multiplier: float
    depth_multiplier: float
    spread_multiplier: float

REGIME_PARAMETERS = {
    MarketRegime.NORMAL: RegimeParameters(
        interval_volatility=0.0010,
        drift=0.0,
        volume_multiplier=1.0,
        depth_multiplier=1.0,
        spread_multiplier=1.0,
    ),
    MarketRegime.HIGH_VOLATILITY: RegimeParameters(
        interval_volatility=0.0030,
        drift=0.0,
        volume_multiplier=1.3,
        depth_multiplier=0.8,
        spread_multiplier=1.8,
    ),
    MarketRegime.LOW_LIQUIDITY: RegimeParameters(
        interval_volatility=0.0015,
        drift=0.0,
        volume_multiplier=0.35,
        depth_multiplier=0.30,
        spread_multiplier=3.0,
    ),
    MarketRegime.WIDE_SPREAD: RegimeParameters(
        interval_volatility=0.0012,
        drift=0.0,
        volume_multiplier=0.8,
        depth_multiplier=0.7,
        spread_multiplier=4.0,
    ),
    MarketRegime.TRENDING: RegimeParameters(
        interval_volatility=0.0010,
        drift=0.0004,
        volume_multiplier=1.0,
        depth_multiplier=0.9,
        spread_multiplier=1.2,
    ),
}

@dataclass(frozen=True)
class MarketGeneratorConfig:
    start_time: pd.Timestamp
    intervals: int = 78
    interval_minutes: int = 5
    initial_mid_price: float = 100.0
    total_market_volume: float = 7_800_000
    base_depth: float = 50_000
    base_spread_bps: float = 2.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.start_time, pd.Timestamp):
            raise TypeError("start_time must be a pandas Timestamp")

        if self.intervals <= 0:
            raise ValueError("intervals must be positive")

        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")

        positive_values = {
            "initial_mid_price": self.initial_mid_price,
            "total_market_volume": self.total_market_volume,
            "base_depth": self.base_depth,
            "base_spread_bps": self.base_spread_bps,
        }

        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

def generate_market(
    config: MarketGeneratorConfig,
    regime: MarketRegime = MarketRegime.NORMAL,
) -> list[MarketSnapshot]:
    """Generate one reproducible synthetic market session."""

    parameters = REGIME_PARAMETERS[regime]
    rng = np.random.default_rng(config.seed)

    expected_profile = expected_volume_profile(config.intervals)

    # The lognormal distribution keeps the multiplier positive.
    volume_noise = rng.lognormal(
        mean=-0.5 * 0.25**2,
        sigma=0.25,
        size=config.intervals,
    )

    snapshots: list[MarketSnapshot] = []
    mid_price = config.initial_mid_price

    for index in range(config.intervals):
        timestamp = config.start_time + pd.Timedelta(
            minutes=index * config.interval_minutes
        )

        realised_volume = (
            config.total_market_volume
            * expected_profile[index]
            * parameters.volume_multiplier
            * volume_noise[index]
        )

        # Intervals with greater expected activity also receive more depth.
        relative_activity = (
            expected_profile[index] / expected_profile.mean()
        )

        common_depth = (
            config.base_depth
            * parameters.depth_multiplier
            * math.sqrt(relative_activity)
            * rng.lognormal(mean=-0.5 * 0.20**2, sigma=0.20)
        )

        # Introduce modest bid/ask depth imbalance.
        imbalance = rng.uniform(-0.15, 0.15)

        bid_depth = common_depth * (1.0 + imbalance)
        ask_depth = common_depth * (1.0 - imbalance)

        spread_noise = max(0.5, 1.0 + 0.10 * rng.normal())

        spread_bps = (
            config.base_spread_bps
            * parameters.spread_multiplier
            * spread_noise
        )

        spread = mid_price * spread_bps / 10_000

        bid_price = mid_price - spread / 2.0
        ask_price = mid_price + spread / 2.0

        snapshots.append(
            MarketSnapshot(
                timestamp=timestamp,
                bid_price=bid_price,
                ask_price=ask_price,
                market_volume=realised_volume,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                volatility=parameters.interval_volatility,
                regime=regime.value,
            )
        )

        interval_return = (
            parameters.drift
            + parameters.interval_volatility * rng.normal()
        )

        mid_price *= math.exp(interval_return)

    return snapshots