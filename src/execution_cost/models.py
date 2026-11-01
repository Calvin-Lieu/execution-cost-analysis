from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import pandas as pd


class Side(Enum):
    """Direction of a parent or child order."""

    BUY = 1
    SELL = -1


def require_finite(name: str, value: float) -> None:
    """Reject NaN and positive/negative infinity."""

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class MarketSnapshot:
    """Market conditions observed at one point in time."""

    timestamp: pd.Timestamp
    bid_price: float
    ask_price: float
    market_volume: float
    bid_depth: float
    ask_depth: float
    volatility: float
    regime: str = "normal"

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, pd.Timestamp):
            raise TypeError("timestamp must be a pandas Timestamp")

        numeric_fields = {
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "market_volume": self.market_volume,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "volatility": self.volatility,
        }

        for name, value in numeric_fields.items():
            require_finite(name, value)

        if self.bid_price <= 0 or self.ask_price <= 0:
            raise ValueError("bid and ask prices must be positive")

        if self.ask_price < self.bid_price:
            raise ValueError("ask price cannot be below bid price")

        if self.market_volume < 0:
            raise ValueError("market volume cannot be negative")

        if self.bid_depth < 0 or self.ask_depth < 0:
            raise ValueError("market depth cannot be negative")

        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")

        if not self.regime:
            raise ValueError("regime cannot be empty")


@dataclass(frozen=True)
class ParentOrder:
    """The complete order that an execution strategy must schedule."""

    order_id: str
    side: Side
    quantity: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id cannot be empty")

        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side")

        require_finite("quantity", self.quantity)

        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

        if not isinstance(self.start_time, pd.Timestamp):
            raise TypeError("start_time must be a pandas Timestamp")

        if not isinstance(self.end_time, pd.Timestamp):
            raise TypeError("end_time must be a pandas Timestamp")

        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")


@dataclass(frozen=True)
class ChildOrder:
    """One scheduled portion of a parent order."""

    parent_order_id: str
    timestamp: pd.Timestamp
    quantity: float

    def __post_init__(self) -> None:
        if not self.parent_order_id:
            raise ValueError("parent_order_id cannot be empty")

        if not isinstance(self.timestamp, pd.Timestamp):
            raise TypeError("timestamp must be a pandas Timestamp")

        require_finite("quantity", self.quantity)

        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class Fill:
    """A completed execution of some child-order quantity."""

    parent_order_id: str
    timestamp: pd.Timestamp
    quantity: float
    price: float
    fee: float = 0.0

    def __post_init__(self) -> None:
        if not self.parent_order_id:
            raise ValueError("parent_order_id cannot be empty")

        if not isinstance(self.timestamp, pd.Timestamp):
            raise TypeError("timestamp must be a pandas Timestamp")

        for name, value in {
            "quantity": self.quantity,
            "price": self.price,
            "fee": self.fee,
        }.items():
            require_finite(name, value)

        if self.quantity <= 0:
            raise ValueError("fill quantity must be positive")

        if self.price <= 0:
            raise ValueError("fill price must be positive")

        if self.fee < 0:
            raise ValueError("fee cannot be negative")


@dataclass(frozen=True)
class ExecutionResult:
    """All fills generated while executing one parent order."""

    parent_order: ParentOrder
    fills: tuple[Fill, ...] = ()

    def __post_init__(self) -> None:
        for fill in self.fills:
            if fill.parent_order_id != self.parent_order.order_id:
                raise ValueError("fill belongs to a different parent order")

            if not (
                self.parent_order.start_time
                <= fill.timestamp
                <= self.parent_order.end_time
            ):
                raise ValueError("fill lies outside the execution window")

        tolerance = max(1e-9, self.parent_order.quantity * 1e-12)

        if self.filled_quantity > self.parent_order.quantity + tolerance:
            raise ValueError("fills exceed the parent-order quantity")

    @property
    def filled_quantity(self) -> float:
        return math.fsum(fill.quantity for fill in self.fills)

    @property
    def unfilled_quantity(self) -> float:
        remaining = self.parent_order.quantity - self.filled_quantity
        return max(0.0, remaining)

    @property
    def total_fees(self) -> float:
        return math.fsum(fill.fee for fill in self.fills)

    @property
    def completion_rate(self) -> float:
        return self.filled_quantity / self.parent_order.quantity