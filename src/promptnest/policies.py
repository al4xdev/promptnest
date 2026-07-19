"""Execution and retry policies for PromptNest."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

RetryMode = Literal["fixed", "exponential_full_jitter"]


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Bound the number of logical jobs admitted by the runner."""

    workers: int = 32
    queue_capacity: int = 128

    def __post_init__(self) -> None:
        if isinstance(self.workers, bool) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        if isinstance(self.queue_capacity, bool) or self.queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Describe retry classification and delay behavior."""

    max_attempts: int = 3
    timeout_s: float = 300.0
    mode: RetryMode = "exponential_full_jitter"
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    retry_if: Callable[[BaseException], bool] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("retry delays cannot be negative")
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s cannot be less than base_delay_s")

    @classmethod
    def fixed(
        cls,
        *,
        max_attempts: int,
        delay_s: float,
        timeout_s: float,
    ) -> RetryPolicy:
        """Create the legacy retry-all, fixed-delay policy."""
        return cls(
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            mode="fixed",
            base_delay_s=delay_s,
            max_delay_s=delay_s,
            retry_if=lambda error: True,
        )

    def should_retry(self, error: BaseException) -> bool:
        if self.retry_if is not None:
            return self.retry_if(error)
        return isinstance(error, (TimeoutError, ConnectionError, RetryableAdapterError))

    def delay_for(
        self,
        attempt: int,
        *,
        error: BaseException,
        random_source: random.Random,
    ) -> float:
        retry_after = error.retry_after_s if isinstance(error, RetryableAdapterError) else None
        if self.mode == "fixed":
            calculated = self.base_delay_s
        else:
            ceiling = min(
                self.max_delay_s,
                self.base_delay_s * (2 ** max(0, attempt - 1)),
            )
            calculated = random_source.uniform(0, ceiling)
        return max(calculated, retry_after or 0.0)


class RetryableAdapterError(Exception):
    """A normalized transient adapter error, optionally carrying Retry-After."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_s: float | None = None,
    ) -> None:
        if retry_after_s is not None and retry_after_s < 0:
            raise ValueError("retry_after_s cannot be negative")
        self.retry_after_s = retry_after_s
        super().__init__(message)
