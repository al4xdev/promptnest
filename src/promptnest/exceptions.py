"""Public exception hierarchy for promptnest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PromptNestError(Exception):
    """Base class for every promptnest-owned error."""


class ConfigurationError(PromptNestError, ValueError):
    """Raised when the fluent runner is configured inconsistently."""


class InvocationError(PromptNestError):
    """Raised after an adapter call exhausts its retry budget."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: BaseException,
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ChunkFailure:
    """One failed fragment or consolidation operation."""

    key: Any
    fragment_index: int | None
    error: BaseException


class ChunkProcessingError(PromptNestError):
    """Raised when one or more chunks cannot be processed."""

    def __init__(self, message: str, failures: list[ChunkFailure]) -> None:
        self.failures = failures
        super().__init__(message)
