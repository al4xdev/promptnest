"""Structural contracts used by the promptnest core."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

ResultModel = TypeVar("ResultModel", bound=BaseModel)


@runtime_checkable
class LLMAdapter(Protocol):
    """A structured asynchronous prompt invocation."""

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel: ...


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Observed provider token usage."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ObservedResult(Generic[ResultModel]):
    """A structured result accompanied by provider usage metadata."""

    value: ResultModel
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class StreamDelta:
    """One non-empty text delta emitted by a streaming adapter."""

    text: str


@dataclass(frozen=True, slots=True)
class StreamCompleted(Generic[ResultModel]):
    """The final validated result and optional provider usage for a stream."""

    value: ResultModel
    usage: TokenUsage | None = None


StreamEvent = StreamDelta | StreamCompleted[ResultModel]


@runtime_checkable
class ObservedLLMAdapter(Protocol):
    """Optional richer adapter protocol used for token reconciliation."""

    async def invoke_observed(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ObservedResult[ResultModel]: ...


@runtime_checkable
class StreamingLLMAdapter(Protocol):
    """Optional adapter protocol exposing text deltas and a validated result."""

    def stream(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent[ResultModel]]: ...
