"""Structural contracts used by the promptnest core."""

from __future__ import annotations

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


@runtime_checkable
class ObservedLLMAdapter(Protocol):
    """Optional richer adapter protocol used for token reconciliation."""

    async def invoke_observed(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ObservedResult[ResultModel]: ...
