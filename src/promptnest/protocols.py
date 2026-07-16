"""Structural contracts used by the promptnest core."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

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
