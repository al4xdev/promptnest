"""Escape-hatch adapter for arbitrary async callables."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from promptnest.adapters._coerce import coerce_output

ResultModel = TypeVar("ResultModel", bound=BaseModel)
AsyncStructuredCallable = Callable[..., Awaitable[Any]]


class CallableAdapter:
    """Adapt an async Python callable to the PromptNest protocol."""

    def __init__(self, function: AsyncStructuredCallable) -> None:
        self.function = function

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        result = await self.function(prompt, output_model, **kwargs)
        return coerce_output(result, output_model)
