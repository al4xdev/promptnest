"""Adapter for LangChain chat models supporting structured output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from promptnest.adapters._coerce import coerce_output

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class LangChainAdapter:
    """Invoke ``BaseChatModel.with_structured_output`` asynchronously."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        options = dict(kwargs)
        config = options.pop("config", None)
        model = self.model.bind(**options) if options else self.model
        runnable = model.with_structured_output(output_model)
        result = await runnable.ainvoke(prompt, config=config)
        return coerce_output(result, output_model)
