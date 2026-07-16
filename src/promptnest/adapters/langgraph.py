"""Adapter for local or remote LangGraph runnables."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from promptnest.adapters._coerce import coerce_output

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class AsyncGraph(Protocol):
    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...


GraphInputBuilder = Callable[[str, type[BaseModel], Mapping[str, Any]], Any]
GraphOutputSelector = Callable[[Any, type[BaseModel]], Any]


def _default_input(
    prompt: str,
    output_model: type[BaseModel],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "output_schema": output_model.model_json_schema(),
        **options,
    }


def _default_output(result: Any, output_model: type[BaseModel]) -> Any:  # noqa: ARG001
    return getattr(result, "value", result)


class LangGraphAdapter:
    """Map PromptNest calls to a compiled graph or ``RemoteGraph``."""

    def __init__(
        self,
        graph: AsyncGraph,
        *,
        input_builder: GraphInputBuilder = _default_input,
        output_selector: GraphOutputSelector = _default_output,
        invoke_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.graph = graph
        self.input_builder = input_builder
        self.output_selector = output_selector
        self.invoke_options = dict(invoke_options or {})

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        options = {**self.invoke_options, **kwargs}
        config = options.pop("config", None)
        graph_input = self.input_builder(prompt, output_model, options)
        result = await self.graph.ainvoke(graph_input, config=config)
        selected = self.output_selector(result, output_model)
        return coerce_output(selected, output_model)
