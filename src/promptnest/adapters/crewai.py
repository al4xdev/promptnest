"""Adapter for CrewAI crews with structured final output."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from promptnest.adapters._coerce import coerce_output

ResultModel = TypeVar("ResultModel", bound=BaseModel)
CrewFactory = Callable[[type[BaseModel]], Any]
CrewInputsBuilder = Callable[[str, type[BaseModel], Mapping[str, Any]], dict[str, Any]]


def _default_inputs(
    prompt: str,
    output_model: type[BaseModel],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "output_schema": output_model.model_json_schema(),
        **options,
    }


class CrewAIAdapter:
    """Create and asynchronously kick off a Crew for each structured call."""

    def __init__(
        self,
        crew_factory: CrewFactory,
        *,
        inputs_builder: CrewInputsBuilder = _default_inputs,
    ) -> None:
        self.crew_factory = crew_factory
        self.inputs_builder = inputs_builder

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        crew = self.crew_factory(output_model)
        inputs = self.inputs_builder(prompt, output_model, kwargs)
        result = await crew.kickoff_async(inputs=inputs)

        pydantic_output = getattr(result, "pydantic", None)
        if pydantic_output is not None:
            return coerce_output(pydantic_output, output_model)
        json_output = getattr(result, "json_dict", None)
        if json_output is not None:
            return coerce_output(json_output, output_model)
        raw_output = getattr(result, "raw", result)
        return coerce_output(raw_output, output_model)
