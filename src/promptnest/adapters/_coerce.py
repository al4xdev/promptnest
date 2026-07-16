"""Shared structured-output coercion for framework adapters."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

ResultModel = TypeVar("ResultModel", bound=BaseModel)


def coerce_output(value: Any, output_model: type[ResultModel]) -> ResultModel:
    """Validate common framework result shapes as ``output_model``."""
    if isinstance(value, output_model):
        return value
    if isinstance(value, BaseModel):
        return output_model.model_validate(value.model_dump(mode="json"))
    if isinstance(value, str):
        return output_model.model_validate_json(value)
    return output_model.model_validate(value)
