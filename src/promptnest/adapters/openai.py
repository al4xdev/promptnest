"""Adapter for OpenAI and Azure OpenAI asynchronous SDK clients."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class OpenAIAdapter:
    """Use the SDK structured-output parser with an injected async client."""

    def __init__(
        self,
        client: Any,
        *,
        default_model: str | None = None,
    ) -> None:
        self.client = client
        self.default_model = default_model

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        options = dict(kwargs)
        model = options.pop("model", None) or self.default_model
        if not model:
            raise ValueError("OpenAIAdapter requires a model")

        messages = options.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=output_model,
            **options,
        )
        choice = response.choices[0]
        parsed = choice.message.parsed
        if parsed is None:
            raise ValueError(
                "OpenAI returned no parsed output "
                f"(finish_reason={choice.finish_reason!r})"
            )
        return output_model.model_validate(parsed)
