"""Adapter for OpenAI and Azure OpenAI asynchronous SDK clients."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from promptnest.policies import RetryableAdapterError
from promptnest.protocols import ObservedResult, TokenUsage

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
        return (await self.invoke_observed(prompt, output_model, **kwargs)).value

    async def invoke_observed(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ObservedResult[ResultModel]:
        options = dict(kwargs)
        model = options.pop("model", None) or self.default_model
        if not model:
            raise ValueError("OpenAIAdapter requires a model")

        messages = options.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=output_model,
                **options,
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429:
                raise RetryableAdapterError(
                    "OpenAI rate limit exceeded",
                    retry_after_s=_retry_after_seconds(exc),
                ) from exc
            raise
        choice = response.choices[0]
        parsed = choice.message.parsed
        if parsed is None:
            raise ValueError(
                f"OpenAI returned no parsed output (finish_reason={choice.finish_reason!r})"
            )
        usage = getattr(response, "usage", None)
        token_usage = None
        if usage is not None:
            token_usage = TokenUsage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0)),
                output_tokens=int(getattr(usage, "completion_tokens", 0)),
            )
        return ObservedResult(output_model.model_validate(parsed), token_usage)


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None
