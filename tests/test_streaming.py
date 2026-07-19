"""Streaming contract, TTFT metrics and retry-safety tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from promptnest import (
    ChunkProcessingError,
    InvocationError,
    PromptNest,
    RetryPolicy,
    StreamCompleted,
    StreamDelta,
    StreamUpdate,
)

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class Result(BaseModel):
    value: str


class StreamingAdapter:
    def __init__(self, *, fail_before: bool = False, fail_after: bool = False) -> None:
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.attempts = 0

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        del kwargs
        return output_model.model_validate({"value": prompt})

    async def stream(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta | StreamCompleted[ResultModel]]:
        del kwargs
        self.attempts += 1
        if self.fail_before and self.attempts == 1:
            raise TimeoutError("before first delta")
        await asyncio.sleep(0.002)
        yield StreamDelta("first")
        if self.fail_after:
            raise TimeoutError("after first delta")
        await asyncio.sleep(0.001)
        yield StreamDelta("second")
        yield StreamCompleted(output_model.model_validate({"value": prompt}))


@pytest.mark.asyncio
async def test_streaming_delivers_context_and_records_ttft_distribution() -> None:
    updates: list[StreamUpdate] = []
    runner = (
        PromptNest.have(StreamingAdapter(), {"one": ["hello"]})
        .set_execution_config(workers=1, queue_capacity=1)
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
        .set_streaming(on_delta=updates.append)
        .set_pre_prompt("{chunk_text}", Result)
    )

    await runner.get_chunks_result()

    assert [update.text for update in updates] == ["first", "second"]
    assert all(update.key == "one" and update.stage == "fragment" for update in updates)
    metrics = runner.streaming_metrics
    assert metrics["streams"] == 1
    assert metrics["ttft_ms"]["count"] == 1
    assert metrics["ttft_ms"]["p99"] >= 1
    assert metrics["inter_delta_ms"]["count"] == 1
    assert metrics["observations"][0]["delta_count"] == 2


@pytest.mark.asyncio
async def test_streaming_retries_before_first_delta() -> None:
    adapter = StreamingAdapter(fail_before=True)
    runner = (
        PromptNest.have(adapter, {"one": ["hello"]})
        .set_retry_policy(
            RetryPolicy(max_attempts=2, timeout_s=1, base_delay_s=0, max_delay_s=0)
        )
        .set_streaming()
        .set_pre_prompt("{chunk_text}", Result)
    )

    await runner.get_chunks_result()

    assert adapter.attempts == 2
    assert runner.streaming_metrics["observations"][0]["attempt"] == 2


@pytest.mark.asyncio
async def test_streaming_does_not_retry_after_visible_delta() -> None:
    adapter = StreamingAdapter(fail_after=True)
    updates: list[StreamUpdate] = []
    runner = (
        PromptNest.have(adapter, {"one": ["hello"]})
        .set_retry_policy(
            RetryPolicy(max_attempts=3, timeout_s=1, base_delay_s=0, max_delay_s=0)
        )
        .set_streaming(on_delta=updates.append)
        .set_pre_prompt("{chunk_text}", Result)
    )

    with pytest.raises(ChunkProcessingError) as captured:
        await runner.get_chunks_result()

    assert adapter.attempts == 1
    assert [update.text for update in updates] == ["first"]
    error = captured.value.failures[0].error
    assert isinstance(error, InvocationError)
    assert error.attempts == 1


@pytest.mark.asyncio
async def test_streaming_rejects_non_streaming_adapter() -> None:
    class CompleteOnlyAdapter:
        async def invoke(
            self,
            prompt: str,
            output_model: type[ResultModel],
            **kwargs: Any,
        ) -> ResultModel:
            del kwargs
            return output_model.model_validate({"value": prompt})

    runner = (
        PromptNest.have(CompleteOnlyAdapter(), {"one": ["hello"]})
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
        .set_streaming()
        .set_pre_prompt("{chunk_text}", Result)
    )

    with pytest.raises(ChunkProcessingError, match="fragment 0"):
        await runner.get_chunks_result()
