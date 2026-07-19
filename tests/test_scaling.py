"""Bounded execution, provider policy, retry and recovery tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from promptnest import (
    ChunkProcessingError,
    PromptNest,
    Provider,
    ProviderPolicy,
    ProviderPool,
    RetryableAdapterError,
    RetryPolicy,
    SQLiteCheckpointStore,
)
from promptnest.providers import InvocationContext

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class Result(BaseModel):
    value: str


class TrackingAdapter:
    def __init__(
        self,
        *,
        delay_s: float = 0,
        fail_consolidation: bool = False,
    ) -> None:
        self.delay_s = delay_s
        self.fail_consolidation = fail_consolidation
        self.prompts: list[str] = []
        self.active = 0
        self.max_active = 0

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        del kwargs
        self.prompts.append(prompt)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.fail_consolidation and prompt.startswith("["):
                raise RuntimeError("consolidation failed")
            return output_model.model_validate({"value": prompt})
        finally:
            self.active -= 1


async def source(jobs: int) -> AsyncIterator[tuple[int, list[str]]]:
    for index in range(jobs):
        yield index, [f"job-{index}"]


@pytest.mark.asyncio
async def test_async_source_applies_real_backpressure() -> None:
    adapter = TrackingAdapter(delay_s=0.002)
    runner = (
        PromptNest.from_async(adapter, source(100))
        .set_execution_config(workers=2, queue_capacity=4)
        .set_concurrency(2)
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
        .set_pre_prompt("{chunk_text}", Result)
    )

    await runner.get_chunks_result()

    assert len(runner.partial_answers) == 100
    assert runner.execution_metrics["queue_high_watermark"] == 4
    assert runner.execution_metrics["admission_waits"] > 0
    assert adapter.max_active == 2


@pytest.mark.asyncio
async def test_provider_pool_routes_with_independent_concurrency() -> None:
    first = TrackingAdapter(delay_s=0.002)
    second = TrackingAdapter(delay_s=0.002)
    pool = ProviderPool(
        {
            "even": Provider(first, ProviderPolicy(max_concurrency=1)),
            "odd": Provider(second, ProviderPolicy(max_concurrency=2)),
        },
        router=lambda context: "even" if int(context.key) % 2 == 0 else "odd",
    )
    runner = (
        PromptNest.from_async(pool, source(20))
        .set_execution_config(workers=8, queue_capacity=4)
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
        .set_pre_prompt("{chunk_text}", Result)
    )

    await runner.get_chunks_result()

    assert first.max_active == 1
    assert second.max_active == 2
    assert pool.provider_name(InvocationContext(key=2, stage="fragment")) == "even"


def test_full_jitter_and_retry_after_are_bounded() -> None:
    policy = RetryPolicy(
        max_attempts=3,
        timeout_s=1,
        base_delay_s=2,
        max_delay_s=8,
    )
    import random

    delay = policy.delay_for(
        2,
        error=RetryableAdapterError("limited", retry_after_s=3),
        random_source=random.Random(1),
    )
    assert 3 <= delay <= 4


@pytest.mark.asyncio
async def test_consolidation_recovery_reuses_fragment_checkpoints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    first_store = SQLiteCheckpointStore(path)
    failing = TrackingAdapter(fail_consolidation=True)
    first = (
        PromptNest.have(failing, {"chapter": ["one", "two"]})
        .set_execution_config(workers=1, queue_capacity=1)
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
        .set_checkpoint_store(first_store, run_id="run", run_revision="v1")
        .set_pre_prompt("{chunk_text}", Result)
    )
    with pytest.raises(ChunkProcessingError):
        await first.get_chunks_result()
    await first_store.close()
    assert failing.prompts == ["one", "two", '[{"value": "one"}, {"value": "two"}]']

    second_store = SQLiteCheckpointStore(path)
    recovered = TrackingAdapter()
    second = (
        PromptNest.have(recovered, {"chapter": ["one", "two"]})
        .set_execution_config(workers=1, queue_capacity=1)
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
        .set_checkpoint_store(second_store, run_id="run", run_revision="v1")
        .set_pre_prompt("{chunk_text}", Result)
    )
    await second.get_chunks_result()
    await second_store.close()

    assert recovered.prompts == ['[{"value": "one"}, {"value": "two"}]']


@pytest.mark.asyncio
async def test_checkpoint_revision_mismatch_is_rejected(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    await store.prepare("run", "v1")
    with pytest.raises(Exception, match="revision"):
        await store.prepare("run", "v2")
    await store.close()
