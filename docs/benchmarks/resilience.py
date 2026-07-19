"""Deterministic failure-under-load verification for PromptNest."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from promptnest import PromptNest
from promptnest.exceptions import ChunkProcessingError

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class Result(BaseModel):
    value: str


class LoadAdapter:
    def __init__(self, *, fail_first: bool = False, delay_s: float = 0.001) -> None:
        self.fail_first = fail_first
        self.delay_s = delay_s
        self.attempts: defaultdict[str, int] = defaultdict(int)
        self.active = 0
        self.max_active = 0
        self.cancelled = 0

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        del kwargs
        self.attempts[prompt] += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_s)
            if self.fail_first and self.attempts[prompt] == 1:
                raise RuntimeError("deterministic first-attempt failure")
            return output_model.model_validate({"value": prompt})
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


def runner(
    adapter: LoadAdapter,
    jobs: int,
    concurrency: int = 8,
) -> PromptNest[int, Result, BaseModel]:
    return (
        PromptNest.have(adapter, {index: [f"job-{index}"] for index in range(jobs)})
        .set_concurrency(concurrency)
        .set_pre_prompt("{chunk_text}", Result)
    )


async def retries_scenario(jobs: int) -> dict[str, Any]:
    adapter = LoadAdapter(fail_first=True)
    subject = runner(adapter, jobs).set_retry_config(max_attempts=2, delay_s=0, timeout_s=1)
    started = time.perf_counter()
    await subject.get_chunks_result()
    attempts = sum(adapter.attempts.values())
    return {
        "name": "retry_recovery",
        "passed": attempts == jobs * 2 and len(subject.partial_answers) == jobs,
        "duration_ms": round((time.perf_counter() - started) * 1_000, 3),
        "jobs": jobs,
        "adapter_calls": attempts,
        "retries": attempts - jobs,
        "max_active_calls": adapter.max_active,
    }


async def timeout_scenario(jobs: int) -> dict[str, Any]:
    adapter = LoadAdapter(delay_s=0.05)
    subject = runner(adapter, jobs).set_retry_config(max_attempts=1, delay_s=0, timeout_s=0.001)
    started = time.perf_counter()
    failed = False
    try:
        await subject.get_chunks_result()
    except ChunkProcessingError:
        failed = True
    await asyncio.sleep(0)
    return {
        "name": "timeout_cleanup",
        "passed": failed and adapter.active == 0,
        "duration_ms": round((time.perf_counter() - started) * 1_000, 3),
        "jobs": jobs,
        "active_calls_after_failure": adapter.active,
        "cancelled_calls": adapter.cancelled,
    }


async def partial_failure_scenario(jobs: int) -> dict[str, Any]:
    class PartialAdapter(LoadAdapter):
        async def invoke(
            self,
            prompt: str,
            output_model: type[ResultModel],
            **kwargs: Any,
        ) -> ResultModel:
            index = int(prompt.rsplit("-", 1)[1])
            if index % 2:
                raise RuntimeError("deterministic partial failure")
            return await super().invoke(prompt, output_model, **kwargs)

    adapter = PartialAdapter()
    subject = runner(adapter, jobs).set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
    started = time.perf_counter()
    await subject.get_chunks_result(discard_defective_chunks=True)
    expected = (jobs + 1) // 2
    return {
        "name": "partial_failure_retention",
        "passed": len(subject.partial_answers) == expected,
        "duration_ms": round((time.perf_counter() - started) * 1_000, 3),
        "jobs": jobs,
        "successful_jobs": len(subject.partial_answers),
        "failed_jobs": jobs - len(subject.partial_answers),
    }


async def cancellation_scenario(jobs: int) -> dict[str, Any]:
    adapter = LoadAdapter(delay_s=10)
    subject = runner(adapter, jobs).set_retry_config(max_attempts=1, delay_s=0, timeout_s=30)
    task = asyncio.create_task(subject.get_chunks_result())
    while adapter.active == 0:
        await asyncio.sleep(0.001)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    return {
        "name": "caller_cancellation_cleanup",
        "passed": adapter.active == 0 and adapter.cancelled > 0,
        "jobs": jobs,
        "active_calls_after_cancellation": adapter.active,
        "cancelled_calls": adapter.cancelled,
    }


async def verify(jobs: int = 100) -> dict[str, Any]:
    scenarios = [
        await retries_scenario(jobs),
        await timeout_scenario(jobs),
        await partial_failure_scenario(jobs),
        await cancellation_scenario(jobs),
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "passed": all(scenario["passed"] for scenario in scenarios),
        "scenarios": scenarios,
    }


async def main() -> None:
    payload = await verify()
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if len(sys.argv) == 3 and sys.argv[1] == "--output":
        output = Path(sys.argv[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
