"""Optional real-provider evidence profile for OpenAI-compatible accounts.

Required environment variables:

    OPENAI_API_KEY
    PROMPTNEST_OPENAI_MODEL

This profile is intentionally separate from the deterministic core certificate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from promptnest import PromptNest, Provider, ProviderPolicy, ProviderPool
from promptnest.adapters import OpenAIAdapter


class ProviderResult(BaseModel):
    value: str


async def source(jobs: int) -> AsyncIterator[tuple[int, list[str]]]:
    for index in range(jobs):
        yield index, [f"Return the identifier provider-job-{index}."]


async def run(
    jobs: int,
    concurrency: int,
    requests_per_second: float,
) -> dict[str, Any]:
    model = os.environ.get("PROMPTNEST_OPENAI_MODEL")
    if not model:
        raise SystemExit("PROMPTNEST_OPENAI_MODEL is required")
    adapter = OpenAIAdapter(AsyncOpenAI(), default_model=model)
    pool: ProviderPool[Any] = ProviderPool(
        {
            "openai": Provider(
                adapter,
                ProviderPolicy(
                    max_concurrency=concurrency,
                    requests_per_second=requests_per_second,
                    request_burst=1,
                ),
            )
        }
    )
    runner = (
        PromptNest.from_async(pool, source(jobs))
        .set_execution_config(workers=concurrency, queue_capacity=max(1, concurrency * 2))
        .set_pre_prompt(
            "{chunk_text}",
            ProviderResult,
        )
    )
    started = time.perf_counter()
    await runner.get_chunks_result()
    duration_s = time.perf_counter() - started
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": "openai",
        "model": model,
        "jobs": jobs,
        "duration_s": duration_s,
        "throughput_jobs_s": jobs / duration_s,
        "result_count": len(runner.partial_answers),
        "provider_metrics": pool.metrics(),
        "execution_metrics": runner.execution_metrics,
        "latency_note": (
            "This profile measures complete structured results, not time to first token."
        ),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=2)
    parser.add_argument("--output", type=Path, default=Path("/tmp/openai-provider.json"))
    args = parser.parse_args()
    if args.jobs < 1 or args.concurrency < 1 or args.requests_per_second <= 0:
        parser.error("jobs/concurrency/rate must be positive")
    payload = await run(args.jobs, args.concurrency, args.requests_per_second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
