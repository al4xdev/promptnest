"""Run the reproducible PromptNest core certification profile."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from promptnest.checkpoints import SQLiteCheckpointStore
from promptnest.exceptions import ChunkProcessingError
from promptnest.policies import RetryableAdapterError, RetryPolicy
from promptnest.protocols import ObservedResult, TokenUsage
from promptnest.providers import Provider, ProviderPolicy, ProviderPool
from promptnest.runner import PromptNest

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class CertificationResult(BaseModel):
    value: str


class CertificationAdapter:
    def __init__(
        self,
        *,
        delay_s: float = 0,
        fail_first: bool = False,
        fail_consolidation: bool = False,
    ) -> None:
        self.delay_s = delay_s
        self.fail_first = fail_first
        self.fail_consolidation = fail_consolidation
        self.attempts: defaultdict[str, int] = defaultdict(int)
        self.starts: list[float] = []
        self.active = 0
        self.max_active = 0
        self.cancelled = 0

    async def _call(
        self,
        prompt: str,
        output_model: type[ResultModel],
    ) -> ResultModel:
        self.attempts[prompt] += 1
        self.starts.append(time.monotonic())
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_s)
            if self.fail_first and self.attempts[prompt] == 1:
                raise TimeoutError("deterministic transient failure")
            if self.fail_consolidation and prompt.startswith("["):
                raise RuntimeError("deterministic consolidation failure")
            return output_model.model_validate({"value": prompt})
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        del kwargs
        return await self._call(prompt, output_model)

    async def invoke_observed(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ObservedResult[ResultModel]:
        del kwargs
        value = await self._call(prompt, output_model)
        return ObservedResult(value, TokenUsage(input_tokens=1, output_tokens=0))


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    observed: Any
    requirement: str


async def lazy_source(
    jobs: int,
    *,
    fragments: int,
) -> AsyncIterator[tuple[int, list[str]]]:
    for key in range(jobs):
        yield key, [f"job-{key}-fragment-{index}" for index in range(fragments)]


def routed_pool(
    even: CertificationAdapter,
    odd: CertificationAdapter,
    *,
    even_concurrency: int = 7,
    odd_concurrency: int = 11,
) -> ProviderPool[Any]:
    return ProviderPool(
        {
            "even": Provider(
                even,
                ProviderPolicy(max_concurrency=even_concurrency),
            ),
            "odd": Provider(
                odd,
                ProviderPolicy(max_concurrency=odd_concurrency),
            ),
        },
        router=lambda context: (
            "even" if context.key == "__reduce__" or int(context.key) % 2 == 0 else "odd"
        ),
    )


async def core_profile(jobs: int) -> tuple[list[Gate], dict[str, Any]]:
    even = CertificationAdapter()
    odd = CertificationAdapter()
    pool = routed_pool(even, odd)
    runner = (
        PromptNest.from_async(pool, lazy_source(jobs, fragments=2))
        .set_execution_config(workers=32, queue_capacity=128)
        .set_retry_policy(RetryPolicy(max_attempts=1, timeout_s=5))
        .set_pre_prompt("{chunk_text}", CertificationResult)
        .set_pos_prompt("{partial_answers}", CertificationResult)
    )

    baseline_tasks = len(asyncio.all_tasks())
    peak_tasks = 0
    tracemalloc.start()
    started = time.perf_counter()
    task = asyncio.create_task(runner.get_chunks_result())
    while not task.done():
        peak_tasks = max(peak_tasks, len(asyncio.all_tasks()) - baseline_tasks)
        await asyncio.sleep(0.001)
    await task
    await runner.run_pos_prompt()
    duration_s = time.perf_counter() - started
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    unique_results = len(runner.partial_answers)
    metrics = runner.execution_metrics
    gates = [
        Gate("core_completed", unique_results == jobs, unique_results, f"{jobs} unique results"),
        Gate("duration", duration_s < 30, round(duration_s, 3), "< 30 seconds"),
        Gate(
            "python_memory",
            peak_memory < 128 * 1024 * 1024,
            peak_memory,
            "< 128 MiB",
        ),
        Gate("pending_tasks", peak_tasks <= 40, peak_tasks, "<= 40 above baseline"),
        Gate(
            "queue_bound",
            metrics["queue_high_watermark"] <= 128,
            metrics["queue_high_watermark"],
            "<= 128",
        ),
        Gate(
            "backpressure",
            metrics["admission_waits"] > 0,
            metrics["admission_waits"],
            "> 0 producer waits",
        ),
        Gate("even_concurrency", even.max_active <= 7, even.max_active, "<= 7"),
        Gate("odd_concurrency", odd.max_active <= 11, odd.max_active, "<= 11"),
    ]
    return gates, {
        "duration_s": duration_s,
        "peak_traced_memory_bytes": peak_memory,
        "peak_pending_tasks_above_baseline": peak_tasks,
        "execution": metrics,
    }


async def rate_limit_profile() -> tuple[list[Gate], dict[str, Any]]:
    even = CertificationAdapter()
    odd = CertificationAdapter()
    policy = ProviderPolicy(
        max_concurrency=4,
        requests_per_second=1000,
        request_burst=1,
        tokens_per_second=100,
        token_burst=1,
        token_estimator=lambda prompt, model, options: 1,
    )
    pool: ProviderPool[Any] = ProviderPool(
        {"even": Provider(even, policy), "odd": Provider(odd, policy)},
        router=lambda context: "even" if int(context.key) % 2 == 0 else "odd",
    )
    runner = (
        PromptNest.from_async(pool, lazy_source(20, fragments=1))
        .set_execution_config(workers=8, queue_capacity=4)
        .set_retry_policy(RetryPolicy(max_attempts=1, timeout_s=2))
        .set_pre_prompt("{chunk_text}", CertificationResult)
    )
    await runner.get_chunks_result()
    gaps = [
        later - earlier
        for starts in (even.starts, odd.starts)
        for earlier, later in zip(starts, starts[1:], strict=False)
    ]
    minimum_gap = min(gaps)
    return [
        Gate("request_rate", minimum_gap >= 0.009, minimum_gap, ">= 9 ms"),
        Gate(
            "token_budget",
            all(value["token_wait_s"] > 0 for value in pool.metrics().values()),
            pool.metrics(),
            "token waits observed for both providers",
        ),
    ], {"minimum_start_gap_s": minimum_gap, "provider_metrics": pool.metrics()}


async def retry_profile() -> tuple[list[Gate], dict[str, Any]]:
    adapter = CertificationAdapter(fail_first=True)
    runner = (
        PromptNest.from_async(adapter, lazy_source(100, fragments=1))
        .set_execution_config(workers=16, queue_capacity=16)
        .set_retry_policy(
            RetryPolicy(
                max_attempts=2,
                timeout_s=1,
                base_delay_s=0,
                max_delay_s=0,
            ),
            random_seed=1,
        )
        .set_pre_prompt("{chunk_text}", CertificationResult)
    )
    await runner.get_chunks_result()
    attempts = sum(adapter.attempts.values())
    import random

    jitter_policy = RetryPolicy(
        max_attempts=4,
        timeout_s=1,
        base_delay_s=1,
        max_delay_s=8,
    )
    source = random.Random(7)
    jitter_delays = [
        jitter_policy.delay_for(
            attempt,
            error=TimeoutError(),
            random_source=source,
        )
        for attempt in range(1, 4)
    ]
    retry_after_delay = jitter_policy.delay_for(
        1,
        error=RetryableAdapterError("limited", retry_after_s=3),
        random_source=source,
    )
    return [
        Gate("retry_recovery", attempts == 200, attempts, "exactly 200 attempts"),
        Gate(
            "full_jitter",
            all(0 <= delay <= 2**index for index, delay in enumerate(jitter_delays)),
            jitter_delays,
            "each delay inside its exponential window",
        ),
        Gate(
            "retry_after",
            retry_after_delay >= 3,
            retry_after_delay,
            ">= 3 seconds",
        ),
    ], {
        "attempts": attempts,
        "jitter_delays_s": jitter_delays,
        "retry_after_delay_s": retry_after_delay,
    }


async def cancellation_profile() -> tuple[list[Gate], dict[str, Any]]:
    adapter = CertificationAdapter(delay_s=10)
    runner = (
        PromptNest.from_async(adapter, lazy_source(1000, fragments=1))
        .set_execution_config(workers=32, queue_capacity=128)
        .set_retry_policy(RetryPolicy(max_attempts=1, timeout_s=30))
        .set_pre_prompt("{chunk_text}", CertificationResult)
    )
    task = asyncio.create_task(runner.get_chunks_result())
    while adapter.active == 0:
        await asyncio.sleep(0.001)
    started = time.perf_counter()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    cleanup_s = time.perf_counter() - started
    return [
        Gate(
            "cancellation_cleanup",
            adapter.active == 0 and cleanup_s < 1,
            {"active": adapter.active, "cleanup_s": cleanup_s},
            "zero active calls in < 1 second",
        )
    ], {"cancelled_calls": adapter.cancelled, "cleanup_s": cleanup_s}


async def _checkpoint_child(path: Path, *, fail: bool) -> dict[str, Any]:
    logging.getLogger("promptnest").setLevel(logging.CRITICAL)
    store = SQLiteCheckpointStore(path)
    adapter = CertificationAdapter(fail_consolidation=fail)
    runner = (
        PromptNest.have(adapter, {"key": ["one", "two"]})
        .set_execution_config(workers=1, queue_capacity=1)
        .set_retry_policy(RetryPolicy(max_attempts=1, timeout_s=1))
        .set_checkpoint_store(store, run_id="cert", run_revision="v1")
        .set_pre_prompt("{chunk_text}", CertificationResult)
    )
    failed = False
    try:
        await runner.get_chunks_result()
    except ChunkProcessingError:
        failed = True
    await store.close()
    return {"failed": failed, "calls": sum(adapter.attempts.values())}


async def checkpoint_profile() -> tuple[list[Gate], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="promptnest-cert-") as directory:
        path = Path(directory) / "checkpoints.sqlite3"

        async def run_child(fail: bool) -> dict[str, Any]:
            command = [
                sys.executable,
                "-m",
                "promptnest.certify",
                "--checkpoint-child",
                str(path),
            ]
            if fail:
                command.append("--checkpoint-fail")
            process = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            if process.returncode != 0:
                raise RuntimeError(f"checkpoint child failed: {process.stderr}")
            return cast("dict[str, Any]", json.loads(process.stdout))

        first = await run_child(True)
        recovered = await run_child(False)
        calls = int(recovered["calls"])
    return [
        Gate(
            "checkpoint_resume",
            first == {"failed": True, "calls": 3} and calls == 1,
            {"first_process": first, "second_process": recovered},
            "new process performs one consolidation call only",
        ),
    ], {
        "first_process": first,
        "second_process": recovered,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def latency_profile(samples: int) -> tuple[list[Gate], dict[str, Any]]:
    durations: list[float] = []
    for _ in range(samples):
        adapter = CertificationAdapter()
        runner = (
            PromptNest.from_async(adapter, lazy_source(100, fragments=1))
            .set_execution_config(workers=32, queue_capacity=128)
            .set_retry_policy(RetryPolicy(max_attempts=1, timeout_s=1))
            .set_pre_prompt("{chunk_text}", CertificationResult)
        )
        started = time.perf_counter()
        await runner.get_chunks_result()
        durations.append(time.perf_counter() - started)
    p95_per_job_ms = percentile(durations, 0.95) * 1000 / 100
    p99_batch_ms = percentile(durations, 0.99) * 1000
    return [
        Gate(
            "overhead_p95",
            p95_per_job_ms < 0.5,
            p95_per_job_ms,
            "< 0.5 ms/job",
        ),
        Gate("batch_p99", p99_batch_ms < 500, p99_batch_ms, "< 500 ms"),
    ], {
        "samples": samples,
        "p50_batch_ms": statistics.median(durations) * 1000,
        "p95_overhead_ms_per_job": p95_per_job_ms,
        "p99_batch_ms": p99_batch_ms,
    }


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=False,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit_sha": git("rev-parse", "HEAD"),
        "working_tree": None if status is None else ("clean" if not status else "dirty"),
    }


async def certify(
    *,
    jobs: int = 10_000,
    samples: int = 100,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    logging.getLogger("promptnest").setLevel(logging.CRITICAL)
    started = time.perf_counter()
    state = git_state()
    profiles: dict[str, Any] = {}
    gates: list[Gate] = []
    for name, coroutine in (
        ("core_10k", core_profile(jobs)),
        ("rate_limits", rate_limit_profile()),
        ("retries", retry_profile()),
        ("cancellation", cancellation_profile()),
        ("checkpoint_recovery", checkpoint_profile()),
        ("latency", latency_profile(samples)),
    ):
        profile_gates, metrics = await coroutine
        profiles[name] = metrics
        gates.extend(profile_gates)
    clean = state["working_tree"] == "clean"
    gates_passed = all(gate.passed for gate in gates)
    if clean and gates_passed:
        status = "PASS"
    elif allow_dirty and gates_passed:
        status = "PASS-DEVELOPMENT"
    else:
        status = "FAIL"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "eligible": clean and gates_passed,
        "dirty_override": allow_dirty and not clean,
        "duration_s": round(time.perf_counter() - started, 3),
        "environment": {
            **state,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "configuration": {"jobs": jobs, "latency_samples": samples},
        "gates": [gate.__dict__ for gate in gates],
        "profiles": profiles,
        "claim_boundary": {
            "covered": [
                "bounded backpressure",
                "per-provider concurrency and rate limits",
                "token budget reconciliation",
                "structured cancellation",
                "retry recovery",
                "idempotent checkpoint resume",
            ],
            "excluded": [
                "TTFT",
                "real-provider latency",
                "exactly-once external calls",
                "independent third-party certification",
            ],
        },
    }


def markdown_report(payload: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {gate['name']} | {'PASS' if gate['passed'] else 'FAIL'} | "
        f"`{gate['observed']}` | {gate['requirement']} |"
        for gate in payload["gates"]
    )
    return (
        "# PromptNest core certification\n\n"
        f"**Status:** {payload['status']}  \n"
        f"**Commit:** `{payload['environment']['commit_sha']}`  \n"
        f"**Working tree:** {payload['environment']['working_tree']}  \n"
        f"**Generated:** {payload['generated_at']}  \n\n"
        "| Gate | Status | Observed | Requirement |\n"
        "|---|---|---|---|\n"
        f"{rows}\n\n"
        "This is a reproducible project self-assessment, not a third-party certification.\n"
    )


def write_artifacts(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2, default=str) + "\n"
    markdown = markdown_report(payload)
    (output_dir / "certificate.json").write_text(json_text, encoding="utf-8")
    (output_dir / "certificate.md").write_text(markdown, encoding="utf-8")
    checksum = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    (output_dir / "certificate.sha256").write_text(
        f"{checksum}  certificate.json\n",
        encoding="utf-8",
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=10_000)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/promptnest-certificate"))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--checkpoint-child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-fail", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs < 1 or args.samples < 1:
        parser.error("jobs and samples must be positive")
    return args


async def main() -> None:
    args = arguments()
    if args.checkpoint_child is not None:
        result = await _checkpoint_child(
            args.checkpoint_child,
            fail=args.checkpoint_fail,
        )
        print(json.dumps(result))
        return
    payload = await certify(
        jobs=args.jobs,
        samples=args.samples,
        allow_dirty=args.allow_dirty,
    )
    write_artifacts(payload, args.output_dir)
    print(markdown_report(payload))
    if payload["status"] == "FAIL":
        raise SystemExit(1)


def cli() -> None:
    """Synchronous console-script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
