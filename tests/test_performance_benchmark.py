"""Tests for the reproducible benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docs.benchmarks.performance import (
    Scenario,
    arguments,
    benchmark,
    percentile,
    serialize_scenarios,
    write_result,
)
from docs.benchmarks.resilience import verify
from promptnest.certify import certify, write_artifacts


def scenario(concurrency: int, throughput: float, max_active: int) -> Scenario:
    values = {"min": 1.0, "p50": throughput, "p95": 1.0, "p99": 1.0, "max": 1.0, "mean": 1.0}
    from docs.benchmarks.performance import ResourceProbe

    return Scenario(
        workload="map",
        concurrency=concurrency,
        samples=100,
        jobs_per_sample=10,
        fragments_per_job=1,
        adapter_calls_per_sample=10,
        total_latency_ms=values,
        throughput_jobs_s=values,
        adapter_call_latency_ms=values,
        orchestration_overhead_ms=values,
        max_active_calls=max_active,
        resources=ResourceProbe(
            jobs=1000,
            peak_traced_memory_bytes=1,
            peak_pending_tasks_above_baseline=1,
        ),
    )


def test_r7_percentile_interpolates() -> None:
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    assert percentile([3.0], 0.99) == 3.0


def test_speedup_uses_first_concurrency_per_workload() -> None:
    result = serialize_scenarios([scenario(1, 100.0, 1), scenario(4, 350.0, 4)])
    assert result[0]["p50_throughput_speedup_vs_first_concurrency"] == 1.0
    assert result[1]["p50_throughput_speedup_vs_first_concurrency"] == 3.5


@pytest.mark.asyncio
async def test_benchmark_schema_overhead_resources_and_concurrency() -> None:
    args = arguments(
        [
            "--jobs",
            "8",
            "--samples",
            "3",
            "--warmups",
            "0",
            "--fragments",
            "2",
            "--resource-jobs",
            "16",
            "--concurrency",
            "2",
            "--workloads",
            "map,nested,full",
            "--adapter-delay-ms",
            "0",
        ]
    )
    result = await benchmark(args)

    assert result["schema_version"] == 2
    assert len(result["scenarios"]) == 3
    assert result["scenarios"][0]["orchestration_overhead_ms"]["p95"] >= 0
    assert result["scenarios"][0]["max_active_calls"] <= 2
    assert result["scenarios"][1]["adapter_calls_per_sample"] == 24
    assert result["scenarios"][2]["adapter_calls_per_sample"] == 25
    assert result["scenarios"][0]["resources"]["peak_traced_memory_bytes"] > 0
    assert result["environment"]["commit_sha"]
    assert result["command"]
    assert result["duration_s"] >= 0


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(SystemExit):
        arguments(["--samples", "0"])
    with pytest.raises(SystemExit):
        arguments(["--workloads", "unknown"])


def test_json_output_is_written(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    rendered = write_result({"schema_version": 2}, output)
    assert json.loads(rendered) == {"schema_version": 2}
    assert json.loads(output.read_text(encoding="utf-8")) == {"schema_version": 2}


@pytest.mark.asyncio
async def test_failure_under_load_scenarios_pass() -> None:
    result = await verify(jobs=12)
    assert result["passed"] is True
    assert {item["name"] for item in result["scenarios"]} == {
        "retry_recovery",
        "timeout_cleanup",
        "partial_failure_retention",
        "caller_cancellation_cleanup",
    }


@pytest.mark.asyncio
async def test_certification_smoke_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "promptnest.certify.git_state",
        lambda: {"commit_sha": "test-sha", "working_tree": "dirty"},
    )
    result = await certify(jobs=200, samples=3, allow_dirty=True)
    assert result["status"] == "PASS-DEVELOPMENT"
    assert result["eligible"] is False
    assert all(gate["passed"] for gate in result["gates"])

    write_artifacts(result, tmp_path)
    assert json.loads((tmp_path / "certificate.json").read_text())["status"] == "PASS-DEVELOPMENT"
    assert (tmp_path / "certificate.md").is_file()
    assert (tmp_path / "certificate.sha256").is_file()
