"""Shared doubles and Pydantic models for the test suite."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class ChunkSummary(BaseModel):
    summary: str
    keywords: list[str]


class FinalReport(BaseModel):
    full_summary: str
    all_keywords: list[str]


class MockAdapter:
    def __init__(
        self,
        responses: dict[type[BaseModel], Callable[[], BaseModel]] | None = None,
        *,
        fail_n_times: int = 0,
        delay_s: float = 0,
    ) -> None:
        self.responses = responses or {}
        self.fail_n_times = fail_n_times
        self.delay_s = delay_s
        self.call_count = 0
        self.prompts: list[str] = []
        self.options: list[dict[str, Any]] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        self.call_count += 1
        self.prompts.append(prompt)
        self.options.append(kwargs)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.call_count <= self.fail_n_times:
                raise RuntimeError(f"transient failure {self.call_count}")
            factory = self.responses[output_model]
            return output_model.model_validate(factory())
        finally:
            self.active_calls -= 1


class SelectiveFailAdapter(MockAdapter):
    def __init__(
        self,
        fail_on: set[str],
        responses: dict[type[BaseModel], Callable[[], BaseModel]],
    ) -> None:
        super().__init__(responses)
        self.fail_on = fail_on

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        if any(marker in prompt for marker in self.fail_on):
            self.call_count += 1
            self.prompts.append(prompt)
            raise RuntimeError("selected failure")
        return await super().invoke(prompt, output_model, **kwargs)


RESPONSES: dict[type[BaseModel], Callable[[], BaseModel]] = {
    ChunkSummary: lambda: ChunkSummary(summary="summary", keywords=["one"]),
    FinalReport: lambda: FinalReport(
        full_summary="report",
        all_keywords=["one"],
    ),
}


@pytest.fixture
def mock_adapter() -> MockAdapter:
    return MockAdapter(RESPONSES)


@pytest.fixture
def sample_chunks() -> dict[int, list[str]]:
    return {0: ["first"], 1: ["second"], 2: ["third"]}
