"""Deterministic structured-output tests backed by known LLM JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from promptnest import PromptNest
from promptnest.adapters import CallableAdapter

from .conftest import ChunkSummary, FinalReport

ResultModel = TypeVar("ResultModel", bound=BaseModel)
FIXTURE = Path(__file__).parent / "fixtures" / "llm_responses.json"


@pytest.mark.asyncio
async def test_known_llm_json_drives_a_typed_workflow() -> None:
    responses = json.loads(FIXTURE.read_text(encoding="utf-8"))

    async def deterministic_llm(
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> str:
        assert kwargs == {"model": "fixture-model"}
        payload = responses["final"] if output_model is FinalReport else responses["chunk"]
        return json.dumps(payload)

    runner = (
        PromptNest.have(
            CallableAdapter(deterministic_llm),
            {"introduction": ["known input"], "conclusion": ["known input"]},
        )
        .set_llm_config(model="fixture-model")
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=1)
        .set_pre_prompt("Extract structured data from:\n{chunk_text}", ChunkSummary)
        .set_pos_prompt("Merge these typed values:\n{partial_answers}", FinalReport)
    )

    result = await (await runner.get_chunks_result()).run_pos_prompt()

    assert isinstance(runner.partial_answers["introduction"], ChunkSummary)
    assert result == FinalReport.model_validate(responses["final"])
