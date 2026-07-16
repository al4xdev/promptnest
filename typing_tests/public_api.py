"""Static type contract for the public fluent API."""

from __future__ import annotations

from typing import Any, TypeVar, assert_type

from pydantic import BaseModel

from promptnest import PromptNest
from promptnest.adapters import CallableAdapter

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class Summary(BaseModel):
    summary: str


class Report(BaseModel):
    report: str


async def fake_llm(
    prompt: str,
    output_model: type[ResultModel],
    **kwargs: Any,
) -> ResultModel:
    raise NotImplementedError


async def type_contract() -> None:
    runner = (
        PromptNest.have(CallableAdapter(fake_llm), {"section": ["text"]})
        .set_pre_prompt("{chunk_text}", Summary)
        .set_pos_prompt("{partial_answers}", Report)
    )
    await runner.get_chunks_result()
    assert_type(runner.partial_answers["section"], Summary | list[str])
    assert_type(await runner.run_pos_prompt(), Report)
