"""Behavioral tests for the PromptNest fluent workflow."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from promptnest import (
    ChunkProcessingError,
    ConfigurationError,
    InvocationError,
    PromptNest,
)

from .conftest import (
    RESPONSES,
    ChunkSummary,
    FinalReport,
    MockAdapter,
    SelectiveFailAdapter,
)

PRE = "Summarize {chunk_text}"
POST = "Merge {partial_answers}"


@pytest.mark.asyncio
async def test_full_lifecycle_preserves_original_keys(
    mock_adapter: MockAdapter,
    sample_chunks: dict[int, list[str]],
) -> None:
    runner = (
        PromptNest.have(mock_adapter, sample_chunks)
        .set_llm_config(model="test", temperature=0)
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=5)
        .set_pre_prompt(PRE, ChunkSummary)
        .set_pos_prompt(POST, FinalReport)
    )

    result = await (await runner.get_chunks_result()).run_pos_prompt()

    assert isinstance(result, FinalReport)
    assert runner.partial_answers.keys() == sample_chunks.keys()
    assert mock_adapter.call_count == 4
    assert mock_adapter.options[0] == {"model": "test", "temperature": 0}


@pytest.mark.asyncio
async def test_multiple_fragments_are_consolidated_as_json(
    mock_adapter: MockAdapter,
) -> None:
    runner = (
        PromptNest.have(mock_adapter, {"chapter": ["one", "two"]})
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=5)
        .set_pre_prompt(PRE, ChunkSummary)
    )

    await runner.get_chunks_result()

    assert mock_adapter.call_count == 3
    consolidated = mock_adapter.prompts[-1].removeprefix("Summarize ")
    assert json.loads(consolidated) == [
        {"summary": "summary", "keywords": ["one"]},
        {"summary": "summary", "keywords": ["one"]},
    ]


@pytest.mark.asyncio
async def test_chain_mode_returns_model_json(mock_adapter: MockAdapter) -> None:
    runner = (
        PromptNest.have(mock_adapter, {7: ["text"]})
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=5)
        .set_pre_prompt(PRE, ChunkSummary)
    )
    await runner.get_chunks_result(is_chain=True)

    value = runner.partial_answers[7]
    assert isinstance(value, list)
    assert json.loads(value[0])["summary"] == "summary"


@pytest.mark.asyncio
async def test_key_and_custom_variables_are_rendered(mock_adapter: MockAdapter) -> None:
    runner = (
        PromptNest.have(mock_adapter, {"intro": ["text"]})
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=5)
        .set_pre_prompt(
            "{language}:{key_text}:{chunk_text}",
            ChunkSummary,
            {"language": "pt-BR"},
            use_key=True,
        )
    )
    await runner.get_chunks_result()
    assert mock_adapter.prompts == ["pt-BR:intro:text"]


@pytest.mark.asyncio
async def test_partial_failures_can_be_discarded() -> None:
    adapter = SelectiveFailAdapter({"bad"}, RESPONSES)
    runner = (
        PromptNest.have(adapter, {"good": ["ok"], "bad": ["bad"]})
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=5)
        .set_pre_prompt(PRE, ChunkSummary)
    )

    await runner.get_chunks_result(discard_defective_chunks=True)

    assert runner.partial_answers.keys() == {"good"}


@pytest.mark.asyncio
async def test_all_failures_include_chunk_context() -> None:
    adapter = SelectiveFailAdapter({"bad"}, RESPONSES)
    runner = (
        PromptNest.have(adapter, {"a": ["bad"], "b": ["bad"]})
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=5)
        .set_pre_prompt(PRE, ChunkSummary)
    )

    with pytest.raises(ChunkProcessingError) as captured:
        await runner.get_chunks_result(discard_defective_chunks=True)

    assert {failure.key for failure in captured.value.failures} == {"a", "b"}
    assert all(failure.fragment_index == 0 for failure in captured.value.failures)


@pytest.mark.asyncio
async def test_retry_exhaustion_is_retained_as_failure() -> None:
    adapter = MockAdapter(RESPONSES, fail_n_times=10)
    runner = (
        PromptNest.have(adapter, {"key": ["text"]})
        .set_retry_config(max_attempts=2, delay_s=0, timeout_s=5)
        .set_pre_prompt(PRE, ChunkSummary)
    )

    with pytest.raises(ChunkProcessingError) as captured:
        await runner.get_chunks_result()

    error = captured.value.failures[0].error
    assert isinstance(error, InvocationError)
    assert error.attempts == 2
    assert adapter.call_count == 2


@pytest.mark.asyncio
async def test_timeout_is_reported_as_the_invocation_cause() -> None:
    adapter = MockAdapter(RESPONSES, delay_s=0.05)
    runner = (
        PromptNest.have(adapter, {"key": ["text"]})
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=0.001)
        .set_pre_prompt(PRE, ChunkSummary)
    )

    with pytest.raises(ChunkProcessingError) as captured:
        await runner.get_chunks_result()

    invocation = captured.value.failures[0].error
    assert isinstance(invocation, InvocationError)
    assert isinstance(invocation.last_error, TimeoutError)
    assert adapter.active_calls == 0


@pytest.mark.asyncio
async def test_concurrency_limit_applies_to_adapter_calls() -> None:
    adapter = MockAdapter(RESPONSES, delay_s=0.02)
    runner = (
        PromptNest.have(adapter, {index: [str(index)] for index in range(8)})
        .set_concurrency(2)
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=5)
        .set_pre_prompt(PRE, ChunkSummary)
    )

    await runner.get_chunks_result()
    assert adapter.max_active_calls == 2


@pytest.mark.asyncio
async def test_pos_prompt_receives_keyed_json(mock_adapter: MockAdapter) -> None:
    runner = (
        PromptNest.have(mock_adapter, {"a": ["text"]})
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=5)
        .set_pre_prompt(PRE, ChunkSummary)
        .set_pos_prompt(POST, FinalReport)
    )
    await runner.get_chunks_result()
    await runner.run_pos_prompt()

    payload = json.loads(mock_adapter.prompts[-1].removeprefix("Merge "))
    assert payload == {"a": {"summary": "summary", "keywords": ["one"]}}


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda adapter: PromptNest.have(adapter, {}), "cannot be empty"),
        (
            lambda adapter: PromptNest.have(adapter, {"key": []}),
            "at least one fragment",
        ),
    ],
)
def test_invalid_chunk_input_is_rejected(
    mock_adapter: MockAdapter,
    factory: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        factory(mock_adapter)  # type: ignore[operator]


def test_invalid_templates_and_models_are_rejected(mock_adapter: MockAdapter) -> None:
    runner = PromptNest.have(mock_adapter, {"key": ["text"]})

    with pytest.raises(ConfigurationError, match="chunk_text"):
        runner.set_pre_prompt("missing", ChunkSummary)
    with pytest.raises(ConfigurationError, match="unknown"):
        runner.set_pre_prompt("{chunk_text} {mystery}", ChunkSummary)
    with pytest.raises(ConfigurationError, match="BaseModel"):
        runner.set_pre_prompt("{chunk_text}", str)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="partial_answers"):
        runner.set_pos_prompt("missing", FinalReport)
    with pytest.raises(ConfigurationError, match="reserved"):
        runner.set_pre_prompt(
            "{chunk_text}",
            ChunkSummary,
            {"chunk_text": "override"},
        )
    with pytest.raises(ConfigurationError, match="reserved"):
        runner.set_pos_prompt(
            "{partial_answers}",
            FinalReport,
            {"partial_answers": "override"},
        )


@pytest.mark.asyncio
async def test_calls_out_of_order_are_rejected(mock_adapter: MockAdapter) -> None:
    runner = PromptNest.have(mock_adapter, {"key": ["text"]})
    with pytest.raises(ConfigurationError, match="set_pre_prompt"):
        await runner.get_chunks_result()

    runner.set_pos_prompt(POST, FinalReport)
    with pytest.raises(ConfigurationError, match="get_chunks_result"):
        await runner.run_pos_prompt()


@pytest.mark.parametrize(
    "configure",
    [
        lambda runner: runner.set_retry_config(max_attempts=0),
        lambda runner: runner.set_retry_config(delay_s=-1),
        lambda runner: runner.set_retry_config(timeout_s=0),
        lambda runner: runner.set_concurrency(0),
    ],
)
def test_invalid_runtime_configuration_is_rejected(
    mock_adapter: MockAdapter,
    configure: object,
) -> None:
    runner = PromptNest.have(mock_adapter, {"key": ["text"]})
    with pytest.raises(ConfigurationError):
        configure(runner)  # type: ignore[operator]


def test_output_model_must_be_pydantic(mock_adapter: MockAdapter) -> None:
    class NotAModel:
        pass

    runner = PromptNest.have(mock_adapter, {"key": ["text"]})
    with pytest.raises(ConfigurationError):
        runner.set_pre_prompt(PRE, NotAModel)  # type: ignore[arg-type]


def test_pydantic_base_class_is_accepted_by_runtime_check(mock_adapter: MockAdapter) -> None:
    class EmptyModel(BaseModel):
        pass

    PromptNest.have(mock_adapter, {"key": ["text"]}).set_pre_prompt(PRE, EmptyModel)
