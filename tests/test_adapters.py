"""Unit tests for official framework adapters without external calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from promptnest.adapters import (
    CallableAdapter,
    CrewAIAdapter,
    LangChainAdapter,
    LangGraphAdapter,
    OpenAIAdapter,
)
from promptnest.policies import RetryableAdapterError
from promptnest.protocols import (
    LLMAdapter,
    StreamCompleted,
    StreamDelta,
    StreamingLLMAdapter,
)

from .conftest import ChunkSummary


def test_all_adapters_satisfy_structural_protocol() -> None:
    async def callback(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    adapters = [
        CallableAdapter(callback),
        CrewAIAdapter(lambda model: object()),
        LangChainAdapter(MagicMock()),
        LangGraphAdapter(MagicMock()),
        OpenAIAdapter(MagicMock(), default_model="model"),
    ]
    assert all(isinstance(adapter, LLMAdapter) for adapter in adapters)
    assert isinstance(adapters[-1], StreamingLLMAdapter)


class FakeOpenAIStream:
    def __init__(self, events: list[Any], completion: Any) -> None:
        self.events = events
        self.completion = completion

    async def __aenter__(self) -> FakeOpenAIStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def __aiter__(self) -> Any:
        async def iterate() -> Any:
            for event in self.events:
                yield event

        return iterate()

    async def get_final_completion(self) -> Any:
        return self.completion


@pytest.mark.asyncio
async def test_callable_adapter_validates_dict_output() -> None:
    async def callback(
        prompt: str,
        output_model: type[ChunkSummary],
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert prompt == "hello"
        assert output_model is ChunkSummary
        assert kwargs == {"trace": True}
        return {"summary": "ok", "keywords": []}

    result = await CallableAdapter(callback).invoke(
        "hello",
        ChunkSummary,
        trace=True,
    )
    assert result.summary == "ok"


@pytest.mark.asyncio
async def test_openai_adapter_uses_parsed_structured_output() -> None:
    parsed = ChunkSummary(summary="ok", keywords=[])
    parse = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=parsed),
                    finish_reason="stop",
                )
            ]
        )
    )
    client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )

    result = await OpenAIAdapter(client, default_model="gpt-test").invoke(
        "hello",
        ChunkSummary,
        temperature=0,
    )

    assert result is parsed
    parse.assert_awaited_once()
    assert parse.await_args.kwargs["model"] == "gpt-test"
    assert parse.await_args.kwargs["temperature"] == 0


@pytest.mark.asyncio
async def test_openai_adapter_exposes_observed_token_usage() -> None:
    parsed = ChunkSummary(summary="ok", keywords=[])
    parse = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=parsed),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )
    )
    client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )

    observed = await OpenAIAdapter(
        client,
        default_model="gpt-test",
    ).invoke_observed("hello", ChunkSummary)

    assert observed.value is parsed
    assert observed.usage is not None
    assert observed.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_openai_adapter_streams_deltas_and_final_structured_output() -> None:
    parsed = ChunkSummary(summary="ok", keywords=[])
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(parsed=parsed),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )
    stream_call = MagicMock(
        return_value=FakeOpenAIStream(
            [
                SimpleNamespace(type="content.delta", delta='{"summary":'),
                SimpleNamespace(type="content.delta", delta='"ok"}'),
            ],
            completion,
        )
    )
    client = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(stream=stream_call))
        )
    )

    events = [
        event
        async for event in OpenAIAdapter(
            client, default_model="gpt-test"
        ).stream("hello", ChunkSummary, temperature=0)
    ]

    assert [event.text for event in events if isinstance(event, StreamDelta)] == [
        '{"summary":',
        '"ok"}',
    ]
    completed = next(event for event in events if isinstance(event, StreamCompleted))
    assert completed.value is parsed
    assert completed.usage is not None
    assert completed.usage.total_tokens == 7
    assert stream_call.call_args.kwargs["response_format"] is ChunkSummary


@pytest.mark.asyncio
async def test_openai_adapter_requires_model() -> None:
    with pytest.raises(ValueError, match="requires a model"):
        await OpenAIAdapter(MagicMock()).invoke("hello", ChunkSummary)


@pytest.mark.asyncio
async def test_openai_adapter_normalizes_rate_limit_retry_after() -> None:
    error = RuntimeError("limited")
    error.status_code = 429  # type: ignore[attr-defined]
    error.response = SimpleNamespace(  # type: ignore[attr-defined]
        headers={"retry-after": "2.5"}
    )
    parse = AsyncMock(side_effect=error)
    client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )

    with pytest.raises(RetryableAdapterError) as captured:
        await OpenAIAdapter(client, default_model="gpt-test").invoke(
            "hello",
            ChunkSummary,
        )

    assert captured.value.retry_after_s == 2.5


@pytest.mark.asyncio
async def test_langchain_adapter_binds_options_and_invokes_structured_model() -> None:
    runnable = SimpleNamespace(ainvoke=AsyncMock(return_value={"summary": "ok", "keywords": []}))
    bound = MagicMock()
    bound.with_structured_output.return_value = runnable
    model = MagicMock()
    model.bind.return_value = bound

    result = await LangChainAdapter(model).invoke(
        "hello",
        ChunkSummary,
        temperature=0,
        config={"tags": ["test"]},
    )

    model.bind.assert_called_once_with(temperature=0)
    bound.with_structured_output.assert_called_once_with(ChunkSummary)
    runnable.ainvoke.assert_awaited_once_with(
        "hello",
        config={"tags": ["test"]},
    )
    assert result.summary == "ok"


@pytest.mark.asyncio
async def test_langgraph_adapter_maps_state_and_selects_output() -> None:
    graph = SimpleNamespace(
        ainvoke=AsyncMock(return_value={"result": {"summary": "ok", "keywords": []}})
    )
    adapter = LangGraphAdapter(
        graph,
        input_builder=lambda prompt, model, options: {
            "question": prompt,
            "schema": model.__name__,
            **options,
        },
        output_selector=lambda result, model: result["result"],
    )

    result = await adapter.invoke(
        "hello",
        ChunkSummary,
        config={"configurable": {"thread_id": "1"}},
        language="en",
    )

    graph.ainvoke.assert_awaited_once_with(
        {"question": "hello", "schema": "ChunkSummary", "language": "en"},
        config={"configurable": {"thread_id": "1"}},
    )
    assert result.summary == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crew_output",
    [
        SimpleNamespace(
            pydantic=ChunkSummary(summary="pydantic", keywords=[]),
            json_dict=None,
            raw="",
        ),
        SimpleNamespace(
            pydantic=None,
            json_dict={"summary": "json", "keywords": []},
            raw="",
        ),
        SimpleNamespace(
            pydantic=None,
            json_dict=None,
            raw='{"summary":"raw","keywords":[]}',
        ),
    ],
)
async def test_crewai_adapter_accepts_supported_output_shapes(crew_output: Any) -> None:
    crew = SimpleNamespace(kickoff_async=AsyncMock(return_value=crew_output))
    adapter = CrewAIAdapter(lambda model: crew)

    result = await adapter.invoke("hello", ChunkSummary, language="en")

    assert result.summary in {"pydantic", "json", "raw"}
    crew.kickoff_async.assert_awaited_once()
    assert crew.kickoff_async.await_args.kwargs["inputs"]["prompt"] == "hello"
