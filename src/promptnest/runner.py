"""Provider-neutral orchestration for nested, structured LLM prompts."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Hashable, Mapping, Sequence
from string import Formatter
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel

from promptnest.exceptions import (
    ChunkFailure,
    ChunkProcessingError,
    ConfigurationError,
    InvocationError,
)
from promptnest.protocols import LLMAdapter

PreResult = TypeVar("PreResult", bound=BaseModel)
PostResult = TypeVar("PostResult", bound=BaseModel)
ResultModel = TypeVar("ResultModel", bound=BaseModel)
ChunkKey = TypeVar("ChunkKey", bound=Hashable)
NewChunkKey = TypeVar("NewChunkKey", bound=Hashable)
NewPreResult = TypeVar("NewPreResult", bound=BaseModel)
NewPostResult = TypeVar("NewPostResult", bound=BaseModel)


class PromptNest(Generic[ChunkKey, PreResult, PostResult]):
    """Run a structured map/consolidate/reduce workflow over text chunks."""

    adapter: LLMAdapter
    processed_text_dict: dict[ChunkKey, tuple[str, ...]]
    partial_answers: dict[ChunkKey, PreResult | list[str]]
    _logger: logging.Logger
    _llm_config: dict[str, Any]
    _max_attempts: int
    _delay_s: float
    _timeout_s: float
    _semaphore: asyncio.Semaphore | None
    _concurrency_limit: int | None
    _pre_template: str | None
    _pre_output_model: type[PreResult] | None
    _pre_input_variables: dict[str, Any]
    _use_key: bool
    _pos_template: str | None
    _pos_output_model: type[PostResult] | None
    _pos_input_variables: dict[str, Any]

    def __init__(
        self,
        adapter: LLMAdapter,
        chunks: dict[ChunkKey, tuple[str, ...]],
        logger: logging.Logger,
    ) -> None:
        self.adapter = adapter
        self.processed_text_dict = chunks
        self._logger = logger
        self._llm_config = {}
        self._max_attempts = 3
        self._delay_s = 1.0
        self._timeout_s = 300.0
        self._semaphore = None
        self._concurrency_limit = None
        self._pre_template = None
        self._pre_output_model = None
        self._pre_input_variables = {}
        self._use_key = False
        self._pos_template = None
        self._pos_output_model = None
        self._pos_input_variables = {}
        self.partial_answers = {}

    @classmethod
    def have(
        cls,
        adapter: LLMAdapter,
        processed_text_dict: Mapping[NewChunkKey, Sequence[str]],
        *,
        logger: logging.Logger | None = None,
    ) -> PromptNest[NewChunkKey, BaseModel, BaseModel]:
        """Create a runner bound to an adapter and a non-empty chunk mapping."""
        if not isinstance(adapter, LLMAdapter):
            raise ConfigurationError("adapter must implement LLMAdapter.invoke()")
        if not processed_text_dict:
            raise ConfigurationError("processed_text_dict cannot be empty")

        chunks: dict[NewChunkKey, tuple[str, ...]] = {}
        for key, fragments in processed_text_dict.items():
            if not fragments:
                raise ConfigurationError(f"chunk {key!r} must contain at least one fragment")
            if any(not isinstance(fragment, str) for fragment in fragments):
                raise ConfigurationError(f"all fragments for chunk {key!r} must be strings")
            chunks[key] = tuple(fragments)

        instance: PromptNest[NewChunkKey, BaseModel, BaseModel] = PromptNest(
            adapter,
            chunks,
            logger or logging.getLogger("promptnest"),
        )
        return instance

    def set_llm_config(self, **options: Any) -> PromptNest[ChunkKey, PreResult, PostResult]:
        """Set options forwarded to the adapter on every invocation."""
        self._llm_config = dict(options)
        return self

    def set_retry_config(
        self,
        *,
        max_attempts: int = 3,
        delay_s: float = 1.0,
        timeout_s: float = 300.0,
    ) -> PromptNest[ChunkKey, PreResult, PostResult]:
        """Configure bounded retries and a timeout for each attempt."""
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ConfigurationError("max_attempts must be at least 1")
        if delay_s < 0:
            raise ConfigurationError("delay_s cannot be negative")
        if timeout_s <= 0:
            raise ConfigurationError("timeout_s must be greater than zero")
        self._max_attempts = max_attempts
        self._delay_s = float(delay_s)
        self._timeout_s = float(timeout_s)
        return self

    def set_concurrency(
        self,
        limit: int | None,
    ) -> PromptNest[ChunkKey, PreResult, PostResult]:
        """Limit concurrent adapter calls; ``None`` keeps them unbounded."""
        if limit is not None and (isinstance(limit, bool) or limit < 1):
            raise ConfigurationError("concurrency limit must be a positive integer or None")
        self._concurrency_limit = limit
        self._semaphore = None
        return self

    def set_pre_prompt(
        self,
        template: str,
        output_model: type[NewPreResult],
        input_variables: Mapping[str, Any] | None = None,
        use_key: bool = False,
    ) -> PromptNest[ChunkKey, NewPreResult, PostResult]:
        """Configure the per-fragment prompt and its structured output."""
        variables = dict(input_variables or {})
        reserved = {"chunk_text", "key_text"} & variables.keys()
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ConfigurationError(f"input_variables cannot override reserved names: {names}")
        allowed = {"chunk_text", *variables}
        if use_key:
            allowed.add("key_text")
        self._validate_template(template, required="chunk_text", allowed=allowed)
        self._validate_output_model(output_model)
        self._pre_template = template
        self._pre_output_model = cast("type[PreResult]", output_model)
        self._pre_input_variables = variables
        self._use_key = use_key
        return cast("PromptNest[ChunkKey, NewPreResult, PostResult]", self)

    def set_pos_prompt(
        self,
        template: str,
        output_model: type[NewPostResult],
        input_variables: Mapping[str, Any] | None = None,
    ) -> PromptNest[ChunkKey, PreResult, NewPostResult]:
        """Configure the final reduce prompt and its structured output."""
        variables = dict(input_variables or {})
        if "partial_answers" in variables:
            raise ConfigurationError(
                "input_variables cannot override reserved name: partial_answers"
            )
        self._validate_template(
            template,
            required="partial_answers",
            allowed={"partial_answers", *variables},
        )
        self._validate_output_model(output_model)
        self._pos_template = template
        self._pos_output_model = cast("type[PostResult]", output_model)
        self._pos_input_variables = variables
        return cast("PromptNest[ChunkKey, PreResult, NewPostResult]", self)

    async def get_chunks_result(
        self,
        is_chain: bool = False,
        discard_defective_chunks: bool = False,
    ) -> PromptNest[ChunkKey, PreResult, PostResult]:
        """Process every chunk and populate :attr:`partial_answers`."""
        self._require_pre_prompt()
        coroutines = [
            self._process_key(key, fragments, is_chain, discard_defective_chunks)
            for key, fragments in self.processed_text_dict.items()
        ]

        if discard_defective_chunks:
            tolerant_results = await asyncio.gather(
                *coroutines,
                return_exceptions=True,
            )
            answers: dict[ChunkKey, PreResult | list[str]] = {}
            failures: list[ChunkFailure] = []
            for key, result in zip(
                self.processed_text_dict,
                tolerant_results,
                strict=True,
            ):
                if isinstance(result, BaseException):
                    failures.extend(self._failures_from_exception(key, result))
                else:
                    answers[key] = result
            if not answers:
                raise ChunkProcessingError("all chunks failed", failures)
            self.partial_answers = answers
            return self

        tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
        try:
            strict_results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        self.partial_answers = dict(
            zip(self.processed_text_dict, strict_results, strict=True)
        )
        return self

    async def run_pos_prompt(self) -> PostResult:
        """Reduce the partial answers into one structured final result."""
        if self._pos_template is None or self._pos_output_model is None:
            raise ConfigurationError("set_pos_prompt() must be called first")
        if not self.partial_answers:
            raise ConfigurationError("get_chunks_result() must be called before run_pos_prompt()")

        payload = {
            str(key): self._serializable_answer(answer)
            for key, answer in self.partial_answers.items()
        }
        variables = {
            "partial_answers": json.dumps(payload, ensure_ascii=False),
            **self._pos_input_variables,
        }
        prompt = self._pos_template.format_map(variables)
        return await self._invoke_with_retry(prompt, self._pos_output_model)

    async def _process_key(
        self,
        key: ChunkKey,
        fragments: tuple[str, ...],
        is_chain: bool,
        discard_defective_chunks: bool,
    ) -> PreResult | list[str]:
        calls = [
            self._invoke_fragment(key, index, fragment)
            for index, fragment in enumerate(fragments)
        ]
        if discard_defective_chunks:
            raw_results = await asyncio.gather(*calls, return_exceptions=True)
            successful: list[PreResult] = []
            failures: list[ChunkFailure] = []
            for index, result in enumerate(raw_results):
                if isinstance(result, BaseException):
                    failures.extend(self._failures_from_exception(key, result, index))
                else:
                    successful.append(result)
            if not successful:
                raise ChunkProcessingError(f"chunk {key!r} failed", failures)
        else:
            tasks = [asyncio.create_task(call) for call in calls]
            try:
                successful = await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if isinstance(exc, ChunkProcessingError):
                    raise
                raise ChunkProcessingError(
                    f"chunk {key!r} failed",
                    self._failures_from_exception(key, exc),
                ) from exc

        final_answer = successful[0]
        if len(successful) > 1:
            consolidated = json.dumps(
                [answer.model_dump(mode="json") for answer in successful],
                ensure_ascii=False,
            )
            try:
                final_answer = await self._invoke_pre_prompt(key, consolidated)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ChunkProcessingError(
                    f"consolidation for chunk {key!r} failed",
                    self._failures_from_exception(key, exc),
                ) from exc

        if is_chain:
            return [final_answer.model_dump_json()]
        return final_answer

    async def _invoke_fragment(
        self,
        key: ChunkKey,
        index: int,
        fragment: str,
    ) -> PreResult:
        try:
            return await self._invoke_pre_prompt(key, fragment)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ChunkProcessingError(
                f"fragment {index} from chunk {key!r} failed",
                self._failures_from_exception(key, exc, index),
            ) from exc

    async def _invoke_pre_prompt(self, key: ChunkKey, text: str) -> PreResult:
        template, output_model = self._require_pre_prompt()
        variables: dict[str, Any] = {
            "chunk_text": text,
            **self._pre_input_variables,
        }
        if self._use_key:
            variables["key_text"] = key
        return await self._invoke_with_retry(template.format_map(variables), output_model)

    async def _invoke_with_retry(
        self,
        prompt: str,
        output_model: type[ResultModel],
    ) -> ResultModel:
        last_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with self._invocation_slot():
                    return await asyncio.wait_for(
                        self.adapter.invoke(
                            prompt,
                            output_model,
                            **self._llm_config,
                        ),
                        timeout=self._timeout_s,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    "promptnest invocation failed",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self._max_attempts,
                        "error": repr(exc),
                    },
                )
                if attempt < self._max_attempts and self._delay_s:
                    await asyncio.sleep(self._delay_s)

        assert last_error is not None
        raise InvocationError(
            f"adapter invocation failed after {self._max_attempts} attempts",
            attempts=self._max_attempts,
            last_error=last_error,
        ) from last_error

    def _invocation_slot(self) -> _InvocationSlot:
        if self._concurrency_limit is not None and self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._concurrency_limit)
        return _InvocationSlot(self._semaphore)

    def _require_pre_prompt(self) -> tuple[str, type[PreResult]]:
        if self._pre_template is None or self._pre_output_model is None:
            raise ConfigurationError("set_pre_prompt() must be called first")
        return self._pre_template, self._pre_output_model

    @staticmethod
    def _validate_output_model(output_model: type[BaseModel]) -> None:
        if not isinstance(output_model, type) or not issubclass(output_model, BaseModel):
            raise ConfigurationError("output_model must be a Pydantic BaseModel subclass")

    @staticmethod
    def _validate_template(template: str, *, required: str, allowed: set[str]) -> None:
        if not isinstance(template, str) or not template.strip():
            raise ConfigurationError("prompt template must be a non-empty string")
        try:
            fields = {
                field_name.split(".", 1)[0].split("[", 1)[0]
                for _, field_name, _, _ in Formatter().parse(template)
                if field_name
            }
        except ValueError as exc:
            raise ConfigurationError(f"invalid prompt template: {exc}") from exc
        if required not in fields:
            raise ConfigurationError(f"prompt template must include {{{required}}}")
        unknown = fields - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ConfigurationError(f"prompt template has unknown variables: {names}")

    @staticmethod
    def _serializable_answer(answer: PreResult | list[str]) -> Any:
        if isinstance(answer, BaseModel):
            return answer.model_dump(mode="json")
        return answer

    @staticmethod
    def _failures_from_exception(
        key: ChunkKey,
        error: BaseException,
        fragment_index: int | None = None,
    ) -> list[ChunkFailure]:
        if isinstance(error, ChunkProcessingError):
            return error.failures
        return [ChunkFailure(key=key, fragment_index=fragment_index, error=error)]


class _InvocationSlot:
    def __init__(self, semaphore: asyncio.Semaphore | None) -> None:
        self._semaphore = semaphore

    async def __aenter__(self) -> None:
        if self._semaphore is not None:
            await self._semaphore.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        if self._semaphore is not None:
            self._semaphore.release()
