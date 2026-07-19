"""Provider-neutral orchestration for nested, structured LLM prompts."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterable, Hashable, Mapping, Sequence
from string import Formatter
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel

from promptnest.checkpoints import CheckpointStore, canonical_job_id
from promptnest.exceptions import (
    ChunkFailure,
    ChunkProcessingError,
    ConfigurationError,
    InvocationError,
)
from promptnest.policies import ExecutionConfig, RetryPolicy
from promptnest.protocols import LLMAdapter
from promptnest.providers import (
    InvocationContext,
    ProviderObservation,
    ProviderPolicy,
    ProviderPool,
)

PreResult = TypeVar("PreResult", bound=BaseModel)
PostResult = TypeVar("PostResult", bound=BaseModel)
ResultModel = TypeVar("ResultModel", bound=BaseModel)
ChunkKey = TypeVar("ChunkKey", bound=Hashable)
NewChunkKey = TypeVar("NewChunkKey", bound=Hashable)
NewPreResult = TypeVar("NewPreResult", bound=BaseModel)
NewPostResult = TypeVar("NewPostResult", bound=BaseModel)

ChunkSource = AsyncIterable[tuple[ChunkKey, Sequence[str]]]
_STOP = object()


class PromptNest(Generic[ChunkKey, PreResult, PostResult]):
    """Run a structured map/consolidate/reduce workflow over text chunks."""

    def __init__(
        self,
        adapter_or_pool: LLMAdapter | ProviderPool[Any],
        chunks: dict[ChunkKey, tuple[str, ...]],
        logger: logging.Logger,
        *,
        source: ChunkSource[ChunkKey] | None = None,
    ) -> None:
        if isinstance(adapter_or_pool, ProviderPool):
            self.adapter: Any = adapter_or_pool
            self._provider_pool = adapter_or_pool
            self._custom_pool = True
        else:
            self.adapter = adapter_or_pool
            self._provider_pool = ProviderPool.single(adapter_or_pool)
            self._custom_pool = False
        self.processed_text_dict = chunks
        self.partial_answers: dict[ChunkKey, PreResult | list[str]] = {}
        self.execution_metrics: dict[str, Any] = {}
        self._source = source
        self._source_consumed = False
        self._logger = logger
        self._llm_config: dict[str, Any] = {}
        self._retry_policy = RetryPolicy.fixed(
            max_attempts=3,
            delay_s=1.0,
            timeout_s=300.0,
        )
        self._random = random.Random()
        self._execution = ExecutionConfig()
        self._concurrency_limit: int | None = None
        self._pre_template: str | None = None
        self._pre_output_model: type[PreResult] | None = None
        self._pre_input_variables: dict[str, Any] = {}
        self._use_key = False
        self._pos_template: str | None = None
        self._pos_output_model: type[PostResult] | None = None
        self._pos_input_variables: dict[str, Any] = {}
        self._checkpoint_store: CheckpointStore | None = None
        self._checkpoint_run_id: str | None = None
        self._checkpoint_revision: str | None = None

    @classmethod
    def have(
        cls,
        adapter: LLMAdapter | ProviderPool[Any],
        processed_text_dict: Mapping[NewChunkKey, Sequence[str]],
        *,
        logger: logging.Logger | None = None,
    ) -> PromptNest[NewChunkKey, BaseModel, BaseModel]:
        """Create a runner from a non-empty materialized mapping."""
        cls._validate_adapter(adapter)
        if not processed_text_dict:
            raise ConfigurationError("processed_text_dict cannot be empty")
        chunks = {
            key: cls._validate_fragments(key, fragments)
            for key, fragments in processed_text_dict.items()
        }
        return PromptNest(
            adapter,
            chunks,
            logger or logging.getLogger("promptnest"),
        )

    @classmethod
    def from_async(
        cls,
        adapter: LLMAdapter | ProviderPool[Any],
        source: AsyncIterable[tuple[NewChunkKey, Sequence[str]]],
        *,
        logger: logging.Logger | None = None,
    ) -> PromptNest[NewChunkKey, BaseModel, BaseModel]:
        """Create a runner from a lazily consumed asynchronous source."""
        cls._validate_adapter(adapter)
        if not isinstance(source, AsyncIterable):
            raise ConfigurationError("source must be an AsyncIterable")
        return PromptNest(
            adapter,
            {},
            logger or logging.getLogger("promptnest"),
            source=source,
        )

    @staticmethod
    def _validate_adapter(adapter: object) -> None:
        if not isinstance(adapter, (LLMAdapter, ProviderPool)):
            raise ConfigurationError(
                "adapter must implement LLMAdapter.invoke() or be a ProviderPool"
            )

    @staticmethod
    def _validate_fragments(key: object, fragments: Sequence[str]) -> tuple[str, ...]:
        if not fragments:
            raise ConfigurationError(f"chunk {key!r} must contain at least one fragment")
        if any(not isinstance(fragment, str) for fragment in fragments):
            raise ConfigurationError(f"all fragments for chunk {key!r} must be strings")
        return tuple(fragments)

    def set_llm_config(self, **options: Any) -> PromptNest[ChunkKey, PreResult, PostResult]:
        self._llm_config = dict(options)
        return self

    def set_retry_config(
        self,
        *,
        max_attempts: int = 3,
        delay_s: float = 1.0,
        timeout_s: float = 300.0,
    ) -> PromptNest[ChunkKey, PreResult, PostResult]:
        try:
            self._retry_policy = RetryPolicy.fixed(
                max_attempts=max_attempts,
                delay_s=delay_s,
                timeout_s=timeout_s,
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        return self

    def set_retry_policy(
        self,
        policy: RetryPolicy,
        *,
        random_seed: int | None = None,
    ) -> PromptNest[ChunkKey, PreResult, PostResult]:
        if not isinstance(policy, RetryPolicy):
            raise ConfigurationError("policy must be a RetryPolicy")
        self._retry_policy = policy
        self._random = random.Random(random_seed)
        return self

    def set_execution_config(
        self,
        *,
        workers: int = 32,
        queue_capacity: int = 128,
    ) -> PromptNest[ChunkKey, PreResult, PostResult]:
        try:
            self._execution = ExecutionConfig(workers, queue_capacity)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if not self._custom_pool and self._concurrency_limit is None:
            self._provider_pool = ProviderPool.single(
                cast("LLMAdapter", self.adapter),
                policy=ProviderPolicy(max_concurrency=workers),
            )
        return self

    def set_concurrency(
        self,
        limit: int | None,
    ) -> PromptNest[ChunkKey, PreResult, PostResult]:
        if limit is not None and (isinstance(limit, bool) or limit < 1):
            raise ConfigurationError("concurrency limit must be a positive integer or None")
        if self._custom_pool:
            raise ConfigurationError(
                "set_concurrency() cannot override a ProviderPool; configure ProviderPolicy"
            )
        self._concurrency_limit = limit
        self._provider_pool = ProviderPool.single(
            cast("LLMAdapter", self.adapter),
            policy=ProviderPolicy(max_concurrency=limit or self._execution.workers),
        )
        return self

    def set_checkpoint_store(
        self,
        store: CheckpointStore,
        *,
        run_id: str,
        run_revision: str,
    ) -> PromptNest[ChunkKey, PreResult, PostResult]:
        if not isinstance(store, CheckpointStore):
            raise ConfigurationError("store must implement CheckpointStore")
        if not run_id or not run_revision:
            raise ConfigurationError("run_id and run_revision must be non-empty")
        self._checkpoint_store = store
        self._checkpoint_run_id = run_id
        self._checkpoint_revision = run_revision
        return self

    def set_pre_prompt(
        self,
        template: str,
        output_model: type[NewPreResult],
        input_variables: Mapping[str, Any] | None = None,
        use_key: bool = False,
    ) -> PromptNest[ChunkKey, NewPreResult, PostResult]:
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
        self._require_pre_prompt()
        if self._source_consumed:
            raise ConfigurationError("get_chunks_result() can only be called once per runner")
        self._source_consumed = True
        await self._prepare_checkpoint()

        queue: asyncio.Queue[object] = asyncio.Queue(self._execution.queue_capacity)
        answers: dict[ChunkKey, PreResult | list[str]] = {}
        failures: list[ChunkFailure] = []
        seen: set[ChunkKey] = set()
        admission_waits = 0
        queue_high_watermark = 0
        started = time.perf_counter()

        async def producer() -> None:
            nonlocal admission_waits, queue_high_watermark
            produced = 0
            async for key, fragments in self._iter_source():
                if key in seen:
                    raise ConfigurationError(f"duplicate chunk key {key!r}")
                seen.add(key)
                validated = self._validate_fragments(key, fragments)
                self.processed_text_dict[key] = validated
                if queue.full():
                    admission_waits += 1
                await queue.put((key, validated))
                produced += 1
                queue_high_watermark = max(queue_high_watermark, queue.qsize())
            if produced == 0:
                raise ConfigurationError("source cannot be empty")
            for _ in range(self._execution.workers):
                await queue.put(_STOP)

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is _STOP:
                        return
                    key, fragments = cast(
                        "tuple[ChunkKey, tuple[str, ...]]",
                        item,
                    )
                    try:
                        answers[key] = await self._process_key(
                            key,
                            fragments,
                            is_chain,
                            discard_defective_chunks,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if not discard_defective_chunks:
                            raise
                        failures.extend(self._failures_from_exception(key, exc))
                finally:
                    queue.task_done()

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(producer())
                for _ in range(self._execution.workers):
                    group.create_task(worker())
        except BaseExceptionGroup as group:
            error = self._find_group_error(group)
            if error is not None:
                raise error from None
            raise

        if not answers:
            raise ChunkProcessingError("all chunks failed", failures)
        self.partial_answers = {
            key: answers[key] for key in self.processed_text_dict if key in answers
        }
        self.execution_metrics = {
            "jobs": len(seen),
            "workers": self._execution.workers,
            "queue_capacity": self._execution.queue_capacity,
            "queue_high_watermark": queue_high_watermark,
            "admission_waits": admission_waits,
            "duration_s": time.perf_counter() - started,
            "provider_metrics": self._provider_pool.metrics(),
        }
        return self

    async def _iter_source(self) -> AsyncIterable[tuple[ChunkKey, Sequence[str]]]:
        if self._source is not None:
            async for item in self._source:
                yield item
            return
        for item in self.processed_text_dict.items():
            yield item

    async def run_pos_prompt(self) -> PostResult:
        if self._pos_template is None or self._pos_output_model is None:
            raise ConfigurationError("set_pos_prompt() must be called first")
        if not self.partial_answers:
            raise ConfigurationError("get_chunks_result() must be called before run_pos_prompt()")

        cached = await self._load_checkpoint("__reduce__", "reduce", self._pos_output_model)
        if cached is not None:
            return cached
        payload = {
            str(key): self._serializable_answer(answer)
            for key, answer in self.partial_answers.items()
        }
        variables = {
            "partial_answers": json.dumps(payload, ensure_ascii=False),
            **self._pos_input_variables,
        }
        prompt = self._pos_template.format_map(variables)
        result, observation = await self._invoke_with_retry(
            InvocationContext(key="__reduce__", stage="reduce"),
            prompt,
            self._pos_output_model,
        )
        await self._save_checkpoint(
            "__reduce__",
            "reduce",
            result,
            observation=observation,
        )
        return result

    async def _process_key(
        self,
        key: ChunkKey,
        fragments: tuple[str, ...],
        is_chain: bool,
        discard_defective_chunks: bool,
    ) -> PreResult | list[str]:
        output_model = self._require_pre_prompt()[1]
        successful: list[PreResult] = []
        failures: list[ChunkFailure] = []
        for index, fragment in enumerate(fragments):
            cached = await self._load_checkpoint(
                canonical_job_id(key),
                "fragment",
                output_model,
                fragment_index=index,
            )
            if cached is not None:
                successful.append(cached)
                continue
            try:
                result, observation = await self._invoke_pre_prompt(
                    key,
                    fragment,
                    stage="fragment",
                    fragment_index=index,
                )
                await self._save_checkpoint(
                    canonical_job_id(key),
                    "fragment",
                    result,
                    fragment_index=index,
                    observation=observation,
                )
                successful.append(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures.extend(self._failures_from_exception(key, exc, index))
                if not discard_defective_chunks:
                    raise ChunkProcessingError(
                        f"fragment {index} from chunk {key!r} failed",
                        failures,
                    ) from exc

        if not successful:
            raise ChunkProcessingError(f"chunk {key!r} failed", failures)

        final_answer = successful[0]
        if len(successful) > 1:
            job_id = canonical_job_id(key)
            cached = await self._load_checkpoint(
                job_id,
                "consolidation",
                output_model,
            )
            if cached is not None:
                final_answer = cached
            else:
                consolidated = json.dumps(
                    [answer.model_dump(mode="json") for answer in successful],
                    ensure_ascii=False,
                )
                try:
                    final_answer, observation = await self._invoke_pre_prompt(
                        key,
                        consolidated,
                        stage="consolidation",
                    )
                    await self._save_checkpoint(
                        job_id,
                        "consolidation",
                        final_answer,
                        observation=observation,
                    )
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

    async def _invoke_pre_prompt(
        self,
        key: ChunkKey,
        text: str,
        *,
        stage: str,
        fragment_index: int | None = None,
    ) -> tuple[PreResult, ProviderObservation]:
        template, output_model = self._require_pre_prompt()
        variables: dict[str, Any] = {
            "chunk_text": text,
            **self._pre_input_variables,
        }
        if self._use_key:
            variables["key_text"] = key
        return await self._invoke_with_retry(
            InvocationContext(
                key=key,
                stage=cast("Any", stage),
                fragment_index=fragment_index,
            ),
            template.format_map(variables),
            output_model,
        )

    async def _invoke_with_retry(
        self,
        context: InvocationContext,
        prompt: str,
        output_model: type[ResultModel],
    ) -> tuple[ResultModel, ProviderObservation]:
        last_error: BaseException | None = None
        policy = self._retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            try:
                async with asyncio.timeout(policy.timeout_s):
                    return await self._provider_pool.invoke(
                        context,
                        prompt,
                        output_model,
                        self._llm_config,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    "promptnest invocation failed",
                    extra={
                        "attempt": attempt,
                        "max_attempts": policy.max_attempts,
                        "error": repr(exc),
                    },
                )
                if attempt >= policy.max_attempts or not policy.should_retry(exc):
                    break
                delay = policy.delay_for(
                    attempt,
                    error=exc,
                    random_source=self._random,
                )
                if delay:
                    await asyncio.sleep(delay)

        assert last_error is not None
        raise InvocationError(
            f"adapter invocation failed after {attempt} attempts",
            attempts=attempt,
            last_error=last_error,
        ) from last_error

    async def _prepare_checkpoint(self) -> None:
        if self._checkpoint_store is not None:
            assert self._checkpoint_run_id is not None
            assert self._checkpoint_revision is not None
            await self._checkpoint_store.prepare(
                self._checkpoint_run_id,
                self._checkpoint_revision,
            )

    async def _load_checkpoint(
        self,
        job_id: str,
        stage: str,
        output_model: type[ResultModel],
        *,
        fragment_index: int | None = None,
    ) -> ResultModel | None:
        if self._checkpoint_store is None:
            return None
        assert self._checkpoint_run_id is not None
        payload = await self._checkpoint_store.load(
            self._checkpoint_run_id,
            job_id,
            stage,
            fragment_index,
        )
        if payload is None:
            return None
        return output_model.model_validate_json(payload)

    async def _save_checkpoint(
        self,
        job_id: str,
        stage: str,
        result: BaseModel,
        *,
        fragment_index: int | None = None,
        observation: ProviderObservation,
    ) -> None:
        if self._checkpoint_store is None:
            return
        assert self._checkpoint_run_id is not None
        await self._checkpoint_store.save(
            self._checkpoint_run_id,
            job_id,
            stage,
            result.model_dump_json(),
            fragment_index=fragment_index,
            provider=observation.provider,
        )

    def _require_pre_prompt(self) -> tuple[str, type[PreResult]]:
        if self._pre_template is None or self._pre_output_model is None:
            raise ConfigurationError("set_pre_prompt() must be called first")
        return self._pre_template, self._pre_output_model

    @staticmethod
    def _find_group_error(group: BaseExceptionGroup[BaseException]) -> BaseException | None:
        for error in group.exceptions:
            if isinstance(error, BaseExceptionGroup):
                nested = PromptNest._find_group_error(error)
                if nested is not None:
                    return nested
            elif not isinstance(error, asyncio.CancelledError):
                return error
        return None

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
