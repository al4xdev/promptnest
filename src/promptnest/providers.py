"""Named provider routing, concurrency and rate limiting."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

from promptnest.policies import RetryableAdapterError
from promptnest.protocols import (
    LLMAdapter,
    ObservedLLMAdapter,
    StreamCompleted,
    StreamDelta,
    StreamingLLMAdapter,
    TokenUsage,
)

ResultModel = TypeVar("ResultModel", bound=BaseModel)
Stage = Literal["fragment", "consolidation", "reduce"]


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Stable routing context for one provider invocation."""

    key: Any
    stage: Stage
    fragment_index: int | None = None


TokenEstimator = Callable[
    [str, type[BaseModel], Mapping[str, Any]],
    int,
]
ProviderRouter = Callable[[InvocationContext], str]


def default_token_estimator(
    prompt: str,
    output_model: type[BaseModel],
    options: Mapping[str, Any],
) -> int:
    """Conservative dependency-free token estimate."""
    output_budget = int(options.get("max_completion_tokens", options.get("max_tokens", 256)))
    return max(1, (len(prompt) + 3) // 4) + max(0, output_budget)


@dataclass(frozen=True, slots=True)
class ProviderPolicy:
    """Limits applied independently to one provider."""

    max_concurrency: int = 32
    requests_per_second: float | None = None
    request_burst: int = 1
    tokens_per_second: float | None = None
    token_burst: int | None = None
    token_estimator: TokenEstimator = default_token_estimator

    def __post_init__(self) -> None:
        if isinstance(self.max_concurrency, bool) or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if self.requests_per_second is not None and self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be greater than zero")
        if isinstance(self.request_burst, bool) or self.request_burst < 1:
            raise ValueError("request_burst must be a positive integer")
        if self.tokens_per_second is not None and self.tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be greater than zero")
        if self.token_burst is not None and self.token_burst < 1:
            raise ValueError("token_burst must be a positive integer")


@dataclass(frozen=True, slots=True)
class Provider:
    """One named adapter and its independent limits."""

    adapter: LLMAdapter
    policy: ProviderPolicy = ProviderPolicy()


@dataclass(frozen=True, slots=True)
class ProviderObservation:
    provider: str
    usage: TokenUsage | None


@dataclass(frozen=True, slots=True)
class StreamObservation:
    """Timing data for one successful streaming provider invocation."""

    provider: str
    ttft_ms: float
    completion_ms: float
    inter_delta_ms: tuple[float, ...]
    delta_count: int


StreamDeltaHandler = Callable[[str, float], Awaitable[None]]


class StreamInterruptedError(Exception):
    """A stream failed after externally visible output was emitted."""


class _TokenBucket:
    def __init__(
        self,
        rate: float | None,
        capacity: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated_at = clock()
        self._clock = clock
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float) -> float:
        if self.rate is None:
            return 0.0
        if amount > self.capacity:
            raise ValueError(
                f"requested amount {amount} exceeds token bucket capacity {self.capacity}"
            )
        waited = 0.0
        while True:
            async with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self.updated_at)
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.updated_at = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return waited
                delay = (amount - self.tokens) / self.rate
            await asyncio.sleep(delay)
            waited += delay

    async def reconcile(self, reserved: int, actual: int) -> None:
        if self.rate is None or reserved == actual:
            return
        async with self._lock:
            now = self._clock()
            elapsed = max(0.0, now - self.updated_at)
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.updated_at = now
            self.tokens = min(self.capacity, max(0.0, self.tokens + reserved - actual))


class _ProviderRuntime:
    def __init__(self, name: str, provider: Provider) -> None:
        self.name = name
        self.provider = provider
        policy = provider.policy
        self.semaphore = asyncio.Semaphore(policy.max_concurrency)
        self.requests = _TokenBucket(
            policy.requests_per_second,
            float(policy.request_burst),
        )
        token_capacity = policy.token_burst
        if token_capacity is None:
            token_capacity = (
                max(1, int(policy.tokens_per_second)) if policy.tokens_per_second is not None else 1
            )
        self.tokens = _TokenBucket(policy.tokens_per_second, float(token_capacity))
        self.active = 0
        self.max_active = 0
        self.request_wait_s = 0.0
        self.token_wait_s = 0.0
        self._cooldown_until = 0.0
        self._cooldown_lock = asyncio.Lock()

    async def _wait_for_cooldown(self) -> None:
        async with self._cooldown_lock:
            delay = max(0.0, self._cooldown_until - time.monotonic())
        if delay:
            await asyncio.sleep(delay)

    async def _apply_retry_after(self, delay_s: float | None) -> None:
        if delay_s is None:
            return
        async with self._cooldown_lock:
            self._cooldown_until = max(
                self._cooldown_until,
                time.monotonic() + delay_s,
            )

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        options: Mapping[str, Any],
    ) -> tuple[ResultModel, ProviderObservation]:
        policy = self.provider.policy
        estimated = policy.token_estimator(prompt, output_model, options)
        await self._wait_for_cooldown()
        self.request_wait_s += await self.requests.acquire(1)
        self.token_wait_s += await self.tokens.acquire(estimated)
        async with self.semaphore:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                adapter = self.provider.adapter
                usage: TokenUsage | None = None
                try:
                    if isinstance(adapter, ObservedLLMAdapter):
                        observed = await adapter.invoke_observed(
                            prompt,
                            output_model,
                            **options,
                        )
                        result = observed.value
                        usage = observed.usage
                    else:
                        result = await adapter.invoke(prompt, output_model, **options)
                except RetryableAdapterError as exc:
                    await self._apply_retry_after(exc.retry_after_s)
                    raise
                if usage is not None:
                    await self.tokens.reconcile(estimated, usage.total_tokens)
                return result, ProviderObservation(self.name, usage)
            finally:
                self.active -= 1

    async def invoke_streaming(
        self,
        prompt: str,
        output_model: type[ResultModel],
        options: Mapping[str, Any],
        on_delta: StreamDeltaHandler,
    ) -> tuple[ResultModel, ProviderObservation, StreamObservation]:
        adapter = self.provider.adapter
        if not isinstance(adapter, StreamingLLMAdapter):
            raise TypeError(f"provider {self.name!r} does not support streaming")
        policy = self.provider.policy
        estimated = policy.token_estimator(prompt, output_model, options)
        await self._wait_for_cooldown()
        self.request_wait_s += await self.requests.acquire(1)
        self.token_wait_s += await self.tokens.acquire(estimated)
        async with self.semaphore:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            started = time.perf_counter()
            first_delta_at: float | None = None
            previous_delta_at: float | None = None
            gaps: list[float] = []
            delta_count = 0
            completed: StreamCompleted[ResultModel] | None = None
            try:
                try:
                    async for event in adapter.stream(prompt, output_model, **options):
                        if isinstance(event, StreamDelta):
                            if not event.text:
                                continue
                            now = time.perf_counter()
                            if first_delta_at is None:
                                first_delta_at = now
                            if previous_delta_at is not None:
                                gaps.append((now - previous_delta_at) * 1000)
                            previous_delta_at = now
                            delta_count += 1
                            await on_delta(event.text, (now - started) * 1000)
                        elif isinstance(event, StreamCompleted):
                            completed = event
                        else:
                            raise TypeError(f"unsupported stream event {type(event).__name__}")
                except RetryableAdapterError as exc:
                    await self._apply_retry_after(exc.retry_after_s)
                    if delta_count:
                        raise StreamInterruptedError(
                            "stream failed after emitting visible output"
                        ) from exc
                    raise
                except Exception as exc:
                    if delta_count and not isinstance(exc, StreamInterruptedError):
                        raise StreamInterruptedError(
                            "stream failed after emitting visible output"
                        ) from exc
                    raise
                if completed is None:
                    raise ValueError("stream ended without a StreamCompleted event")
                if first_delta_at is None:
                    raise ValueError("stream completed without a non-empty text delta")
                if completed.usage is not None:
                    await self.tokens.reconcile(estimated, completed.usage.total_tokens)
                finished = time.perf_counter()
                return (
                    completed.value,
                    ProviderObservation(self.name, completed.usage),
                    StreamObservation(
                        provider=self.name,
                        ttft_ms=(first_delta_at - started) * 1000,
                        completion_ms=(finished - started) * 1000,
                        inter_delta_ms=tuple(gaps),
                        delta_count=delta_count,
                    ),
                )
            finally:
                self.active -= 1


class ProviderPool(Generic[ResultModel]):
    """Route calls to named providers with independent policies."""

    def __init__(
        self,
        providers: Mapping[str, Provider],
        *,
        router: ProviderRouter | None = None,
    ) -> None:
        if not providers:
            raise ValueError("providers cannot be empty")
        self._runtimes = {
            name: _ProviderRuntime(name, provider) for name, provider in providers.items()
        }
        default_name = next(iter(providers))
        self._router = router or (lambda context: default_name)

    @classmethod
    def single(
        cls,
        adapter: LLMAdapter,
        *,
        policy: ProviderPolicy | None = None,
    ) -> ProviderPool[Any]:
        return cls({"default": Provider(adapter, policy or ProviderPolicy())})

    def provider_name(self, context: InvocationContext) -> str:
        name = self._router(context)
        if name not in self._runtimes:
            raise ValueError(f"provider router returned unknown provider {name!r}")
        return name

    async def invoke(
        self,
        context: InvocationContext,
        prompt: str,
        output_model: type[ResultModel],
        options: Mapping[str, Any],
    ) -> tuple[ResultModel, ProviderObservation]:
        name = self.provider_name(context)
        return await self._runtimes[name].invoke(prompt, output_model, options)

    async def invoke_streaming(
        self,
        context: InvocationContext,
        prompt: str,
        output_model: type[ResultModel],
        options: Mapping[str, Any],
        on_delta: StreamDeltaHandler,
    ) -> tuple[ResultModel, ProviderObservation, StreamObservation]:
        name = self.provider_name(context)
        return await self._runtimes[name].invoke_streaming(
            prompt, output_model, options, on_delta
        )

    def metrics(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "max_active": runtime.max_active,
                "request_wait_s": runtime.request_wait_s,
                "token_wait_s": runtime.token_wait_s,
            }
            for name, runtime in self._runtimes.items()
        }
