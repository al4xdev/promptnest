# Changelog

All notable changes to PromptNest are documented here.

## 0.3.0

- Add a provider-neutral streaming adapter contract and contextual delta callback.
- Add per-stream TTFT, completion, and inter-delta observations with p50/p95/p99 summaries.
- Add structured-output streaming to the OpenAI adapter without a second provider call.
- Suppress automatic retries after visible output to prevent duplicate stream delivery.
- Add synthetic TTFT certification gates and an optional real-provider TTFT profile.

## 0.2.0

- Add bounded producer/worker execution and lazy `AsyncIterable` input.
- Add named provider routing with concurrency, request-rate, and token budgets.
- Add exponential full-jitter retries and normalized retry-after errors.
- Add durable SQLite checkpoints and idempotent consolidation recovery.
- Add a reproducible 10,000-job core certification command and CI evidence.

## 0.1.0

- Initial public release.
- Typed asynchronous map/consolidate/reduce runner.
- Structured Pydantic output with retries, timeouts, partial failure handling,
  and concurrency limits.
- Official OpenAI, LangChain, LangGraph, CrewAI, and callable adapters.
- Deterministic JSON contract tests and editable-install consumer tests.
