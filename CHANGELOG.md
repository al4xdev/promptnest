# Changelog

All notable changes to PromptNest are documented here.

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
