# PromptNest core certification

The core certification is a deterministic project self-assessment. It is not an independent
third-party certification and does not measure a real model or provider network.

Run it from a clean checkout:

```fish
uv sync --locked --dev
uv run --locked promptnest-certify --output-dir /tmp/promptnest-certificate
```

The command produces:

- `certificate.json`: structured environment, gates, observations and scope;
- `certificate.md`: human-readable PASS/FAIL report;
- `certificate.sha256`: checksum for the JSON evidence.

A dirty working tree is ineligible unless `--allow-dirty` is explicitly supplied for development;
that override produces `PASS-DEVELOPMENT`, remains ineligible, and must not be presented as release
evidence. Only a clean run can emit `PASS`.

## Mandatory gates

The standard profile processes 10,000 lazy logical jobs with two fragments, nested consolidation,
and reduce. It uses 32 workers, a queue capacity of 128, and two providers with concurrency limits
of 7 and 11.

PASS requires:

- 10,000 unique results with no loss or duplication;
- completion under 30 seconds with the deterministic adapter;
- peak traced Python memory below 128 MiB;
- at most 40 asyncio tasks above the baseline;
- queue high-watermark no greater than 128 and at least one blocked admission;
- both provider concurrency limits respected;
- observable request/token rate limiting;
- exactly one successful retry for each injected transient failure;
- cancellation cleanup in under one second with zero active calls;
- recovery of a failed consolidation without repeating checkpointed fragments;
- 100 latency samples with p95 overhead below 0.5 ms/job and batch p99 below 500 ms.

The certification workflow runs on Python 3.12 for pull requests and `master`. Release tags attach
the JSON, Markdown, and checksum to the GitHub Release.

## Claim boundary

A PASS supports claims about PromptNest's Python orchestration: bounded backpressure, independent
provider limits, token-budget enforcement, retry recovery, structured cancellation, and
idempotent checkpoint resume.

It certifies synthetic TTFT/inter-delta instrumentation, but not real model/provider TTFT or
latency, exactly-once external calls, encrypted
checkpoints, automatic provider failover, or third-party validation. Optional real-provider
profiles may provide additional evidence but do not determine the core badge.

An optional OpenAI profile is included for account-specific evidence:

```fish
set -x OPENAI_API_KEY "..."
set -x PROMPTNEST_OPENAI_MODEL "gpt-4.1-mini"
uv run --extra openai python docs/benchmarks/openai_provider.py \
    --jobs 20 \
    --concurrency 4 \
    --requests-per-second 2 \
    --output /tmp/promptnest-openai-provider.json
```

It records complete structured-result latency and provider-limit behavior. It incurs provider
costs, requires credentials, and is never executed by the mandatory CI profile.
