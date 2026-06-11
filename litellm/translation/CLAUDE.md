# litellm/translation

v2 of core LLM translation: hub-and-spoke through a frozen IR, shipped behind a
per-provider opt-in flag so it runs beside v1 until a provider is differential-green
and the harness stays green with its flag on. Inbound parsers map an accepted request
schema into the IR, provider serializers map the IR onto one wire format, and
`dispatch.route` is the single v1/v2 fork. This file is the seed the agents working
here land against; keep it in sync when the layout or conventions move.

## What goes where

```
translation/
├── __init__.py     # public surface ONLY; everything else is private
├── ir.py           # frozen IR: product types are frozen dataclasses, sums are
│                   #   Expression tagged unions, collections are Block / Map
├── errors.py       # failures as values; every error carries .summary
├── boundary.py     # the typed boundary: freeze/thaw + as_* accessors. Untyped
│                   #   data enters here and nowhere else
├── dispatch.py     # route(): per-provider allowlist + same-family fast path
├── inbound/        # one subpackage per accepted schema: raw dict -> IR
│   └── openai_chat/
├── providers/      # one subpackage per wire format: IR -> body. Pure, no I/O
│   └── anthropic/
└── engine/
    └── pipeline.py # composition + the public translate entry point
```

The injected I/O port (functional core, imperative shell) lands with the
response and stream increment, when there is actual I/O to inject; this slice is
a pure request transform with no network.

## Conventions

- Composition over inheritance: a provider is a module of pure functions, not a base
  class. No imports from the v1 stack (`litellm.llms`, `litellm.main`, ...); the
  decoupling is enforced by `tests/test_litellm/translation/test_import_contract.py`,
  which stands in for an import-linter contract until one is wired into CI.
- Failures as values: nothing here raises. Fallible steps return `Result`; the
  boundary returns a `BoundaryError` whose `.summary` lists every field failure; the
  FastAPI layer converts at the edge.
- No mutation: IR types are frozen and collections are `Block`/`Map`. Build dicts by
  collecting present `(key, value)` pairs and calling `dict`, never by subscript-assign.
- Tagged unions plus `match`; fully typed, no `Any`; one concern per file under a
  ~400-line soft cap; tests mirror this tree under `tests/test_litellm/translation/`.
- Built with Expression (`Block`, `Map`, `Result`, `Option`, `tagged_union`). The
  parent guidelines mention algebraic effects; we model I/O as injected ports instead
  (Mateo's scope doc), the same testability without a generator-effects runtime.

## How parity is proven

Characterization then differential, never mixed with a behavior change. The differential
corpus (`test_differential_anthropic_request.py`) runs v1 (`map_openai_params` then
`transform_request`) and v2 over the same requests and asserts identical normalized JSON.
That corpus is the covered surface: the anthropic flag turns on only for the shapes
pinned there. A behavior change ships as its own snapshot-diffed PR, never inside a port.

## Current scope

This slice is OpenAI-chat-in to Anthropic-out request translation, differential-green
on the corpus. Not yet here, each its own follow-up: response and stream parsing; the
other inbound schemas (`anthropic_messages`, `google_genai`, `responses`, `completions`);
the other providers (bedrock converse/invoke, vertex, azure, `openai_compat`); and
wiring `dispatch.route` into the live `completion()` path. To add a provider, write
`providers/<name>/serialize.py`, register it in `engine/pipeline._SERIALIZERS`, add a
differential corpus, and leave the flag off until it is differential-green.
