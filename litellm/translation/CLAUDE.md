# litellm/translation

v2 of core LLM translation: hub-and-spoke through a frozen IR, shipped behind a
per-provider opt-in flag so it runs beside v1 until a provider is differential-green
and the harness stays green with its flag on. Inbound parsers map an accepted request
schema into the IR, provider serializers map the IR onto one wire format, and
`dispatch.route` is the single v1/v2 fork. This file is the package's map; keep it
true when the layout or conventions move.

## What goes where

```
translation/
├── __init__.py     # public surface ONLY; everything else is private
├── CLAUDE.md       # this file
├── ruff.toml       # folder rule gates (see "Deterministic checks" below)
├── pyrightconfig.json  # strict; the folder must stay at 0 errors
├── ir.py           # frozen IR: product types are frozen dataclasses, sums are
│                   #   Expression tagged unions; opaque JSON rides as JsonBlob
├── errors.py       # failures as values; every error carries .summary; the ONLY
│                   #   module allowed near `raise` (it defines none today)
├── boundary.py     # the typed boundary: parse() = arktype calling convention over
│                   #   frozen pydantic v2 wire models; as_plain_json() admits
│                   #   opaque JSON sub-trees (leaf-checked, deep-copied)
├── deps.py         # TranslationDeps: every ambient litellm input (model map
│                   #   lookups, drop_params/modify_params) enters as a value
├── dispatch.py     # route(): per-provider allowlist + same-family fast path
├── inbound/        # one subpackage per accepted schema: raw <-> OpenAI shapes
│   └── openai_chat/
│       ├── schema.py    # pydantic wire models, extra="forbid" + strict=True
│       ├── messages.py  # wire messages -> IR messages (hot path)
│       ├── parse.py     # top-level parse + semantic checks
│       ├── response.py  # IR ChatResponse -> chat-completion body
│       └── stream.py    # IR stream events -> chunk bodies (pure fold)
├── providers/      # one subpackage per wire format. Pure, no I/O
│   └── anthropic/
│       ├── serialize.py # body assembly in v1 transform_request order
│       ├── messages.py  # IR messages -> anthropic dicts (placeholder, dedupe,
│       │                #   id sanitize, final-assistant rstrip)
│       ├── tools.py     # tool defs, name sanitize maps, schema whitelists
│       ├── params.py    # sampling gates, thinking/effort, model detection
│       ├── response.py  # response JSON -> IR (json_tool_call rewrite here)
│       └── stream.py    # SSE lines -> IR stream events
└── engine/
    ├── pipeline.py # prepare (pure, drives the fallback decision) -> send;
    │               #   async-first, the seam owns the one sync wrapper
    ├── http.py     # the injected HttpPort + ExecuteError values
    └── stream.py   # the ONE accumulator: lines -> IR events -> chunks
```

The v1-side adapter lives OUTSIDE the package in `litellm/translation_seam.py`:
deps building from litellm ambient state, ModelResponse/ModelResponseStream
envelope adaptation, the `completion()` fork, and the
`litellm.translation_v2_providers` allowlist (env:
`LITELLM_TRANSLATION_V2_PROVIDERS`). Streaming and `modify_params=True`
traffic stay on v1 until their seams land; response headers passthrough into
`_hidden_params` and pooled async clients are noted follow-ups there.

## The fail-closed contract (the most important invariant)

`parse_request` accounts for every inbound field. Anything outside the proven
surface returns a typed error instead of being dropped:

- unknown field / unrecognized union tag -> `TranslationError.unsupported`
- malformed known field -> `TranslationError.boundary` (every failure listed)
- shapes v1 serves through unported paths -> `unsupported` with a reason naming
  the v1 path (http:// image download, $ref inlining, JSON-repair, legacy
  function_call, server-tool reconstruction, body-order-dependent parameter
  interplay, modify_params behaviors)

The dispatch seam falls back to v1 on ANY error, so flag-on can never silently
lose prompt caching, thinking, structured outputs, or anything else (audit F1).
When you add a field to a wire model you are widening the proven surface: add
corpus shapes in the same commit.

## Conventions (each backed by a deterministic check)

- Composition over inheritance: a provider is a module of pure functions. No
  imports from the v1 stack — enforced by the import-linter contracts in
  pyproject.toml (`lint-imports`) plus the AST test
  `tests/test_litellm/translation/test_import_contract.py`. The ONE allowed
  litellm import is `litellm.constants` (env-seeded leaf constants). Anything
  else ambient comes in through `deps.TranslationDeps`.
- Failures as values: nothing raises (semgrep `translation-no-raise`).
  try/except around stdlib/pydantic calls that themselves raise is fine; the
  except arm returns an Error value.
- No mutation; no module-level mutable state (semgrep `translation-no-mutation`,
  `translation-no-module-level-mutable-state` — the latter is the rule that
  would have caught audit F2). A local build-then-freeze accumulator that never
  escapes its scope may carry an inline `# nosemgrep: <rule-id>` plus a
  justification; keep these rare and obvious.
- Exhaustiveness pattern (the only Expression-compatible form pyright strict
  proves): match on the union's `Literal` tag with one arm per case and
  `assert_never(x.tag)` AFTER the match. A `case never:` capture arm is flagged
  by strict as unmatchable — don't use it.
- Fully typed, zero `Any`: pyright strict via the folder pyrightconfig.json
  (0 errors is the bar — restructure, don't suppress) and ruff `ANN` rules.
- Never-nester / size caps: ruff PLR1702 (preview), C901, PLR0915; ~400-line
  soft cap per file.
- Hot-path exception to ceremony (deliberate, perf-budgeted): inbound message
  conversion returns `value | TranslationError` plain unions internally and
  lifts into `Result`/`Block` once at the public boundary; `ir._case_maker`
  builds tagged-union cases without the stock `__init__`; `Nothing` is
  identity-checked where a request has hundreds of blocks. Budget: v2 <= 1.5x
  v1 CPU at 600-message histories — re-check with
  `python -m tests.test_litellm.translation.bench_request_translation`.
- `JsonBlob` ownership: blobs are created fresh at the parse boundary and may
  be emitted into the returned body without copying (the body takes ownership;
  the IR is dropped). Never share a blob into two bodies and never put one in
  module state.

## Deterministic checks and how to run them

`make lint-translation` runs everything: ruff (this folder's ruff.toml, which
the repo-wide `ruff check` also picks up hierarchically), pyright strict,
semgrep (`.semgrep/rules/python/translation/translation-v2-tenets.yml`, runs in
the repo's semgrep workflow), import-linter (contracts in pyproject.toml, runs
in the code-quality workflow), and the package's tests. CI: test-linting
(ruff/black/mypy), test-semgrep, test-code-quality (pyright + lint-imports),
test-unit-misc (tests). Format with black (CI enforces black, not ruff format).

## How parity is proven

Characterization then differential, never mixed with a behavior change. The
differential corpus (`tests/test_litellm/translation/test_differential_anthropic_request.py`)
runs v1 (`map_openai_params` + `transform_request`) and v2 over the same
requests and asserts identical normalized JSON, across current-generation
model ids and all four Claude-Code feature surfaces (caching, thinking, tools,
structured outputs). That corpus is the covered surface: the anthropic flag
turns on only for shapes pinned there. Notable transform-seam facts the corpus
pins: v1's output includes `json_mode: true` for response_format requests (the
HTTP layer pops it before the wire — v2's engine must do the same), and the
output_format schema-filter note order inherits v1's set-iteration order.
A behavior change ships as its own snapshot-diffed PR, never inside a port.

## Current scope

OpenAI-chat-in -> Anthropic-out request translation, differential-green on a
46-shape corpus, fail-closed everywhere else. Not yet here, each its own
follow-up: response and stream parsing (engine/http.py, engine/stream.py); the
other inbound schemas (`anthropic_messages`, `google_genai`, `responses`,
`completions`); the other providers (bedrock converse/invoke, vertex, azure,
`openai_compat`); wiring `dispatch.route` into the live `completion()` path
with the `LITELLM_TRANSLATION_V2_PROVIDERS` allowlist. To add a provider:
write `providers/<name>/`, register it in `engine/pipeline._SERIALIZERS`, add
a differential corpus, keep the flag off until differential-green.
