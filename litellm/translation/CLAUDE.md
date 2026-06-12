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
│       ├── response.py  # IR ChatResponse -> chat-completion body; carries a
│       │                #   ResponseDialect (anthropic | bedrock_converse)
│       │                #   because v1's outbound shapes are per-provider
│       └── stream.py    # IR stream events -> chunk bodies (pure fold); same
│                        #   dialect idea via ChunkDialect on StreamState
├── providers/      # one subpackage per wire format. Pure, no I/O
│   ├── anthropic/
│   │   ├── serialize.py # body assembly in v1 transform_request order
│   │   ├── messages.py  # IR messages -> anthropic dicts (placeholder, dedupe,
│   │   │                #   id sanitize, final-assistant rstrip)
│   │   ├── tools.py     # tool defs, name sanitize maps, schema whitelists
│   │   ├── params.py    # sampling gates, thinking/effort, model detection
│   │   ├── response.py  # response JSON -> IR (json_tool_call rewrite here)
│   │   └── stream.py    # SSE lines -> IR stream events
│   ├── bedrock_converse/  # the one genuinely distinct bedrock wire format
│   │   ├── serialize.py # messages/system/inferenceConfig/toolConfig/
│   │   │                #   additionalModelRequestFields assembly
│   │   ├── messages.py  # IR -> converse blocks (cachePoint, toolUse dedupe,
│   │   │                #   reasoning-first sort, blank-assistant drop)
│   │   ├── tools.py     # toolSpec build, bedrock tool-name normalization,
│   │   │                #   cachePoint ttl gates, json_tool_call rf tool
│   │   ├── params.py    # claude gates, thinking clamp, converse finish map
│   │   ├── response.py  # converse JSON -> IR (properties-unwrap json rewrite)
│   │   └── stream.py    # PARSED converse events -> IR events (pinned at the
│   │                    #   parsed-event seam; AWS framing is botocore's)
│   ├── bedrock_invoke/  # anthropic serializer + envelope deltas ONLY:
│   │   ├── serialize.py # pop model/stream, inject anthropic_version, spoof
│   │   │                #   model for response_format (v1's json-tool forcing)
│   │   ├── response.py  # re-export of anthropic parse_response (invoke
│   │   │                #   response body IS anthropic wire format)
│   │   └── stream.py    # re-export of anthropic parse_event (invoke stream
│   │                    #   = anthropic events over AWS framing)
│   └── openai_compat/   # the same-family hub serializer (GPT first consumer)
│       ├── guard.py     # raw-shape fidelity guard run BEFORE parse: shapes
│       │                #   the IR cannot round-trip losslessly (string stop,
│       │                #   message name, image detail, max-tokens key split,
│       │                #   tool-arg spacing) fall back to v1 as typed errors
│       ├── serialize.py # v1's five-touch passthrough body assembly
│       ├── messages.py  # IR -> openai wire messages (inverse of inbound)
│       ├── params.py    # o-series/gpt-5 family gates (fail closed until
│       │                #   their param families are ported), user gate
│       ├── response.py  # mirrors convert_to_model_response_object (the LIVE
│       │                #   normalizer; transform_response is dead on the SDK
│       │                #   path); rides the outbound body on ChatResponse.wire
│       └── stream.py    # SSE chunk -> wire_chunk events normalized to the
│                        #   SDK-dump shape; the openai chunk dialect folds them
└── engine/
    ├── pipeline.py # prepare (pure, drives the fallback decision) -> send;
    │               #   per-provider serializer/parser/dialect tables; the
    │               #   wire_body step strips transform-seam markers
    │               #   (json_mode; converse additionalModelRequestFields.stream)
    ├── http.py     # the injected HttpPort + ExecuteError values
    └── stream.py   # the ONE accumulator: lines OR parsed events -> IR
                    #   events -> chunks (fold_lines / fold_events)
```

The v1-side adapter lives OUTSIDE the package in `litellm/translation_seam.py`:
deps building from litellm ambient state, ModelResponse/ModelResponseStream
envelope adaptation (per-provider usage construction, because `Usage` only
serializes explicitly-set fields), the `completion()` forks (anthropic branch
and the bedrock branch's converse/invoke routes, selected by
`BedrockModelInfo.get_bedrock_route`), and the
`litellm.translation_v2_providers` allowlist (env:
`LITELLM_TRANSLATION_V2_PROVIDERS`). On bedrock, AWS auth params split out of
the body BEFORE parsing (envelope, not payload) and SigV4 signs AFTER
`wire_body` finalizes the bytes (sign-after-body-final, or bedrock 403s).
Streaming and `modify_params=True` traffic stay on v1 until their seams land;
response headers passthrough into `_hidden_params` and pooled async clients
are noted follow-ups there.

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
anthropic corpus (`tests/test_litellm/translation/test_differential_anthropic_request.py`)
runs v1 (`map_openai_params` + `transform_request`) and v2 over the same
requests and asserts identical normalized JSON, across current-generation
model ids and all four Claude-Code feature surfaces (caching, thinking, tools,
structured outputs). The bedrock gates
(`test_differential_bedrock_{request,response,stream}.py`) are two-sided over
the characterization corpus copied verbatim into
`tests/test_litellm/translation/characterization_bedrock/`: v1-at-HEAD must
still equal the committed snapshot (drift guard) AND v2 must equal the
snapshot byte-for-byte (canonical JSON), plus a quirk corpus (4.5 cache ttl,
parallel tool config, bedrock name normalization, thinking clamp, tool_choice
gates) referenced against v1 in-process. Streams compare v2's fold against
the REAL decoders inside `CustomStreamWrapper`, pinned at the parsed-event
seam (AWS event-stream framing is botocore plumbing). Those corpora are the
covered surface: a provider's flag turns on only for shapes pinned there.
Notable transform-seam facts the corpora pin: v1's anthropic output includes
`json_mode: true` for response_format requests, and converse's
`get_optional_params` invocation leaves `stream` inside
`additionalModelRequestFields` — both are popped by `engine.pipeline.wire_body`
before the wire, exactly like v1's HTTP layers. `DIFFERENTIAL_REPORT.md` is
the merged artifact; regenerate with
`python -m tests.test_litellm.translation.generate_differential_report`.
A behavior change ships as its own snapshot-diffed PR, never inside a port.

## Current scope

OpenAI-chat-in to four providers out — `anthropic`, `bedrock_converse`,
`bedrock_invoke`, `openai_compat` — request, response, and stream
translation, differential-green (anthropic: 46-shape corpus + responses +
stream replays; bedrock: the characterization corpus per route + quirk
corpus; openai: 17-shape request corpus + 17 typed-fallback rows + response
and SDK-chunk stream replays), fail-closed everywhere else, with
non-streaming flag-gated seams live in `completion()` for the anthropic and
bedrock routes (the openai seam fork is integrator scope and NOT wired).
Deliberate bedrock fallback surfaces (each names the v1 path): non-Claude
bedrock models, native structured outputs (outputConfig), adaptive-effort
output_config/beta, response_format+stream (fake_stream), response_format
with thinking on invoke (the model spoof crossing), tool history without
tools (modify_params dummy tool), empty user text on converse
(string-vs-list ambiguity), provisioned `model_id`, `guardrailConfig`, and
`<thinking>`-tagged text in converse responses. Deliberate openai fallback
surfaces: o-series and gpt-5 model families (their param-rewrite configs are
unported), every raw shape the IR cannot round-trip byte-identically (the
guard's list: string stop, both max-tokens keys, message `name`, image
`detail`/`format`, consecutive same-role turns, single-text content lists,
empty tools/stop lists, non-canonical tool-argument JSON spacing, null/
non-function tool_calls), the `user` param (model-list gated in v1),
`response_format` on gpt-4/gpt-3.5-turbo-16k, `stream_options`, file blocks
(v1 downloads http pdf file_ids in-transform), and `http://` image URLs. On
streams, the trailing `choices: []` usage chunk passes through verbatim and
the wrapper's synthesized final usage chunk stays a seam/envelope concern.
Not yet here, each its own follow-up: streaming seams live; the other
inbound schemas (`anthropic_messages`, `google_genai`, `responses`,
`completions`); the other providers (vertex, azure); the same-family fast
path (waits on the opaque-body relay). To add a provider: write
`providers/<name>/`, register it in `engine/pipeline._SERIALIZERS` /
`_RESPONSE_PARSERS` / `_RESPONSE_DIALECTS` (plus `_RAW_GUARDS` when the
inbound schema is the provider's own family), add a differential corpus,
keep the flag off until differential-green.
