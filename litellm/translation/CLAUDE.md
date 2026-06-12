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
│       │                #   ResponseDialect (anthropic | bedrock_converse |
│       │                #   gemini) because v1's outbound shapes are
│       │                #   per-provider
│       └── stream.py    # IR stream events -> chunk bodies (pure fold); same
│                        #   dialect idea via ChunkDialect on StreamState;
│                        #   the gemini dialect consumes composite chunk
│                        #   events (cumulative tool index + the
│                        #   seen-tool-calls stop->tool_calls rewrite ride
│                        #   StreamState)
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
│   │   │                #   model for response_format (v1's json-tool forcing;
│   │   │                #   RESPONSE_FORMAT_SPOOF_MODEL is shared with vertex)
│   │   ├── response.py  # re-export of anthropic parse_response (invoke
│   │   │                #   response body IS anthropic wire format)
│   │   └── stream.py    # re-export of anthropic parse_event (invoke stream
│   │                    #   = anthropic events over AWS framing)
│   ├── openai_compat/   # the same-family hub serializer (GPT first consumer)
│   │   ├── guard.py     # raw-shape fidelity guard run BEFORE parse: shapes
│   │   │                #   the IR cannot round-trip losslessly (string stop,
│   │   │                #   message name, image detail, max-tokens key
│   │   │                #   split) fall back to v1 as typed errors; tool
│   │   │                #   argument strings ride ToolUse.arguments_raw
│   │   │                #   verbatim, so spacing never falls back
│   │   ├── serialize.py # v1's five-touch passthrough body assembly
│   │   ├── messages.py  # IR -> openai wire messages (inverse of inbound)
│   │   ├── params.py    # o-series/gpt-5 family gates (fail closed until
│   │   │                #   their param families are ported), user gate
│   │   ├── response.py  # mirrors convert_to_model_response_object (the LIVE
│   │   │                #   normalizer; transform_response is dead on the SDK
│   │   │                #   path); rides the outbound body on ChatResponse.wire
│   │   └── stream.py    # SSE chunk -> wire_chunk events normalized to the
│   │                    #   SDK-dump shape; the openai chunk dialect folds
│   │                    #   them; make_parse_line is the ONE data:-line
│   │                    #   decode every same-family provider composes
│   ├── google_genai/    # ONE generateContent family for BOTH google routes:
│   │   │                #   providers "vertex_ai" and "gemini" are the same
│   │   │                #   serializer parameterized by the drift list
│   │   │                #   (AI Studio refuses https media + forwards
│   │   │                #   function-call ids on gemini-3+); auth/host/api-
│   │   │                #   version differences are envelope, never here
│   │   ├── guard.py     # raw guard run BEFORE parse: message `name` beside
│   │   │                #   cache markers falls back (the IR drops `name`,
│   │   │                #   so its bytes are invisible to the cache-marker
│   │   │                #   token bound while v1 token-counts them)
│   │   ├── serialize.py # cache-marker gate (conservative token bound:
│   │   │                #   UTF-8 bytes + per-message margin < 1024 proves
│   │   │                #   v1 skips the context-cache network call; any
│   │   │                #   media beside markers fails closed), 3-way
│   │   │                #   structured-
│   │   │                #   output fork (responseJsonSchema regex vs
│   │   │                #   responseSchema+propertyOrdering capability vs
│   │   │                #   schema-as-user-message), generationConfig with
│   │   │                #   v1's exact snake/camel key mix
│   │   ├── messages.py  # contents/system_instruction; tool-result runs flush
│   │   │                #   as their own user turns; signature-bearing
│   │   │                #   thinking -> thoughtSignature parts
│   │   ├── tools.py     # function_declarations + toolConfig (no name munging)
│   │   ├── schema.py    # pure ports of v1's _build_vertex_schema pipeline
│   │   │                #   ($ref/$defs fail closed)
│   │   ├── params.py    # thinkingConfig budget/level fork, finish map,
│   │   │                #   gemini-3 default temperature
│   │   ├── response.py  # candidates/functionCall/thought parts -> IR;
│   │   │                #   usageMetadata modality math (cached/thoughts)
│   │   └── stream.py    # parsed GenerateContentResponse events -> the
│   │                    #   composite `chunk` StreamEvent (complete tool args
│   │                    #   per chunk; mid-stream error objects are loud)
│   ├── vertex_anthropic/ # anthropic serializer + envelope deltas ONLY
│   │   │                #   (the bedrock_invoke pattern): pop model/stream/
│   │   │                #   json_mode (the anthropic handler pops json_mode
│   │   │                #   BEFORE transform on this path), inject
│   │   │                #   anthropic_version: vertex-2023-10-16, shared
│   │   │                #   json-tool model spoof; betas never emitted for
│   │   │                #   the v2 surface (vertex suppresses them)
│   │   ├── serialize.py
│   │   ├── response.py  # anthropic parse + request-model restore
│   │   └── stream.py    # re-export of anthropic parse_event
│   ├── azure/           # openai_compat + azure gates ONLY:
│   │   ├── guard.py     # openai guard + cache_control (azure never strips
│   │   │                #   it) + explicit stream:false (key reaches the wire)
│   │   ├── params.py    # api-version gates (tool_choice, response_format
│   │   │                #   json-tool strategy), o/gpt-5 detection on
│   │   │                #   base_model-or-model (deps.api_version/base_model)
│   │   ├── serialize.py # azure gates then the openai_compat body verbatim
│   │   ├── response.py  # re-export of openai parse_response (same live
│   │   │                #   normalizer; json_mode requests fail closed)
│   │   └── stream.py    # openai parser + per-chunk model re-attach and the
│   │                    #   SDK service_tier default; "azure" chunk dialect
│   ├── azure_ai/        # the Foundry override set + the Claude route:
│   │   ├── guard.py     # azure guard + the text-only content-list flatten
│   │   ├── serialize.py # azure_ai gates (grok, model-map tool_choice) then
│   │   │                #   the openai_compat body
│   │   ├── response.py  # openai parser + the azure_ai/{model} rename
│   │   ├── stream.py    # re-export of the azure parser ("azure" dialect)
│   │   └── claude.py    # anthropic serializer/parsers re-exported; NO
│   │                    #   response-format model spoof (v1 maps with the
│   │                    #   real model); billing-header blocks fail closed
│   └── xai/             # Grok over openai_compat (httpx path: NO model
│       │                #   prefix anywhere, transform_response is LIVE):
│       ├── guard.py     # web_search_options (v1's Responses-bridge reroute
│       │                #   sits ABOVE the seam) + use_xai_oauth (PKCE I/O
│       │                #   in validate_environment) + explicit stream:false
│       │                #   + nested tool `strict` keys; then the openai
│       │                #   guard with the user-only message-name fallback
│       │                #   (v1 strips non-user names, so the IR drop IS v1)
│       ├── params.py    # v1's RAISE-unless-drop_params supported-list truth
│       │                #   as typed fallbacks (max_completion_tokens always;
│       │                #   stop/frequency_penalty/reasoning_effort per model
│       │                #   family; reasoning via deps over xai/{model})
│       ├── serialize.py # xai gates -> openai_compat assemble_body ->
│       │                #   function-level strict strip + user/
│       │                #   reasoning_effort emission
│       ├── response.py  # openai parser + the xai usage post-steps (reasoning
│       │                #   fold, total normalize, num_sources_used ->
│       │                #   web_search_requests); the finish_reason "" chain
│       │                #   needs no arm: v1's own fix is dead, both sides
│       │                #   map "" -> "stop" in the live Choices constructor
│       └── stream.py    # SSE dict-path parser: per-chunk usage fold with
│                        #   the folded usage attached ONLY to the choices:[]
│                        #   tail (v1's wrapper strips it elsewhere), no
│                        #   extras/system_fingerprint passthrough (the
│                        #   chunk_parser rebuild drops them), reasoning
│                        #   rename, refusal forwarded when it rides a
│                        #   role/content delta, tool_call type "function"
│                        #   default; the "xai" chunk dialect folds it
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
seam (AWS event-stream framing is botocore plumbing; gemini's SSE line
splitting and accumulated-json fallback are likewise transport in front of
`ModelResponseIterator.chunk_parser`). The google gates
(`test_differential_google_{request,response,stream}.py` over
`characterization_google/`, vendored verbatim from
mateo/translation-characterization-providers) are two-sided the same way,
with the vertex token fetch stubbed at `VertexBase.get_access_token` and a
quirk corpus pinning the 3-way structured-output fork and the
vertex-vs-AI-Studio drift list. Those corpora are the
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

OpenAI-chat-in to eleven providers out — `anthropic`, `bedrock_converse`,
`bedrock_invoke`, `openai_compat`, `vertex_ai` (gemini route), `gemini`
(AI Studio), `vertex_anthropic`, `azure`, `azure_ai`,
`azure_ai_anthropic`, `xai` — request, response, and stream translation,
differential-green (anthropic: 46-shape corpus + responses + stream
replays; bedrock and google: the characterization corpus per route + quirk
corpora; openai: 17-shape request corpus + 17 typed-fallback rows +
response and SDK-chunk stream replays; azure: 19-shape request corpus + the
vendored characterization corpus second gate + content-filter
response/stream rows with the azure dialect's per-chunk model re-read;
azure_ai: the Foundry override-set and no-spoof Claude-route corpora;
xai: a 21-shape generated characterization corpus — provenance v1
in-process at HEAD, zero recorded vendor fixtures exist — two-sided over
`tests/test_litellm/translation/characterization_xai/` plus 7 rows pinning
v1's UnsupportedParamsError raises and the line-seam stream replays),
fail-closed everywhere else, with non-streaming flag-gated seams live in
`completion()` for the anthropic, bedrock, and google routes (the
openai/azure seam forks are integrator scope and NOT wired; the google
forks live in `litellm/translation_seam_google.py` and
`litellm/translation_seam_google_send.py` and route via v1's own
`get_vertex_ai_model_route`). ALL forks share one `_raw_openai_body`:
every caller-set OpenAI param rides into the parse so unknown ones fall
back typed, never silently dropped; all forks gate through
`dispatch.route`, and the anthropic/google sends share the engine's
injected-HttpPort skeleton (bedrock stays separate: SigV4 signs after
the body is final).
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
empty tools/stop lists, null/non-function tool_calls — tool ARGUMENT
strings themselves ride verbatim via ToolUse.arguments_raw, so real
compact-spaced replayed histories are served), the `user` param
(model-list gated in v1),
`response_format` on gpt-4/gpt-3.5-turbo-16k, `stream_options`, file blocks
(v1 downloads http pdf file_ids in-transform), and `http://` image URLs. On
streams, the trailing `choices: []` usage chunk passes through verbatim and
the wrapper's synthesized final usage chunk stays a seam/envelope concern.
Deliberate google fallback surfaces: cache markers whose conservative token
bound (UTF-8 bytes + per-message margin over the whole request) reaches
gemini's 1024-token cache minimum (context-cache create is network I/O;
below the bound v1 provably skips the call and ignores the markers), any
media block in a marker-bearing request (v1 token-counts images at 250),
message `name` beside cache markers (raw guard: the IR drops `name`, so
its bytes cannot be bounded post-parse while v1 token-counts them),
gs:// and http:// media plus AI-Studio https media (downloads), https media
without a recognizable extension, file/pdf parts (inbound), params outside
the IR (n, seed, penalties, modalities, audio, web_search_options,
service_tier, labels — inbound boundary fallback), hosted tools,
reasoning_effort xhigh/max (v1 raises), gemini-3 thinking budgets (read
`litellm.enable_gemini_default_thinking_level_low`), gemini-3
reasoning_effort+thinking (v1's conflict check raises), models without
`supports_system_messages` carrying system prompts, multi-candidate /
flagged / promptFeedback-blocked responses, inline media outputs, and
data:-URI tool results (functionResponse.parts). The vertex_anthropic
fallbacks mirror bedrock_invoke (spoof crossing, output_config/format)
plus beta-emitting shapes. Ambient globals force v1 at the seam:
`vertex_ai_safety_settings`, `custom_prompt_dict`, `modify_params`.
Deliberate azure fallback surfaces (each names the v1 path): cache_control
anywhere (azure forwards it verbatim), explicit `stream: false`, the
api-version gates (tool_choice pre-2023-12, `required` on 2024-05,
response_format pre-2024-08 or on gpt-3.5/gpt-35 — v1's synthetic json-tool
strategy and its json_mode response conversion are unported, so json_mode
requests never reach v2), o-series/gpt-5 on `base_model or model`, and for
azure_ai: text-only content lists (v1 flattens them), model-map-gated
tool_choice, grok models, and `x-anthropic-billing-header` system blocks on
the Claude route. None of the azure family fast-paths; the api-version
URL/query, deployment SDK client, api-key-vs-AD-token auth and the
`.../anthropic/v1/messages` rewrite are envelope (seam scope, unwired here).
Deliberate xai fallback surfaces (each names the v1 path): every param v1's
supported-list gate RAISES on (max_completion_tokens on every grok model —
the rename arm is dead code; stop on grok-3-mini/grok-4/grok-code-fast;
frequency_penalty on grok-4/grok-code-fast; reasoning_effort on
non-reasoning models per the model map), `web_search_options` (v1 reroutes
to the Responses-API bridge above the chat seam), `use_xai_oauth` (browser
PKCE flow inside validate_environment), explicit `stream: false` (the httpx
path keeps the key on the wire), user-message `name` (v1 forwards it; non-
user names are STRIPPED by v1, so those serve through the IR drop), tool
definitions with `strict` keys below the function level (v1 deletes every
depth), the openai guard's raw shapes, and the parse-level unknowns v1
passes through for grok (presence/frequency penalties on supported
families, seed, logprobs, top_logprobs, logit_bias, n, stream_options,
web_search_options). The xai completion() fork is NOT wired (integrator
scope, like openai/azure); when it lands these are HARD OBLIGATIONS, not
notes: the in-package `use_xai_oauth` guard arm is defense-in-depth ONLY
and unreachable through `_raw_openai_body` (use_xai_oauth is a litellm
param, never in non_default_params — the canary
`test_use_xai_oauth_guard_reachability_facts` pins the classification), so
the fork MUST either include the kwarg in the raw body it routes (making
the guard live) or fall back on it BEFORE building deps, and MUST pin that
with a completion()-level `use_xai_oauth=True` fallback test before the
xai flag can turn on; route through `dispatch.route` with provider "xai";
keep the bare wire model (the B1 re-prefix arm must never fire: xai
presets no model); synthesize the final stream usage chunk from the
passthrough `choices: []` tail (non-tail usage is withheld exactly like
v1's wrapper, so the synthesis covers the tail-chunk shape only).
Not yet here, each its own follow-up: streaming seams live; the other
inbound schemas (`anthropic_messages`, `google_genai`, `responses`,
`completions`); the same-family fast path (waits on the opaque-body
relay). To add a provider: write
`providers/<name>/`, register it in `engine/pipeline._SERIALIZERS` /
`_RESPONSE_PARSERS` / `_RESPONSE_DIALECTS` (plus `_RAW_GUARDS` when the
inbound schema is the provider's own family), add a differential corpus,
keep the flag off until differential-green.
