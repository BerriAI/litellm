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
├── dispatch.py     # route(): per-provider allowlist + same-family fast path;
│                   #   NeverPortProvider/NEVER_PORT — the 23 permanent v1
│                   #   fallbacks AS CODE (disjointness + typed-fallback
│                   #   tests in test_dispatch.py; promotion = a deliberate
│                   #   two-file edit). baseten, cohere(v1-route) and the
│                   #   wave-1b drops stay OUT (own canaries / route
│                   #   predicate in the future cohere module)
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
│   │   ├── stream.py    # SSE chunk -> wire_chunk events normalized to the
│   │   │                #   SDK-dump shape (service_tier preset to None:
│   │   │                #   the validated SDK chunk materializes it even
│   │   │                #   when the wire omits the key, and v1's wrapper
│   │   │                #   copies it onto every chunk); the openai chunk
│   │   │                #   dialect folds them; make_parse_line is the ONE
│   │   │                #   data:-line decode every same-family provider
│   │   │                #   composes
│   │   └── httpx_chunk.py # the ONE httpx (dict-path) chunk normalizer
│   │                    #   behind every BaseModelResponseIterator-style
│   │                    #   v1 decode: make_parse_event(HttpxChunkPolicy)
│   │                    #   owns the shared rebuild (extras/fingerprint
│   │                    #   dropped, tool_call type "function" default,
│   │                    #   usage only on the choices:[] tail, the
│   │                    #   reasoning -> reasoning_content rewrite by
│   │                    #   ReasoningMode, strict-envelope keys, usage-fold
│   │                    #   hook). Consumers: xai (rename + usage fold) and
│   │                    #   cometapi (copy_both + strict envelope); wave-2b
│   │                    #   dict-path providers (groq's pop == "rename",
│   │                    #   openrouter's unconditional variant) EXTEND
│   │                    #   ReasoningMode with their arm in the same commit
│   │                    #   as the consumer + its differential rows — the
│   │                    #   critic rejects another copy of this machinery
│   │                    #   (critic-wave2a M2; no consumer, no arm)
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
│   │   └── stream.py    # openai parser + per-chunk model re-attach;
│   │                    #   "azure" chunk dialect (the SDK service_tier
│   │                    #   preset is the shared openai parser's)
│   ├── azure_ai/        # the Foundry override set + the Claude route:
│   │   ├── guard.py     # azure guard + the text-only content-list flatten
│   │   ├── serialize.py # azure_ai gates (grok, model-map tool_choice) then
│   │   │                #   the openai_compat body
│   │   ├── response.py  # openai parser + the azure_ai/{model} rename
│   │   ├── stream.py    # re-export of the azure parser ("azure" dialect)
│   │   └── claude.py    # anthropic serializer/parsers re-exported; NO
│   │                    #   response-format model spoof (v1 maps with the
│   │                    #   real model); billing-header blocks fail closed
│   ├── compat_sdk/      # the SDK-path openai-compat family in ONE
│   │   │                #   subpackage (the google_genai "one family,
│   │   │                #   parameterized" precedent): wave-1a together_ai,
│   │   │                #   cerebras, nvidia_nim, lm_studio, llamafile,
│   │   │                #   lambda_ai, nebius, novita, wandb,
│   │   │                #   featherless_ai, nscale, hyperbolic, volcengine;
│   │   │                #   wave-1b shims ai21_chat (the get_llm_provider
│   │   │                #   coercion is CONDITIONAL on ai21_chat_models
│   │   │                #   membership — uncoerced "ai21" traffic has no
│   │   │                #   row and falls back, canary-pinned), dashscope,
│   │   │                #   docker_model_runner, empower, friendliai,
│   │   │                #   galadriel, github, inception, meta_llama,
│   │   │                #   morph, v0, zai, vercel_ai_gateway (its
│   │   │                #   dedicated elif is dead code — the compat list
│   │   │                #   matches first); wave-1b JSON-registry rows
│   │   │                #   (providers.json), ONLY the 14 LlmProviders
│   │   │                #   enum members — the dynamic JSONProviderConfig
│   │   │                #   = base list minus tool params unless
│   │   │                #   supports_function_calling({slug}/{m}), plus
│   │   │                #   param_mappings mct renames; the 7 NON-members
│   │   │                #   (veniceai, abliteration, llamagate, gmi,
│   │   │                #   sarvam, aihubmix, crusoe) ride v1's generic
│   │   │                #   openai fallback arms (no provider config at
│   │   │                #   param/transform time) and stay DROPPED,
│   │   │                #   canary-pinned. aiml is dropped the same way
│   │   │                #   (its config class is unregistered at HEAD).
│   │   │                #   Wave 2a: perplexity, sambanova, deepinfra,
│   │   │                #   moonshot — all SDK-path (cometapi, wave-2a's
│   │   │                #   httpx member, moved to compat_httpx at the
│   │   │                #   sibling merge: that family IS the
│   │   │                #   dedicated-elif shape; critic-wave1b
│   │   │                #   reconciliation).
│   │   │                #   All ride v1's big openai elif into the SDK, so
│   │   │                #   the body is openai_compat.assemble_body after
│   │   │                #   per-provider gates; the response parser is
│   │   │                #   openai_compat's verbatim (same live normalizer)
│   │   │                #   and the {provider}/{wire_model} re-prefix is
│   │   │                #   the SEAM's preset arm, never parser scope;
│   │   │                #   streams are the "openai" chunk dialect (pinned
│   │   │                #   per provider by wrapper replays). baseten is
│   │   │                #   DELIBERATELY ABSENT: its streams ride a
│   │   │                #   dedicated legacy wrapper branch
│   │   │                #   (handle_baseten_chunk), so it stays a typed v1
│   │   │                #   fallback (canary-pinned).
│   │   ├── checks.py    # the generic supported-list checker BOTH compat
│   │   │                #   families compose: unsupported_against (the
│   │   │                #   _check_valid_arg mirror incl. the top_k
│   │   │                #   extra_body default arm), BASE_LIST,
│   │   │                #   base_list_unsupported, the user notes
│   │   │                #   (critic-wave1b M3 concept split)
│   │   ├── json_registry.py # the providers.json mechanism: the dynamic
│   │   │                #   JSONProviderConfig fc-capability fork + the
│   │   │                #   param_mappings JSON_RENAME set, co-located so
│   │   │                #   the mirror test reconciles ONE module;
│   │   │                #   membership (JSON_REGISTRY_PROVIDERS) stays
│   │   │                #   beside ALLOWED in params.py
│   │   ├── params.py    # per-provider supported-list truths as pure gates
│   │   │                #   (v1 RAISES-unless-drop_params on anything off
│   │   │                #   the list); capability gates read deps over the
│   │   │                #   LOAD-BEARING {provider}/{model} map keys
│   │   │                #   (together_ai/sambanova function calling,
│   │   │                #   cerebras/perplexity/deepinfra/moonshot
│   │   │                #   reasoning); nvidia_nim's static per-model
│   │   │                #   table; moonshot's kimi-thinking-preview
│   │   │                #   exclusion and rewrite fallbacks (tool_choice
│   │   │                #   required -> synthetic message, reasoning-model
│   │   │                #   fill_reasoning_content), deepinfra's
│   │   │                #   tool_choice∉{auto,none} raise, sambanova's
│   │   │                #   non-text content fallback (v1's lossy flatten);
│   │   │                #   ALLOWED — the per-provider MAXIMAL
│   │   │                #   allowed sets, the ONE source the gates narrow
│   │   │                #   and serialization derives emission from
│   │   ├── serialize.py # frozen CompatProfile per provider (gate + mct
│   │   │                #   rename / together's rf-text drop / meta_llama's
│   │   │                #   non-json_schema response_format drop / wave-2a
│   │   │                #   named deltas: deepinfra's tool_choice drop +
│   │   │                #   zero-temperature floor, moonshot's temperature
│   │   │                #   pop/clamp VALUE rewrites, the sambanova/
│   │   │                #   moonshot text-content-list flatten with the
│   │   │                #   request-wide multimodal skip; user and
│   │   │                #   reasoning_effort emission DERIVED from
│   │   │                #   params.ALLOWED, never cached as booleans) ->
│   │   │                #   gates -> openai_compat assemble_body ->
│   │   │                #   deltas; PROFILES is the family registry and
│   │   │                #   SERIALIZERS its derived serializer table —
│   │   │                #   pipeline splices them whole, one line per
│   │   │                #   TABLE
│   │   └── guard.py     # explicit stream:false (the SDK serializes the
│   │   │                #   key; absent-vs-false is lost in the IR — and
│   │   │                #   the httpx path keeps it on the wire, so the
│   │   │                #   arm covers the httpx members too), then
│   │   │                #   the shared openai guard with the full
│   │   │                #   message-name fallback (nobody here strips
│   │   │                #   names); GUARDS is the COMPLETE per-provider
│   │   │                #   table (overrides: dashscope/zai preserve
│   │   │                #   cache_control on the wire; docker_model_runner/
│   │   │                #   publicai flatten content lists) — pipeline
│   │   │                #   splices it whole
│   ├── compat_httpx/    # the httpx-path shim family (dedicated
│   │   │                #   completion() elifs, transforms LIVE, NO seam
│   │   │                #   model preset — the xai routing shape): heroku,
│   │   │                #   bedrock_mantle, minimax, compactifai,
│   │   │                #   amazon_nova, datarobot, gradient_ai, ovhcloud,
│   │   │                #   lemonade (param config unregistered at HEAD;
│   │   │                #   the elif threads LemonadeChatConfig explicitly
│   │   │                #   — facts canary), and cometapi (wave-2a's httpx
│   │   │                #   member, moved here at the sibling merge — the
│   │   │                #   dedicated elif IS this family's routing fact)
│   │   ├── params.py    # per-provider allowed sets over the shared
│   │   │                #   compat_sdk/checks.py checker (gradient_ai's
│   │   │                #   own map RAISES even on
│   │   │                #   user; bedrock_mantle reasoning is capability-
│   │   │                #   gated; amazon_nova's static list serves
│   │   │                #   reasoning_effort unconditionally)
│   │   ├── serialize.py # frozen HttpxProfile rows (mct rename +
│   │   │                #   response_model_prefix data) -> gates ->
│   │   │                #   openai_compat assemble_body -> deltas
│   │   ├── guard.py     # the compat_sdk default (stream:false + full
│   │   │                #   name fallback) + heroku content-list flatten
│   │   │                #   and minimax cache_control arms; complete GUARDS
│   │   ├── response.py  # TWO v1 construction styles as family DATA
│   │   │                #   (RESPONSE_STYLES, the seam fork must read it;
│   │   │                #   the construction-arm gate makes the
│   │   │                #   response_dialect() shortcut fail tests):
│   │   │                #   "openai" = cdr (heroku/minimax/ovhcloud/
│   │   │                #   cometapi);
│   │   │                #   "openai_like" = ModelResponse(**json) direct —
│   │   │                #   NO stop->tool_calls rewrite, different pydantic
│   │   │                #   dump — with the {prefix}/{REQUEST model}
│   │   │                #   overwrite for compactifai / amazon-nova (the
│   │   │                #   literal hyphen) / lemonade; the seam's
│   │   │                #   to_model_response grew the "openai_like" arm
│   │   └── stream.py    # BOTH policies over openai_compat.httpx_chunk.
│   │                    #   make_parse_event (the B1 factory lift — no
│   │                    #   copies): the family policy (reasoning rename,
│   │                    #   no extras, usage only on the choices:[] tail —
│   │                    #   the xai policy minus the usage fold; the BASE
│   │                    #   OpenAIChatCompletionStreamingHandler rebuild)
│   │                    #   and cometapi's strict-envelope/copy-both
│   │                    #   policy; LINE_PARSERS is the per-provider table
│   │                    #   the streaming seam must select from; folds
│   │                    #   with the "xai" ChunkDialect (the generic httpx
│   │                    #   dict path); ovhcloud's custom handler is dead
│   │                    #   code in v1 (canary); provider error chunks are
│   │                    #   a LOUD v2 boundary error where v1's BASE
│   │                    #   handler silently swallows them (deliberate
│   │                    #   fail-closed divergence on a failure path,
│   │                    #   two-sided-pinned + a named report row; v1's
│   │                    #   cometapi handler RAISES instead — its policy
│   │                    #   row mirrors that raise)
│   ├── cohere/          # wave-2b-beta: the cohere V2 chat wire — the
│   │   │                #   DEFAULT route at HEAD for BOTH provider names
│   │   │                #   ("cohere" and "cohere_chat" resolve to
│   │   │                #   CohereV2ChatConfig and one main.py elif); the
│   │   │                #   legacy "v1/" route and the explicit "v2/"
│   │   │                #   model prefix (an envelope strip) are route
│   │   │                #   predicates in the guard, never Literal rows
│   │   ├── guard.py     # v1-route/v2-prefix predicates + explicit
│   │   │                #   stream:false + the shared openai guard (FULL
│   │   │                #   message-name fallback: the transform is the
│   │   │                #   inherited GPT one, names ride verbatim)
│   │   ├── params.py    # supported-list truths as typed fallbacks
│   │   │                #   (response_format/parallel_tool_calls/thinking/
│   │   │                #   reasoning_effort RAISE in v1; user is silently
│   │   │                #   dropped upstream — fallback, v1 serves)
│   │   ├── serialize.py # openai_compat assemble_body + renames (top_p->p,
│   │   │                #   stop->stop_sequences, mct->max_tokens),
│   │   │                #   tool_choice DROPPED (supported list, no map
│   │   │                #   arm), top_k emitted verbatim top-level (the
│   │   │                #   generic passthrough — wire-proven, NOT the
│   │   │                #   extra_body arm)
│   │   ├── response.py  # cohere-native body -> normalized chat-completion
│   │   │                #   body on ChatResponse.wire (content join,
│   │   │                #   citations->annotations, tool-call message
│   │   │                #   REPLACEMENT, usage.tokens — finish is ALWAYS
│   │   │                #   "stop", the wire id/finish are never read)
│   │   └── stream.py    # bare-JSON line seam (NO data:/[DONE] framing) ->
│   │                    #   GenericStreamingChunk payloads folded by the
│   │                    #   inbound "generic" chunk dialect; the v1 parser
│   │                    #   reads message-end off the ``event`` key while
│   │                    #   the real wire sends ``type`` — both regimes
│   │                    #   pinned (wrapper end-of-stream synthesis = seam
│   │                    #   scope)
│   ├── groq/            # wave-2b-beta: groq over openai_compat (httpx
│   │   │                #   path, bare wire model)
│   │   ├── guard.py     # explicit stream:false + the shared openai guard
│   │   ├── params.py    # thinking raise, user drop, reasoning_effort
│   │   │                #   capability gate (groq/{m}), the json_schema
│   │   │                #   THREE-WAY fork (native serve / workaround
│   │   │                #   fallback / BadRequestError fallback)
│   │   ├── serialize.py # assemble_body + mct rename + reasoning_effort +
│   │   │                #   top_k (extra_body->wire merge) + assistant
│   │   │                #   None-strip
│   │   ├── response.py  # openai parse + verbatim wire + the service_tier
│   │   │                #   clamp (a MISSING service_tier key fails closed
│   │   │                #   — v1's getattr crashes); F2 arm; construction
│   │   │                #   arm "openai_like"
│   │   └── stream.py    # httpx_chunk factory, reasoning="rename" (groq's
│   │                    #   pop == rename, verified — no new arm); "xai"
│   │                    #   chunk dialect
│   ├── mistral/         # wave-2b-beta: mistral over openai_compat (httpx
│   │   │                #   path, dedicated elif, bare wire model)
│   │   ├── guard.py     # own name matrix (tool-role names kept by v1;
│   │   │                #   the image branch forwards every name) + the
│   │   │                #   shared openai guard with skip_name_fallback;
│   │   │                #   NO stream:false arm (v1's map only copies
│   │   │                #   stream=True — explicit False never reaches
│   │   │                #   the wire)
│   │   ├── params.py    # user (silent drop) + thinking/reasoning_effort
│   │   │                #   (magistral system-prompt injection / raise)
│   │   │                #   as typed fallbacks
│   │   ├── serialize.py # assemble_body + mct rename, tool_choice
│   │   │                #   required->any (dict form silently dropped),
│   │   │                #   $id/$schema strip at v1's depth cap, top_k
│   │   │                #   verbatim; two-branch message munge (image ->
│   │   │                #   verbatim base output; else flatten +
│   │   │                #   MistralToolCallMessage + empty-assistant
│   │   │                #   removal + None strip)
│   │   ├── response.py  # the two raw pre-steps (empty->None FIRST, then
│   │   │                #   magistral content-list collapse: LAST text
│   │   │                #   wins, thinking joins -> reasoning_content)
│   │   │                #   then the shared openai parser
│   │   └── stream.py    # content-list normalize pre-step over the
│   │                    #   httpx_chunk factory (reasoning="rename" + the
│   │                    #   passthrough_delta_keys axis admitting
│   │                    #   thinking_blocks); "xai" chunk dialect
│   ├── sagemaker_chat/  # wave-2b-beta: SageMaker Messages-API endpoints —
│   │   │                #   the BASE GPT config over SigV4 transport (the
│   │   │                #   sign-after-body-final bedrock precedent;
│   │   │                #   endpoint name == model; sagemaker_nova is
│   │   │                #   deliberately unregistered)
│   │   ├── guard.py     # explicit stream:false + the shared openai guard
│   │   ├── params.py    # thinking/reasoning_effort raises, user drop,
│   │   │                #   the gpt-4-named-endpoint response_format raise
│   │   ├── serialize.py # assemble_body verbatim + top_k (mct passes
│   │   │                #   VERBATIM — the widest list in the wave)
│   │   ├── response.py  # re-export of the shared openai parser (bare
│   │   │                #   wire model, construction arm "openai")
│   │   └── stream.py    # openai parser + the litellm-validation post-step
│   │                    #   (psf:None on every delta, tool type null ->
│   │                    #   "function") at the AWS event-stream
│   │                    #   PARSED-event seam; "openai" chunk dialect
│   ├── watsonx/         # wave-2b-beta: watsonx /ml/v1/text/chat over the
│   │   │                #   OpenAILikeChatHandler route (auth incl. the
│   │   │                #   IAM-token network POST is envelope; project/
│   │   │                #   space ids ride TranslationDeps)
│   │   ├── guard.py     # the shared openai guard (full name fallback);
│   │   │                #   NO stream:false arm (the wire ALWAYS carries
│   │   │                #   the stream key)
│   │   ├── params.py    # mct/parallel_tool_calls/thinking raises, the
│   │   │                #   top_k legacy watsonx_text ValueError, user
│   │   │                #   drop, deployment/ models, missing project+
│   │   │                #   space (v1 raises 401) — all typed fallbacks
│   │   ├── serialize.py # assemble_body + stream-always + model_id +
│   │   │                #   project_id/space_id injection + tools
│   │   │                #   strict/additionalProperties-False strip +
│   │   │                #   tool_choice_option split + reasoning_effort
│   │   │                #   verbatim
│   │   ├── response.py  # openai parse for validation, verbatim wire with
│   │   │                #   the LIVE "watsonx/{wire_model}" prefix (the
│   │   │                #   seam constructs with usage_style
│   │   │                #   "openai_like"); F2 non-string-model arm
│   │   └── stream.py    # databricks ModelResponseIterator mirror ->
│   │                    #   generic payloads ("generic" dialect; data:-
│   │                    #   OPTIONAL line seam; v1's silent swallows of
│   │                    #   malformed lines/chunks are PINNED DIVERGENCES)
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
│       └── stream.py    # SSE dict-path parser = httpx_chunk.make_parse_event
│                        #   with the xai policy (reasoning="rename" + the
│                        #   per-chunk usage fold hook); the folded usage
│                        #   attaches ONLY to the choices:[] tail (v1's
│                        #   wrapper strips it elsewhere), no extras/
│                        #   system_fingerprint passthrough, refusal
│                        #   forwarded when it rides a role/content delta,
│                        #   tool_call type "function" default; the "xai"
│                        #   chunk dialect folds it
└── engine/
    ├── pipeline.py # prepare (pure, drives the fallback decision) -> send;
    │               #   per-provider serializer/parser/dialect tables —
    │               #   family packages register as DATA (compat_sdk's
    │               #   SERIALIZERS spliced in whole; one line per table
    │               #   per family, never per provider); the wire_body
    │               #   step strips transform-seam markers (json_mode;
    │               #   converse additionalModelRequestFields.stream)
    ├── http.py     # the injected HttpPort + ExecuteError values
    └── stream.py   # the ONE accumulator: lines OR parsed events -> IR
                    #   events -> chunks (fold_lines / fold_events)
```

The v1-side adapter lives OUTSIDE the package in `litellm/translation_seam.py`:
deps building from litellm ambient state, ModelResponse/ModelResponseStream
envelope adaptation (per-provider usage construction, because `Usage` only
serializes explicitly-set fields; on stream chunks the seam presets
`citations: None` for content chunks ONLY when the body lacks the key, so a
wire-carried value — perplexity — survives, pinned by the wave-2a citations
stream row), the `completion()` forks (anthropic branch
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

OpenAI-chat-in to seventy-one providers out (wave-2b-beta adds
`cohere`/`cohere_chat` (one module), `mistral`, `watsonx`,
`sagemaker_chat`, and `groq` — see their paragraphs below) —
`anthropic`,
`bedrock_converse`, `bedrock_invoke`, `openai_compat`, `vertex_ai` (gemini
route), `gemini` (AI Studio), `vertex_anthropic`, `azure`, `azure_ai`,
`azure_ai_anthropic`, `xai`, the thirteen wave-1a compat_sdk providers
(`together_ai`, `cerebras`, `nvidia_nim`, `lm_studio`, `llamafile`,
`lambda_ai`, `nebius`, `novita`, `wandb`, `featherless_ai`, `nscale`,
`hyperbolic`, `volcengine`), the twenty-seven wave-1b compat_sdk rows
(shims `ai21_chat`, `dashscope`, `docker_model_runner`, `empower`,
`friendliai`, `galadriel`, `github`, `inception`, `meta_llama`, `morph`,
`v0`, `zai`, `vercel_ai_gateway`; JSON-registry enum members `publicai`,
`helicone`, `xiaomi_mimo`, `scaleway`, `synthetic`, `apertis`, `nano-gpt`,
`poe`, `chutes`, `assemblyai`, `charity_engine`, `neosantara`,
`tensormesh`, `parasail`), and the nine wave-1b compat_httpx providers
(`heroku`, `bedrock_mantle`, `minimax`, `compactifai`, `amazon_nova`,
`datarobot`, `gradient_ai`, `ovhcloud`, `lemonade`), and the five wave-2a
providers (`perplexity`, `sambanova`, `deepinfra`, `moonshot` on the SDK
path, `cometapi` on the httpx path) — request, response, and stream
translation,
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
v1's UnsupportedParamsError raises and the line-seam stream replays;
compat_sdk: per-provider generated corpora vs v1 in-process at HEAD
(`test_differential_compat_sdk_{request,response,stream}.py` over
`_compat_sdk_corpus.py` — served rows, UnsupportedParamsError raise rows,
preset-model re-prefix response rows, per-provider wrapper stream replays,
and supported-list mirror drift gates over every model-map row; wave-2a
adds the VALUE-rewrite IDENTICAL rows — moonshot temperature pop/clamp,
deepinfra zero-temp floor and tool_choice drop, the flatten rows — the
perplexity citations-dormancy response row and wire-citations stream row,
and cometapi's dedicated no-prefix/line-seam gates in
`test_differential_cometapi_{response,stream}.py`)),
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
web_search_options).
Deliberate compat_sdk (wave-1a) fallback surfaces (each names the v1
path): every IR-carried param a provider's supported list excludes — v1's
`_check_valid_arg` raises UnsupportedParamsError or drops under
drop_params, so the typed fallback serves v1's own behavior
(max_completion_tokens on nscale/hyperbolic — both rename arms are dead
code behind the list gate; tools/tool_choice/response_format on
featherless_ai, on volcengine (response_format only), on together_ai
models without the supports_function_calling map flag, and on
nvidia_nim's reduced static-table models; parallel_tool_calls on
cerebras/featherless_ai/nscale/hyperbolic/volcengine; reasoning_effort
everywhere except capability-flagged cerebras models; response_format on
base-list providers when the model is literally named gpt-4 /
gpt-3.5-turbo-16k); `user` wherever a provider's own list doesn't carry
it (v1's base list gates it on openai model-list membership and silently
drops it otherwise — only cerebras and hyperbolic emit it);
volcengine `thinking` (v1 packs the verbatim dict into extra_body for the
SDK to merge top-level); lm_studio's bare-`schema` response_format wrap
(non-canonical inbound shape, parse rejects it); explicit `stream: false`
(the SDK serializes the key; absent-vs-false is lost in the IR); the
openai guard's raw shapes with the FULL message-name fallback (no config
here strips names); and the parse-level unknowns (seed, penalties,
logprobs, n, logit_bias, stream_options, ...) regardless of whether a
given provider's list serves or raises on them. The compat_sdk
completion() forks are NOT wired (integrator scope, like openai/azure/
xai); when they land, the seam must pre-set
`ModelResponse(model=f"{provider}/{model}")` exactly like openai.py:
676-677 so `_to_model_response_openai`'s re-prefix arm reproduces
cdr:699-710 (pinned per provider by the preset-model differential rows),
and baseten must NEVER be added to the family without resolving its
dedicated legacy wrapper stream branch (handle_baseten_chunk — the
test_baseten_drop_canary evidence).
Wave-1b additions to that family inherit every surface above, plus
(each names the v1 path): ai21_chat's own list (top_p RAISES; n/seed are
parse-level unknowns v1 serves); inception's list (top_p RAISES;
reasoning_effort is served UNCONDITIONALLY — emission derived from
ALLOWED); morph (everything but stream RAISES) and v0 (everything but
stream/tools/tool_choice RAISES); zai (mct/response_format RAISE; verbatim
top-level `thinking` for capability models is an unported emission, typed
fallback); meta_llama's non-json_schema response_format drop (a serializer
delta, both sides drop json_object silently); dashscope/zai cache_control
preserved on the wire (guard arm — the v2 serializer strips it);
docker_model_runner/publicai content-list flatten (guard arm — ANY
list-form content falls back); the JSON-registry function-calling fork
(tools/tool_choice/parallel_tool_calls RAISE unless
supports_function_calling({slug}/{m})). DROPPED with re-evaluate canaries:
aiml (config class unregistered at HEAD — the generic fallback stack
renames mct; registration would flip it to verbatim) and the 7 non-enum
JSON providers (no provider config at param/transform time).
Deliberate compat_httpx (wave-1b) surfaces: the same family fallbacks
(supported-list raises incl. gradient_ai's map-level raise on `user`,
explicit stream:false — the httpx path keeps the key on the wire — the
openai guard's raw shapes with the full message-name fallback, parse-level
unknowns), heroku's content-list flatten and minimax's preserved
cache_control (guard arms), minimax `thinking` (verbatim top-level copy,
unported emission), bedrock_mantle reasoning_effort on non-flagged models
(v1 raises). The compat_httpx completion() forks are NOT wired (integrator
scope); when they land these are HARD OBLIGATIONS: NO model preset (fresh
ModelResponse — the xai R4 rule; the compat_sdk preset arm must never fire
for this family), to_model_response MUST select the style from
compat_httpx.response.RESPONSE_STYLES ("openai" = cdr for heroku/minimax/
ovhcloud/cometapi; "openai_like" = ModelResponse(**json) for the other six
— the seam arm added for this family), streams fold with the "xai"
ChunkDialect over compat_httpx.parse_line, and the family parser's
fail-closed non-string wire `model` arm must stay ahead of the fork
(verifier-longtail F2): v1's OpenAILike construction raises pydantic
ValidationError BEFORE the prefix overwrite, so on
compactifai/amazon_nova/lemonade a wired fork that bypassed the parser's
arm would SERVE what v1 raises on — keep the typed fallback (pinned by
the family's ValidationError differential rows and the datarobot canary;
`model: None` stays served, v1 constructs it).
Deliberate wave-2a fallback surfaces (each names the v1 path): every
supported-list raise (perplexity stop/tools/tool_choice/parallel and
reasoning_effort on non-reasoning models; sambanova tools/tool_choice/
parallel on models without the sambanova/{m} supports_function_calling
flag; deepinfra parallel_tool_calls, reasoning_effort off
deepinfra/{m}-flagged models, and its CUSTOM tool_choice raise on every
value outside {auto, none}; moonshot tools/tool_choice on
kimi-thinking-preview models); the v1 body REWRITES v2 deliberately does
not reproduce (moonshot tool_choice="required" appends a synthetic user
message; moonshot reasoning models inject reasoning_content into
assistant tool-call history; sambanova's lossy flatten of non-text
content lists, which also skips the base image transforms); `top_k`
family-wide (the shared default arm: v1 packs it into extra_body via the
provider-specific passthrough and the SDK/handler merges it top-level —
wire-proven for all 18 members; critic-wave2a M1 flipped the reason text
from the transform-output-only "silently drops" reading that
verifier-wave1a F6 had pinned); `user` everywhere (model-list gated in
v1); and the family's shared guard/parse fallbacks. The VALUE rewrites are SERVED, never
fallbacks (the wave-2a brief): moonshot temperature pop (reasoning
models) and >1 -> int 1 clamp, deepinfra temperature-0 -> 0.0001 on
mistralai/Mistral-7B-Instruct-v0.1, deepinfra tool_choice auto/none
silently dropped, and the text-only content-list flatten with moonshot's
request-wide multimodal skip — each pinned by IDENTICAL differential
rows. cometapi fork obligations (a compat_httpx family row since the
sibling merge; when the integrator wires it): the fork must NOT preset a
model (fresh ModelResponse, bare wire model — pinned by the no-prefix
rows, the xai R4 shape), must route streams through
compat_httpx.LINE_PARSERS["cometapi"] with the "xai" chunk dialect at the
SSE line seam (NOT the SDK wrapper arm), and inherits the
synthesized-final-usage contract from the openai/xai ports.
Streaming-seam obligations carried from wave 2a (verifier-wave2a W1/W2 —
no impact while streaming stays on v1, pinned by
`test_reasoning_stream_seam_obligation_canary`): the SDK family's openai
chunk parser typed-errors on `reasoning_content` deltas as "unreachable",
but wave 2a SERVES `reasoning_effort` on perplexity/deepinfra (and 1a on
cerebras) whose real streams carry reasoning deltas v1's wrapper serves —
the streaming seam must teach the openai parser + dialect reasoning
deltas (and decide finish+content interleaves) BEFORE flag-on streaming
for reasoning-capable family members; and the line decoders
(make_parse_line consumers) error loudly on malformed SSE lines that
v1's BaseModelResponseIterator silently swallows — the PR #30138
lenient-boundary call belongs to that seam, not the parsers.
The xai completion() fork is NOT wired (integrator
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
Deliberate wave-2b-beta cohere fallback surfaces (providers/cohere — both
provider names; each names the v1 path): the legacy ``v1/`` route (the v1
chat wire is DON'T-PORT; route predicate in the guard) and the explicit
``v2/`` model prefix (main.py strips it before transform — an envelope
rewrite v2 does not reproduce); every supported-list raise
(response_format, parallel_tool_calls, thinking, reasoning_effort —
NOTE the get_optional_params drift: the cohere arm runs the LEGACY
CohereChatConfig map, byte-equivalent to the v2 config's over the shared
list, probed at HEAD); ``user`` (silently dropped upstream, v1 serves the
drop); n/seed/frequency_penalty/presence_penalty (v1 SERVES them — renamed
n->num_generations — but they are parse-level unknowns, so v1 keeps
serving); explicit stream:false; message ``name`` (forwarded verbatim by
the GPT transform, full-name guard arm). SERVED quirks pinned IDENTICAL:
tool_choice silently dropped (in the supported list with no map arm),
top_k emitted verbatim top-level (wire-proven — the generic passthrough,
NOT the extra_body arm), mct->max_tokens, response finish_reason ALWAYS
"stop" (v1 never reads the wire finish; the fresh-Choices default
survives even on tool calls), the wire response id IGNORED (ambient
chatcmpl id), tool-call responses REPLACING the message (text content
lost, annotations kwarg explicit). Cohere stream/fork obligations (the
forks are NOT wired; integrator scope): the response construction arm is
"openai" with NO model preset and the request model riding the body; the
chunk fold dialect is "generic" (the wrapper's GenericStreamingChunk arm)
over providers/cohere.parse_line — a BARE-JSON line seam (v1's iterator
json.loads each line; no data:/[DONE] framing); the generic streaming
seam must mint a FRESH chunk id per chunk (v1's wrapper does), must NOT
preset citations/system_fingerprint (to_model_response_stream's openai
preset would diverge — the cohere stream gate's local adapter is the
contract), and OWNS v1's end-of-stream synthesis: the real (type-keyed)
cohere wire never matches the parser's event-keyed message-end arm, so
v1 synthesizes the trailing finish chunk at StopIteration ("tool_calls"
when tool deltas were seen, else "stop") and, under include_usage, a
final usage chunk with token-counter ESTIMATES (the wire usage never
reaches the wrapper on that regime). One PINNED DIVERGENCE (fail-closed
on a failure path, the compat_httpx error-chunk precedent): non-str
stream ``content.text`` — v1 silently swallows the chunk, v2 errors
loudly (named report row).
Deliberate wave-2b-beta mistral fallback surfaces (providers/mistral; each
names the v1 path): every supported-list raise (frequency_penalty,
presence_penalty, n, logprobs, web_search_options, and
thinking/reasoning_effort on NON-magistral models); the magistral
reasoning-prompt INJECTION (v1 rewrites the message list with a constant
system prompt at transform time — v2 falls back on thinking/
reasoning_effort everywhere, one arm for both regimes); ``user`` (silently
dropped upstream — dossier drift: researcher-4 listed it as a raise);
``seed`` (v1 packs extra_body.random_seed, parse-level fallback); tool-role
message ``name`` (v1 keeps it; non-tool names serve through the IR drop);
ANY message ``name`` beside image/file content (the image branch forwards
names verbatim); string-form stop; single-text content lists (v1 flattens
— the conservative shared arm, the sambanova precedent); text lists
flattening to "" (v1's truthy assignment keeps the LIST form). SERVED
quirks pinned IDENTICAL: explicit stream:false (v1's map only copies True
— NO guard arm, the IR collapse IS v1's drop), mct->max_tokens, top_k
verbatim top-level (wire-proven — researcher-4 listed it as a raise, the
probe refutes it), tool_choice required->any with the DICT form silently
dropped, tools $id/$schema strip at v1's exact call-site cap —
``max_depth=DEFAULT_MAX_RECURSE_DEPTH`` (=100 at HEAD; v2 imports the same
``litellm.constants`` symbol, so the env-overridable value cannot drift;
the signature's ``max_depth=10`` default is dead there, verifier F2)
(additionalProperties/strict KEPT — port the code, not the docstring),
multi-text flatten, MistralToolCallMessage rebuild, empty-assistant
removal, per-message None strip, the image branch's verbatim
base-transform passthrough, response empty-content->None BEFORE the
magistral content-list collapse (LAST text block wins; thinking texts
join "\\n" into reasoning_content), bare wire model, and the stream
content-list normalize (thinking_blocks signature "mistral" riding the
delta via the httpx_chunk factory's wave-2b-beta passthrough_delta_keys
axis — mistral is the axis's consumer). codestral reuses MistralConfig in
v1 but is NOT registered here (stays a v1 fallback; re-evaluate with its
own dossier). Mistral fork obligations (NOT wired; integrator scope): no
model preset (bare wire model, the xai R4 rule), construction arm
"openai", streams fold with the "xai" dialect over
providers/mistral.parse_line (standard data:/[DONE] SSE).
Deliberate wave-2b-beta watsonx fallback surfaces (providers/watsonx; each
names the v1 path): max_completion_tokens (RAISES — the OpenAILike rename
is dead behind the list gate), parallel_tool_calls/thinking/logit_bias/
web_search_options (raises), top_k and every legacy watsonx-text param
(the get_optional_params watsonx-only arm raises a bare ValueError naming
the watsonx_text provider — a DIFFERENT raise class, pinned), ``user``
(silent drop), ``deployment/`` models (deployment URLs, api_version pop,
no model_id/project_id payload — envelope), missing project AND space id
in deps (v1's _get_api_params raises WatsonXAIError 401), message ``name``
(forwarded verbatim — full-name guard arm), n/seed/penalties/logprobs/
top_logprobs (parse-level; v1 serves verbatim). SERVED quirks pinned
IDENTICAL: the body ALWAYS carries ``stream`` (the openai_like handler
re-adds ``stream or False`` unconditionally, so explicit false == absent —
NO guard arm), ``model`` AND ``model_id`` both set to the wire model,
project_id-or-space_id injection (project wins; ids ride
``TranslationDeps.watsonx_project_id``/``watsonx_space_id`` — the future
seam fork must run v1's _get_api_params resolution chain), tools
``strict`` stripped at every depth + ``additionalProperties`` removed only
where False, tool_choice auto/none/required -> ``tool_choice_option`` with
the dict form riding ``tool_choice`` verbatim, response_format (object AND
json_schema) + reasoning_effort verbatim (json_mode is NEVER set —
the OpenAILike json_mode machinery is dormant), responses constructed
ModelResponse(**json)-direct with the LIVE ``watsonx/{wire_model}`` prefix
(model None -> the literal "watsonx/"; non-string wire model fails closed
— the verifier-longtail F2 arm), and streams through the databricks
GenericStreamingChunk iterator (researcher-4 drift: NOT the plain openai
dialect) folded by the "generic" dialect — wire finish maps through
map_finish_reason (IBM time_limit/cancelled/error -> stop, max_tokens ->
length; unknown strings are conservative loud errors), name-only tool
starts ride with validated arguments "", mid-stream usage is stripped, the
choices=[] usage tail passes through (seam contract), and a stream without
a wire finish gets v1's SYNTHESIZED trailing stop (seam scope, the cohere
contract). PINNED DIVERGENCES (fail-closed on failure paths): non-JSON
stream lines and chunks failing ModelResponseStream validation — v1's
iterator swallows both silently, v2 errors loudly (named report rows).
Watsonx fork obligations (NOT wired; integrator scope): construction arm
"openai_like" (NEVER "openai" — the construction-arm gate applies), no
model preset, deps built with the resolved project/space ids, streams fold
with the "generic" dialect over providers/watsonx.parse_line.
Deliberate wave-2b-beta sagemaker_chat fallback surfaces
(providers/sagemaker_chat; each names the v1 path): thinking/
reasoning_effort (raises — the BASE GPT list, the widest in the wave),
``user`` (silent drop), response_format on endpoints literally named
gpt-4/gpt-3.5-turbo-16k (the base-list name gate raises — endpoint names
are arbitrary so the corner is REACHABLE), explicit stream:false (rides
the wire), aws_* kwargs (v1 places them in optional_params and the BODY
carries them — wire-probed aws_region_name; v2's parse rejects the
unknown key, v1 serves), message ``name`` (forwarded), and the
parse-level unknowns (n/seed/penalties/logprobs/logit_bias/
web_search_options — v1 serves all verbatim). SERVED pins:
max_completion_tokens VERBATIM (no rename), top_k top-level
(wire-proven), bare wire model on responses (cdr, construction arm
"openai", NO preset), and streams at the AWS event-stream PARSED-event
seam (botocore framing is transport — the bedrock precedent): the shared
openai parser + a post-step mirroring the decoder's litellm validation
(provider_specific_fields: None materialized on every delta, tool_call
type missing-or-null -> "function") with wrong-typed delta strings LOUD
(v1's StreamingChatCompletionChunk validation raises — not the watsonx
swallow). sagemaker_nova shares the main.py branch but carries its own
config overrides and stays a typed v1 fallback (canary in the request
gate). SigV4 + the /invocations[-response-stream] URL split are envelope;
the fork must sign AFTER wire_body finalizes (the bedrock rule).
Deliberate wave-2b-beta groq fallback surfaces (providers/groq; each names
the v1 path): ``thinking`` (raises), ``reasoning_effort`` on models
without the ``groq/{m}`` supports_reasoning flag (raises), ``user``
(silent drop), ``response_format`` with a json_schema on NON-native models
(v1 serves its json_tool_call WORKAROUND — tools + forced tool_choice +
json_mode + fake_stream, the cross-plane rewrite v2 deliberately does not
reproduce; researcher-4's prescribed shape), the same plus user tools (v1
raises ``litellm.BadRequestError`` — the wave's one
non-UnsupportedParamsError request raise, exact type pinned), explicit
stream:false, message ``name``, and the parse-level unknowns
(seed/n/penalties/logit_bias/logprobs/top_logprobs/service_tier/
web_search_options/max_retries — v1 serves or silently drops each).
SERVED pins: mct->max_tokens (groq's map forwards POSITIONALLY so the
OpenAILike rename runs — its own replace_...=False default is never
threaded), top_k via extra_body -> hh's wire merge (wire-equivalent
top-level emission), response_format json_object verbatim (fake_stream is
a routing key hh pops), json_schema VERBATIM on native-schema models
(llama-4 / gpt-oss / kimi-k2-0905 map rows; capability via deps over
groq/{m}), assistant-message None-strip (content: None never reaches the
wire), reasoning_effort verbatim on flagged models, responses
ModelResponse(**json)-direct (BARE wire model — the prefix arm is dead,
custom_llm_provider=None) with x_groq extras surviving and the
service_tier CLAMP ({auto,default,flex}, null/unknown -> "auto") — a
response body MISSING the service_tier key fails closed (v1's clamp reads
getattr with no default and CRASHES with AttributeError, probed), and
streams through the httpx_chunk factory reasoning="rename" (groq's pop ==
the existing mode, verified — no new ReasoningMode arm) with truthy error
chunks loud on both sides (v1 raises -> MidStreamFallbackError; error:
null serves, the value-check pin). Groq fork obligations (NOT wired;
integrator scope): construction arm "openai_like", no model preset,
streams fold with the "xai" dialect over providers/groq.parse_line, and
the main.py groq elif merges ``GroqChatConfig.get_config()`` CLASS-ATTR
state into optional_params (main.py:2338-2344) — ambient module state v2
cannot see, so the fork MUST fall back to v1 when that config is
non-empty (the ambient-globals rule: vertex_ai_safety_settings/
custom_prompt_dict precedent).
Not yet here, each its own follow-up: streaming seams live; the other
inbound schemas (`anthropic_messages`, `google_genai`, `responses`,
`completions`); the same-family fast path (waits on the opaque-body
relay). To add a provider that is a pure param-surface delta over an
existing wire family, add DATA rows to that family's package instead of a
new subpackage (the `compat_sdk`/`compat_httpx` shape, THE wave-1b/2
convention): an allowed frozenset + `ALLOWED` row in the family's
params.py, a profile row in its serialize.py registry (named gates ride
the profile's gate fn; deterministic VALUE rewrites ride its
`rewrite`/delta fields — wave-2a's moonshot/deepinfra rows are the
template), one dispatch `Provider` Literal line, and a corpus `SPECS`
row — pipeline already splices the family's exported
`SERIALIZERS`/`GUARDS` (and compat_httpx `PARSERS`) tables whole (one
`**` line per table per FAMILY; never add per-provider rows for a family
member; guard overrides go in the family's `_OVERRIDES`, the complete
`GUARDS` table derives from `ALLOWED`). The shared supported-list checker
is `compat_sdk/checks.py`: IMPORT it (`from ..compat_sdk import checks`),
NEVER copy `_CHECKS` or its helpers into a new family — copies of that
machinery get rejected, exactly like the httpx_chunk normalizer's
(critic-wave2a M3; critic-longtail NIT-6). HEADROOM: compat_sdk/params.py
is near the 720 hard cap — it is FULL; new FAMILIES (the wave-1b
openai_like shims, the httpx no-prefix group) get their OWN family package
per researcher-4, never more rows in that file (critic-wave2a M3) — and
the registration-completeness gate
(`test_differential_compat_sdk_request.py`) fails any registered provider
without a differential corpus row — for BOTH families. To add a provider with its own wire
format: write `providers/<name>/`, register it in
`engine/pipeline._SERIALIZERS` / `_RESPONSE_PARSERS` / `_RESPONSE_DIALECTS`
(plus `_RAW_GUARDS` when the inbound schema is the provider's own family),
add a differential corpus, keep the flag off until differential-green.
