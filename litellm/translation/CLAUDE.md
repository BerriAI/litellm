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
│   ├── deepseek/        # wave-2b-alpha own module (httpx dedicated elif
│   │   │                #   main.py:1942, transforms LIVE, bare wire model)
│   │   ├── guard.py     # explicit stream:false + the shared openai guard
│   │   │                #   (full message-name fallback: nothing strips)
│   │   ├── params.py    # base list + thinking/reasoning_effort
│   │   │                #   (UNCONDITIONAL, every deepseek model);
│   │   │                #   non-text content lists and thinking-mode
│   │   │                #   assistant history fall back (v1's lossy
│   │   │                #   flatten / _fill_reasoning_content rewrites)
│   │   ├── serialize.py # openai_compat body + the compat_sdk flatten
│   │   │                #   delta (imported, never copied) + the thinking
│   │   │                #   rewrite ({"type":"enabled"} only:
│   │   │                #   budget_tokens discarded, reasoning_effort
│   │   │                #   rewritten in and its key never on the wire,
│   │   │                #   a present non-enabled dict shadows it)
│   │   ├── response.py  # shared openai parser re-export (no v1 override;
│   │   │                #   NO deepseek/ prefix — the xai R4 shape)
│   │   └── stream.py    # httpx_chunk family policy (v1 = the BASE
│   │                    #   OpenAIChatCompletionStreamingHandler) +
│   │                    #   make_parse_line; "xai" chunk dialect
│   ├── fireworks_ai/    # wave-2b-alpha own module (httpx dedicated elif
│   │   │                #   main.py:2198; the WIRE model differs from the
│   │   │                #   request model and the response model carries
│   │   │                #   the fireworks_ai/ prefix INSIDE the parser)
│   │   ├── guard.py     # explicit stream:false + the shared openai guard
│   │   │                #   (cache_control needs NO arm: v1 strips it
│   │   │                #   recursively == the IR drop, served)
│   │   ├── params.py    # fireworks' OWN list (user SERVED; top_k mention
│   │   │                #   dead — extra_body crossing) + three capability
│   │   │                #   forks over fireworks_ai/{m}; the deps read is
│   │   │                #   STRICTLY NARROWER than v1's get_provider_info
│   │   │                #   default-true + hyphen-boundary scan (one-
│   │   │                #   direction mirror gate; honest drift notes);
│   │   │                #   response_format WITH tools falls back (v1's
│   │   │                #   json_mode cross-plane machinery); legacy-defs
│   │   │                #   tools fall back at the shared inbound arm
│   │   ├── serialize.py # openai_compat body + accounts/fireworks/models/
│   │   │                #   model rewrite + mct->max_tokens rename +
│   │   │                #   tool_choice required->any + function-level
│   │   │                #   strict strip (shared strip_function_strict) +
│   │   │                #   the #transform=inline image suffix (literal
│   │   │                #   "vision" substring gate; data: urls exempt;
│   │   │                #   the ambient disable global is a SEAM
│   │   │                #   obligation) + user/reasoning_effort verbatim
│   │   ├── response.py  # the shared make_direct_parser (OpenAILike
│   │   │                #   DIRECT construction, "openai_like" seam arm)
│   │   │                #   with the fireworks_ai/{WIRE model} policy +
│   │   │                #   the tool-calls-in-content repair fallback
│   │   │                #   (v1 mints uuid4 ids; the pure parser fails
│   │   │                #   closed)
│   │   └── stream.py    # httpx_chunk family policy (v1 = the BASE
│   │                    #   handler; chunks keep the BARE wire model);
│   │                    #   joins the ONE error-chunk PINNED DIVERGENCE
│   ├── hosted_vllm/     # wave-2b-alpha own module (httpx dedicated elif
│   │   │                #   main.py:2619, transforms LIVE, bare wire model)
│   │   ├── guard.py     # explicit stream:false + the shared openai guard
│   │   │                #   (its custom-tool / assistant-thinking_blocks
│   │   │                #   arms double as this provider's rewrite
│   │   │                #   fallbacks; file parts fall back inbound)
│   │   ├── params.py    # base list + thinking/reasoning_effort
│   │   │                #   (UNCONDITIONAL — no capability fork)
│   │   ├── serialize.py # openai_compat body + recursive tools cleaning
│   │   │                #   (strict any-depth; additionalProperties only
│   │   │                #   when false) + the thinking budget-band rewrite
│   │   │                #   (>=10000 high / >=5000 medium / >=2000 low /
│   │   │                #   else minimal; disabled+adaptive dropped;
│   │   │                #   explicit reasoning_effort WINS)
│   │   ├── response.py  # shared openai parser re-export (no v1 override;
│   │   │                #   NO hosted_vllm/ prefix)
│   │   └── stream.py    # httpx_chunk family policy (v1 = the BASE
│   │                    #   handler); joins the ONE error-chunk PINNED
│   │                    #   DIVERGENCE row
│   ├── huggingface/     # wave-2b-alpha own module — the api_base
│   │   │                #   (dedicated endpoint) ROUTE ONLY (httpx elif
│   │   │                #   main.py:3185, bare wire model)
│   │   ├── guard.py     # explicit stream:false + the shared openai guard
│   │   │                #   (the api_base arm sends messages VERBATIM)
│   │   ├── params.py    # the ROUTE gate first: deps.api_base None ->
│   │   │                #   EVERYTHING falls back (the router route's
│   │   │                #   3-segment names fetch the HF provider mapping
│   │   │                #   over HTTP INSIDE the transform; 1-segment
│   │   │                #   names crash v1); then the plain base list +
│   │   │                #   top_k (non-compat top-level passthrough)
│   │   ├── serialize.py # openai_compat body verbatim (v1's
│   │   │                #   ChatCompletionRequest passthrough: model and
│   │   │                #   messages untouched, mct verbatim) + top_k
│   │   ├── response.py  # shared openai parser re-export (no v1 override;
│   │   │                #   NO huggingface/ prefix)
│   │   └── stream.py    # httpx_chunk family policy (v1 = the BASE
│   │                    #   handler); joins the ONE PINNED DIVERGENCE
│   ├── openrouter/      # wave-2b-alpha own module (httpx dedicated elif
│   │   │                #   main.py:3354, transforms LIVE, bare wire model)
│   │   ├── guard.py     # explicit stream:false + the cache-capable-model
│   │   │                #   arm (claude/gemini/minimax/glm/z-ai substrings
│   │   │                #   KEEP cache_control + get the message-level
│   │   │                #   move; other models' base strip == the IR drop,
│   │   │                #   served) + the shared openai guard
│   │   ├── params.py    # base list + top_k (top-level passthrough:
│   │   │                #   openrouter is NOT a compat provider — wire-
│   │   │                #   proven) + reasoning_effort/thinking on DUAL-
│   │   │                #   keyed reasoning models (openrouter/{m} OR bare
│   │   │                #   {m}; openrouter/-prefixed ids answer False —
│   │   │                #   v1's prefix-strip); thinking falls back BOTH
│   │   │                #   ways (verbatim emission / UPE)
│   │   ├── serialize.py # openai_compat body + usage:{include:true}
│   │   │                #   ALWAYS + top_k/reasoning_effort verbatim
│   │   ├── response.py  # shared openai parser re-export (bare wire
│   │   │                #   model); usage.cost rides the dump verbatim and
│   │   │                #   the _hidden_params cost header is a FORK
│   │   │                #   obligation (gate-pinned)
│   │   └── stream.py    # OWN v1 handler -> per-provider policy: the
│   │                    #   "unconditional" ReasoningMode arm (added with
│   │                    #   this consumer), key-presence error raise,
│   │                    #   strict id/created/model/choices envelope +
│   │                    #   the missing-delta pre-step
│   ├── snowflake/       # wave-2b-alpha own module (httpx dedicated elif
│   │   │                #   main.py:4286; a genuine wire mapping; the
│   │   │                #   response model is snowflake/{wire model}
│   │   │                #   INSIDE the parser)
│   │   ├── guard.py     # the shared openai guard ONLY (messages ride
│   │   │                #   VERBATIM in v1 — no transform); deliberately
│   │   │                #   NO stream:false arm: stream is ALWAYS a body
│   │   │                #   key (absent == false on the wire), so v2
│   │   │                #   SERVES explicit false
│   │   ├── params.py    # the small static list (mct/stop/n/seed/
│   │   │                #   parallel RAISE; user silently dropped) +
│   │   │                #   top_k (non-compat top-level passthrough,
│   │   │                #   wire-proven via respx — the elif overwrites
│   │   │                #   the caller's client, so the mock-transport
│   │   │                #   helper can't reach it)
│   │   ├── serialize.py # openai_compat body + the ALWAYS-on stream key
│   │   │                #   + tools -> tool_spec + tool_choice -> object
│   │   │                #   (required -> any; dict fn -> name ARRAY) +
│   │   │                #   top_k verbatim
│   │   ├── response.py  # content_list pre-rewrite (choices[0]: text
│   │   │                #   concat + tool_use -> tool_calls with
│   │   │                #   json.dumps(input) DEFAULT-separator
│   │   │                #   arguments) + the shared direct parser with
│   │   │                #   the snowflake/{wire-or-empty} prefix policy;
│   │   │                #   "openai_like" seam arm; the request model in
│   │   │                #   _hidden_params["model"] is a fork obligation
│   │   └── stream.py    # httpx_chunk family policy (v1 = the BASE
│   │                    #   handler); real Cortex delta shapes UNPINNED
│   │                    #   upstream; joins the ONE PINNED DIVERGENCE
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

OpenAI-chat-in to seventy-one providers out — `anthropic`,
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
`datarobot`, `gradient_ai`, `ovhcloud`, `lemonade`), the five wave-2a
providers (`perplexity`, `sambanova`, `deepinfra`, `moonshot` on the SDK
path, `cometapi` on the httpx path), and the wave-2b-alpha own modules
(`deepseek`, `openrouter`, `hosted_vllm`, `fireworks_ai`, `snowflake`,
`huggingface` on its api_base route) — request, response, and stream
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
Deliberate wave-2b-alpha (deepseek) fallback surfaces (each names the v1
path): non-text content lists (v1's ALWAYS-on flatten drops non-text
parts and skips the base image transforms — v1 serves its lossy flatten);
assistant history in thinking mode on supports_reasoning("deepseek/{m}")
models (v1's _fill_reasoning_content patches every assistant message
missing reasoning_content with provider_specific_fields promotion or a
single-space placeholder — v1 serves its rewrite; the fill is INERT on
non-reasoning models like deepseek-chat, where the same shape SERVES,
pinned both ways); top_k (the wire-proven extra_body merge — pinned by a
mock-transport completion() row per the wave-2b wire-prove rule); user
(model-list gated, v1 silently drops); explicit stream:false (guard arm);
and the family guard/parse fallbacks (web_search_options is SERVED by
v1's base list at HEAD — a dossier drift fact — and falls back at the
inbound boundary like every parse-level unknown). The thinking rewrite
itself is SERVED, never a fallback: {"type":"enabled"} verbatim-only with
budget_tokens discarded, reasoning_effort != "none" rewritten to it (the
key never reaches the wire), a present non-enabled thinking dict shadows
reasoning_effort entirely — all pinned by IDENTICAL rows. deepseek fork
obligations (when the integrator wires it): NO model preset (fresh
ModelResponse, bare wire model — the xai R4 rule, pinned by the no-prefix
response rows), streams fold providers/deepseek.parse_line with the "xai"
chunk dialect at the SSE line seam, and the error-chunk PINNED DIVERGENCE
(v1's base handler swallows; v2 is loud) extends to deepseek exactly like
the compat_httpx family.
Deliberate wave-2b-alpha (openrouter) fallback surfaces (each names the
v1 path): thinking BOTH ways (reasoning-capable models: v1 serves the
wire dict VERBATIM, an unported emission — the zai/minimax precedent;
non-capable models: v1 raises UnsupportedParamsError, like
reasoning_effort off the dual-keyed capability); cache_control on
cache-capable models (claude/gemini/minimax/glm/z-ai substrings — v1
keeps it and moves message-level markers into the last content block;
on every OTHER model the base strip == the IR drop, SERVED and pinned);
`transforms`/`models`/`route` (v1's top-level passthrough — the map's
extra_body packing is DEAD at runtime, a dossier drift fact); user
(model-list gated); explicit stream:false (guard arm). SERVED, never
fallbacks: top_k (the top-level passthrough, wire-proven — openrouter is
NOT in openai_compatible_providers so there is no extra_body crossing),
reasoning_effort on dual-key-capable models (verbatim emission), and the
ALWAYS-injected `usage: {"include": true}` body key. openrouter fork
obligations (when the integrator wires it): NO model preset (fresh
ModelResponse, bare wire model); streams fold providers/openrouter
.parse_line (its OWN strict/raising policy row — NOT the family default)
with the "xai" chunk dialect; and the HARD OBLIGATION from the response
gate — v1 copies wire `usage.cost` into `_hidden_params
["additional_headers"]["llm_provider-x-litellm-response-cost"]` as a
float (the cost-calculator feed); the v2 body retains usage.cost
verbatim, so the fork MUST rebuild that header from the body before the
openrouter flag can turn on (pinned by
test_v1_cost_hidden_param_is_the_fork_obligation and the report's
SEAM CONTRACT row). The envelope (HTTP-Referer/X-Title headers, the
litellm.OpenrouterConfig class-attr extra_body merge in the elif) stays
fork scope.
Deliberate wave-2b-alpha (hosted_vllm) fallback surfaces (each names the
v1 path): custom-type tools (the shared openai guard — v1 synthesizes
function tools from them, asserted in-process); assistant
`thinking_blocks` (the shared openai guard — v1 prepends them as content
blocks); `file` content parts (inbound boundary — v1 converts video files
to `video_url` blocks); top_k (the extra_body crossing, wire-proven);
user (model-list gated); explicit stream:false (guard arm). SERVED,
never fallbacks: reasoning_effort verbatim (UNCONDITIONALLY in v1's
list — no capability fork), the thinking budget-band rewrite (>=10000
high / >=5000 medium / >=2000 low / else-incl.-absent minimal; disabled
and adaptive DROPPED; an explicit reasoning_effort WINS — all pinned by
IDENTICAL rows), and the recursive tools cleaning (`strict` removed at
every depth and any value; `additionalProperties` removed ONLY when
false — contrast xai's function-level-only strip). hosted_vllm fork
obligations: NO model preset (bare wire model); streams fold
providers/hosted_vllm.parse_line (the FAMILY policy; the base-handler
error-chunk PINNED DIVERGENCE covers it) with the "xai" chunk dialect.
Deliberate wave-2b-alpha (fireworks_ai) fallback surfaces (each names the
v1 path): tools/parallel_tool_calls/tool_choice/reasoning_effort wherever
the fireworks_ai/{m} map flag is not explicitly true — the deps read is
STRICTLY NARROWER than v1's get_provider_info default-true +
hyphen-boundary scan (one-direction soundness pinned by
test_capability_mirror_is_one_direction: zero v1-False/v2-True rows at
HEAD; v1 serves or raises per its own gate, the typed fallback reproduces
it either way; re-evaluate with a deps map-scan surface if tools fallback
volume hurts); response_format WITH tools (v1 pops response_format and
sets the json_mode routing marker — the groq-shape cross-plane
machinery); legacy-def tool schemas (the SHARED inbound arm; v1 inlines
$refs via unpack_legacy_defs); assistant provider_specific_fields /
thinking_blocks (v1 POPS both); pdf/file parts (v1 migrates to
image_url); n / logprobs / penalties / prompt_truncate_length /
context_length_exceeded_behavior (v1 serves — top-level for the OpenAI
ones, extra_body crossing for the provider-native two); top_k (extra_body
crossing, wire-proven); explicit stream:false; on the response side, the
tool-calls-in-content repair shape (v1 synthesizes a uuid4 tool_call —
the pure parser fails closed; reserved json_tool_call contents serve
verbatim, pinned) and non-string wire models (v1's ModelResponse(**json)
raises ValidationError). SERVED, never fallbacks: the accounts/fireworks/
models/{m} model rewrite (kept verbatim for accounts/-prefixed and
#-bearing ids), mct->max_tokens, tool_choice required->any, the
function-level strict strip (deeper strict keys ride verbatim — v1 keeps
them), the #transform=inline image suffix (LITERAL "vision" substring
gate on the REWRITTEN model; data: urls exempt), cache_control stripped
(== the IR drop), and user/reasoning_effort verbatim. fireworks_ai fork
obligations: NO seam preset and the "openai_like" construction arm with
the parser-owned fireworks_ai/{WIRE model} prefix (NOT the request
model); the ambient litellm.disable_add_transform_inline_image_block
global (and the per-request litellm_params flag) must FORCE v1 at the
seam before flag-on — the serializer encodes the default-enabled arm
only; response _hidden_params.additional_headers ride the seam's
headers-passthrough follow-up.
Deliberate wave-2b-alpha (snowflake) fallback surfaces (each names the
v1 path): max_completion_tokens / stop / n / seed / penalties /
parallel_tool_calls (v1's small static list RAISES — no rename arms
anywhere); user (silently dropped; the list never carries it); message
`name` and every other openai-guard raw shape (v1 sends messages
VERBATIM — its transform never calls super, so no flatten and no base
image transforms; the guard's fallbacks are all the v1-serves kind); on
the response side, non-string wire models (v1's ModelResponse(**json)
raises ValidationError). SERVED, never fallbacks: explicit
`stream: false` (the ONE wave-2b provider where the family guard arm
would be WRONG — v1 puts `stream` in every body, default false, so
absent and explicit-false are the same wire byte, pinned); top_k (the
non-compat TOP-LEVEL passthrough, wire-proven via respx — the snowflake
elif OVERWRITES the caller's client at main.py:4390, so the
mock-transport helper cannot reach it); tools -> Snowflake `tool_spec`
(missing parameters get the empty-object default); tool_choice -> the
object shape (auto/none verbatim types, required -> any, the dict
function form -> {"type":"tool","name":[fn]} with the ARRAY); the
content_list response rewrite (choices[0] text concat + tool_use ->
OpenAI tool_calls with json.dumps DEFAULT-separator arguments,
byte-pinned) and the snowflake/{wire-or-EMPTY} response prefix.
snowflake fork obligations: NO seam preset and the "openai_like"
construction arm; the request model must ride
_hidden_params["model"] (v1's cost-calculator feed, dump-invisible —
gate-pinned); account_id -> URL synthesis and KEYPAIR_JWT/PAT headers
are pure envelope (header-only auth, researcher-4's correction). Real
Cortex STREAM delta shapes are unpinned upstream: a content_list delta
would be a loud v2 error where v1 serves a chunk that silently lost it
— get real fixtures before flag-on streaming.
Deliberate wave-2b-alpha (huggingface) fallback surfaces (each names the
v1 path): the WHOLE router route — `deps.api_base` None falls back every
request typed (v1's 3-segment `provider/org/model` names fetch the HF
inference-provider mapping over a BLOCKING HTTP GET inside
transform_request and rewrite the model to providerId; 2-segment names
skip the fetch but run the base transforms + router URL synthesis;
1-segment names CRASH v1 with the split ValueError — all pinned); on the
api_base route, `max_retries` (NOT popped on this arm — v1 serves it
INTO the body, an oddity the row pins), user (model-list gated), message
`name` and the other openai-guard raw shapes (v1 sends messages
VERBATIM), and explicit stream:false (the body carries every
optional_params key). SERVED: the verbatim body (model untouched, mct
verbatim, no renames) plus top_k (the non-compat TOP-LEVEL passthrough).
huggingface fork obligations: the fork MUST thread
`litellm_params["api_base"]` into `deps.api_base` (the route gate keys
on it — None means every request falls back; the env
HF_API_BASE/HUGGINGFACE_API_BASE chain resolves in v1's envelope ABOVE
the transform and does NOT arm the api_base transform route, so the
fork must thread exactly the litellm_params value, not the env) and
keep the bare wire model (no preset, "openai" construction arm).
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
Streaming obligations added at the wave-2b-alpha fix round
(verifier-wave2b-alpha F1/F2/F3, fixed ONCE in the shared
`openai_compat/httpx_chunk.py` factory so every dict-path consumer
inherits them): a reasoning/refusal-bearing FINISH delta is the LOUD
finish-chunk fallback (v1's wrapper interleaves it as a separate chunk
the fold cannot reproduce — the pre-fix factory served an empty finish
chunk with the text silently GONE; two-sided rows in the openrouter and
xai stream gates, all base-handler consumers inherit), and a non-string
non-None reasoning value is a loud error (v1 serves it then RAISES
APIError at stream end in the chunk builder). One NAMED zero-data
divergence remains for the streaming seam to inherit deliberately: v1
serves a mid-stream empty-string reasoning delta as its own chunk while
the v2 fold swallows the empty delta (byte-identical otherwise) —
pinned by the openrouter gate's named-obligation row; decide
serve-vs-swallow at the seam before flag-on streaming.
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
