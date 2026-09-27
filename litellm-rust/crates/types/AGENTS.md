- `litellm-types` owns shared API data contracts and their serialization
  - A type belongs here when it describes a request, response, event, or value that consumers must agree on independently of how a call executes
  - Being public, serializable, or used by several crates is not sufficient
  - These are intended boundaries, not a claim that every existing item follows them

- Organize public contracts by API format: `messages`, `chat_completions`, and `responses`
  - Use names such as `litellm_types::messages::MessagesRequest`, without an Anthropic prefix solely because Anthropic designed Messages
  - Existing `llms::openai`, `llms::anthropic_messages`, and chat types under `utils` are legacy locations, not patterns for new modules
  - Keep one canonical definition and import path when moving a contract, updating consumers together instead of adding duplicate models or compatibility re-exports

- Shared request/response bodies, message and content-block enums, usage records, tool-call chunks, stream-event payloads, and protocol error bodies belong here
  - This includes LiteLLM's normalized response contracts and extensions, not just exact upstream schemas
  - `ChatCompletionsResponse` currently represents the response handed to the host, so replacing it with a supposedly more complete upstream schema must not silently change that contract
  - Messages web-search result/error schemas and encrypted-content fields belong to the Messages format regardless of which providers implement them

- Shared LiteLLM input data such as `ProviderSpecificHeader` and `ProviderSpecificHeaders` also belongs here
  - Selecting entries for a provider belongs in `core-utils`, and applying headers belongs in `http`
  - A genuinely provider-specific wire value may retain its provider name: `AnthropicBeta` and `BetaSet` describe the `anthropic-beta` header
  - Parsing, formatting, deduplication, and value equality belong with those types
  - Choosing required betas, OAuth companions, header precedence, or credentials belongs in `llms` and the auth crates

- Allow deterministic constructors, accessors, serialization, schema generation, and validation of the represented data shape
  - `ContentBlock::text`, `ResponsesWsEvent::model`, and `Recognized::known` are examples
  - Exact value conversions such as `EffortLevel` to the matching `ReasoningEffort` are acceptable
  - Clamping effort, choosing a thinking budget, rewriting content, mapping finish reasons, computing normalized usage, and translating between API formats are policy or transformations and belong outside this crate, even when they are pure functions

- Keep call envelopes and execution state in their owning crates
  - `MessagesCall`, `MessagesShaping`, prepared provider requests, and the response wrapper containing a live stream belong in `core`
  - Provider config traits, `MessagesTransformContext`, `MessagesModelCapabilities`, `ThinkingBudgets`, `StreamShape`, and transformer state belong in `llms`
  - Catalog records and pricing belong in `model-catalog`, which may reuse wire enums such as `ReasoningEffort`
  - Host hooks, Python objects, credentials, clients, timeouts, and routing decisions do not become API payload types merely because they cross a crate boundary

- Stream-event data belongs here, but live streams, decoders, framing, buffering, and stream lifecycle decisions do not
  - Keep SSE and AWS framing in `framer`, provider decoding and conversion in `llms`, and call orchestration in `core`
  - `ResponsesWsEvent` belongs here
  - `ResponsesWsTransformResult` wraps the output of a provider transformation rather than a wire event and lives in `llms::base_llm::responses::transformation`
  - Protocol error payloads may live here, while operational errors remain in the crate that raises them

- Keep provider-only wire envelopes and transformation-specific projections local until there is a shared API contract to expose
  - `InvokeChunkPayload` and the permissive `ReplayedWebSearchResult` projection serve decoding or rendering and should not be promoted unchanged into public schemas
  - Reuse existing shared message contracts inside batch and token-counting adapters without moving every adapter struct into this crate
  - Shared normalization intermediates such as the prompt factory's `Conversation` stay with their algorithms in `core-utils`

- Keep this crate at the bottom of the dependency graph
  - Use data, serialization, and optional schema libraries without depending on workspace execution crates, async runtimes, transport clients, or Python bindings
  - No network or filesystem I/O, environment lookup, clock access, model catalog lookup, or global configuration reads belong here
  - Defaults must describe the data contract, not select runtime policy

- Represent known discriminators such as `tool_use` and `server_tool_use` with enum variants
  - Preserve each contract's existing treatment of unknown variants, extra fields, missing fields, and explicit nulls
  - `Recognized<T>` deliberately retains values that fail typed parsing, including wrong-shaped values, so use it only where permissive passthrough is already part of the contract
  - Typing an opaque field must neither reject previously accepted inputs nor accept malformed inputs previously rejected
  - Do not silently discard unknown data or tighten a partial projection into a stricter public schema

- Test observable serialization, malformed-input rejection, unknown-value preservation, and value semantics in this crate
  - Follow the workspace test-placement and `rstest` rules
  - Test provider transformations, header policy, and stream execution in their owning crates
  - Do not test import locations or Rust source structure as substitutes for behavior
