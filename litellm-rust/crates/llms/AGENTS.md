# rules

## Scope

- This crate mirrors `litellm/llms/`: base config traits in `src/base_llm/<format>/`, provider transformations in `src/<provider>/<format>/transformation.rs`
- Transport (clients, media fetching, header helpers, transport errors) lives in `litellm-http`. See `../inference/AGENTS.md` for crate layering

## Layering

- `litellm-llms-types` owns API data contracts
- `src/base_llm/<format>/` owns the adapter contract and provider-independent machinery. It never imports a provider or embeds provider policy in trait defaults, normalization or context defaults
- `src/<provider>/<format>/` owns that provider's implementation and policy
- `inference-<format>` owns call orchestration
- These are the intended boundaries, not a claim that all code already satisfies them

## Provider folders

- `src/<provider>/` holds provider-wide policy: credentials, auth, endpoints, model capabilities. `common_utils.rs` means shared across that provider's formats, not across providers
- Shared constants and wire types used by several of a provider's formats go in `litellm-llms-types/src/providers/<provider>/`
- A provider may explicitly reuse another provider's helper when its policy applies to the backend (Bedrock, Vertex and Azure reuse `anthropic/messages` shaping for Claude). That does not make the policy format-wide
- Pure payload rewrites belong with transformations, not transport handlers, even in a file named `handler.rs`

## AGENTS.md convention

- Every file is `# rules` then `# references`, both concise bullets. Split a long `# rules` into `##` topic sections
- A folder gets an AGENTS.md only when it adds something: a deviation, an explicit reuse of another provider, or upstream docs no other file lists. A folder without one follows its nearest parent
- Every `src/<provider>/<format>/` folder gets one, since it talks to its own upstream endpoint
- A rule lives in the highest file where it holds. Never restate a parent's rule
- Each upstream URL appears in exactly one AGENTS.md, the one owning that contract. Others point to that file by path
  - Format specs: `litellm-llms-types/src/formats/<format>/`
  - Provider wire-type docs: `litellm-llms-types/src/providers/<provider>/`
  - Provider-wide docs (auth, errors, regions): `src/<provider>/`
  - One host's endpoint docs: `src/<provider>/<format>/`

## Nested files

- `src/base_llm/messages/`
- `src/anthropic/`, with `batches/`, `chat/`, `count_tokens/` and `messages/`
- `src/aws_textract/ocr/`
- `src/azure_ai/messages/` and `src/azure_ai/ocr/`
- `src/bedrock/`, with `audio_transcription/`, `chat/` and `messages/`
- `src/cohere/ocr/`, `src/deepseek/messages/`, `src/mistral/ocr/`, `src/reducto/ocr/`
- `src/openai/responses/` and `src/openai_like/chat/`
- `src/vertex_ai/messages/` and `src/vertex_ai/ocr/`

## Python pairs

- Python paths identify counterparts but do not dictate Rust module names or class hierarchy. Preserve behavior and concepts, not structure
- Keep operation and parameter names when responsibilities match. Rust types keep the Python name with Rust acronym casing (`BaseOCRConfig` -> `BaseOcrConfig`). Private Python helpers drop the leading underscore
- Give Rust-only helpers distinct responsibility names instead of reusing trait method names
- Use trait defaults for unchanged inherited behavior and explicit delegation for shared provider behavior. Config traits represent real provider contracts, so do not recreate inheritance with extra traits
- The base OCR and Mistral OCR pairs are the reference when aligning transformations

## Method order

- OCR configs: supported params, credential metadata and connection resolution, health-check input, param mapping, env validation, URL, request, async request, response, async response, error conversion
- Other formats: param support and mapping, env validation, URL, request, response, then Rust-only runtime hooks
- Constants and data types before the config, private helpers after it in operation order, tests last

## Known partial ports

- `AnthropicConfig`, `AmazonConverseConfig` and the non-OCR base traits are partial. Preserve their acceptance gates, passthrough and host fallback contracts
- `OpenAiResponsesApiConfig` implements WebSocket transformations and direct HTTP Responses, without Python's model-specific param rewriting or Responses-to-Chat emulation
- Chat `supported_openai_param_mappings` lists OpenAI/provider name pairs, unlike Python's name-only list. Audio `map_transcription_params` is a Rust filtering helper
- Auth resolution is split between configs and route preparation in `inference-<format>`

## OCR specifics

- `BaseOcrConfig::prepare_request` matches Python's HTTP-handler preparation. `validate_request_body` is Rust-only
- `src/base_llm/ocr/error.rs` and `document.rs` are Rust-only: the OCR error taxonomy shared with the route and inline-document helpers
- Mistral auth key handling and top-level response-extra preservation differ from Python. Layout refactors must keep those behaviors and their tests

## Code

- Providers choose auth and header policy, shared auth and HTTP infrastructure apply it. Generic config lookup uses the existing settings utilities
- Pick traversal by operation (per-block map, filter, or whole-message when blocks depend on each other). Add an abstraction only for a repeated responsibility

## Tests

- Named `#[rstest]` cases instead of loops, `#[fixture]` for setup, `#[with(...)]` for overrides
- Verify observable auth precedence, headers, serialization, passthrough and transformations, never code structure
