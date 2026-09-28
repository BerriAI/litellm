litellm-llms mirrors `litellm/llms/`: base config traits, provider transformations, and the OCR request handler in `base_llm/ocr/handler.rs`. Transport code (clients, media fetching, header helpers, transport errors) lives in `litellm-http`. See `../core/AGENTS.md` for how the crates layer.

## Python/Rust transformation pairs

Use the base OCR and Mistral OCR pairs as the reference when aligning transformations. Use `src/<provider>/<format>/transformation.rs` for provider transformations. Python paths identify counterparts but do not dictate Rust module names

Keep corresponding operation names and parameter names when their responsibilities match. Rust types retain the Python semantic name with Rust acronym casing (`BaseOCRConfig` / `BaseOcrConfig`, `MistralOCRConfig` / `MistralOcrConfig`). Private Python helpers can drop their leading underscore. Give Rust adapter helpers distinct responsibility names rather than duplicating trait method names

Order OCR config methods as supported parameters, credential metadata and connection resolution, health-check input, parameter mapping, environment validation, URL construction, request transformation, async request transformation, response transformation, async response transformation, and error conversion. Put constants and data types before the config, private helpers after it in operation order, and tests last. Rust-only trait hooks follow the corresponding Python methods

Use trait defaults for unchanged inherited behavior and explicit delegation for shared provider behavior. Keep typed inputs, ownership, `Result`, and async I/O idiomatic. A matching path or symbol identifies the counterpart, not a claim of full behavioral parity

Use named `#[rstest]` cases for independent input/output scenarios instead of loops or repeated calls in one test. Inject reusable setup with `#[fixture]` arguments and use `#[with(...)]` for fixture overrides. Keep assertions about the same result together

Base OCR currently keeps response models next to `BaseOcrConfig` in `src/base_llm/ocr/transformation.rs`. This is legacy placement, not an exception to the shared API contract ownership in `litellm-types`. Rust context/environment types support the runtime. `BaseOcrConfig::prepare_request` corresponds to Python's HTTP-handler preparation rather than a `BaseOCRConfig` method, and `validate_request_body` is a Rust-only hook. `src/base_llm/ocr/error.rs` and `src/base_llm/ocr/document.rs` are Rust-only: the OCR error taxonomy shared with the route, and inline-document helpers shared by several providers

For Mistral, `async_transform_ocr_request` uses the base default in both languages. `resolve_headers` and `build_ocr_url` implement the respective environment and URL operations, and `normalize_response` implements the typed part of response transformation. Existing auth key/header handling and top-level response-extra preservation differ between languages; layout refactors must preserve those behaviors and verify them with the existing tests

For non-OCR pairs, order corresponding methods as parameter support/mapping, environment validation, URL construction, request transformation, and response transformation, followed by Rust-only runtime hooks. Auth resolution remains split between configs and route preparation in litellm-core. Chat `supported_openai_param_mappings` describes accepted OpenAI/provider name pairs, unlike Python's `get_supported_openai_params` name list. Audio `map_transcription_params` remains a Rust filtering helper

Azure Messages maps to `llms/azure_ai/anthropic/messages_transformation.py`; Bedrock Converse maps to `llms/bedrock/chat/converse_transformation.py`. `AnthropicConfig`, `AmazonConverseConfig`, and the non-OCR base traits are partial ports. `OpenAiResponsesApiConfig` implements WebSocket transformations and a direct HTTP Responses path. Its HTTP path does not implement Python model-specific parameter rewriting or Responses-to-Chat emulation. Preserve their acceptance gates, passthrough behavior, and host fallback contracts when aligning layout

## Provider and format boundaries

The same ownership rule applies to Messages, Responses, Chat Completions, OCR, and other API formats. `litellm-types` owns shared API data contracts. `llms/src/base_llm/<format>/` owns provider adapter contracts and shared transformation machinery. `llms/src/<provider>/<format>/` owns provider implementations and policy. `core/src/<format>/` owns call orchestration. Repeating a format name identifies the API each layer handles, not duplicate ownership of its schema. These boundaries also apply between modules in the same crate

A provider adapter may explicitly reuse another provider's transformation helper when that policy applies to its backend, such as Bedrock's Claude adapter using Anthropic payload shaping. Reuse across hosts of the same model family does not make the policy format-wide. Keep provider policy out of shared trait defaults and generic normalization, and keep shared execution contexts limited to inputs the adapter contract actually needs. Pure payload rewrites belong with transformations, not transport handlers

- These are intended boundaries, not a claim that all existing code already satisfies them
- Preserve behavior and conceptual boundaries. Python names and layout are reference points, not requirements to reproduce its class hierarchy or helper structure
- Provider directories own provider behavior. API formats and their public data contracts are independent of the provider that originated them
- Shared `base_llm` contracts must not import provider implementations or provider-specific transformation policy
- Config traits represent actual provider contracts. Use composition and existing helpers instead of recreating inheritance with unnecessary traits or delegation layers
- Providers choose authentication and header policy. Shared auth and HTTP infrastructure apply those decisions
- Generic configuration lookup belongs in the existing settings utilities, not in a provider directory
- Closures are idiomatic Rust, but a `Vec<ContentBlock> -> Vec<ContentBlock>` helper is not automatically a useful abstraction
- Choose traversal for the operation: per-block mapping, filtering, or whole-message processing when blocks depend on one another
- Add an abstraction only when it clarifies a repeated responsibility
- Verify observable auth precedence, headers, serialization, passthrough, and transformations, not code structure
