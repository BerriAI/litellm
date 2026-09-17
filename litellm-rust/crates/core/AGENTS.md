litellm-core is the LiteLLM SDK in Rust — it makes the LLM call. Each top-level call is a module under `src/<route>/` exposing a public entrypoint named after the route (`messages::messages()`, the Rust equivalent of `litellm.messages()`): you call it and get a typed non-streaming response back.

A route module owns the call entrypoint, runtime types, provider/auth/URL resolution, and the handler that performs the HTTP call. Provider code and base config traits live under `src/llms/`, mirroring their Python source paths. This applies to every API surface: shared orchestration stays in its route module (`ocr/`, `chat_completions/`, `messages/`, `audio_transcription/`, or `responses/`), while provider transformations live under the corresponding Python-mirrored `llms/<provider>/` path. Import implementations directly from their canonical paths; do not add a `src/providers/` layer or compatibility re-exports. Shared provider resolution lives under `src/litellm_core_utils/get_llm_provider_logic.rs`. Handlers belong in core, never in a host crate

Not here: serving HTTP (axum routes, extractors), config file reading, rollout state, databases, or host-specific callback execution. Core owns lifecycle sequencing and callback payload construction; hosts execute the selected integrations. Env reads are limited to credential fallback in a route's `prepare.rs`.

Routes (messages, ocr, realtime) and providers (anthropic, mistral, openai) are modules, not crates.

## Python/Rust transformation pairs

Use the base OCR and Mistral OCR pairs as the reference when aligning transformations. Derive `src/<relative_path>.rs` from `litellm/<relative_path>.py`, preserving meaningful basenames such as `messages_transformation`

Keep corresponding operation names and parameter names when their responsibilities match. Rust types retain the Python semantic name with Rust acronym casing (`BaseOCRConfig` / `BaseOcrConfig`, `MistralOCRConfig` / `MistralOcrConfig`). Private Python helpers can drop their leading underscore. Give Rust adapter helpers distinct responsibility names rather than duplicating trait method names

Order OCR config methods as supported parameters, credential metadata and connection resolution, health-check input, parameter mapping, environment validation, URL construction, request transformation, async request transformation, response transformation, async response transformation, and error conversion. Put constants and data types before the config, private helpers after it in operation order, and tests last. Rust-only trait hooks follow the corresponding Python methods

Use trait defaults for unchanged inherited behavior and explicit delegation for shared provider behavior. Keep typed inputs, ownership, `Result`, and async I/O idiomatic. A matching path or symbol identifies the counterpart, not a claim of full behavioral parity

Use named `#[rstest]` cases for independent input/output scenarios instead of loops or repeated calls in one test. Inject reusable setup with `#[fixture]` arguments and use `#[with(...)]` for fixture overrides. Keep assertions about the same result together

For base OCR, Python response models correspond to `src/ocr/types.rs`; Rust context/environment types support the runtime. `BaseOcrConfig::prepare_request` corresponds to Python's HTTP-handler preparation rather than a `BaseOCRConfig` method, and `validate_request_body` is a Rust-only hook

For Mistral, `async_transform_ocr_request` uses the base default in both languages. `resolve_headers` and `build_ocr_url` implement the respective environment and URL operations, and `normalize_response` implements the typed part of response transformation. Existing auth key/header handling and top-level response-extra preservation differ between languages; layout refactors must preserve those behaviors and verify them with the existing tests

For non-OCR pairs, order corresponding methods as parameter support/mapping, environment validation, URL construction, request transformation, and response transformation, followed by Rust-only runtime hooks. Auth resolution remains split between configs and route preparation. Chat `supported_openai_param_mappings` describes accepted OpenAI/provider name pairs, unlike Python's `get_supported_openai_params` name list. Audio `map_transcription_params` remains a Rust filtering helper

Azure Messages maps to `llms/azure_ai/anthropic/messages_transformation.py`; Bedrock Converse maps to `llms/bedrock/chat/converse_transformation.py`. `AnthropicConfig`, `AmazonConverseConfig`, and the non-OCR base traits are partial ports. `OpenAiResponsesApiConfig` currently implements only the WebSocket surface. Preserve their acceptance gates, passthrough behavior, and host fallback contracts when aligning layout
