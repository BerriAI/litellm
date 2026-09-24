litellm-llms mirrors `litellm/llms/`: base config traits, provider transformations, and the OCR request handler in `base_llm/ocr/handler.rs`. Transport code (clients, media fetching, header helpers, transport errors) lives in `litellm-http`. See `../core/AGENTS.md` for how the crates layer.

## Python/Rust transformation pairs

Use the base OCR and Mistral OCR pairs as the reference when aligning transformations. Derive `src/<relative_path>.rs` from `litellm/llms/<relative_path>.py`, preserving meaningful basenames such as `messages_transformation`

Keep corresponding operation names and parameter names when their responsibilities match. Rust types retain the Python semantic name with Rust acronym casing (`BaseOCRConfig` / `BaseOcrConfig`, `MistralOCRConfig` / `MistralOcrConfig`). Private Python helpers can drop their leading underscore. Give Rust adapter helpers distinct responsibility names rather than duplicating trait method names

Order OCR config methods as supported parameters, credential metadata and connection resolution, health-check input, parameter mapping, environment validation, URL construction, request transformation, async request transformation, response transformation, async response transformation, and error conversion. Put constants and data types before the config, private helpers after it in operation order, and tests last. Rust-only trait hooks follow the corresponding Python methods

Use trait defaults for unchanged inherited behavior and explicit delegation for shared provider behavior. Keep typed inputs, ownership, `Result`, and async I/O idiomatic. A matching path or symbol identifies the counterpart, not a claim of full behavioral parity

Use named `#[rstest]` cases for independent input/output scenarios instead of loops or repeated calls in one test. Inject reusable setup with `#[fixture]` arguments and use `#[with(...)]` for fixture overrides. Keep assertions about the same result together

For base OCR, Python response models live next to `BaseOcrConfig` in `src/base_llm/ocr/transformation.rs`, as they do in Python; Rust context/environment types support the runtime. `BaseOcrConfig::prepare_request` corresponds to Python's HTTP-handler preparation rather than a `BaseOCRConfig` method, and `validate_request_body` is a Rust-only hook. `src/base_llm/ocr/error.rs` and `src/base_llm/ocr/document.rs` are Rust-only: the OCR error taxonomy shared with the route, and inline-document helpers shared by several providers

For Mistral, `async_transform_ocr_request` uses the base default in both languages. `resolve_headers` and `build_ocr_url` implement the respective environment and URL operations, and `normalize_response` implements the typed part of response transformation. Existing auth key/header handling and top-level response-extra preservation differ between languages; layout refactors must preserve those behaviors and verify them with the existing tests

For non-OCR pairs, order corresponding methods as parameter support/mapping, environment validation, URL construction, request transformation, and response transformation, followed by Rust-only runtime hooks. Auth resolution remains split between configs and route preparation in litellm-core. Chat `supported_openai_param_mappings` describes accepted OpenAI/provider name pairs, unlike Python's `get_supported_openai_params` name list. Audio `map_transcription_params` remains a Rust filtering helper

Azure Messages maps to `llms/azure_ai/anthropic/messages_transformation.py`; Bedrock Converse maps to `llms/bedrock/chat/converse_transformation.py`. `AnthropicConfig`, `AmazonConverseConfig`, and the non-OCR base traits are partial ports. `OpenAiResponsesApiConfig` currently implements only the WebSocket surface. Preserve their acceptance gates, passthrough behavior, and host fallback contracts when aligning layout
