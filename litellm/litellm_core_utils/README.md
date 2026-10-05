## Folder Contents

This folder contains general-purpose utilities that are used in multiple places in the codebase. 

Core files:
- `streaming_handler.py`: The core streaming logic + streaming related helper utils 
- `core_helpers.py`: code used in `types/` - e.g. `map_finish_reason`. 
- `exception_mapping_utils.py`: utils for mapping exceptions to openai-compatible error types. 
- `default_encoding.py`: code for loading the default Python tokenizer and bundled cache
- `get_llm_provider_logic.py`: code for inferring the LLM provider from a given model name. 
- `duration_parser.py`: code for parsing durations - e.g. "1d", "1mo", "10s"
- `api_route_to_call_types.py`: mapping of API routes to their corresponding CallTypes (e.g., `/chat/completions` -> [acompletion, completion])

Tokenizer factories return Python tokenizer objects by default. Set `LITELLM_RUST=1` or call `litellm.rust(True)` before constructing tokenizers to select the Rust backend through `Route.TOKENIZER` in the Rust catalog. Missing native bindings or unsupported native features fall back to Python. Existing tokenizer objects keep their selected backend. Rust-backed tokenizer objects carry the read-only `tiktoken.Encoding` / `tokenizers.Tokenizer` surface and are immutable: `enable_padding`, `enable_truncation` and `add_tokens` stay on the Python tokenizer.
