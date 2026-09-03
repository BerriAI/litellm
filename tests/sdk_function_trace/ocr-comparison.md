# OCR Python and Rust comparison

Audited implementation revision: `edcba483b2`. The implementations do not match in function contracts, call structure, or all tested response behavior. This audit changes the source listing coverage, not OCR runtime behavior

Run both source listings from the repository root:

```bash
python3 tests/sdk_function_trace/list_python_steps.py --route ocr --signatures --calls
uv run tests/sdk_function_trace/list_rust_steps.py --route ocr --signatures --calls
```

Both cover Mistral, Azure AI Mistral, Azure Document Intelligence, Vertex Mistral, and Vertex DeepSeek. Listings show declarations and source call sites, not executed traces

## Function contracts

Comparing Python `BaseOCRConfig` with Rust `OcrProviderConfig`, omitting `self` and language-specific ownership details:

| Python | Rust | Difference |
| --- | --- | --- |
| `get_supported_ocr_params(model)` | `supported_ocr_params()` | Name and model argument |
| `get_api_key_env_var()` | No corresponding method | Missing contract |
| `map_ocr_params(non_default_params, optional_params, model)` | `map_ocr_params(non_default_params)` | Missing accumulator and model |
| `validate_environment(headers, model, api_key, api_base, litellm_params, **kwargs)` | Separate auth/key/header helpers | Different contract |
| `get_complete_url(api_base, model, optional_params, litellm_params, **kwargs)` | `complete_url(api_base, model, optional_params, env_lookup)` | Name and context |
| `transform_ocr_request(model, document, optional_params, headers, **kwargs)` | `transform_ocr_request(model, document, optional_params)` | Missing headers and extra context |
| `async_transform_ocr_request(...)` | No corresponding method | Missing async override |
| `transform_ocr_response(model, raw_response, logging_obj, **kwargs)` | `transform_ocr_response(model, response_json)` | Missing HTTP metadata, logging and extra context |
| `async_transform_ocr_response(...)` | No corresponding method | Missing async override |
| `get_error_class(error_message, status_code, headers)` | Central Rust error mapping | Different contract |

Python's default mapper returns the supplied `optional_params`; Rust's filters `non_default_params`. Provider overrides must also be compared

Python maps parameters during SDK preparation, before HTTP-handler environment validation and URL construction. Rust resolves auth and URL before mapping parameters in `prepare_provider_request`. Python has async provider transforms; both native entrypoints execute the same Rust async route using synchronous transform hooks, with polling and document downloading in gateway helpers

The native bindings also accept `optional_params` and `timeout_seconds`, while the Python SDK accepts `**kwargs` and `timeout`. Public SDK calls with Rust enabled still execute Python preparation before entering Rust, so matching SDK responses would not prove matching standalone Rust steps

## Runtime results

Built the native extension from the audited source using `cargo build -p litellm-python-bridge --features extension-module --offline`. Supplied that build's functions through `use_litellm_rust` dependency injection. Ran public `litellm.ocr` and `litellm.aocr` with Rust disabled and enabled against identical local HTTP response fixtures, requiring one request per invocation

Successful `model_dump()` results and failure exception classes were compared. These checks cover Mistral response outcomes only, not request equality, error messages, live providers, or every execution branch

| Mistral response fixture | Sync | Async | Observation |
| --- | --- | --- | --- |
| Valid page/model/usage | Match | Match | Same normalized response |
| Model omitted | Match | Match | Both use the requested model |
| `model: null` | Different | Different | Python rejects; Rust uses the requested model |
| `pages: null` | Different | Different | Python rejects; Rust returns an empty array |
| Invalid page element | Match | Match | Both reject during response validation |

Six of ten fixture/mode comparisons match, four differ. Rust's Mistral response transform conflates missing values with explicit nulls through `as_array`/`as_str` fallbacks. Python preserves explicit nulls into response validation, which rejects them

## Other provider gaps found in source

Azure Document Intelligence's Python configuration supports `pages`, `features`, and `req_format`; Rust lists only `pages`. Python normalizes parameters before URL construction; Rust normalizes pages during URL construction

Python preserves Azure `content`, `tables`, and `keyValuePairs`, and supports retaining the native operation payload. Rust's `OcrResponseData` has no corresponding fields, and its Azure transform does not preserve those values

Azure and Vertex async document transforms and Azure polling also use different helper contracts. Their runtime equivalence was not tested in this audit
