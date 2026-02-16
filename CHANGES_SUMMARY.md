# Changes Summary

## 1. Large payload truncation

> Addressed the block from the right side

Truncate large payloads before `json.dumps` in logging to cut ~3–4s serialization cost.

- **File**: `litellm/litellm_core_utils/litellm_logging.py`
- **Details**: See [truncation_change.md](./truncation_change.md)
- **Tests**: `tests/test_litellm/litellm_core_utils/test_litellm_logging.py::TestTruncateLargePayloadForLogging` (15 tests)

## 2. ORJSONResponse bypass for serialize_response

> Addressed the request block (~5.5s serialize_response / jsonable_encoder overhead)

Bypass FastAPI `jsonable_encoder` for non-streaming responses by converting Pydantic/dict responses to plain dicts and returning `ORJSONResponse`. This eliminates the major serialize_response cost for large embeddings and completions.

- **File**: `litellm/proxy/common_request_processing.py`
- **Change**: `_response_to_json_serializable()` converts responses to JSON-serializable dicts; when applicable, we return `ORJSONResponse` directly instead of letting FastAPI serialize the response.
- **Tests**: `tests/test_litellm/proxy/test_common_request_processing.py::TestResponseToJsonSerializable` (7 tests, including 4 backward compatibility tests)
