"""Behavior pins: requestBody appears in app.openapi() for raw-body passthrough routes.

These routes read the request body via ``await request.body()`` / ``orjson.loads``
rather than a typed Pydantic parameter, so FastAPI emits no requestBody by default.
The ``openapi_extra={"requestBody": ...}`` decorator argument injects the schema
without changing handler behavior.

Covered routes:
    - POST /moderations and /v1/moderations
    - POST /rerank, /v1/rerank, /v2/rerank
    - POST /audio/speech and /v1/audio/speech
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def openapi_schema(app):
    """Cached app.openapi() result for the module — expensive call, run once."""
    return app.openapi()


# ---------------------------------------------------------------------------
# moderations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/moderations", "/v1/moderations"])
def test_moderations_has_request_body(openapi_schema, path):
    """POST /moderations and /v1/moderations must expose a requestBody schema."""
    post_op = openapi_schema["paths"][path]["post"]
    assert "requestBody" in post_op, f"No requestBody on POST {path}"


@pytest.mark.parametrize("path", ["/moderations", "/v1/moderations"])
def test_moderations_input_is_required(openapi_schema, path):
    """``input`` must be listed in ``required``."""
    schema = openapi_schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert "input" in schema.get("required", []), f"'input' not in required on POST {path}"


@pytest.mark.parametrize("path", ["/moderations", "/v1/moderations"])
def test_moderations_input_accepts_string_or_array(openapi_schema, path):
    """``input`` must accept both a plain string and an array of strings (oneOf)."""
    schema = openapi_schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
    input_schema = schema["properties"]["input"]
    # Must use oneOf to cover both string and array-of-strings shapes
    assert "oneOf" in input_schema, (
        f"'input' on POST {path} must use oneOf to accept string-or-array; got: {input_schema}"
    )
    type_values = {branch.get("type") for branch in input_schema["oneOf"]}
    assert "string" in type_values, "oneOf must include a string branch"
    assert "array" in type_values, "oneOf must include an array branch"


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank", "/v2/rerank"])
def test_rerank_has_request_body(openapi_schema, path):
    """POST /rerank, /v1/rerank, /v2/rerank must expose a requestBody schema."""
    post_op = openapi_schema["paths"][path]["post"]
    assert "requestBody" in post_op, f"No requestBody on POST {path}"


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank", "/v2/rerank"])
def test_rerank_required_fields(openapi_schema, path):
    """``model``, ``query``, and ``documents`` must all be required."""
    schema = openapi_schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
    required = schema.get("required", [])
    for field in ("model", "query", "documents"):
        assert field in required, f"'{field}' not in required on POST {path}"


# ---------------------------------------------------------------------------
# audio/speech
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/audio/speech", "/v1/audio/speech"])
def test_audio_speech_has_request_body(openapi_schema, path):
    """POST /audio/speech and /v1/audio/speech must expose a requestBody schema."""
    post_op = openapi_schema["paths"][path]["post"]
    assert "requestBody" in post_op, f"No requestBody on POST {path}"


@pytest.mark.parametrize("path", ["/audio/speech", "/v1/audio/speech"])
def test_audio_speech_required_fields(openapi_schema, path):
    """``model``, ``input``, and ``voice`` must all be required."""
    schema = openapi_schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
    required = schema.get("required", [])
    for field in ("model", "input", "voice"):
        assert field in required, f"'{field}' not in required on POST {path}"
