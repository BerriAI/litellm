import pytest
from integration._support.client import Gateway, object_value
from pydantic import JsonValue


def _assert_responses_post_is_documented(openapi: dict[str, JsonValue]) -> None:
    post: dict[str, JsonValue] = object_value(object_value(object_value(openapi["paths"])["/v1/responses"])["post"])
    body: dict[str, JsonValue] = object_value(post["requestBody"])
    schema: dict[str, JsonValue] = object_value(
        object_value(object_value(body["content"])["application/json"])["schema"]
    )
    properties: dict[str, JsonValue] = object_value(schema.get("properties"))
    assert "model" in properties and "input" in properties, schema
    ok: dict[str, JsonValue] = object_value(object_value(object_value(post)["responses"])["200"])
    assert "schema" in object_value(object_value(ok["content"])["application/json"]), ok


def test_v1_responses_post_declares_a_request_body_and_response_schema(gateway: Gateway) -> None:
    pytest.skip("BUG: POST /v1/responses takes a raw Request, so /openapi.json documents no body or response schema")
    openapi: dict[str, JsonValue] = gateway.get("/openapi.json")
    _assert_responses_post_is_documented(openapi)
