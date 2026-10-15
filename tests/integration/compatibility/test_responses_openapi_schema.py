from typing import Final

from integration._support.client import Gateway, object_value, string_value
from pydantic import JsonValue


def _operation(openapi: dict[str, JsonValue], path: str, method: str) -> dict[str, JsonValue]:
    return object_value(object_value(object_value(openapi["paths"])[path])[method])


def _ok_schema_properties(openapi: dict[str, JsonValue], operation: dict[str, JsonValue]) -> dict[str, JsonValue]:
    ok: Final = object_value(object_value(operation["responses"])["200"])
    schema: Final = object_value(object_value(object_value(ok["content"])["application/json"])["schema"])
    if "$ref" in schema:
        name: Final = string_value(schema["$ref"]).rsplit("/", 1)[-1]
        return object_value(object_value(object_value(object_value(openapi["components"])["schemas"])[name])["properties"])
    assert "properties" in schema, ok
    return object_value(schema["properties"])


def test_v1_responses_post_declares_a_request_body_and_response_schema(gateway: Gateway) -> None:
    openapi: dict[str, JsonValue] = gateway.get("/openapi.json")
    post: Final = _operation(openapi, "/v1/responses", "post")
    body: Final = object_value(post["requestBody"])
    schema: Final = object_value(object_value(object_value(body["content"])["application/json"])["schema"])
    properties: Final = object_value(schema.get("properties"))
    assert {"model", "input", "instructions", "tools", "previous_response_id", "background", "stream"} <= set(
        properties
    ), sorted(properties)
    assert {"id", "object", "output", "usage"} <= set(_ok_schema_properties(openapi, post)), post["responses"]
    assert "tool_calls" in object_value(
        object_value(object_value(object_value(openapi["components"])["schemas"])["Message"])["properties"]
    )


def test_v1_responses_by_id_routes_declare_response_schemas(gateway: Gateway) -> None:
    openapi: dict[str, JsonValue] = gateway.get("/openapi.json")
    get: Final = _operation(openapi, "/v1/responses/{response_id}", "get")
    assert {"id", "object", "output"} <= set(_ok_schema_properties(openapi, get)), get["responses"]
    delete: Final = _operation(openapi, "/v1/responses/{response_id}", "delete")
    assert {"id", "object", "deleted"} <= set(_ok_schema_properties(openapi, delete)), delete["responses"]
    items: Final = _operation(openapi, "/v1/responses/{response_id}/input_items", "get")
    assert {"data", "object", "has_more"} <= set(_ok_schema_properties(openapi, items)), items["responses"]
