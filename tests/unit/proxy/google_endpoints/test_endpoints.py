"""
Test for google_endpoints/endpoints.py
"""

from typing import Final

import pytest
from dotenv import load_dotenv
from pydantic import JsonValue, ValidationError
from starlette.requests import Request

from litellm.proxy.google_endpoints.endpoints import _google_count_tokens_request, google_count_tokens
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse


@pytest.mark.parametrize(
    ("system_key", "schema_key"),
    (("systemInstruction", "parameters"), ("system_instruction", "parametersJsonSchema")),
)
def test_google_count_tokens_translates_system_and_function_declarations(
    system_key: str, schema_key: str
) -> None:
    schema: Final = {"type": "object", "properties": {"query": {"type": "string"}}}
    request_body: Final[dict[str, JsonValue]] = {
        "contents": [{"role": "user", "parts": [{"text": "Look this up"}]}],
        system_key: {"parts": [{"text": "Use the provided tool."}]},
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "lookup",
                        "description": "Find a record",
                        schema_key: schema,
                    }
                ]
            }
        ],
    }
    request: Final = _google_count_tokens_request("claude-sonnet", request_body)

    assert request.system == [{"type": "text", "text": "Use the provided tool."}]
    assert request.tools == [
        {
            "name": "lookup",
            "description": "Find a record",
            "input_schema": schema,
        }
    ]
    assert request.messages == [{"role": "user", "content": "Look this up"}]


@pytest.mark.parametrize(
    ("schema_key", "expected_schema"),
    (
        ("parameters", {"type": "object", "properties": {"query": {"type": "string"}}}),
        ("parametersJsonSchema", {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}}),
    ),
)
def test_google_count_tokens_normalizes_parameters_only(
    schema_key: str, expected_schema: dict[str, JsonValue]
) -> None:
    request: Final = _google_count_tokens_request(
        "claude-sonnet",
        {
            "contents": [{"role": "user", "parts": [{"text": "Look this up"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "lookup",
                            schema_key: {
                                "type": "OBJECT",
                                "properties": {"query": {"type": "STRING"}},
                            },
                        }
                    ]
                }
            ],
        },
    )

    assert request.tools == [{"name": "lookup", "input_schema": expected_schema}]

load_dotenv()



@pytest.mark.asyncio
async def test_proxy_gemini_to_openai_like_model_token_counting():
    """
    Test the token counting endpoint for proxing gemini to openai-like models.
    """
    response: TokenCountDetailsResponse = await google_count_tokens(
        request=Request(
            scope={
                "type": "http",
                "parsed_body": (
                    ["contents"],
                    {"contents": [{"parts": [{"text": "Hello, how are you?"}]}]},
                ),
            }
        ),
        model_name="volcengine/foo",
    )

    assert response.get("totalTokens") > 0


@pytest.mark.parametrize("system_field", ("systemInstruction", "system_instruction"))
@pytest.mark.parametrize("parts", ({"text": "Follow these rules"}, [{"text": "Follow these rules"}]))
@pytest.mark.parametrize("tools", (None, []))
def test_google_count_tokens_accepts_equivalent_system_parts_and_no_tools(
    system_field: str, parts: JsonValue, tools: JsonValue
) -> None:
    contents: Final = [{"role": "user", "parts": [{"text": "Count this request"}]}]
    request: Final = _google_count_tokens_request(
        "counted", {"contents": contents, system_field: {"parts": parts}, "tools": tools}
    )

    assert request.messages == [{"role": "user", "content": "Count this request"}]
    assert request.system == [{"type": "text", "text": "Follow these rules"}]
    assert request.tools is None
    assert request.contents == contents


@pytest.mark.parametrize(
    "parts",
    (
        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
        [{"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}],
    ),
)
def test_google_count_tokens_rejects_media_only_system_parts(parts: JsonValue) -> None:
    with pytest.raises(ValidationError):
        _google_count_tokens_request(
            "counted",
            {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}], "systemInstruction": {"parts": parts}},
        )
