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
                "headers": [],
                "parsed_body": (
                    ["contents"],
                    {"contents": [{"parts": [{"text": "Hello, how are you?"}]}]},
                ),
            }
        ),
        model_name="volcengine/foo",
    )

    assert response.get("totalTokens") > 0


_DECLARATION: Final = {"name": "lookup", "description": "Find a record"}
_SYSTEM: Final = {"systemInstruction": {"parts": [{"text": "Follow these rules"}]}}
_NO_SYSTEM_PARTS: Final = {"systemInstruction": {"parts": []}}
_TOOLS: Final = {"tools": [{"functionDeclarations": [_DECLARATION]}]}
_TURN: Final = {"role": "user", "parts": [{"text": "Count this request"}]}
_CONTENTS: Final = {"contents": [_TURN]}


@pytest.mark.parametrize(
    ("spelled", "canonical"),
    (
        ({"systemInstruction": {"parts": {"text": "Follow these rules"}}}, _SYSTEM),
        ({"system_instruction": {"parts": {"text": "Follow these rules"}}}, _SYSTEM),
        ({"systemInstruction": {"parts": [{"text": "Follow these rules"}, None]}}, _SYSTEM),
        ({"systemInstruction": {"parts": None}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {"parts": [{"text": None}, {"text": "Follow these rules"}]}}, _SYSTEM),
        ({"systemInstruction": {"parts": {"text": None}}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {"parts": {"text": None, "inlineData": None}}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {"parts": {"text": None, "thought": False}}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {"parts": [{"thoughtSignature": "c2ln"}, {"text": "Follow these rules"}]}}, _SYSTEM),
        ({"systemInstruction": {"parts": [{}]}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {}}, _NO_SYSTEM_PARTS),
        ({"tools": None}, {}),
        ({"tools": [None]}, {}),
        ({"tools": [{"functionDeclarations": None}]}, {}),
        ({"tools": {"functionDeclarations": [_DECLARATION]}}, _TOOLS),
        ({"tools": [{"functionDeclarations": _DECLARATION}]}, _TOOLS),
        ({"tools": {"functionDeclarations": _DECLARATION}}, _TOOLS),
        ({"contents": _TURN}, _CONTENTS),
        ({"contents": [None, _TURN]}, _CONTENTS),
        ({"contents": None}, {"contents": []}),
        ({"contents": [{"role": "user", "parts": {"text": "Count this request"}}]}, _CONTENTS),
        ({"contents": [{"role": "user", "parts": [None, {"text": "Count this request"}]}]}, _CONTENTS),
        ({"contents": [{"role": "user", "parts": None}]}, {"contents": [{"role": "user", "parts": []}]}),
    ),
)
def test_google_count_tokens_counts_every_google_spelling_of_a_repeated_field_like_its_list_form(
    spelled: dict[str, JsonValue], canonical: dict[str, JsonValue]
) -> None:
    contents: Final = [{"role": "user", "parts": [{"text": "Count this request"}]}]

    assert _google_count_tokens_request("counted", {"contents": contents, **spelled}) == _google_count_tokens_request(
        "counted", {"contents": contents, **canonical}
    )


@pytest.mark.parametrize(
    "malformed",
    (
        {"tools": "lookup"},
        {"tools": [{"functionDeclarations": "lookup"}]},
        {"systemInstruction": {"parts": 7}},
        {"contents": 5},
        {"contents": ["Count this request"]},
    ),
)
def test_google_count_tokens_rejects_a_scalar_where_google_expects_an_object_or_list(
    malformed: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValidationError):
        _google_count_tokens_request("counted", {"contents": [_TURN], **malformed})


@pytest.mark.parametrize(
    "parts",
    (
        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
        [{"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}],
        {"text": None, "inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
        [{"text": "Follow these rules"}, {"text": None, "fileData": {"fileUri": "gs://bucket/rules.pdf"}}],
    ),
)
def test_google_count_tokens_rejects_media_only_system_parts(parts: JsonValue) -> None:
    with pytest.raises(ValidationError):
        _google_count_tokens_request(
            "counted",
            {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}], "systemInstruction": {"parts": parts}},
        )


def test_google_count_tokens_keeps_counting_system_text_that_sits_beside_media() -> None:
    text_with_media: Final = {"text": "Follow these rules", "inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}

    assert _google_count_tokens_request(
        "counted", {**_CONTENTS, "systemInstruction": {"parts": [text_with_media]}}
    ) == _google_count_tokens_request("counted", {**_CONTENTS, **_SYSTEM})
