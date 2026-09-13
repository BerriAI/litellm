"""
Tests for `response_format` on the responses API.

`response_format` is the spelling `litellm.completion()` uses for structured output.
The responses API equivalent is `text.format`, and `response_format` used to be dropped
by the `ResponsesAPIOptionalRequestParams` filter in
`ResponsesAPIRequestUtils.get_requested_response_api_optional_param` before any
validation ran -- so the call succeeded with no format enforced and no error raised.

These assert on the params that would be sent to the provider rather than on a
successful return, because the call returned successfully in the broken case too.
"""

import os
import sys
from unittest.mock import patch

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

FLAGS_SCHEMA = {
    "type": "object",
    "properties": {"flags": {"type": "array", "items": {"type": "string"}}},
    "required": ["flags"],
    "additionalProperties": False,
}


def _capture_request_params(**responses_kwargs):
    """Call litellm.responses and return the optional request params bound for the provider."""
    captured = {}

    def mock_handler(
        model,
        input,
        responses_api_provider_config,
        response_api_optional_request_params,
        custom_llm_provider,
        litellm_params,
        logging_obj,
        _is_async=False,
        **kwargs,
    ):
        captured["params"] = response_api_optional_request_params
        return ResponsesAPIResponse(
            id="resp_123",
            object="response",
            created_at=1741476542,
            status="completed",
            model=model,
            output=[],
            usage=ResponseAPIUsage(input_tokens=10, output_tokens=20, total_tokens=30),
            error=None,
            incomplete_details=None,
        )

    with patch(
        "litellm.responses.main.base_llm_http_handler.response_api_handler",
        new=mock_handler,
    ):
        litellm.responses(
            model="gpt-4o",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            input="Review this draft.",
            **responses_kwargs,
        )

    return captured["params"]


def test_response_format_json_schema_reaches_request_as_text_format():
    """A json_schema response_format is hoisted into text.format with the schema intact."""
    params = _capture_request_params(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Flags", "strict": True, "schema": FLAGS_SCHEMA},
        }
    )

    # The bug: params carried no "text" key at all, so nothing constrained the output.
    assert "text" in params, "response_format should be mapped onto the text parameter"
    assert params["text"]["format"] == {
        "type": "json_schema",
        "name": "Flags",
        "strict": True,
        "schema": FLAGS_SCHEMA,
    }

    # The chat-completions spelling must not also be forwarded to the provider.
    assert "response_format" not in params


def test_response_format_accepts_pydantic_model():
    """response_format=<BaseModel> converts the same way text_format=<BaseModel> does."""

    class Flags(BaseModel):
        flags: list[str]

    params = _capture_request_params(response_format=Flags)

    text_format = params["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "Flags"
    assert "flags" in text_format["schema"]["properties"]


def test_response_format_json_schema_without_strict():
    """`strict` is optional in a hand-written response_format; omitting it is not an error."""
    params = _capture_request_params(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Flags", "schema": FLAGS_SCHEMA},
        }
    )

    assert params["text"]["format"] == {
        "type": "json_schema",
        "name": "Flags",
        "schema": FLAGS_SCHEMA,
    }


@pytest.mark.parametrize("format_type", ["json_object", "text"])
def test_response_format_without_schema(format_type):
    """
    The schema-less formats carry no json_schema block. These previously raised
    KeyError in the conversion helper rather than mapping across.
    """
    params = _capture_request_params(response_format={"type": format_type})

    assert params["text"]["format"] == {"type": format_type}


def test_explicit_text_takes_precedence_over_response_format():
    """text is the native spelling, so it wins when both are supplied."""
    native_text = {"format": {"type": "json_schema", "name": "Native", "schema": FLAGS_SCHEMA}}

    params = _capture_request_params(
        text=native_text,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Alias", "strict": True, "schema": FLAGS_SCHEMA},
        },
    )

    assert params["text"]["format"]["name"] == "Native"


def test_text_format_takes_precedence_over_response_format():
    """text_format is the responses-API spelling, so it also wins over the alias."""

    class Native(BaseModel):
        flags: list[str]

    params = _capture_request_params(
        text_format=Native,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Alias", "strict": True, "schema": FLAGS_SCHEMA},
        },
    )

    assert params["text"]["format"]["name"] == "Native"


def test_no_format_leaves_text_unset():
    """Absent every spelling, nothing is invented."""
    assert "text" not in _capture_request_params()


@pytest.mark.parametrize(
    "bad_format",
    [
        {},
        {"json_schema": {"name": "Flags", "schema": FLAGS_SCHEMA}},  # `type` omitted
    ],
)
def test_response_format_without_type_raises(bad_format):
    """
    A format with no readable `type` cannot be converted. It must raise rather than
    return None: response_format is consumed before the request is built, so returning
    None would send the request unconstrained -- reintroducing, for malformed input,
    exactly the silent drop this conversion exists to remove.

    The helper raises ValueError; responses() surfaces it through exception_type() as
    BadRequestError, which is what a caller actually sees.
    """
    with pytest.raises(litellm.BadRequestError, match="Could not read a `type`"):
        _capture_request_params(response_format=bad_format)


def test_text_format_without_type_raises():
    """Same guarantee for the text_format spelling, which previously raised KeyError."""
    with pytest.raises(litellm.BadRequestError, match="Could not read a `type`"):
        _capture_request_params(text_format={"json_schema": {"name": "Flags"}})
