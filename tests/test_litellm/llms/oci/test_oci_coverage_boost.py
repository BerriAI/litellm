"""
Coverage-boost tests for the OCI provider happy paths.

Covers:
  - litellm/llms/oci/common_utils.py  (sign_with_manual_credentials, routing)
  - litellm/llms/oci/chat/generic.py  (message adaptation, tool conversion, streaming)
  - litellm/llms/oci/chat/cohere.py   (message adaptation, response parsing, streaming)
  - litellm/llms/oci/chat/transformation.py  (OCIChatConfig methods, stream wrappers)

All tests are self-contained and require no real OCI credentials or network access.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import httpx

from litellm import ModelResponse
from litellm.llms.oci.chat.cohere import (
    _extract_text_content,
    adapt_messages_to_cohere_standard,
    handle_cohere_response,
    handle_cohere_stream_chunk,
)
from litellm.llms.oci.chat.generic import (
    adapt_messages_to_generic_oci_standard,
    adapt_messages_to_generic_oci_standard_tool_response,
    adapt_tool_definition_to_oci_standard,
    adapt_tools_to_openai_standard,
    handle_generic_stream_chunk,
)
from litellm.llms.oci.chat.transformation import OCIChatConfig, get_vendor_from_model
from litellm.llms.oci.common_utils import (
    OCIError,
    sign_with_manual_credentials,
    sign_oci_request,
    validate_oci_environment,
)
from litellm.types.llms.oci import OCIVendors, OCIToolCall


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MANUAL_CREDS = {
    "oci_user": "ocid1.user.oc1..xxx",
    "oci_fingerprint": "aa:bb:cc:dd",
    "oci_tenancy": "ocid1.tenancy.oc1..xxx",
    "oci_compartment_id": "ocid1.compartment.oc1..xxx",
    "oci_key": "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
}

_API_BASE = "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat"

_COHERE_MODEL = "cohere.command-r-plus"
_GENERIC_MODEL = "meta.llama-3-70b-instruct"


# ===========================================================================
# common_utils.py — sign_with_manual_credentials happy paths
# ===========================================================================


@patch("litellm.llms.oci.common_utils._CRYPTOGRAPHY_AVAILABLE", True)
@patch("litellm.llms.oci.common_utils.load_private_key_from_str")
@patch("litellm.llms.oci.common_utils.padding")
@patch("litellm.llms.oci.common_utils.hashes")
def test_sign_with_manual_credentials_inline_key(mock_hashes, mock_padding, mock_load_key):
    """sign_with_manual_credentials succeeds with an inline oci_key string."""
    mock_key = MagicMock()
    mock_key.sign.return_value = b"fake_signature"
    mock_load_key.return_value = mock_key

    result_headers, body = sign_with_manual_credentials(
        {}, _MANUAL_CREDS, {"key": "val"}, _API_BASE
    )

    assert "authorization" in result_headers
    assert result_headers["authorization"].startswith('Signature version="1"')
    assert "rsa-sha256" in result_headers["authorization"]
    assert isinstance(body, bytes)
    mock_key.sign.assert_called_once()


@patch("litellm.llms.oci.common_utils._CRYPTOGRAPHY_AVAILABLE", True)
@patch("litellm.llms.oci.common_utils.load_private_key_from_file")
@patch("litellm.llms.oci.common_utils.padding")
@patch("litellm.llms.oci.common_utils.hashes")
def test_sign_with_manual_credentials_key_file(mock_hashes, mock_padding, mock_load_file):
    """sign_with_manual_credentials falls back to oci_key_file when oci_key absent."""
    mock_key = MagicMock()
    mock_key.sign.return_value = b"sig_from_file"
    mock_load_file.return_value = mock_key

    creds = {**_MANUAL_CREDS, "oci_key_file": "/tmp/key.pem"}
    creds_no_inline = {k: v for k, v in creds.items() if k != "oci_key"}

    result_headers, body = sign_with_manual_credentials(
        {}, creds_no_inline, {}, _API_BASE
    )

    assert "authorization" in result_headers
    mock_load_file.assert_called_once_with("/tmp/key.pem")


@patch("litellm.llms.oci.common_utils._CRYPTOGRAPHY_AVAILABLE", True)
@patch("litellm.llms.oci.common_utils.load_private_key_from_str")
@patch("litellm.llms.oci.common_utils.padding")
@patch("litellm.llms.oci.common_utils.hashes")
def test_sign_with_manual_credentials_authorization_contains_key_id(
    mock_hashes, mock_padding, mock_load_key
):
    """Authorization header encodes tenancy/user/fingerprint as key ID."""
    mock_key = MagicMock()
    mock_key.sign.return_value = b"sig"
    mock_load_key.return_value = mock_key

    result_headers, _ = sign_with_manual_credentials(
        {}, _MANUAL_CREDS, {}, _API_BASE
    )

    auth = result_headers["authorization"]
    assert 'keyId="ocid1.tenancy.oc1..xxx/ocid1.user.oc1..xxx/aa:bb:cc:dd"' in auth


def test_sign_with_manual_credentials_non_string_oci_key_raises():
    """Passing a non-string oci_key raises OCIError(400)."""
    bad_creds = {**_MANUAL_CREDS, "oci_key": 12345}
    with pytest.raises(OCIError) as exc_info:
        sign_with_manual_credentials({}, bad_creds, {}, _API_BASE)
    assert exc_info.value.status_code == 400
    assert "oci_key must be a string" in str(exc_info.value)


# ---------------------------------------------------------------------------
# common_utils.py — sign_oci_request routing
# ---------------------------------------------------------------------------


def test_sign_oci_request_routes_to_signer_when_present():
    """sign_oci_request delegates to sign_with_oci_signer when oci_signer is set."""
    signer = MagicMock()
    signer.do_request_sign.return_value = None
    headers, body = sign_oci_request(
        {}, {"oci_signer": signer}, {"data": 1}, _API_BASE
    )
    signer.do_request_sign.assert_called_once()
    assert isinstance(body, bytes)


@patch("litellm.llms.oci.common_utils._CRYPTOGRAPHY_AVAILABLE", True)
@patch("litellm.llms.oci.common_utils.load_private_key_from_str")
@patch("litellm.llms.oci.common_utils.padding")
@patch("litellm.llms.oci.common_utils.hashes")
def test_sign_oci_request_routes_to_manual_when_no_signer(
    mock_hashes, mock_padding, mock_load_key
):
    """sign_oci_request delegates to sign_with_manual_credentials when oci_signer absent."""
    mock_key = MagicMock()
    mock_key.sign.return_value = b"sig"
    mock_load_key.return_value = mock_key

    headers, body = sign_oci_request({}, _MANUAL_CREDS, {}, _API_BASE)
    assert "authorization" in headers


# ---------------------------------------------------------------------------
# common_utils.py — _require_cryptography happy path
# ---------------------------------------------------------------------------


def test_require_cryptography_available_does_not_raise():
    """_require_cryptography() should not raise when the package is importable."""
    from litellm.llms.oci.common_utils import _require_cryptography

    with patch("litellm.llms.oci.common_utils._CRYPTOGRAPHY_AVAILABLE", True):
        _require_cryptography()  # must not raise


# ===========================================================================
# generic.py — adapt_messages_to_generic_oci_standard
# ===========================================================================


def test_adapt_generic_user_message_string_content():
    messages = [{"role": "user", "content": "Hello!"}]
    result = adapt_messages_to_generic_oci_standard(messages)
    assert len(result) == 1
    assert result[0].role == "USER"
    assert result[0].content[0].text == "Hello!"


def test_adapt_generic_assistant_message():
    messages = [{"role": "assistant", "content": "Hi there!"}]
    result = adapt_messages_to_generic_oci_standard(messages)
    assert result[0].role == "ASSISTANT"
    assert result[0].content[0].text == "Hi there!"


def test_adapt_generic_system_message():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    result = adapt_messages_to_generic_oci_standard(messages)
    assert result[0].role == "SYSTEM"
    assert result[0].content[0].text == "You are a helpful assistant."


def test_adapt_generic_tool_message():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "42 degrees",
        }
    ]
    result = adapt_messages_to_generic_oci_standard(messages)
    assert result[0].role == "TOOL"
    assert result[0].toolCallId == "call_abc123"
    assert result[0].content[0].text == "42 degrees"


def test_adapt_generic_assistant_tool_call_message():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
                }
            ],
        }
    ]
    result = adapt_messages_to_generic_oci_standard(messages)
    assert result[0].role == "ASSISTANT"
    assert result[0].toolCalls is not None
    assert len(result[0].toolCalls) == 1
    tc = result[0].toolCalls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == '{"city": "Rome"}'


def test_adapt_generic_multipart_content():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
    ]
    result = adapt_messages_to_generic_oci_standard(messages)
    assert len(result[0].content) == 2
    assert result[0].content[0].text == "Look at this:"
    assert result[0].content[1].imageUrl.url == "https://example.com/img.png"


# ---------------------------------------------------------------------------
# generic.py — adapt_messages_to_generic_oci_standard_tool_response
# ---------------------------------------------------------------------------


def test_adapt_generic_tool_response_direct():
    result = adapt_messages_to_generic_oci_standard_tool_response(
        "tool", "call_999", "The answer is 42"
    )
    assert result.role == "TOOL"
    assert result.toolCallId == "call_999"
    assert result.content[0].text == "The answer is 42"


# ---------------------------------------------------------------------------
# generic.py — adapt_tool_definition_to_oci_standard
# ---------------------------------------------------------------------------


def test_adapt_tool_definition_basic():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Retrieve current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    result = adapt_tool_definition_to_oci_standard(tools, OCIVendors.GENERIC)
    assert len(result) == 1
    tool_def = result[0]
    assert tool_def.name == "get_weather"
    assert tool_def.type == "FUNCTION"
    assert tool_def.parameters is not None


def test_adapt_tool_definition_resolves_refs():
    """$ref/$defs schemas are inlined before being sent to OCI."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "do_thing",
                "parameters": {
                    "$defs": {"Loc": {"type": "string"}},
                    "type": "object",
                    "properties": {"location": {"$ref": "#/$defs/Loc"}},
                },
            },
        }
    ]
    result = adapt_tool_definition_to_oci_standard(tools, OCIVendors.GENERIC)
    props = result[0].parameters["properties"]
    assert props["location"] == {"type": "string"}


# ---------------------------------------------------------------------------
# generic.py — adapt_tools_to_openai_standard
# ---------------------------------------------------------------------------


def test_adapt_tools_to_openai_standard():
    oci_tool = OCIToolCall(
        id="call_abc",
        type="FUNCTION",
        name="search",
        arguments='{"query": "hello"}',
    )
    result = adapt_tools_to_openai_standard([oci_tool])
    assert len(result) == 1
    assert result[0].id == "call_abc"
    assert result[0].type == "function"
    assert result[0].function["name"] == "search"


def test_adapt_tools_to_openai_standard_generates_id_when_absent():
    oci_tool = OCIToolCall(
        id=None,
        type="FUNCTION",
        name="lookup",
        arguments="{}",
    )
    result = adapt_tools_to_openai_standard([oci_tool])
    assert result[0].id.startswith("call_")


# ---------------------------------------------------------------------------
# generic.py — handle_generic_stream_chunk
# ---------------------------------------------------------------------------


def test_handle_generic_stream_chunk_text_content():
    chunk = {
        "message": {
            "content": [{"type": "TEXT", "text": "Hello from OCI"}],
            "role": "ASSISTANT",
        },
        "finishReason": None,
        "index": 0,
    }
    result = handle_generic_stream_chunk(chunk)
    assert result.choices[0].delta.content == "Hello from OCI"
    assert result.choices[0].finish_reason is None


def test_handle_generic_stream_chunk_complete_finish_reason():
    chunk = {"finishReason": "COMPLETE", "index": 0}
    result = handle_generic_stream_chunk(chunk)
    assert result.choices[0].finish_reason == "stop"


def test_handle_generic_stream_chunk_max_tokens_finish_reason():
    chunk = {"finishReason": "MAX_TOKENS", "index": 0}
    result = handle_generic_stream_chunk(chunk)
    assert result.choices[0].finish_reason == "length"


def test_handle_generic_stream_chunk_tool_calls_finish_reason():
    chunk = {"finishReason": "TOOL_CALLS", "index": 0}
    result = handle_generic_stream_chunk(chunk)
    assert result.choices[0].finish_reason == "tool_calls"


def test_handle_generic_stream_chunk_no_message():
    """Chunks without a message key should still parse without error."""
    chunk = {"finishReason": "COMPLETE", "index": 1}
    result = handle_generic_stream_chunk(chunk)
    assert result.choices[0].delta.content == ""
    assert result.choices[0].finish_reason == "stop"


# ===========================================================================
# cohere.py — _extract_text_content
# ===========================================================================


def test_extract_text_content_none():
    assert _extract_text_content(None) == ""


def test_extract_text_content_string():
    assert _extract_text_content("hello") == "hello"


def test_extract_text_content_list():
    content = [
        {"type": "text", "text": "foo"},
        {"type": "text", "text": "bar"},
    ]
    assert _extract_text_content(content) == "foobar"


def test_extract_text_content_list_skips_non_text():
    content = [
        {"type": "image_url", "url": "https://x.com/img.png"},
        {"type": "text", "text": "only this"},
    ]
    assert _extract_text_content(content) == "only this"


def test_extract_text_content_non_string_non_list():
    assert _extract_text_content(42) == "42"


# ===========================================================================
# cohere.py — adapt_messages_to_cohere_standard
# ===========================================================================


def test_adapt_cohere_user_in_history():
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "user", "content": "current question"},
    ]
    history = adapt_messages_to_cohere_standard(messages)
    assert len(history) == 1
    assert history[0].role == "USER"
    assert history[0].message == "first question"


def test_adapt_cohere_assistant_in_history():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
    ]
    history = adapt_messages_to_cohere_standard(messages)
    assert len(history) == 2
    chatbot_msg = history[1]
    assert chatbot_msg.role == "CHATBOT"
    assert chatbot_msg.message == "answer"


def test_adapt_cohere_assistant_with_tool_calls_in_history():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"x": 1}'},
                }
            ],
        },
        {"role": "user", "content": "thanks"},
    ]
    history = adapt_messages_to_cohere_standard(messages)
    assert len(history) == 1
    assert history[0].role == "CHATBOT"
    assert history[0].toolCalls is not None
    assert history[0].toolCalls[0].name == "calc"


def test_adapt_cohere_tool_result_in_history():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"x": 1}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "result: 42",
        },
        {"role": "user", "content": "ok"},
    ]
    history = adapt_messages_to_cohere_standard(messages)
    tool_msg = next(m for m in history if m.role == "TOOL")
    assert tool_msg.toolResults[0].call.name == "calc"
    assert tool_msg.toolResults[0].outputs[0]["output"] == "result: 42"


# ===========================================================================
# cohere.py — handle_cohere_response
# ===========================================================================


_COHERE_RESPONSE_JSON = {
    "modelId": "cohere.command-r-plus",
    "modelVersion": "1.0",
    "chatResponse": {
        "apiFormat": "COHERE",
        "text": "Hello from Cohere!",
        "finishReason": "COMPLETE",
        "usage": {
            "promptTokens": 10,
            "completionTokens": 5,
            "totalTokens": 15,
        },
    },
}


def test_handle_cohere_response_complete():
    model_response = ModelResponse()
    result = handle_cohere_response(
        _COHERE_RESPONSE_JSON, _COHERE_MODEL, model_response
    )
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message["content"] == "Hello from Cohere!"
    assert result.usage.prompt_tokens == 10


def test_handle_cohere_response_max_tokens():
    resp = {
        **_COHERE_RESPONSE_JSON,
        "chatResponse": {
            **_COHERE_RESPONSE_JSON["chatResponse"],
            "finishReason": "MAX_TOKENS",
        },
    }
    model_response = ModelResponse()
    result = handle_cohere_response(resp, _COHERE_MODEL, model_response)
    assert result.choices[0].finish_reason == "length"


def test_handle_cohere_response_tool_call():
    resp = {
        **_COHERE_RESPONSE_JSON,
        "chatResponse": {
            **_COHERE_RESPONSE_JSON["chatResponse"],
            "finishReason": "TOOL_CALL",
            "toolCalls": [{"name": "get_time", "parameters": {"tz": "UTC"}}],
        },
    }
    model_response = ModelResponse()
    result = handle_cohere_response(resp, _COHERE_MODEL, model_response)
    assert result.choices[0].finish_reason == "tool_calls"
    tool_calls = result.choices[0].message["tool_calls"]
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "get_time"


# ===========================================================================
# cohere.py — handle_cohere_stream_chunk
# ===========================================================================


def test_handle_cohere_stream_chunk_text():
    chunk = {"apiFormat": "COHERE", "text": "streaming text", "finishReason": None}
    result = handle_cohere_stream_chunk(chunk)
    assert result.choices[0].delta.content == "streaming text"
    assert result.choices[0].finish_reason is None


def test_handle_cohere_stream_chunk_complete():
    chunk = {"apiFormat": "COHERE", "text": "", "finishReason": "COMPLETE"}
    result = handle_cohere_stream_chunk(chunk)
    assert result.choices[0].finish_reason == "stop"


def test_handle_cohere_stream_chunk_max_tokens():
    chunk = {"apiFormat": "COHERE", "text": "", "finishReason": "MAX_TOKENS"}
    result = handle_cohere_stream_chunk(chunk)
    assert result.choices[0].finish_reason == "length"


def test_handle_cohere_stream_chunk_tool_call():
    chunk = {"apiFormat": "COHERE", "text": "", "finishReason": "TOOL_CALL"}
    result = handle_cohere_stream_chunk(chunk)
    assert result.choices[0].finish_reason == "tool_calls"


# ===========================================================================
# transformation.py — get_vendor_from_model
# ===========================================================================


def test_get_vendor_cohere():
    assert get_vendor_from_model("cohere.command-r-plus") == OCIVendors.COHERE


def test_get_vendor_generic_llama():
    assert get_vendor_from_model("meta.llama-3-70b-instruct") == OCIVendors.GENERIC


def test_get_vendor_generic_xai():
    assert get_vendor_from_model("xai.grok-4") == OCIVendors.GENERIC


def test_get_vendor_generic_google():
    assert get_vendor_from_model("google.gemini-2-flash") == OCIVendors.GENERIC


# ===========================================================================
# transformation.py — OCIChatConfig methods
# ===========================================================================


class TestOCIChatConfigGetCompleteUrl:
    def test_returns_chat_endpoint_from_region(self):
        config = OCIChatConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=_GENERIC_MODEL,
            optional_params={"oci_region": "eu-frankfurt-1"},
            litellm_params={},
        )
        assert url == (
            "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
            "/20231130/actions/chat"
        )

    def test_respects_explicit_api_base(self):
        config = OCIChatConfig()
        url = config.get_complete_url(
            api_base="https://custom.endpoint.com/",
            api_key=None,
            model=_GENERIC_MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.endpoint.com/20231130/actions/chat"


class TestOCIChatConfigGetErrorClass:
    def test_returns_oci_error(self):
        config = OCIChatConfig()
        err = config.get_error_class("boom", 503, {})
        assert isinstance(err, OCIError)
        assert err.status_code == 503


class TestOCIChatConfigSignRequest:
    @patch("litellm.llms.oci.common_utils._CRYPTOGRAPHY_AVAILABLE", True)
    @patch("litellm.llms.oci.common_utils.load_private_key_from_str")
    @patch("litellm.llms.oci.common_utils.padding")
    @patch("litellm.llms.oci.common_utils.hashes")
    def test_sign_request_delegates(self, mock_hashes, mock_padding, mock_load_key):
        mock_key = MagicMock()
        mock_key.sign.return_value = b"sig"
        mock_load_key.return_value = mock_key

        config = OCIChatConfig()
        headers, body = config.sign_request(
            headers={},
            optional_params=_MANUAL_CREDS,
            request_data={"hello": "world"},
            api_base=_API_BASE,
        )
        assert "authorization" in headers
        assert isinstance(body, bytes)


class TestOCIChatConfigValidateEnvironment:
    def test_with_signer_skips_credential_check(self):
        """If oci_signer is provided, validate_environment must NOT raise."""
        config = OCIChatConfig()
        signer = MagicMock()
        result = config.validate_environment(
            headers={},
            model=_GENERIC_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"oci_signer": signer},
            litellm_params={},
        )
        assert result["content-type"] == "application/json"

    def test_raises_when_messages_empty(self):
        config = OCIChatConfig()
        with pytest.raises(OCIError) as exc_info:
            config.validate_environment(
                headers={},
                model=_GENERIC_MODEL,
                messages=[],
                optional_params={"oci_signer": MagicMock()},
                litellm_params={},
            )
        assert exc_info.value.status_code == 400


class TestOCIChatConfigGetOptionalParams:
    def _config(self):
        return OCIChatConfig()

    def test_cohere_maps_stop_to_stop_sequences(self):
        config = self._config()
        result = config._get_optional_params(
            OCIVendors.COHERE, {"stop": ["END"]}
        )
        assert "stopSequences" in result
        assert result["stopSequences"] == ["END"]

    def test_generic_maps_max_tokens(self):
        config = self._config()
        result = config._get_optional_params(
            OCIVendors.GENERIC, {"max_tokens": 512}
        )
        assert result["maxTokens"] == 512

    def test_tool_choice_string_auto_converted_to_dict(self):
        config = self._config()
        result = config._get_optional_params(
            OCIVendors.GENERIC, {"tool_choice": "auto"}
        )
        assert result["toolChoice"] == {"type": "AUTO"}

    def test_tool_choice_string_none_converted_to_dict(self):
        config = self._config()
        result = config._get_optional_params(
            OCIVendors.GENERIC, {"tool_choice": "none"}
        )
        assert result["toolChoice"] == {"type": "NONE"}

    def test_tool_choice_string_required_converted_to_dict(self):
        config = self._config()
        result = config._get_optional_params(
            OCIVendors.GENERIC, {"tool_choice": "required"}
        )
        assert result["toolChoice"] == {"type": "REQUIRED"}

    def test_response_format_json_generic(self):
        config = self._config()
        result = config._get_optional_params(
            OCIVendors.GENERIC, {"response_format": {"type": "json_object"}}
        )
        assert result["responseFormat"]["type"] == "JSON_OBJECT"

    def test_tools_adapted_for_cohere(self):
        config = self._config()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                        "required": ["msg"],
                    },
                },
            }
        ]
        result = config._get_optional_params(OCIVendors.COHERE, {"tools": tools})
        # tools should be CohereTool objects
        assert len(result["tools"]) == 1
        assert result["tools"][0].name == "echo"

    def test_tools_adapted_for_generic(self):
        config = self._config()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]
        result = config._get_optional_params(OCIVendors.GENERIC, {"tools": tools})
        assert len(result["tools"]) == 1
        assert result["tools"][0].name == "search"


class TestOCIChatConfigTransformRequest:
    _base_params = {**_MANUAL_CREDS}

    def test_generic_model_transform(self):
        config = OCIChatConfig()
        result = config.transform_request(
            model=_GENERIC_MODEL,
            messages=[{"role": "user", "content": "hello"}],
            optional_params=self._base_params,
            litellm_params={},
            headers={},
        )
        assert result["compartmentId"] == _MANUAL_CREDS["oci_compartment_id"]
        chat_req = result["chatRequest"]
        assert chat_req["apiFormat"] == "GENERIC"

    def test_cohere_model_transform(self):
        config = OCIChatConfig()
        result = config.transform_request(
            model=_COHERE_MODEL,
            messages=[{"role": "user", "content": "tell me a joke"}],
            optional_params=self._base_params,
            litellm_params={},
            headers={},
        )
        chat_req = result["chatRequest"]
        assert chat_req["apiFormat"] == "COHERE"
        assert chat_req["message"] == "tell me a joke"

    def test_cohere_model_with_system_preamble(self):
        config = OCIChatConfig()
        result = config.transform_request(
            model=_COHERE_MODEL,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            optional_params=self._base_params,
            litellm_params={},
            headers={},
        )
        assert result["chatRequest"]["preambleOverride"] == "You are helpful."

    def test_raises_without_compartment_id(self):
        config = OCIChatConfig()
        params = {k: v for k, v in _MANUAL_CREDS.items() if k != "oci_compartment_id"}
        with pytest.raises(OCIError) as exc_info:
            config.transform_request(
                model=_GENERIC_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                optional_params=params,
                litellm_params={},
                headers={},
            )
        assert exc_info.value.status_code == 400
        assert "oci_compartment_id" in str(exc_info.value)


# ===========================================================================
# transformation.py — OCIStreamWrapper.chunk_creator
# ===========================================================================


class TestOCIStreamWrapperChunkCreator:
    def _make_wrapper(self, model: str) -> "OCIStreamWrapper":
        from litellm.llms.oci.chat.transformation import OCIStreamWrapper

        return OCIStreamWrapper(
            completion_stream=iter([]),
            model=model,
            custom_llm_provider="oci",
            logging_obj=MagicMock(),
        )

    def test_cohere_chunk_dispatched_correctly(self):
        wrapper = self._make_wrapper(_COHERE_MODEL)
        payload = json.dumps({"apiFormat": "COHERE", "text": "hi", "finishReason": None})
        result = wrapper.chunk_creator(f"data:{payload}")
        assert result.choices[0].delta.content == "hi"

    def test_generic_chunk_dispatched_correctly(self):
        wrapper = self._make_wrapper(_GENERIC_MODEL)
        payload = json.dumps({
            "finishReason": "COMPLETE",
            "index": 0,
        })
        result = wrapper.chunk_creator(f"data:{payload}")
        assert result.choices[0].finish_reason == "stop"

    def test_raises_on_non_data_prefix(self):
        wrapper = self._make_wrapper(_GENERIC_MODEL)
        with pytest.raises(ValueError, match="does not start with 'data:'"):
            wrapper.chunk_creator("event: done")

    def test_raises_on_non_string_chunk(self):
        wrapper = self._make_wrapper(_GENERIC_MODEL)
        with pytest.raises(ValueError, match="not a string"):
            wrapper.chunk_creator({"bad": "type"})


# ===========================================================================
# transformation.py — get_sync_custom_stream_wrapper
# ===========================================================================


def test_get_sync_custom_stream_wrapper_returns_wrapper():
    from litellm.llms.oci.chat.transformation import OCIStreamWrapper

    config = OCIChatConfig()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_text.return_value = iter(
        ['data:{"finishReason":"COMPLETE","index":0}']
    )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    wrapper = config.get_sync_custom_stream_wrapper(
        model=_GENERIC_MODEL,
        custom_llm_provider="oci",
        logging_obj=MagicMock(),
        api_base=_API_BASE,
        headers={"authorization": "Signature ..."},
        data={"chatRequest": {}},
        messages=[{"role": "user", "content": "hi"}],
        client=mock_client,
        signed_json_body=b'{"chatRequest":{}}',
    )

    assert isinstance(wrapper, OCIStreamWrapper)
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_get_async_custom_stream_wrapper_returns_wrapper():
    from litellm.llms.oci.chat.transformation import OCIStreamWrapper

    config = OCIChatConfig()

    async def _fake_aiter_text():
        yield 'data:{"finishReason":"COMPLETE","index":0}'

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_text = _fake_aiter_text

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    wrapper = await config.get_async_custom_stream_wrapper(
        model=_GENERIC_MODEL,
        custom_llm_provider="oci",
        logging_obj=MagicMock(),
        api_base=_API_BASE,
        headers={"authorization": "Signature ..."},
        data={"chatRequest": {}},
        messages=[{"role": "user", "content": "hi"}],
        client=mock_client,
        signed_json_body=b'{"chatRequest":{}}',
    )

    assert isinstance(wrapper, OCIStreamWrapper)
