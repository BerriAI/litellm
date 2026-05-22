"""
Tests for litellm.litellm_core_utils.redact_messages.should_redact_message_logging

Covers the proxy flow where headers arrive in litellm_params["metadata"]["headers"]
but litellm_params["litellm_metadata"] is None.
"""

from types import SimpleNamespace

import pytest

import litellm
from litellm.litellm_core_utils.redact_messages import (
    _redact_responses_api_output,
    perform_redaction,
    should_redact_message_logging,
)
from litellm.responses.main import mock_responses_api_response


@pytest.fixture(autouse=True)
def _reset_global_redaction():
    """Ensure the global setting is off for every test."""
    original = litellm.turn_off_message_logging
    litellm.turn_off_message_logging = False
    yield
    litellm.turn_off_message_logging = original


def _make_model_call_details(
    metadata_headers=None,
    litellm_metadata=None,
    metadata=None,
    standard_callback_dynamic_params=None,
):
    """Build a model_call_details dict that mimics real proxy/SDK flows."""
    litellm_params = {}
    if metadata is not None:
        litellm_params["metadata"] = metadata
    elif metadata_headers is not None:
        litellm_params["metadata"] = {"headers": metadata_headers}
    else:
        litellm_params["metadata"] = {}

    # get_litellm_params always sets this key (even when value is None)
    litellm_params["litellm_metadata"] = litellm_metadata

    details = {"litellm_params": litellm_params}
    if standard_callback_dynamic_params is not None:
        details["standard_callback_dynamic_params"] = standard_callback_dynamic_params
    return details


class TestShouldRedactMessageLogging:
    """Unit tests for should_redact_message_logging()."""

    # ---- proxy flow: headers in metadata, litellm_metadata is None ----

    def test_enable_redaction_via_x_header_proxy_flow(self):
        """x-litellm-enable-message-redaction header should enable redaction
        even when litellm_metadata is None (proxy path)."""
        details = _make_model_call_details(
            metadata_headers={"x-litellm-enable-message-redaction": "true"},
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    def test_enable_redaction_via_old_header_proxy_flow(self):
        """litellm-enable-message-redaction header should enable redaction
        even when litellm_metadata is None (proxy path)."""
        details = _make_model_call_details(
            metadata_headers={"litellm-enable-message-redaction": "true"},
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    def test_disable_redaction_via_header_proxy_flow(self):
        """Core helper still honors the explicit disable-redaction header."""
        litellm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata_headers={"litellm-disable-message-redaction": "true"},
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    def test_disable_redaction_via_header_when_global_off(self):
        """litellm-disable-message-redaction is still honored when global redaction is off."""
        details = _make_model_call_details(
            metadata_headers={"litellm-disable-message-redaction": "true"},
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    # ---- SDK direct-call flow: headers in litellm_metadata ----

    def test_enable_redaction_via_header_in_litellm_metadata(self):
        """Headers inside litellm_metadata (SDK direct call) should work."""
        details = _make_model_call_details(
            litellm_metadata={
                "headers": {"x-litellm-enable-message-redaction": "true"}
            },
        )
        assert should_redact_message_logging(details) is True

    # ---- no headers at all ----

    def test_no_headers_defaults_to_global_off(self):
        """Without headers, falls back to global setting (False)."""
        details = _make_model_call_details(
            metadata_headers=None,
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    def test_no_headers_global_on(self):
        """Without headers, respects global turn_off_message_logging=True."""
        litellm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata_headers=None,
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    # ---- dynamic params take precedence ----

    def test_dynamic_param_enables_redaction(self):
        """Dynamic turn_off_message_logging=True should enable redaction."""
        details = _make_model_call_details(
            metadata_headers={},
            litellm_metadata=None,
            standard_callback_dynamic_params={"turn_off_message_logging": True},
        )
        assert should_redact_message_logging(details) is True

    def test_dynamic_param_false_overrides_header(self):
        """Dynamic turn_off_message_logging=False should take precedence over enable header."""
        details = _make_model_call_details(
            metadata_headers={"x-litellm-enable-message-redaction": "true"},
            litellm_metadata=None,
            standard_callback_dynamic_params={"turn_off_message_logging": False},
        )
        assert should_redact_message_logging(details) is False

    def test_dynamic_param_false_overrides_global_redaction(self):
        """Dynamic turn_off_message_logging=False should take precedence."""
        litellm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata_headers={},
            litellm_metadata=None,
            standard_callback_dynamic_params={"turn_off_message_logging": False},
        )
        assert should_redact_message_logging(details) is False

    # ---- non-dict metadata safety ----

    def test_both_metadata_fields_none(self):
        """When both litellm_metadata and metadata are None, should not raise."""
        details = _make_model_call_details(
            metadata=None,
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    def test_both_metadata_fields_none_global_on(self):
        """When both metadata fields are None but global is on, should still return True."""
        litellm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata=None,
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is True


class TestPerformRedaction:
    def test_redacts_standard_logging_and_responses_api_dicts(self):
        details = {
            "messages": [{"role": "user", "content": "sensitive input"}],
            "prompt": "sensitive prompt",
            "input": "sensitive input",
            "standard_logging_object": {
                "messages": [{"role": "user", "content": "sensitive input"}],
                "response": {
                    "output": [
                        {"text": "top-level text"},
                        {"content": [{"text": "nested text"}]},
                        {"type": "reasoning", "summary": [{"text": "reasoning"}]},
                    ],
                    "usage": {"total_tokens": 1},
                },
            },
        }
        result = {
            "output": [
                {"text": "top-level result"},
                {"content": [{"text": "nested result"}]},
                {"type": "reasoning", "summary": [{"text": "reasoning result"}]},
            ],
            "usage": {"total_tokens": 1},
        }

        redacted = perform_redaction(details, result)

        assert details["messages"] == [
            {"role": "user", "content": "redacted-by-litellm"}
        ]
        assert details["prompt"] == ""
        assert details["input"] == ""

        logged_response = details["standard_logging_object"]["response"]
        assert logged_response["usage"] == {"total_tokens": 1}
        assert logged_response["output"][0]["text"] == "redacted-by-litellm"
        assert logged_response["output"][1]["content"][0]["text"] == (
            "redacted-by-litellm"
        )
        assert logged_response["output"][2]["summary"][0]["text"] == (
            "redacted-by-litellm"
        )

        assert redacted["usage"] == {"total_tokens": 1}
        assert redacted["output"][0]["text"] == "redacted-by-litellm"
        assert redacted["output"][1]["content"][0]["text"] == "redacted-by-litellm"
        assert redacted["output"][2]["summary"][0]["text"] == "redacted-by-litellm"
        assert result["output"][0]["text"] == "top-level result"

    def test_redacts_model_response_dict_choices(self):
        result = {
            "choices": [
                {
                    "message": {
                        "content": "message content",
                        "reasoning_content": "message reasoning",
                        "thinking_blocks": ["thinking"],
                        "audio": {"data": "audio"},
                    }
                },
                {
                    "delta": {
                        "content": "delta content",
                        "reasoning_content": "delta reasoning",
                        "thinking_blocks": ["delta thinking"],
                        "audio": {"data": "audio"},
                    }
                },
            ]
        }

        redacted = perform_redaction({}, result)

        message = redacted["choices"][0]["message"]
        assert message["content"] == "redacted-by-litellm"
        assert message["reasoning_content"] == "redacted-by-litellm"
        assert message["thinking_blocks"] is None
        assert message["audio"] is None

        delta = redacted["choices"][1]["delta"]
        assert delta["content"] == "redacted-by-litellm"
        assert delta["reasoning_content"] == "redacted-by-litellm"
        assert delta["thinking_blocks"] is None
        assert delta["audio"] is None

    def test_redacts_standard_logging_model_response_dict_choices(self):
        details = {
            "standard_logging_object": {
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "message content",
                                "reasoning_content": "message reasoning",
                                "thinking_blocks": ["thinking"],
                                "audio": {"data": "audio"},
                            }
                        },
                        {
                            "delta": {
                                "content": "delta content",
                                "reasoning_content": "delta reasoning",
                                "thinking_blocks": ["delta thinking"],
                                "audio": {"data": "audio"},
                            }
                        },
                    ]
                }
            }
        }

        perform_redaction(details, None)

        choices = details["standard_logging_object"]["response"]["choices"]
        message = choices[0]["message"]
        assert message["content"] == "redacted-by-litellm"
        assert message["reasoning_content"] == "redacted-by-litellm"
        assert message["thinking_blocks"] is None
        assert message["audio"] is None

        delta = choices[1]["delta"]
        assert delta["content"] == "redacted-by-litellm"
        assert delta["reasoning_content"] == "redacted-by-litellm"
        assert delta["thinking_blocks"] is None
        assert delta["audio"] is None

    def test_redacts_object_choices_inside_model_response_dict(self):
        result = {
            "choices": [
                litellm.Choices(
                    message=litellm.Message(
                        content="message content",
                        role="assistant",
                        reasoning_content="message reasoning",
                    )
                )
            ]
        }

        redacted = perform_redaction({}, result)

        choice = redacted["choices"][0]
        assert choice.message.content == "redacted-by-litellm"
        assert choice.message.reasoning_content == "redacted-by-litellm"

    def test_redacts_response_output_objects_with_top_level_text(self):
        output_items = [
            SimpleNamespace(text="top-level output"),
            "non-dict output item",
        ]

        _redact_responses_api_output(output_items)

        assert output_items[0].text == "redacted-by-litellm"
        assert output_items[1] == "non-dict output item"

    def test_skips_non_dict_response_output_items(self):
        result = {
            "output": [
                "non-dict output item",
                {"content": [{"text": "nested result"}]},
            ]
        }

        redacted = perform_redaction({}, result)

        assert redacted["output"][0] == "non-dict output item"
        assert redacted["output"][1]["content"][0]["text"] == "redacted-by-litellm"

    def test_redacts_responses_api_response_object(self):
        response = mock_responses_api_response("sensitive output")

        redacted = perform_redaction({}, response)

        assert redacted.output[0].content[0].text == "redacted-by-litellm"
        assert response.output[0].content[0].text == "sensitive output"

    def test_redacts_proxy_server_request_body_messages(self):
        """perform_redaction must scrub the proxy's body snapshot, not just
        the top-level messages / prompt / input on model_call_details."""
        details = {
            "messages": [{"role": "user", "content": "sensitive input"}],
            "litellm_params": {
                "proxy_server_request": {
                    "url": "http://localhost/v1/chat/completions",
                    "method": "POST",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "user",
                                "content": "CANARY_INPUT_should_be_redacted",
                            }
                        ],
                        "prompt": "fallback prompt",
                        "input": "fallback input",
                    },
                }
            },
        }

        perform_redaction(details, None)

        body = details["litellm_params"]["proxy_server_request"]["body"]
        assert body["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
        assert body["prompt"] == ""
        assert body["input"] == ""

    def test_redact_proxy_server_request_body_is_safe_when_missing(self):
        """No KeyError / TypeError when litellm_params / proxy_server_request /
        body are absent, None, or not dicts."""
        # litellm_params missing
        perform_redaction({}, None)
        # litellm_params present, proxy_server_request missing
        perform_redaction({"litellm_params": {}}, None)
        # proxy_server_request present, body is None
        perform_redaction(
            {"litellm_params": {"proxy_server_request": {"body": None}}}, None
        )
        # proxy_server_request is not a dict
        perform_redaction(
            {"litellm_params": {"proxy_server_request": "not-a-dict"}}, None
        )

    def test_redacts_provider_specific_fields_on_object_choices(self):
        """Anthropic reasoning content lives in both message.thinking_blocks
        AND message.provider_specific_fields.thinking_blocks. The flat field
        is scrubbed; ensure the duplicate is too (plus the signature blob)."""
        result = litellm.ModelResponse(
            choices=[
                litellm.Choices(
                    message=litellm.Message(
                        content="message content",
                        role="assistant",
                        reasoning_content="message reasoning",
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "CHAIN_OF_THOUGHT"}
                        ],
                        provider_specific_fields={
                            "reasoning_content": "psf reasoning",
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "CHAIN_OF_THOUGHT",
                                    "signature": "RAW_SIGNATURE_BLOB",
                                }
                            ],
                        },
                    )
                )
            ]
        )

        redacted = perform_redaction({}, result)

        psf = redacted.choices[0].message.provider_specific_fields
        assert psf["reasoning_content"] == "redacted-by-litellm"
        assert psf["thinking_blocks"] is None

    def test_redacts_provider_specific_fields_bedrock_reasoning_content_blocks(self):
        """Bedrock converse populates provider_specific_fields.reasoningContentBlocks.
        Covered by code symmetry — assert it via the dict path."""
        details = {
            "standard_logging_object": {
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "answer",
                                "reasoning_content": "flat reasoning",
                                "provider_specific_fields": {
                                    "reasoningContentBlocks": [
                                        {
                                            "reasoningText": {
                                                "text": "BEDROCK_THOUGHT",
                                                "signature": "sig",
                                            }
                                        }
                                    ],
                                },
                            }
                        }
                    ]
                }
            }
        }

        perform_redaction(details, None)

        choice = details["standard_logging_object"]["response"]["choices"][0]
        psf = choice["message"]["provider_specific_fields"]
        assert psf["reasoningContentBlocks"] is None

    def test_redacts_provider_specific_fields_on_dict_delta(self):
        """Streaming-style dict path: choice['delta']['provider_specific_fields']."""
        result = {
            "choices": [
                {
                    "delta": {
                        "content": "delta content",
                        "reasoning_content": "delta reasoning",
                        "thinking_blocks": ["delta thinking"],
                        "provider_specific_fields": {
                            "thinking_blocks": [
                                {"type": "thinking", "thinking": "STREAMED_THOUGHT"}
                            ],
                            "reasoning_content": "psf streamed reasoning",
                        },
                    }
                }
            ]
        }

        redacted = perform_redaction({}, result)

        psf = redacted["choices"][0]["delta"]["provider_specific_fields"]
        assert psf["thinking_blocks"] is None
        assert psf["reasoning_content"] == "redacted-by-litellm"

    def test_redact_provider_specific_fields_is_safe_when_absent(self):
        """Messages without provider_specific_fields (the common case) must
        not raise."""
        result = litellm.ModelResponse(
            choices=[
                litellm.Choices(message=litellm.Message(content="hi", role="assistant"))
            ]
        )
        redacted = perform_redaction({}, result)
        assert redacted.choices[0].message.content == "redacted-by-litellm"

    def test_redacts_additional_args_complete_input_dict_messages(self):
        """perform_redaction must scrub the provider-native request payload
        stashed on additional_args.complete_input_dict. Anthropic-shape
        content blocks (list of dicts) and the OpenAI prompt / input keys
        must all be wiped."""
        details = {
            "additional_args": {
                "complete_input_dict": {
                    "model": "claude-3-7-sonnet",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "CANARY_INPUT_should_be_redacted",
                                }
                            ],
                        }
                    ],
                    "prompt": "fallback prompt",
                    "input": "fallback input",
                }
            }
        }

        perform_redaction(details, None)

        cid = details["additional_args"]["complete_input_dict"]
        assert cid["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
        assert cid["prompt"] == ""
        assert cid["input"] == ""

    def test_redact_additional_args_complete_input_dict_is_safe_when_missing(self):
        """No KeyError / TypeError when additional_args / complete_input_dict
        are absent, None, or not dicts."""
        # additional_args missing
        perform_redaction({}, None)
        # additional_args present, complete_input_dict missing
        perform_redaction({"additional_args": {}}, None)
        # complete_input_dict is None
        perform_redaction({"additional_args": {"complete_input_dict": None}}, None)
        # additional_args is not a dict
        perform_redaction({"additional_args": "not-a-dict"}, None)
