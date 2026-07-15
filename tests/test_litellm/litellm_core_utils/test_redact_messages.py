"""
Tests for litellm.litellm_core_utils.redact_messages.should_redact_message_logging

Covers the proxy flow where headers arrive in litellm_params["metadata"]["headers"]
but litellm_params["litellm_metadata"] is None.
"""

from types import SimpleNamespace

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.redact_messages import (
    _redact_responses_api_output,
    perform_redaction,
    redact_streaming_responses_for_custom_logger,
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
        """provider_specific_fields is a provider-native grab-bag carrying
        output content (reasoning, citations, web-search/tool/code-interpreter
        results, compaction). Redaction wholesale-clears it — every key gone,
        including otherwise-benign metadata — so no content can slip through."""
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
                            "citations": ["CANARY_CITATION"],
                            "web_search_results": ["CANARY_SEARCH"],
                            "tool_results": ["CANARY_TOOL"],
                            "code_interpreter_results": ["CANARY_CODE"],
                            "compaction_blocks": ["CANARY_COMPACTION"],
                            "container": {"id": "benign-container-id"},
                        },
                    )
                )
            ]
        )

        redacted = perform_redaction({}, result)

        psf = redacted.choices[0].message.provider_specific_fields
        assert psf == {}
        assert "CANARY" not in str(psf)
        assert "RAW_SIGNATURE_BLOB" not in str(psf)

    def test_redacts_provider_specific_fields_multi_provider_dict_path(self):
        """The dict path (standard_logging_object) wholesale-clears the same
        grab-bag across providers — Bedrock reasoning/citations, Gemini thought
        signatures, MCP tool results, RAG search results — not just the old
        reasoning keys."""
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
                                        {"reasoningText": {"text": "BEDROCK_THOUGHT"}}
                                    ],
                                    "citationsContent": ["CANARY_BEDROCK_CITE"],
                                    "thought_signatures": ["CANARY_SIGNATURE"],
                                    "server_side_tool_invocations": ["CANARY_SSTI"],
                                    "mcp_call_results": ["CANARY_MCP"],
                                    "search_results": ["CANARY_RAG"],
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
        assert psf == {}
        assert "CANARY" not in str(psf)
        assert "BEDROCK_THOUGHT" not in str(psf)

    def test_redacts_provider_specific_fields_on_dict_delta(self):
        """Streaming-style dict path: choice['delta']['provider_specific_fields']
        is wholesale-cleared too (streaming-only keys like compaction_delta
        included)."""
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
                            "compaction_delta": {"content": "CANARY_COMPACTION_DELTA"},
                            "citation": {"text": "CANARY_CITATION_DELTA"},
                        },
                    }
                }
            ]
        }

        redacted = perform_redaction({}, result)

        psf = redacted["choices"][0]["delta"]["provider_specific_fields"]
        assert psf == {}
        assert "CANARY" not in str(psf)
        assert "STREAMED_THOUGHT" not in str(psf)

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
        """perform_redaction wholesale-redacts the provider-native request
        payload stashed on additional_args.complete_input_dict: every input
        key is dropped and only the non-input `model` survives."""
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
        assert cid == {"redacted-by-litellm": True, "model": "claude-3-7-sonnet"}
        assert "messages" not in cid
        assert "prompt" not in cid
        assert "input" not in cid
        assert "CANARY_INPUT_should_be_redacted" not in str(cid)

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

    def test_redacts_system_prompt_in_proxy_server_request_body(self):
        """The Anthropic-native top-level `system` prompt (and the
        `system_prompt` / `instructions` variants) carry user content just
        like messages — they must be scrubbed from the proxy body snapshot."""
        details = {
            "litellm_params": {
                "proxy_server_request": {
                    "body": {
                        "model": "claude-3-7-sonnet",
                        "messages": [{"role": "user", "content": "hi"}],
                        "system": "CANARY_SYSTEM_should_be_redacted",
                    }
                }
            },
        }

        perform_redaction(details, None)

        body = details["litellm_params"]["proxy_server_request"]["body"]
        assert body["system"] == "redacted-by-litellm"

    def test_redacts_system_prompt_in_complete_input_dict(self):
        """Provider-native system-prompt keys (Anthropic `system`, the
        `system_prompt` variant, Responses API `instructions`) are dropped by
        the wholesale redaction of the wire-format request body."""
        details = {
            "additional_args": {
                "complete_input_dict": {
                    "model": "claude-3-7-sonnet",
                    "system": [
                        {"type": "text", "text": "CANARY_SYSTEM_should_be_redacted"}
                    ],
                    "system_prompt": "CANARY_should_be_redacted",
                    "instructions": "CANARY_should_be_redacted",
                }
            }
        }

        perform_redaction(details, None)

        cid = details["additional_args"]["complete_input_dict"]
        assert cid == {"redacted-by-litellm": True, "model": "claude-3-7-sonnet"}
        assert "system" not in cid
        assert "system_prompt" not in cid
        assert "instructions" not in cid
        assert "CANARY_SYSTEM_should_be_redacted" not in str(cid)

    def test_redacts_gemini_native_fields_in_complete_input_dict(self):
        """Gemini/Vertex wire-format requests carry the user turn in `contents`
        and the system prompt in `system_instruction` / `systemInstruction`,
        not the OpenAI-style `messages` / `system` keys — the wholesale
        redaction drops them all, preserving only `model`."""
        details = {
            "additional_args": {
                "complete_input_dict": {
                    "model": "gemini-2.0-flash",
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": "CANARY_INPUT_should_be_redacted"}],
                        }
                    ],
                    "system_instruction": {
                        "parts": [{"text": "CANARY_SYSTEM_should_be_redacted"}]
                    },
                    "systemInstruction": {
                        "parts": [{"text": "CANARY_SYSTEM_should_be_redacted"}]
                    },
                }
            }
        }

        perform_redaction(details, None)

        cid = details["additional_args"]["complete_input_dict"]
        assert cid == {"redacted-by-litellm": True, "model": "gemini-2.0-flash"}
        assert "contents" not in cid
        assert "system_instruction" not in cid
        assert "systemInstruction" not in cid
        assert "CANARY" not in str(cid)

    def test_redacts_gemini_native_fields_in_proxy_server_request_body(self):
        """Same Gemini/Vertex native input fields can also land in the proxy
        body snapshot — `contents` / `system_instruction` must be scrubbed."""
        details = {
            "litellm_params": {
                "proxy_server_request": {
                    "body": {
                        "model": "gemini-2.0-flash",
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": "CANARY_INPUT_should_be_redacted"}],
                            }
                        ],
                        "system_instruction": {
                            "parts": [{"text": "CANARY_SYSTEM_should_be_redacted"}]
                        },
                    }
                }
            },
        }

        perform_redaction(details, None)

        body = details["litellm_params"]["proxy_server_request"]["body"]
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "redacted-by-litellm"}]}
        ]
        assert body["system_instruction"] == "redacted-by-litellm"

    def test_complete_input_dict_wholesale_redacts_embeddings_instances(self):
        """Vertex embeddings carry user input under `instances` — a key the old
        allowlist never enumerated. Wholesale redaction drops it regardless."""
        details = {
            "additional_args": {
                "complete_input_dict": {
                    "model": "text-embedding-004",
                    "instances": [{"content": "CANARY_INPUT_should_be_redacted"}],
                }
            }
        }

        perform_redaction(details, None)

        cid = details["additional_args"]["complete_input_dict"]
        assert cid == {"redacted-by-litellm": True, "model": "text-embedding-004"}
        assert "instances" not in cid
        assert "CANARY_INPUT_should_be_redacted" not in str(cid)

    def test_complete_input_dict_wholesale_redacts_rerank_query_documents(self):
        """Cohere/Bedrock rerank carry user input under `query` / `documents` —
        wholesale redaction of the native body drops both."""
        details = {
            "additional_args": {
                "complete_input_dict": {
                    "model": "rerank-english-v3.0",
                    "query": "CANARY_QUERY_should_be_redacted",
                    "documents": ["CANARY_DOC_should_be_redacted"],
                }
            }
        }

        perform_redaction(details, None)

        cid = details["additional_args"]["complete_input_dict"]
        assert cid == {"redacted-by-litellm": True, "model": "rerank-english-v3.0"}
        assert "query" not in cid
        assert "documents" not in cid
        assert "CANARY" not in str(cid)

    def test_complete_input_dict_wholesale_redacts_passthrough_arbitrary(self):
        """An arbitrary/passthrough provider key with no allowlist entry must
        still be dropped — proving the redaction has no allowlist dependency."""
        details = {
            "additional_args": {
                "complete_input_dict": {
                    "model": "some-provider/some-model",
                    "totally_unknown_provider_key": "CANARY_should_be_redacted",
                }
            }
        }

        perform_redaction(details, None)

        cid = details["additional_args"]["complete_input_dict"]
        assert cid == {
            "redacted-by-litellm": True,
            "model": "some-provider/some-model",
        }
        assert "totally_unknown_provider_key" not in cid
        assert "CANARY_should_be_redacted" not in str(cid)

    def test_redacts_rerank_keys_in_proxy_server_request_body(self):
        """The keyed proxy body snapshot scrubs the rerank input keys
        (`query` / `documents`) while leaving non-input keys (`model`, `user`)
        intact for downstream consumers."""
        details = {
            "litellm_params": {
                "proxy_server_request": {
                    "body": {
                        "model": "rerank-english-v3.0",
                        "user": "user-123",
                        "query": "CANARY_QUERY_should_be_redacted",
                        "documents": ["CANARY_DOC_should_be_redacted"],
                    }
                }
            },
        }

        perform_redaction(details, None)

        body = details["litellm_params"]["proxy_server_request"]["body"]
        assert body["query"] == "redacted-by-litellm"
        assert body["documents"] == ["redacted-by-litellm"]
        assert body["model"] == "rerank-english-v3.0"
        assert body["user"] == "user-123"

    def test_redacts_ocr_document_in_proxy_server_request_body(self):
        """The OCR route lands the user's `document` as a top-level key in the
        proxy body snapshot — it must be scrubbed while `model` survives."""
        details = {
            "litellm_params": {
                "proxy_server_request": {
                    "body": {
                        "model": "mistral/mistral-ocr-latest",
                        "document": {
                            "type": "document_url",
                            "document_url": "CANARY_DOC_should_be_redacted",
                        },
                    }
                }
            },
        }

        perform_redaction(details, None)

        body = details["litellm_params"]["proxy_server_request"]["body"]
        assert body["document"] == "redacted-by-litellm"
        assert body["model"] == "mistral/mistral-ocr-latest"
        assert "CANARY_DOC_should_be_redacted" not in str(body)

    def test_redacts_vertex_provider_metadata_in_standard_logging_response(self):
        details = {
            "standard_logging_object": {
                "messages": [{"role": "user", "content": "sensitive prompt"}],
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "sensitive answer",
                                "role": "assistant",
                            }
                        }
                    ],
                    "vertex_ai_grounding_metadata": [
                        {"webSearchQueries": ["sensitive search term"]}
                    ],
                    "vertex_ai_url_context_metadata": [
                        {"urlMetadata": [{"retrievedUrl": "https://example.com"}]}
                    ],
                },
            }
        }

        perform_redaction(details, None)

        response = details["standard_logging_object"]["response"]
        assert response["choices"][0]["message"]["content"] == "redacted-by-litellm"
        assert response["vertex_ai_grounding_metadata"] == []
        assert response["vertex_ai_url_context_metadata"] == []

    def test_redacts_vertex_provider_metadata_on_streaming_model_response(self):
        response = litellm.ModelResponse(
            id="resp-1",
            choices=[
                litellm.Choices(
                    message=litellm.Message(
                        content="sensitive answer",
                        role="assistant",
                    )
                )
            ],
            model="gemini-2.5-flash",
        )
        setattr(
            response,
            "vertex_ai_grounding_metadata",
            [{"webSearchQueries": ["sensitive search term"]}],
        )
        response._hidden_params["vertex_ai_grounding_metadata"] = [
            {"webSearchQueries": ["sensitive search term"]}
        ]

        details = {
            "stream": True,
            "complete_streaming_response": response,
        }

        perform_redaction(details, response)

        assert response.choices[0].message.content == "redacted-by-litellm"
        assert getattr(response, "vertex_ai_grounding_metadata") == []
        assert "vertex_ai_grounding_metadata" not in response._hidden_params

    def test_redacts_vertex_provider_metadata_from_metadata_hidden_params(self):
        """Streaming success_handler copies _hidden_params into metadata before redaction."""
        details = {
            "stream": True,
            "litellm_params": {
                "metadata": {
                    "hidden_params": {
                        "response_cost": 0.01,
                        "vertex_ai_grounding_metadata": [
                            {"webSearchQueries": ["sensitive search term"]}
                        ],
                        "vertex_ai_url_context_metadata": [
                            {"urlMetadata": [{"retrievedUrl": "https://example.com"}]}
                        ],
                        "vertex_ai_safety_ratings": [{"category": "HARM"}],
                        "vertex_ai_citation_metadata": [{"citations": ["source"]}],
                    }
                }
            },
        }

        perform_redaction(details, None)

        hidden_params = details["litellm_params"]["metadata"]["hidden_params"]
        assert hidden_params["response_cost"] == 0.01
        assert "vertex_ai_grounding_metadata" not in hidden_params
        assert "vertex_ai_url_context_metadata" not in hidden_params
        assert "vertex_ai_safety_ratings" not in hidden_params
        assert "vertex_ai_citation_metadata" not in hidden_params

    def test_redact_async_complete_streaming_response(self):
        """Test that async_complete_streaming_response is properly redacted."""
        response_obj = litellm.ModelResponse(
            choices=[
                litellm.Choices(
                    message=litellm.Message(content="secret content", role="assistant")
                )
            ]
        )

        model_call_details = {
            "messages": [{"role": "user", "content": "hi"}],
            "prompt": "hi",
            "input": "hi",
            "stream": True,
            "async_complete_streaming_response": response_obj,
        }

        perform_redaction(model_call_details, result=None)

        redacted_response = model_call_details["async_complete_streaming_response"]
        assert redacted_response.choices[0].message.content == "redacted-by-litellm"

    def test_redact_complete_streaming_response(self):
        """Test that complete_streaming_response is properly redacted."""
        response_obj = litellm.ModelResponse(
            choices=[
                litellm.Choices(
                    message=litellm.Message(content="secret content", role="assistant")
                )
            ]
        )

        model_call_details = {
            "messages": [{"role": "user", "content": "hi"}],
            "prompt": "hi",
            "input": "hi",
            "stream": True,
            "complete_streaming_response": response_obj,
        }

        perform_redaction(model_call_details, result=None)

        redacted_response = model_call_details["complete_streaming_response"]
        assert redacted_response.choices[0].message.content == "redacted-by-litellm"

    def test_streaming_responses_untouched_when_disabled(self):
        response_obj = litellm.ModelResponse(
            choices=[
                litellm.Choices(
                    message=litellm.Message(content="secret content", role="assistant")
                )
            ]
        )

        model_call_details = {
            "messages": [{"role": "user", "content": "hi"}],
            "prompt": "hi",
            "input": "hi",
            "stream": True,
            "async_complete_streaming_response": response_obj,
        }

        perform_redaction(model_call_details, result=None, redact_streaming_responses=False)

        assert response_obj.choices[0].message.content == "secret content"


class TestRedactStreamingResponsesForCustomLogger:
    def _model_call_details(self):
        response_obj = litellm.ModelResponse(
            choices=[
                litellm.Choices(
                    message=litellm.Message(content="secret content", role="assistant")
                )
            ]
        )
        return {
            "stream": True,
            "async_complete_streaming_response": response_obj,
        }, response_obj

    def test_opted_out_logger_gets_redacted_copy(self):
        model_call_details, response_obj = self._model_call_details()
        opted_out_logger = CustomLogger(message_logging=False)

        redacted_details = redact_streaming_responses_for_custom_logger(
            model_call_details=model_call_details, custom_logger=opted_out_logger
        )

        redacted_response = redacted_details["async_complete_streaming_response"]
        assert redacted_response.choices[0].message.content == "redacted-by-litellm"
        assert response_obj.choices[0].message.content == "secret content"
        assert model_call_details["async_complete_streaming_response"] is response_obj

    def test_compliant_logger_gets_shared_response(self):
        model_call_details, response_obj = self._model_call_details()
        compliant_logger = CustomLogger()

        result_details = redact_streaming_responses_for_custom_logger(
            model_call_details=model_call_details, custom_logger=compliant_logger
        )

        assert result_details is model_call_details
        assert response_obj.choices[0].message.content == "secret content"
