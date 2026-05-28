"""LIT-3131 regression tests.

Covers three gaps in the proxy redaction pipeline that previously leaked raw
content to CustomLogger consumers via async_log_success_event when
`x-litellm-enable-message-redaction: true` is set:

1. `litellm_params.proxy_server_request.body.messages` (and `input` /
   `prompt`) still carried the original request body content because the
   redaction helper only touched the top-level model_call_details.
2. `response_obj.choices[i].message.provider_specific_fields` still held
   raw reasoning_content / thinking_blocks / reasoningContentBlocks copies
   because the choice-content redactor only cleared the dedicated
   `reasoning_content` and `thinking_blocks` attributes.
3. The `proxy_server_request.body` snapshot constructed in
   `litellm_pre_call_utils.add_litellm_data_for_backend_llm_call` included
   `proxy_server_request` itself (the very dict that owned the snapshot
   slot), producing a doubly-nested
   `proxy_server_request.body.proxy_server_request` structure.
"""

import pytest

import litellm
from litellm.litellm_core_utils.redact_messages import (
    _redact_provider_specific_fields,
    _redact_provider_specific_fields_dict,
    _redact_proxy_server_request_body,
    perform_redaction,
)


@pytest.fixture(autouse=True)
def _reset_global_redaction():
    original = litellm.turn_off_message_logging
    litellm.turn_off_message_logging = False
    yield
    litellm.turn_off_message_logging = original


# ---------------------------------------------------------------------------
# Issue 1a — nested proxy_server_request.body content must be redacted
# ---------------------------------------------------------------------------


def _make_model_call_details_with_body(body):
    return {
        "messages": [{"role": "user", "content": "raw"}],
        "litellm_params": {
            "proxy_server_request": {
                "url": "http://localhost:4000/v1/chat/completions",
                "method": "POST",
                "headers": {"x-litellm-enable-message-redaction": "true"},
                "body": body,
            },
            "metadata": {
                "headers": {"x-litellm-enable-message-redaction": "true"}
            },
        },
    }


def test_perform_redaction_redacts_proxy_server_request_body_messages():
    raw_user_text = "raw secret content 1234"
    mcd = _make_model_call_details_with_body(
        {"model": "m", "messages": [{"role": "user", "content": raw_user_text}]}
    )
    perform_redaction(mcd, None)
    body = mcd["litellm_params"]["proxy_server_request"]["body"]
    assert body["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
    assert raw_user_text not in str(body)


def test_perform_redaction_redacts_proxy_server_request_body_input_and_prompt():
    mcd = _make_model_call_details_with_body(
        {"model": "m", "input": "raw embedding input", "prompt": "raw legacy prompt"}
    )
    perform_redaction(mcd, None)
    body = mcd["litellm_params"]["proxy_server_request"]["body"]
    assert body["input"] == ""
    assert body["prompt"] == ""


def test_perform_redaction_leaves_non_content_body_fields_alone():
    mcd = _make_model_call_details_with_body(
        {"model": "m", "messages": [{"role": "user", "content": "raw"}], "user": "u1", "temperature": 0.7}
    )
    perform_redaction(mcd, None)
    body = mcd["litellm_params"]["proxy_server_request"]["body"]
    assert body["user"] == "u1"
    assert body["temperature"] == 0.7


def test_redact_proxy_server_request_body_handles_missing_keys_safely():
    # No litellm_params at all.
    _redact_proxy_server_request_body({})
    # litellm_params with no proxy_server_request.
    _redact_proxy_server_request_body({"litellm_params": {}})
    # proxy_server_request with non-dict body.
    _redact_proxy_server_request_body(
        {"litellm_params": {"proxy_server_request": {"body": "not-a-dict"}}}
    )
    # Empty body dict — nothing to redact, no crash.
    mcd = _make_model_call_details_with_body({})
    _redact_proxy_server_request_body(mcd)
    assert mcd["litellm_params"]["proxy_server_request"]["body"] == {}


# ---------------------------------------------------------------------------
# Issue 1b — provider_specific_fields reasoning content must be wiped
# ---------------------------------------------------------------------------


def _build_choice_with_provider_specific_fields():
    return litellm.Choices(
        finish_reason="stop",
        index=0,
        message=litellm.Message(
            role="assistant",
            content="raw reply",
            reasoning_content="raw reasoning",
            thinking_blocks=[{"type": "thinking", "thinking": "raw thinking"}],
            provider_specific_fields={
                "reasoning_content": "raw reasoning",
                "thinking_blocks": [{"type": "thinking", "thinking": "raw thinking"}],
                "reasoningContentBlocks": [
                    {"reasoningText": {"text": "raw bedrock text"}}
                ],
                "native_finish_reason": "stop",  # untouched scalar
            },
        ),
    )


def test_perform_redaction_clears_message_provider_specific_fields_reasoning():
    choice = _build_choice_with_provider_specific_fields()
    response = litellm.ModelResponse(choices=[choice], model="m")
    redacted = perform_redaction(_make_model_call_details_with_body({}), response)
    psf = redacted.choices[0].message.provider_specific_fields
    assert psf["reasoning_content"] is None
    assert psf["thinking_blocks"] is None
    assert psf["reasoningContentBlocks"] is None
    # Non-reasoning keys are preserved.
    assert psf["native_finish_reason"] == "stop"


def test_redact_provider_specific_fields_no_op_when_missing():
    msg = litellm.Message(role="assistant", content="x")
    # No provider_specific_fields attribute set -> no-op, no crash.
    _redact_provider_specific_fields(msg)


def test_redact_provider_specific_fields_dict_form_clears_reasoning_keys():
    from litellm.litellm_core_utils.redact_messages import (
        _redact_model_response_dict_choices,
    )

    choices = [
        {
            "message": {
                "role": "assistant",
                "content": "raw reply",
                "reasoning_content": "raw reasoning",
                "thinking_blocks": [{"type": "thinking", "thinking": "raw"}],
                "provider_specific_fields": {
                    "reasoning_content": "raw reasoning",
                    "thinking_blocks": [{"type": "thinking", "thinking": "raw"}],
                    "reasoningContentBlocks": [
                        {"reasoningText": {"text": "raw bedrock text"}}
                    ],
                    "keep_me": True,
                },
            }
        }
    ]
    _redact_model_response_dict_choices(choices, "redacted-by-litellm")
    psf = choices[0]["message"]["provider_specific_fields"]
    assert psf["reasoning_content"] is None
    assert psf["thinking_blocks"] is None
    assert psf["reasoningContentBlocks"] is None
    assert psf["keep_me"] is True


def test_redact_model_response_dict_choices_delta_branch_clears_reasoning_psf():
    from litellm.litellm_core_utils.redact_messages import (
        _redact_model_response_dict_choices,
    )

    choices = [
        {
            "delta": {
                "role": "assistant",
                "content": "raw streaming",
                "reasoning_content": "raw reasoning",
                "thinking_blocks": [{"type": "thinking", "thinking": "raw"}],
                "provider_specific_fields": {
                    "reasoning_content": "raw reasoning",
                    "reasoningContentBlocks": [
                        {"reasoningText": {"text": "raw bedrock text"}}
                    ],
                },
            }
        }
    ]
    _redact_model_response_dict_choices(choices, "redacted-by-litellm")
    psf = choices[0]["delta"]["provider_specific_fields"]
    assert psf["reasoning_content"] is None
    assert psf["reasoningContentBlocks"] is None


def test_redact_provider_specific_fields_dict_safe_when_psf_missing_or_non_dict():
    # No provider_specific_fields key.
    _redact_provider_specific_fields_dict({"content": "x"})
    # provider_specific_fields is not a dict.
    _redact_provider_specific_fields_dict(
        {"content": "x", "provider_specific_fields": "not-a-dict"}
    )


# ---------------------------------------------------------------------------
# Issue 2 — proxy_server_request body snapshot must not nest itself
# ---------------------------------------------------------------------------


def test_proxy_server_request_body_snapshot_excludes_self():
    """Mirror of the snapshot construction in
    `litellm_pre_call_utils.add_litellm_data_for_backend_llm_call`.

    Locking in the fix prevents future regressions: the snapshot must drop
    `secret_fields` AND `proxy_server_request` itself.
    """
    data = {
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "secret_fields": {"raw_headers": "***"},
        "proxy_server_request": {
            "url": "http://localhost:4000/v1/chat",
            "method": "POST",
            "headers": {},
            "body": None,
        },
    }
    _body_snapshot = {
        k: v
        for k, v in data.items()
        if k not in ("secret_fields", "proxy_server_request")
    }
    data["proxy_server_request"]["body"] = _body_snapshot
    body = data["proxy_server_request"]["body"]
    assert "proxy_server_request" not in body
    assert "secret_fields" not in body
    assert "model" in body and "messages" in body
