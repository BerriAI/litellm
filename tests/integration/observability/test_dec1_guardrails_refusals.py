"""Contracts for how a guardrail refusal or a guardrail failure reaches the caller on every inference surface.

A Generic Guardrail API block raises `GuardrailRaisedException(status_code=400)`, which the proxy answers as a 400
with the OpenAI error envelope on `/v1/chat/completions` and `/v1/responses` and with the Anthropic envelope
(`{"type": "error", "error": {"type": "invalid_request_error", ...}}`) on `/v1/messages`. A pre_call refusal on a
streaming request is answered before the first byte as a plain JSON 400. A post_call refusal that can only be
decided after chunks were sent ends the stream in band with an error frame. The `litellm_content_filter` and
`custom_code` guardrails refuse with an `HTTPException` whose dict detail becomes `error.provider_specific_fields`.

Guardrail failures: a Generic Guardrail API guardrail fails closed by default, and the failure is raised as a plain
exception, so the caller gets 500 (not 400). `unreachable_fallback: fail_open` lets the request through only when
the service is unreachable (connect error, timeout, or a 502/503/504 answer); any other error status still fails
closed. `fail_on_error: false` lets every guardrail error through. These are the exact paths the Rust port of the
front door and of `/v1/messages` must reproduce byte for byte, because SDKs map the status and the envelope to
exception classes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import object_value
from integration.observability.dec1_guardrails_support import (
    PLAIN_ANSWER,
    STREAM_WORDS,
    Rig,
    boot_rig,
    chat,
    chat_text,
    dead_base,
    error_message,
    generic,
    marker,
    messages,
    prompts_seen_by_provider,
    rails_called,
    refused_input,
    refused_output,
    responses,
    sse_payloads,
    streamed_content,
)
from pydantic import JsonValue

PRE: Final = "dec1-rf-pre"
POST: Final = "dec1-rf-post"
POST_END_OF_STREAM: Final = "dec1-rf-post-end-of-stream"
FAIL_CLOSED: Final = "dec1-rf-fail-closed"
FAIL_OPEN: Final = "dec1-rf-fail-open"
FAIL_OPEN_STATUS: Final = "dec1-rf-fail-open-status"
LENIENT: Final = "dec1-rf-fail-on-error-false"
WORD_FILTER: Final = "dec1-rf-word-filter"
PATTERN_MASK: Final = "dec1-rf-pattern-mask"
OUTPUT_KEYWORD_MASK: Final = "dec1-rf-output-keyword-mask"
CODE_OUTPUT_BLOCK: Final = "dec1-rf-code-output-block"
CODE_MODIFY: Final = "dec1-rf-code-modify"

BANNED_WORD: Final = "forbiddendec1word"
BANNED_DESCRIPTION: Final = "dec1 banned term"
CODE_REFUSAL: Final = "dec1 custom code refused output"


def _guardrails(policy_url: str) -> list[dict[str, JsonValue]]:
    return [
        generic(PRE, "pre_call", policy_url),
        generic(POST, "post_call", policy_url),
        generic(POST_END_OF_STREAM, "post_call", policy_url, streaming_end_of_stream_only=True),
        generic(FAIL_CLOSED, "pre_call", dead_base()),
        generic(FAIL_OPEN, "pre_call", dead_base(), unreachable_fallback="fail_open"),
        generic(FAIL_OPEN_STATUS, "pre_call", policy_url, unreachable_fallback="fail_open"),
        generic(LENIENT, "pre_call", policy_url, fail_on_error=False),
        {
            "guardrail_name": WORD_FILTER,
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "default_on": False,
                "blocked_words": [{"keyword": BANNED_WORD, "action": "BLOCK", "description": BANNED_DESCRIPTION}],
            },
        },
        {
            "guardrail_name": PATTERN_MASK,
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "default_on": False,
                "patterns": [
                    {"pattern_type": "regex", "pattern": "ACCT-[0-9]{6}", "name": "dec1_account", "action": "MASK"}
                ],
            },
        },
        {
            "guardrail_name": OUTPUT_KEYWORD_MASK,
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "post_call",
                "default_on": False,
                "blocked_words": [{"keyword": "SECRET-1234", "action": "MASK"}],
            },
        },
        {
            "guardrail_name": CODE_OUTPUT_BLOCK,
            "litellm_params": {
                "guardrail": "custom_code",
                "mode": "post_call",
                "default_on": False,
                "custom_code": (
                    "def apply_guardrail(inputs, request_data, input_type):\n"
                    "    for text in inputs.get('texts', []):\n"
                    "        if 'BLOCK-OUT' in text:\n"
                    f"            return block({CODE_REFUSAL!r})\n"
                    "    return allow()\n"
                ),
            },
        },
        {
            "guardrail_name": CODE_MODIFY,
            "litellm_params": {
                "guardrail": "custom_code",
                "mode": "pre_call",
                "default_on": False,
                "custom_code": (
                    "def apply_guardrail(inputs, request_data, input_type):\n"
                    "    return modify(texts=[text.replace('ACCT-', 'ACCOUNT-') for text in inputs.get('texts', [])])\n"
                ),
            },
        },
    ]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with boot_rig(tmp_path_factory.mktemp("dec1-guardrails-refusals"), _guardrails) as value:
        yield value


@pytest.fixture(autouse=True)
def _drained(rig: Rig) -> None:
    rig.drain()


def _with(*names: str) -> dict[str, JsonValue]:
    return {"guardrails": list(names)}


def test_chat_refusal_body_is_the_exact_openai_error_envelope(rig: Rig) -> None:
    """A pre_call block on /v1/chat/completions. The body must be exactly
    {"error": {"message": <reason>, "type": "invalid_request_error", "param": null, "code": "400"}} with status 400
    (read from code: common_request_processing.py handle_llm_api_exception, openai_error_payload.py)."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra=_with(PRE))
    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": {"message": refused_input(PRE), "type": "invalid_request_error", "param": None, "code": "400"}
    }


def test_chat_refusal_reaches_the_openai_sdk_as_a_bad_request_error(rig: Rig) -> None:
    """The OpenAI Python SDK calls chat with the blocking guardrail. It must raise openai.BadRequestError (status
    400) whose message carries the guardrail's reason, which is what SDK users catch today."""
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.gateway.key, max_retries=0)
    with pytest.raises(openai.BadRequestError) as raised:
        client.chat.completions.create(
            model=rig.chat_model,
            messages=[{"role": "user", "content": f"BLOCK-IN {marker()}"}],
            extra_body=_with(PRE),
        )
    assert raised.value.status_code == 400
    assert refused_input(PRE) in raised.value.message


def test_messages_refusal_uses_the_anthropic_error_envelope(rig: Rig) -> None:
    """A pre_call block on /v1/messages for an anthropic deployment (served by the Rust messages path once the
    guardrail lets it through). The caller gets 400 with body {"type": "error", "error": {"type":
    "invalid_request_error", "message": <reason>}} plus optional extra keys, and the provider gets nothing."""
    response: Final = messages(rig, f"BLOCK-IN {marker()}", extra=_with(PRE))
    assert response.status_code == 400, response.text
    body: Final = response.json()
    assert body["type"] == "error", body
    error: Final = object_value(body["error"])
    assert error["type"] == "invalid_request_error", body
    assert error["message"] == refused_input(PRE), body
    assert rig.provider_calls() == ()


def test_messages_refusal_reaches_the_anthropic_sdk_as_a_bad_request_error(rig: Rig) -> None:
    """The Anthropic Python SDK sends the blocked message. It must raise anthropic.BadRequestError with status 400
    and the guardrail's reason in its message."""
    client: Final = anthropic.Anthropic(base_url=rig.base(), api_key=rig.gateway.key, max_retries=0)
    with pytest.raises(anthropic.BadRequestError) as raised:
        client.messages.create(
            model=rig.claude_model,
            max_tokens=32,
            messages=[{"role": "user", "content": f"BLOCK-IN {marker()}"}],
            extra_body=_with(PRE),
        )
    assert raised.value.status_code == 400
    assert refused_input(PRE) in str(raised.value)


def test_streaming_messages_refused_before_the_first_byte_is_a_json_400(rig: Rig) -> None:
    """A streaming /v1/messages request is refused by a pre_call guardrail. Nothing was streamed yet, so the caller
    gets status 400 with an application/json Anthropic envelope, not an event stream, and the provider gets nothing.
    UNVERIFIED: content type read from the generic handler path (anthropic_endpoints/endpoints.py)."""
    response: Final = messages(rig, f"BLOCK-IN {marker()}", stream=True, extra=_with(PRE))
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json"), dict(response.headers)
    assert object_value(response.json()["error"])["message"] == refused_input(PRE)
    assert rig.provider_calls() == ()


def test_messages_post_call_refusal_uses_the_anthropic_envelope_after_the_provider_answered(rig: Rig) -> None:
    """The anthropic provider answers with BLOCK-OUT and the post_call guardrail blocks. The caller gets 400 with
    the Anthropic envelope carrying "dec1 refused output by dec1-rf-post"; the provider was called once."""
    response: Final = messages(rig, f"REPLY-BLOCK-OUT {marker()}", extra=_with(POST))
    assert response.status_code == 400, response.text
    body: Final = response.json()
    assert body["type"] == "error", body
    assert object_value(body["error"])["message"] == refused_output(POST), body
    assert len(rig.provider_calls()) == 1


def test_messages_post_call_masking_rewrites_the_text_block(rig: Rig) -> None:
    """The anthropic provider answers "the code is SECRET-1234" and the generic post_call guardrail masks it. The
    /v1/messages caller must get one text block "the code is [MASKED]"."""
    response: Final = messages(rig, f"REPLY-SECRET {marker()}", extra=_with(POST))
    assert response.status_code == 200, response.text
    assert [block.get("text") for block in response.json()["content"]] == ["the code is [MASKED]"]


def test_messages_pre_call_masking_rewrites_what_the_anthropic_provider_receives(rig: Rig) -> None:
    """A /v1/messages prompt carries SECRET-1234 and the pre_call guardrail masks it. The anthropic provider must
    receive the masked prompt, which pins that the Rust messages path sends the guardrail-modified body."""
    tag: Final = marker()
    response: Final = messages(rig, f"token SECRET-1234 {tag}", extra=_with(PRE))
    assert response.status_code == 200, response.text
    assert prompts_seen_by_provider(rig.provider_calls()) == (f"token [MASKED] {tag}",)


def test_responses_refusal_uses_the_openai_error_envelope(rig: Rig) -> None:
    """A pre_call Generic Guardrail API block on /v1/responses. Unlike a custom_code block (which answers 200 with a
    message item), this answers 400 with error.message equal to the reason and error.code "400"; the provider gets
    nothing."""
    response: Final = responses(rig, f"BLOCK-IN {marker()}", extra=_with(PRE))
    assert response.status_code == 400, response.text
    error: Final = object_value(response.json()["error"])
    assert error["message"] == refused_input(PRE), error
    assert error["code"] == "400", error
    assert rig.provider_calls() == ()


def test_responses_post_call_refusal_answers_400(rig: Rig) -> None:
    """The responses provider answers with BLOCK-OUT and the post_call guardrail blocks it. The caller gets 400
    with "dec1 refused output by dec1-rf-post"."""
    response: Final = responses(rig, f"REPLY-BLOCK-OUT {marker()}", extra=_with(POST))
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_output(POST)


def test_responses_pre_call_masking_rewrites_the_input_the_provider_receives(rig: Rig) -> None:
    """A /v1/responses string input carries SECRET-1234 and the pre_call guardrail masks it. The provider's
    /v1/responses request must carry the masked input."""
    tag: Final = marker()
    response: Final = responses(rig, f"token SECRET-1234 {tag}", extra=_with(PRE))
    assert response.status_code == 200, response.text
    seen: Final = prompts_seen_by_provider(rig.provider_calls())
    assert len(seen) == 1 and "[MASKED]" in seen[0] and "SECRET-1234" not in seen[0], seen


def test_streaming_chat_refused_before_the_first_byte_is_a_json_400(rig: Rig) -> None:
    """A streaming chat request is refused by a pre_call guardrail. The caller gets status 400 with
    content-type application/json and the OpenAI envelope (no SSE frames, no [DONE]), and the provider gets nothing."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", stream=True, extra=_with(PRE))
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json"), dict(response.headers)
    assert error_message(response) == refused_input(PRE)
    assert "[DONE]" not in response.text
    assert rig.provider_calls() == ()


def test_streaming_chat_with_pre_call_masking_streams_and_the_provider_sees_the_masked_prompt(rig: Rig) -> None:
    """A streaming chat prompt carries SECRET-1234 and the pre_call guardrail masks it. The caller gets a 200 event
    stream with the full answer and the provider receives the masked prompt."""
    tag: Final = marker()
    response: Final = chat(rig, f"SECRET-1234 {tag}", stream=True, extra=_with(PRE))
    assert response.status_code == 200, response.text
    assert streamed_content(response.text) == "".join(STREAM_WORDS)
    assert prompts_seen_by_provider(rig.provider_calls()) == (f"[MASKED] {tag}",)


def test_end_of_stream_post_call_block_ends_the_stream_with_an_error_frame_after_the_content(rig: Rig) -> None:
    """A post_call guardrail with `streaming_end_of_stream_only: true` scans the assembled answer once. The answer
    carries BLOCK-OUT, so the caller has already received the content chunks with status 200, and the stream must end
    with one `data: {"error": ...}` frame whose message is the reason and code "400", then `data: [DONE]`.
    UNVERIFIED: the error frame path for a GuardrailRaisedException mid stream (proxy_server.py stream generator)."""
    response: Final = chat(rig, f"REPLY-BLOCK-OUT {marker()}", stream=True, extra=_with(POST_END_OF_STREAM))
    assert response.status_code == 200, response.text
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    payloads: Final = sse_payloads(response.text)
    last: Final = object_value(payloads[-1])
    assert "error" in last, response.text
    error: Final = object_value(last["error"])
    assert error["message"] == refused_output(POST_END_OF_STREAM), error
    assert error["code"] == "400", error
    assert all("error" not in object_value(payload) for payload in payloads[:-1]), response.text


def test_a_clean_stream_with_a_post_call_guardrail_ends_with_stop_and_one_done(rig: Rig) -> None:
    """A streaming chat with the sampling post_call guardrail and a harmless answer. The caller gets every word,
    exactly one `[DONE]`, no error frame, and the peer was asked at least once with input_type "response"."""
    response: Final = chat(rig, f"clean stream {marker()}", stream=True, extra=_with(POST))
    assert response.status_code == 200, response.text
    assert streamed_content(response.text) == "".join(STREAM_WORDS)
    assert response.text.count("data: [DONE]") == 1
    assert all("error" not in object_value(payload) for payload in sse_payloads(response.text))
    assert {kind for _, kind in rails_called(rig.policy_calls())} == {"response"}


def test_content_filter_keyword_block_names_the_keyword_and_guardrail(rig: Rig) -> None:
    """A content filter with a BLOCK keyword and a description refuses a chat naming the word. The caller gets 400
    with message "Content blocked: keyword 'forbiddendec1word' detected (dec1 banned term)" and
    error.provider_specific_fields carrying keyword, description and guardrail_name (read from code:
    content_filter.py _handle_blocked_word_match, exception_utils.py enrich_http_exception_with_guardrail_context)."""
    response: Final = chat(rig, f"say {BANNED_WORD} {marker()}", extra=_with(WORD_FILTER))
    assert response.status_code == 400, response.text
    error: Final = object_value(response.json()["error"])
    assert error["message"] == f"Content blocked: keyword '{BANNED_WORD}' detected ({BANNED_DESCRIPTION})", error
    fields: Final = object_value(error["provider_specific_fields"])
    assert fields["keyword"] == BANNED_WORD, fields
    assert fields["description"] == BANNED_DESCRIPTION, fields
    assert fields["guardrail_name"] == WORD_FILTER, fields
    assert rig.provider_calls() == ()


def test_content_filter_keyword_block_on_messages_uses_the_anthropic_envelope(rig: Rig) -> None:
    """The same content filter refuses a /v1/messages request. The caller gets 400, body type "error", error type
    "invalid_request_error" and the same "Content blocked: keyword ..." message; the provider gets nothing."""
    response: Final = messages(rig, f"say {BANNED_WORD} {marker()}", extra=_with(WORD_FILTER))
    assert response.status_code == 400, response.text
    body: Final = response.json()
    assert body["type"] == "error", body
    error: Final = object_value(body["error"])
    assert error["type"] == "invalid_request_error", body
    assert error["message"] == f"Content blocked: keyword '{BANNED_WORD}' detected ({BANNED_DESCRIPTION})"
    assert rig.provider_calls() == ()


def test_content_filter_keyword_match_is_case_insensitive(rig: Rig) -> None:
    """The prompt carries the banned word in upper case. The content filter lowercases keywords at load and matches
    case-insensitively, so the caller still gets 400 (read from code: content_filter.py blocked_words keyed by
    lower())."""
    response: Final = chat(rig, f"say {BANNED_WORD.upper()} {marker()}", extra=_with(WORD_FILTER))
    assert response.status_code == 400, response.text


def test_content_filter_pattern_mask_rewrites_the_prompt_with_the_pattern_tag(rig: Rig) -> None:
    """A content filter regex pattern named dec1_account with action MASK. A prompt carrying ACCT-123456 reaches the
    provider with "[dec1_account_REDACTED]" in its place and the caller gets 200 (read from code: content_filter.py
    PATTERN_REDACTION_FORMAT)."""
    tag: Final = marker()
    response: Final = chat(rig, f"pay ACCT-123456 now {tag}", extra=_with(PATTERN_MASK))
    assert chat_text(response) == PLAIN_ANSWER
    assert prompts_seen_by_provider(rig.provider_calls()) == (f"pay [dec1_account_REDACTED] now {tag}",)


def test_content_filter_post_call_keyword_mask_rewrites_the_answer(rig: Rig) -> None:
    """A post_call content filter masks the keyword SECRET-1234 in the answer. The caller gets "the code is
    [KEYWORD_REDACTED]" (read from code: content_filter.py KEYWORD_REDACTION_STR)."""
    response: Final = chat(rig, f"REPLY-SECRET {marker()}", extra=_with(OUTPUT_KEYWORD_MASK))
    assert chat_text(response) == "the code is [KEYWORD_REDACTED]"


def test_custom_code_post_call_block_is_a_400_carrying_the_guardrail_and_detection_info(rig: Rig) -> None:
    """A custom_code post_call guardrail returns block() for an answer with BLOCK-OUT. Unlike a pre_call block (200
    with the message), a post_call block answers 400 with message "dec1 custom code refused output" and
    provider_specific_fields {"guardrail": name, "detection_info": {}} plus guardrail_name (read from code:
    custom_code_guardrail.py _process_result)."""
    response: Final = chat(rig, f"REPLY-BLOCK-OUT {marker()}", extra=_with(CODE_OUTPUT_BLOCK))
    assert response.status_code == 400, response.text
    error: Final = object_value(response.json()["error"])
    assert error["message"] == CODE_REFUSAL, error
    fields: Final = object_value(error["provider_specific_fields"])
    assert fields["guardrail"] == CODE_OUTPUT_BLOCK, fields
    assert fields["detection_info"] == {}, fields


def test_custom_code_modify_rewrites_the_prompt_the_provider_receives(rig: Rig) -> None:
    """A custom_code pre_call guardrail returns modify(texts=...) replacing "ACCT-" by "ACCOUNT-". The provider must
    receive the rewritten prompt. UNVERIFIED: that the sandbox accepts a list comprehension."""
    tag: Final = marker()
    response: Final = chat(rig, f"ACCT-9 {tag}", extra=_with(CODE_MODIFY))
    assert chat_text(response) == PLAIN_ANSWER
    assert prompts_seen_by_provider(rig.provider_calls()) == (f"ACCOUNT-9 {tag}",)


def test_an_unreachable_guardrail_fails_closed_with_500_and_no_provider_call(rig: Rig) -> None:
    """A guardrail whose api_base refuses connections and keeps the default fail_closed. The caller gets 500 (the
    failure is raised as a plain Exception, which carries no status) whose message starts with
    "Generic Guardrail API failed", and the provider gets nothing (read from code: generic_guardrail_api.py
    _handle_guardrail_request_error)."""
    response: Final = chat(rig, f"fail closed {marker()}", extra=_with(FAIL_CLOSED))
    assert response.status_code == 500, response.text
    assert error_message(response).startswith("Generic Guardrail API failed"), response.text
    assert rig.provider_calls() == ()


def test_an_unreachable_guardrail_on_messages_fails_closed_with_an_anthropic_api_error(rig: Rig) -> None:
    """The same fail-closed guardrail on /v1/messages. The caller gets 500 with body type "error" and error type
    "api_error", and the provider gets nothing."""
    response: Final = messages(rig, f"fail closed {marker()}", extra=_with(FAIL_CLOSED))
    assert response.status_code == 500, response.text
    body: Final = response.json()
    assert body["type"] == "error", body
    assert object_value(body["error"])["type"] == "api_error", body
    assert rig.provider_calls() == ()


def test_an_unreachable_guardrail_fails_closed_on_a_stream_before_the_first_byte(rig: Rig) -> None:
    """A streaming chat with the fail-closed unreachable guardrail. The caller gets status 500 as JSON before any
    event-stream frame, and the provider gets nothing."""
    response: Final = chat(rig, f"fail closed stream {marker()}", stream=True, extra=_with(FAIL_CLOSED))
    assert response.status_code == 500, response.text
    assert "[DONE]" not in response.text
    assert rig.provider_calls() == ()


def test_an_unreachable_fail_open_guardrail_lets_the_original_prompt_through(rig: Rig) -> None:
    """A guardrail with `unreachable_fallback: fail_open` whose api_base refuses connections. The caller gets 200
    with the answer and the provider receives the original prompt untouched."""
    tag: Final = marker()
    response: Final = chat(rig, f"fail open SECRET-1234 {tag}", extra=_with(FAIL_OPEN))
    assert chat_text(response) == PLAIN_ANSWER
    assert prompts_seen_by_provider(rig.provider_calls()) == (f"fail open SECRET-1234 {tag}",)


def test_fail_open_still_fails_closed_when_the_service_answers_500(rig: Rig) -> None:
    """The fail_open guardrail's service is reachable but answers 500. A 500 is not "unreachable" (only 502, 503
    and 504 are), so the request fails closed: the caller gets 500 and the provider gets nothing (read from code:
    generic_guardrail_api.py `is_unreachable = status_code in (502, 503, 504)`)."""
    response: Final = chat(rig, f"STATUS-500 {marker()}", extra=_with(FAIL_OPEN_STATUS))
    assert response.status_code == 500, response.text
    assert rig.provider_calls() == ()


def test_fail_open_lets_the_request_through_when_the_service_answers_503(rig: Rig) -> None:
    """The fail_open guardrail's service answers 503, which counts as unreachable. The caller gets 200 and the
    provider is called once."""
    response: Final = chat(rig, f"STATUS-503 {marker()}", extra=_with(FAIL_OPEN_STATUS))
    assert chat_text(response) == PLAIN_ANSWER
    assert len(rig.provider_calls()) == 1


def test_fail_open_still_fails_closed_when_the_service_answers_400(rig: Rig) -> None:
    """The fail_open guardrail's service answers 400. That is an error status, not unreachability, so the caller
    gets 500 (not the service's 400) and the provider gets nothing."""
    response: Final = chat(rig, f"STATUS-400 {marker()}", extra=_with(FAIL_OPEN_STATUS))
    assert response.status_code == 500, response.text
    assert rig.provider_calls() == ()


@pytest.mark.parametrize("trigger", ["STATUS-500", "STATUS-400", "BAD-JSON"], ids=["500", "400", "not_json"])
def test_fail_on_error_false_lets_every_guardrail_error_through(rig: Rig, trigger: str) -> None:
    """A guardrail with `fail_on_error: false` whose service answers 500, 400 or a non-JSON 200. Every one of these
    is treated as action NONE: the caller gets 200 and the provider receives the original prompt."""
    tag: Final = marker()
    response: Final = chat(rig, f"{trigger} {tag}", extra=_with(LENIENT))
    assert chat_text(response) == PLAIN_ANSWER
    assert prompts_seen_by_provider(rig.provider_calls()) == (f"{trigger} {tag}",)


def test_fail_on_error_false_still_honours_a_real_block(rig: Rig) -> None:
    """`fail_on_error: false` only forgives errors. When the same guardrail's service answers BLOCKED the caller
    still gets 400 with the reason and the provider gets nothing."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra=_with(LENIENT))
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_input(LENIENT)
    assert rig.provider_calls() == ()
