"""StandardLoggingPayload on the surfaces besides plain chat: embeddings, /v1/messages, /v1/responses and streams.

Same rig as the chat payload tests: `callbacks: ["generic_api", <custom recorder>]`, one scripted provider, custom
prices of 0.001 per input token and 0.002 per output token. Each test sends one request on one surface and reads
the payload the generic_api endpoint received for its `x-litellm-call-id`. The contract is that every surface
produces exactly one payload per request with that surface's own `call_type`, the provider's token counts mapped
onto prompt_tokens and completion_tokens, and the cost those tokens price to. Streams log once, after the last
chunk, with the assembled response.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import list_value, object_value
from integration.observability.dec1_callbacks_support import (
    CHAT_COST,
    CHAT_USAGE,
    EMBEDDING,
    EMBEDDING_COST,
    EMBEDDING_TOKENS,
    MESSAGES,
    MESSAGES_COST,
    MESSAGES_USAGE,
    RECORDER,
    REPLY_TEXT,
    RESPONSES,
    RESPONSES_COST,
    RESPONSES_USAGE,
    CallbackRig,
    callback_rig,
    marker,
    new_call_id,
    post,
    stream,
    stream_chat,
    streamed_text,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-payload-surfaces"), {"callbacks": ["generic_api", RECORDER]}
    ) as created:
        yield created


def _messages_body(text: str) -> dict[str, JsonValue]:
    return {"model": MESSAGES, "max_tokens": 32, "messages": [{"role": "user", "content": text}]}


def test_embedding_payload_has_aembedding_call_type_tokens_and_cost(rig: CallbackRig) -> None:
    """A caller embeds one string and the provider reports 5 prompt tokens. Exactly one payload arrives with
    `call_type` "aembedding", status "success", prompt_tokens 5, completion_tokens 0, total_tokens 5 and
    response_cost 0.005 (5 tokens at 0.001)."""
    call_id: Final = new_call_id()
    response: Final = post(rig, "/v1/embeddings", {"model": EMBEDDING, "input": "dec1 embedding"}, call_id)
    assert response.status_code == 200, response.text
    event: Final = rig.settled_events(call_id, 1)[0]
    assert (event["call_type"], event["status"]) == ("aembedding", "success")
    assert (event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) == (EMBEDDING_TOKENS, 0, 5)
    assert event["response_cost"] == pytest.approx(EMBEDDING_COST)


def test_embedding_payload_names_group_provider_and_model(rig: CallbackRig) -> None:
    """The embedding request names the public model "dec1-embedding", served by openai/text-embedding-3-small.
    The payload reports model_group "dec1-embedding", custom_llm_provider "openai" and model
    "openai/text-embedding-3-small", the deployment's model string with its provider prefix
    (`reconstruct_model_name` in litellm/litellm_core_utils/core_helpers.py)."""
    call_id: Final = new_call_id()
    assert post(rig, "/v1/embeddings", {"model": EMBEDDING, "input": "dec1 embedding"}, call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert (event["model_group"], event["custom_llm_provider"], event["model"]) == (
        EMBEDDING,
        "openai",
        "openai/text-embedding-3-small",
    )


def test_embedding_payload_logs_the_input_text(rig: CallbackRig) -> None:
    """The embedded string must be visible in the payload's `messages` field when message logging is on.
    UNVERIFIED: the exact shape (a bare string, a list, or a message list) was not traced, so the test only
    requires the unique input text to appear inside `messages`; see get_standard_logging_object_payload."""
    call_id: Final = new_call_id()
    text: Final = marker()
    assert post(rig, "/v1/embeddings", {"model": EMBEDDING, "input": text}, call_id).status_code == 200
    assert text in json.dumps(rig.event(call_id)["messages"])


def test_messages_payload_has_anthropic_messages_call_type_tokens_and_cost(rig: CallbackRig) -> None:
    """A caller sends one Anthropic-format request to /v1/messages and the provider reports 9 input and 3 output
    tokens. Exactly one payload arrives with `call_type` "anthropic_messages", prompt_tokens 9,
    completion_tokens 3, total_tokens 12 and response_cost 0.015."""
    call_id: Final = new_call_id()
    response: Final = post(rig, "/v1/messages", _messages_body("dec1 messages"), call_id)
    assert response.status_code == 200, response.text
    event: Final = rig.settled_events(call_id, 1)[0]
    assert (event["call_type"], event["status"]) == ("anthropic_messages", "success")
    assert (event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) == (*MESSAGES_USAGE, 12)
    assert event["response_cost"] == pytest.approx(MESSAGES_COST)


def test_messages_payload_names_the_anthropic_provider_and_logs_request_and_reply(rig: CallbackRig) -> None:
    """For /v1/messages the payload reports custom_llm_provider "anthropic" and model_group "dec1-messages", its
    `messages` contain the caller's user text, and its `response` contains the provider's reply text."""
    call_id: Final = new_call_id()
    text: Final = marker()
    assert post(rig, "/v1/messages", _messages_body(text), call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert (event["custom_llm_provider"], event["model_group"]) == ("anthropic", MESSAGES)
    assert text in json.dumps(event["messages"])
    assert REPLY_TEXT in json.dumps(event["response"])


def test_messages_payload_id_is_the_message_id_the_caller_received(rig: CallbackRig) -> None:
    """The `id` of the /v1/messages payload must be the `id` of the message object returned to the caller."""
    call_id: Final = new_call_id()
    response: Final = post(rig, "/v1/messages", _messages_body("dec1 messages id"), call_id)
    assert response.status_code == 200, response.text
    assert rig.event(call_id)["id"] == response.json()["id"]


def test_responses_payload_has_aresponses_call_type_tokens_and_cost(rig: CallbackRig) -> None:
    """A caller sends one request to /v1/responses and the provider reports 7 input and 2 output tokens. Exactly
    one payload arrives with `call_type` "aresponses", prompt_tokens 7, completion_tokens 2, total_tokens 9 and
    response_cost 0.011."""
    call_id: Final = new_call_id()
    response: Final = post(rig, "/v1/responses", {"model": RESPONSES, "input": "dec1 responses"}, call_id)
    assert response.status_code == 200, response.text
    event: Final = rig.settled_events(call_id, 1)[0]
    assert (event["call_type"], event["status"]) == ("aresponses", "success")
    assert (event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) == (*RESPONSES_USAGE, 9)
    assert event["response_cost"] == pytest.approx(RESPONSES_COST)


def test_responses_payload_logs_input_and_output_text(rig: CallbackRig) -> None:
    """For /v1/responses the payload's `messages` must contain the caller's input text and its `response` must be
    an object whose output carries the provider's reply text."""
    call_id: Final = new_call_id()
    text: Final = marker()
    assert post(rig, "/v1/responses", {"model": RESPONSES, "input": text}, call_id).status_code == 200
    event: Final = rig.event(call_id)
    assert text in json.dumps(event["messages"])
    assert "output" in object_value(event["response"])
    assert REPLY_TEXT in json.dumps(event["response"])


def test_streamed_chat_delivers_one_payload_with_stream_true(rig: CallbackRig) -> None:
    """A caller streams a chat completion to the end. Exactly one payload arrives (not one per chunk), with status
    "success", `stream` true and `call_type` "acompletion"."""
    call_id: Final = new_call_id()
    status, _, sse = stream_chat(rig, call_id)
    assert status == 200 and streamed_text(sse) == REPLY_TEXT, sse
    event: Final = rig.settled_events(call_id, 1)[0]
    assert (event["status"], event["stream"], event["call_type"]) == ("success", True, "acompletion")


def test_streamed_chat_payload_has_the_assembled_reply_tokens_and_cost(rig: CallbackRig) -> None:
    """The provider streams the reply in two content chunks and reports usage 11/4 in the final chunk. The payload's
    `response` must hold the whole assembled assistant text in one choice, and tokens and cost must match the
    non-streamed case: 11, 4, 15 and 0.019."""
    call_id: Final = new_call_id()
    status, _, _ = stream_chat(rig, call_id)
    assert status == 200
    event: Final = rig.event(call_id)
    choice: Final = object_value(list_value(object_value(event["response"])["choices"])[0])
    assert object_value(choice["message"])["content"] == REPLY_TEXT
    assert (event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) == (*CHAT_USAGE, 15)
    assert event["response_cost"] == pytest.approx(CHAT_COST)


def test_streamed_chat_payload_times_the_first_chunk(rig: CallbackRig) -> None:
    """For a stream, `completionStartTime` is when the first chunk arrived: it must lie between startTime and
    endTime, and `response_time` must be completionStartTime - startTime (time to first token), not the full
    duration."""
    call_id: Final = new_call_id()
    assert stream_chat(rig, call_id)[0] == 200
    event: Final = rig.event(call_id)
    start, first, end = event["startTime"], event["completionStartTime"], event["endTime"]
    assert isinstance(start, float) and isinstance(first, float) and isinstance(end, float), event
    assert start <= first <= end
    assert event["response_time"] == pytest.approx(first - start)


def test_streamed_chat_payload_id_matches_the_chunk_id(rig: CallbackRig) -> None:
    """Every SSE chunk the caller receives carries the same completion id. The payload's `id` must be that id."""
    call_id: Final = new_call_id()
    _, _, sse = stream_chat(rig, call_id)
    chunk_ids: Final = {json.loads(line[6:])["id"] for line in sse.splitlines() if line.startswith("data: {")}
    assert len(chunk_ids) == 1, sse
    assert rig.event(call_id)["id"] in chunk_ids


def test_streamed_messages_delivers_one_success_payload_with_usage(rig: CallbackRig) -> None:
    """A caller streams /v1/messages to the end; the provider reports 9 input tokens in message_start and 3 output
    tokens in message_delta. Exactly one payload arrives with status "success", `call_type` "anthropic_messages",
    prompt_tokens 9, completion_tokens 3 and response_cost 0.015."""
    call_id: Final = new_call_id()
    status, sse = stream(rig, "/v1/messages", _messages_body("dec1 messages stream"), call_id)
    assert status == 200 and "message_stop" in sse, sse
    event: Final = rig.settled_events(call_id, 1)[0]
    assert (event["status"], event["call_type"]) == ("success", "anthropic_messages")
    assert (event["prompt_tokens"], event["completion_tokens"]) == MESSAGES_USAGE
    assert event["response_cost"] == pytest.approx(MESSAGES_COST)


def test_streamed_responses_delivers_one_success_payload_with_usage(rig: CallbackRig) -> None:
    """A caller streams /v1/responses to the end; usage 7/2 arrives in the response.completed event. Exactly one
    payload arrives with status "success", `call_type` "aresponses", prompt_tokens 7, completion_tokens 2 and
    response_cost 0.011."""
    call_id: Final = new_call_id()
    status, sse = stream(rig, "/v1/responses", {"model": RESPONSES, "input": "dec1 responses stream"}, call_id)
    assert status == 200 and "response.completed" in sse, sse
    event: Final = rig.settled_events(call_id, 1)[0]
    assert (event["status"], event["call_type"]) == ("success", "aresponses")
    assert (event["prompt_tokens"], event["completion_tokens"]) == RESPONSES_USAGE
    assert event["response_cost"] == pytest.approx(RESPONSES_COST)


def test_every_surface_reports_the_same_key_identity(rig: CallbackRig) -> None:
    """One virtual key with an alias calls chat, embeddings, /v1/messages and /v1/responses. All four payloads must
    carry the same `metadata.user_api_key_alias`, and each `metadata.user_api_key_request_route` must be the route
    that was called."""
    alias: Final = "alias-" + marker()
    routes: Final = (
        ("/v1/chat/completions", {"model": "dec1-chat", "messages": [{"role": "user", "content": "identity"}]}),
        ("/v1/embeddings", {"model": EMBEDDING, "input": "identity"}),
        ("/v1/messages", _messages_body("identity")),
        ("/v1/responses", {"model": RESPONSES, "input": "identity"}),
    )
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias)
        for route, body in routes:
            call_id = new_call_id()
            assert post(rig, route, body, call_id, key=key).status_code == 200, route
            metadata = object_value(rig.event(call_id)["metadata"])
            assert (metadata["user_api_key_alias"], metadata["user_api_key_request_route"]) == (alias, route)
