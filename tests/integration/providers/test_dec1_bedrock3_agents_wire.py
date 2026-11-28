"""Bedrock Agents (`bedrock/agent/<agent id>/<alias id>`) and AgentCore runtimes (`bedrock/agentcore/<runtime ARN>`).

Both are reachable through `/v1/chat/completions` on the `bedrock` provider. An Agents deployment calls
the Agents runtime `InvokeAgent` operation: `POST /agents/<agent>/agentAliases/<alias>/sessions/<session>/text`
with `{"inputText": <last message text>, "enableTrace": true}` (`AmazonInvokeAgentConfig`,
`litellm/llms/bedrock/chat/invoke_agent/transformation.py:76`). The answer is an AWS event stream whose
`chunk` events carry base64 text and whose `trace` events carry token usage in `preProcessingTrace`; the
proxy joins the chunks into one assistant message with `finish_reason: stop`.

An AgentCore deployment calls `POST /runtimes/<URL-encoded runtime ARN>/invocations` with
`{"prompt": <last message text>}`, a session id in `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` (generated
as `litellm-session-<uuid4>` when the caller gives none), an optional user id header and `?qualifier=`
query, and either a Bearer token or SigV4 for the `bedrock-agentcore` service
(`litellm/llms/bedrock/chat/agentcore/transformation.py:81`). The runtime may answer JSON in several
agent framework shapes or SSE; each shape must come back as the same assistant text. These are the
only agent integrations that ride on the plain chat route, so a rewrite of the Bedrock dispatch must
keep both route prefixes.
"""

from __future__ import annotations

import json
import re
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_msgprov_support import query_of
from integration.providers.dec1_bedrock3_support import (
    AGENT_ALIAS_ID,
    AGENT_ID,
    AGENTCORE_ARN,
    agent_chunk,
    agent_reply,
    agent_trace,
    agentcore,
    preprocessing_usage,
    quoted,
    unique_text,
)
from integration.providers.dec1_bedrock_support import (
    TOKEN,
    bedrock,
    body_of,
    path_of,
    sigv4_params,
    sigv4_scope,
)

AGENT_MODEL: Final = f"bedrock/agent/{AGENT_ID}/{AGENT_ALIAS_ID}"
AGENTCORE_MODEL: Final = f"bedrock/agentcore/{AGENTCORE_ARN}"
SESSION_PATH: Final = re.compile(rf"^/agents/{AGENT_ID}/agentAliases/{AGENT_ALIAS_ID}/sessions/([^/]+)/text$")


def _agent_ok(_: Request) -> Reply:
    return agent_reply(agent_trace(preprocessing_usage(30, 8)), agent_chunk("Hello "), agent_chunk("agent"))


def _agentcore_json(_: Request) -> Reply:
    return Reply(
        body=json.dumps({"result": {"role": "assistant", "content": [{"text": "agentcore says hi"}]}}).encode()
    )


# ------------------------------------------------------------------ Bedrock Agents


def test_an_agent_call_posts_input_text_with_trace_enabled_to_the_session_text_path(gateway: Gateway) -> None:
    """A `bedrock/agent/DEC1AGENT1/DEC1ALIAS1` deployment is called with one user turn.

    Bedrock must receive one POST to `/agents/DEC1AGENT1/agentAliases/DEC1ALIAS1/sessions/<session>/text`
    whose body has `inputText` equal to the user text and `enableTrace: true`.
    """
    text: Final = unique_text()
    with bedrock(gateway, _agent_ok, model=AGENT_MODEL) as rig:
        response: Final = rig.chat(text)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert SESSION_PATH.match(path_of(sent)), sent.target
        body: Final = body_of(sent)
        assert body["inputText"] == text
        assert body["enableTrace"] is True


def test_an_agent_call_without_a_session_gets_a_fresh_uuid_session_each_time(gateway: Gateway) -> None:
    """Two calls on the same agent deployment, no `sessionID` given.

    `_get_session_id` falls back to `uuid4()`, so each call's path must carry a UUID shaped session id and
    the two ids must differ: the proxy keeps no agent memory between calls unless the caller asks.
    """
    with bedrock(gateway, _agent_ok, model=AGENT_MODEL) as rig:
        assert rig.chat().status_code == 200
        assert rig.chat().status_code == 200
        posts: Final = rig.posts()
        sessions: Final = [SESSION_PATH.match(path_of(request)).group(1) for request in posts]  # pyright: ignore[reportOptionalMemberAccess]
        assert len(sessions) == 2, sessions
        assert all(re.fullmatch(r"[0-9a-f-]{36}", session) for session in sessions), sessions
        assert sessions[0] != sessions[1]


def test_a_caller_session_id_is_used_in_the_agent_path(gateway: Gateway) -> None:
    """The caller sends `sessionID: dec1-session-0001` in the chat body.

    It reaches `optional_params` as a provider specific field, so the path must name
    `/sessions/dec1-session-0001/text`. UNVERIFIED that the field also stays in the body.
    """
    with bedrock(gateway, _agent_ok, model=AGENT_MODEL) as rig:
        response: Final = rig.chat(sessionID="dec1-session-0001")
        assert response.status_code == 200, response.text
        assert (
            path_of(rig.one_post())
            == f"/agents/{AGENT_ID}/agentAliases/{AGENT_ALIAS_ID}/sessions/dec1-session-0001/text"
        )


def test_only_the_last_message_is_sent_as_the_agent_input(gateway: Gateway) -> None:
    """A system turn, a user turn, an assistant turn and a final user turn `last question`.

    InvokeAgent holds the conversation itself, so `inputText` must be just `last question`.
    """
    with bedrock(gateway, _agent_ok, model=AGENT_MODEL) as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "last question"},
                ]
            }
        )
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["inputText"] == "last question"


def test_agent_chunks_are_joined_into_one_assistant_message_with_trace_usage(gateway: Gateway) -> None:
    """The agent answers a trace with `inputTokens: 30, outputTokens: 8`, then chunks `Hello ` and `agent`.

    The caller must get one choice with `message.content: "Hello agent"`, `finish_reason: stop`, and usage
    30 prompt, 8 completion, 38 total tokens taken from the `preProcessingTrace`.
    """
    with bedrock(gateway, _agent_ok, model=AGENT_MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"][0]["message"]["content"] == "Hello agent"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert (body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"], body["usage"]["total_tokens"]) == (
            30,
            8,
            38,
        )


def test_an_agent_answer_without_trace_usage_reports_zero_tokens(gateway: Gateway) -> None:
    """The agent answers one chunk `bare` and no trace event.

    `_extract_usage_info` starts at zero, so the caller must get content `bare` and `usage.total_tokens: 0`.
    UNVERIFIED that the proxy does not recount tokens locally when the provider reports zero.
    """
    with bedrock(gateway, lambda _: agent_reply(agent_chunk("bare")), model=AGENT_MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"][0]["message"]["content"] == "bare"
        assert body["usage"]["total_tokens"] == 0


def test_a_streamed_agent_call_is_faked_from_one_invoke_agent_request(gateway: Gateway) -> None:
    """`stream: true` on an agent deployment: InvokeAgent has no OpenAI streaming, so `should_fake_stream` is true.

    Bedrock must still get exactly one POST to the session text path, and the caller gets an SSE answer
    whose deltas join to `Hello agent`.
    """
    with bedrock(gateway, _agent_ok, model=AGENT_MODEL) as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        assert SESSION_PATH.match(path_of(rig.one_post()))
        pieces: Final = []
        for line in response.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                for choice in json.loads(line[6:]).get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if isinstance(content, str):
                        pieces.append(content)
        assert "".join(pieces) == "Hello agent"


def test_an_agent_call_with_static_keys_is_signed_for_the_bedrock_service(gateway: Gateway) -> None:
    """The agent deployment uses SigV4 static keys and `aws_region_name: us-east-1` instead of a Bearer key.

    The Agents runtime is signed as service `bedrock` (`invoke_agent/transformation.py:121`), so the
    credential scope must be `(<access key>, us-east-1, bedrock)`.
    """
    with bedrock(gateway, _agent_ok, model=AGENT_MODEL, **sigv4_params()) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert sigv4_scope(rig.one_post())[1:] == ("us-east-1", "bedrock")


def test_a_malformed_agent_model_never_reaches_bedrock(gateway: Gateway) -> None:
    """A deployment `bedrock/agent/ONLYONEPART` lacks the alias segment.

    `_get_agent_id_and_alias_id` raises `Invalid model format. Expected format: 'model=agent/AGENT_ID/ALIAS_ID'`,
    so the caller gets an error status and the peer receives no POST.
    UNVERIFIED: the exact status code.
    """
    with bedrock(gateway, _agent_ok, model="bedrock/agent/ONLYONEPART") as rig:
        response: Final = rig.chat()
        assert response.status_code >= 400, response.text
        assert rig.posts() == ()


# ------------------------------------------------------------------ AgentCore runtimes


def test_an_agentcore_call_posts_the_prompt_to_the_encoded_runtime_invocations_path(gateway: Gateway) -> None:
    """A `bedrock/agentcore/<runtime ARN>` deployment with `aws_bedrock_runtime_endpoint` at the peer.

    Bedrock AgentCore must receive one `POST /runtimes/<ARN with ':' and '/' encoded>/invocations` whose
    body is exactly `{"prompt": <user text>}`.
    """
    text: Final = unique_text()
    with agentcore(gateway, _agentcore_json) as rig:
        response: Final = rig.chat(text)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target.split("?")[0] == f"/runtimes/{quoted(AGENTCORE_ARN)}/invocations", sent.target
        assert body_of(sent) == {"prompt": text}


def test_an_agentcore_call_generates_a_long_session_header_and_asks_for_json_or_sse(gateway: Gateway) -> None:
    """No session given by the caller.

    The request must carry `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: litellm-session-<uuid4>` (at least
    33 characters, the AgentCore minimum) and `Accept: application/json, text/event-stream`.
    """
    with agentcore(gateway, _agentcore_json) as rig:
        assert rig.chat().status_code == 200
        sent: Final = rig.one_post()
        session: Final = sent.headers["x-amzn-bedrock-agentcore-runtime-session-id"]
        assert re.fullmatch(r"litellm-session-[0-9a-f-]{36}", session), session
        assert len(session) >= 33
        assert sent.headers["accept"] == "application/json, text/event-stream"


def test_an_agentcore_runtime_session_and_user_id_become_headers(gateway: Gateway) -> None:
    """The caller passes `runtimeSessionId` (40 chars) and `runtimeUserId: dec1-user` in the chat body.

    They must arrive as `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` and
    `X-Amzn-Bedrock-AgentCore-Runtime-User-Id` headers, and the body must stay `{"prompt": ...}` only.
    """
    session: Final = "dec1-session-" + "a" * 27
    with agentcore(gateway, _agentcore_json) as rig:
        response: Final = rig.chat("hi", runtimeSessionId=session, runtimeUserId="dec1-user")
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.headers["x-amzn-bedrock-agentcore-runtime-session-id"] == session
        assert sent.headers["x-amzn-bedrock-agentcore-runtime-user-id"] == "dec1-user"
        assert body_of(sent) == {"prompt": "hi"}


def test_an_agentcore_qualifier_becomes_the_query_string(gateway: Gateway) -> None:
    """The caller passes `qualifier: DEFAULT`.

    `get_complete_url` appends it as `?qualifier=DEFAULT` (`agentcore/transformation.py:115`).
    """
    with agentcore(gateway, _agentcore_json) as rig:
        assert rig.chat(qualifier="DEFAULT").status_code == 200
        assert query_of(rig.one_post()) == "qualifier=DEFAULT"


def test_an_agentcore_bearer_key_is_sent_as_a_bearer_token_not_sigv4(gateway: Gateway) -> None:
    """The deployment has `api_key` set (a Cognito or JWT token for the runtime).

    `sign_request` sends `Authorization: Bearer <key>` and skips SigV4, so no `x-amz-date` header is sent.
    """
    with agentcore(gateway, _agentcore_json) as rig:
        assert rig.chat().status_code == 200
        sent: Final = rig.one_post()
        assert sent.headers["authorization"] == f"Bearer {TOKEN}"
        assert "x-amz-date" not in sent.headers


def test_an_agentcore_call_with_static_keys_is_signed_for_the_bedrock_agentcore_service(gateway: Gateway) -> None:
    """The deployment has SigV4 static keys and no `api_key`.

    The credential scope's service must be `bedrock-agentcore`, never `bedrock`.
    """
    with agentcore(gateway, _agentcore_json, **sigv4_params()) as rig:
        assert rig.chat().status_code == 200
        assert sigv4_scope(rig.one_post())[2] == "bedrock-agentcore"


def test_an_agentcore_json_result_message_becomes_the_assistant_content(gateway: Gateway) -> None:
    """The runtime answers `application/json` `{"result": {"role": "assistant", "content": [{"text": "agentcore says hi"}]}}`.

    The caller must get `message.content: "agentcore says hi"` and `finish_reason: stop`.
    """
    with agentcore(gateway, _agentcore_json) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        choice: Final = response.json()["choices"][0]
        assert choice["message"]["content"] == "agentcore says hi"
        assert choice["finish_reason"] == "stop"


def test_an_agentcore_strands_response_list_becomes_the_assistant_content(gateway: Gateway) -> None:
    """The runtime answers JSON `{"response": [{"text": "part one "}, {"text": "part two"}]}` (Strands agents).

    The text blocks are joined, so the content must be `part one part two`.
    """

    def respond(_: Request) -> Reply:
        return Reply(body=json.dumps({"response": [{"text": "part one "}, {"text": "part two"}]}).encode())

    with agentcore(gateway, respond) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "part one part two"


def test_an_agentcore_plain_string_result_becomes_the_assistant_content(gateway: Gateway) -> None:
    """The runtime answers JSON `{"result": "plain words"}`; the content must be `plain words`."""

    def respond(_: Request) -> Reply:
        return Reply(body=json.dumps({"result": "plain words"}).encode())

    with agentcore(gateway, respond) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "plain words"


def test_an_agentcore_unknown_json_shape_is_returned_as_raw_json_text(gateway: Gateway) -> None:
    """The runtime answers JSON `{"answer": 42}`, a shape no strategy knows.

    Strategy 4 returns the raw JSON as the content, so the caller's content must parse back to
    `{"answer": 42}`.
    """

    def respond(_: Request) -> Reply:
        return Reply(body=json.dumps({"answer": 42}).encode())

    with agentcore(gateway, respond) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert json.loads(response.json()["choices"][0]["message"]["content"]) == {"answer": 42}


def test_an_agentcore_sse_answer_joins_deltas_and_reads_metadata_usage(gateway: Gateway) -> None:
    """The runtime answers `text/event-stream` with two `contentBlockDelta` events (`Hi ` and `there`) and a
    `metadata.usage` event of 12 in, 3 out, 15 total.

    The caller (not streaming) must get content `Hi there` and usage 12, 3, 15.
    """
    events: Final = [
        {"event": {"contentBlockDelta": {"delta": {"text": "Hi "}}}},
        {"event": {"contentBlockDelta": {"delta": {"text": "there"}}}},
        {"event": {"metadata": {"usage": {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15}}}},
    ]

    def respond(_: Request) -> Reply:
        return Reply(
            content_type="text/event-stream", body="".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
        )

    with agentcore(gateway, respond) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"][0]["message"]["content"] == "Hi there"
        assert (body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"], body["usage"]["total_tokens"]) == (
            12,
            3,
            15,
        )
