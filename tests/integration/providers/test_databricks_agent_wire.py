import asyncio
import base64
import json
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_ENDPOINT: Final = "support-agent"
_INVOCATIONS: Final = f"/serving-endpoints/{_ENDPOINT}/invocations"
_UNIFIED: Final = "/serving-endpoints/responses"
_APP: Final = "/responses"
_TOKEN: Final = "/oidc/v1/token"
_PAT: Final = "synthetic-databricks-pat"
_CLIENT_ID: Final = "synthetic-databricks-sp"
_CLIENT_SECRET: Final = "synthetic-databricks-sp-secret"
_MINTED: Final = "synthetic-databricks-minted-token"
_REPLY: Final = "the export job is waiting on a lock"
_PROMPT: Final = "Why does the export job hang?"
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OWNED_PROXY_BUDGET: Final = int(2 * graceful_stop_seconds() + 120)


def _basic(client_id: str, client_secret: str) -> str:
    return "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()


def _agent_reply(text: str = _REPLY) -> bytes:
    return json.dumps(
        {
            "output": [
                {
                    "type": "message",
                    "id": "msg-1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": _USAGE,
        }
    ).encode()


def _agent_stream(text: str = _REPLY) -> bytes:
    head, tail = text[: len(text) // 2], text[len(text) // 2 :]
    events: Final = (
        {"type": "response.output_text.delta", "item_id": "msg-1", "delta": head},
        {"type": "response.output_text.delta", "item_id": "msg-1", "delta": tail},
        {
            "type": "response.output_item.done",
            "item": {"type": "message", "id": "msg-1", "content": [{"type": "output_text", "text": text}]},
        },
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


def _token_reply() -> Reply:
    return Reply(body=json.dumps({"access_token": _MINTED, "token_type": "Bearer", "expires_in": 3600}).encode())


@dataclass(frozen=True, slots=True)
class _Databricks:
    token_status: Callable[[], int] = lambda: 200
    endpoint_reply: Callable[[Request], Reply] | None = None

    def __call__(self, request: Request) -> Reply:
        if request.target == _TOKEN:
            status: Final = self.token_status()
            if status != 200:
                return Reply(status=status, body=b'{"error":"invalid_client"}')
            return _token_reply()
        if self.endpoint_reply is not None:
            return self.endpoint_reply(request)
        if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return Reply(body=_agent_stream(), content_type="text/event-stream")
        return Reply(body=_agent_reply())


def _calls_to(requests: Sequence[Request], target: str) -> tuple[Request, ...]:
    return tuple(request for request in requests if request.target == target)


def _body(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _sdk_base_url(gateway: Gateway) -> str:
    return f"{str(gateway.client.base_url).rstrip('/')}/v1"


def _register(
    gateway: Gateway,
    cleanups: Callable[[Callable[[], None]], object],
    card_url: str,
    static_headers: dict[str, JsonValue] | None = None,
    **params: JsonValue,
) -> str:
    name: Final = f"dbx-agent-{uuid.uuid4().hex[:12]}"
    created: Final = gateway.request(
        "POST",
        "/v1/agents",
        {
            "agent_name": name,
            "agent_card_params": {
                "protocolVersion": "0.3",
                "name": name,
                "description": "Databricks Agent agent",
                "url": card_url,
                "version": "1.0.0",
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "capabilities": {"streaming": True},
                "skills": [],
            },
            "litellm_params": {"custom_llm_provider": "databricks_agent", **params},
            **({"static_headers": static_headers} if static_headers else {}),
        },
    )
    assert created.status_code == 200, created.text
    agent_id: Final = created.json()["agent_id"]

    def delete() -> None:
        deleted: Final = gateway.request("DELETE", f"/v1/agents/{agent_id}")
        assert deleted.status_code == 200, deleted.text

    cleanups(delete)
    return name


@contextmanager
def _agent(
    gateway: Gateway,
    databricks: _Databricks,
    *,
    url_path: str = "",
    static_headers: dict[str, JsonValue] | None = None,
    **params: JsonValue,
) -> Iterator[tuple[Wire, str]]:
    with wire_server(databricks) as wire, gateway.scenario() as scenario:
        params_with_base: Final = {"api_base": wire.url + url_path, **params}
        yield (
            wire,
            _register(gateway, scenario.cleanups.callback, wire.url + url_path, static_headers, **params_with_base),
        )


def _chat(gateway: Gateway, name: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": f"a2a/{name}", "messages": [{"role": "user", "content": _PROMPT}], **extra},
    )


def _a2a(
    gateway: Gateway, name: str, method: str, marker: str, metadata: dict[str, JsonValue] | None = None
) -> httpx.Response:
    message: Final[dict[str, JsonValue]] = {
        "role": "user",
        "messageId": marker,
        "parts": [{"kind": "text", "text": _PROMPT}],
        **({"metadata": metadata} if metadata is not None else {}),
    }
    return gateway.request(
        "POST",
        f"/a2a/{name}",
        {"jsonrpc": "2.0", "id": marker, "method": method, "params": {"message": message}},
    )


def _sse(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _spend_status(request_id: str) -> str:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return str(rows[0]["status"])


def _assert_invocation(request: Request, bearer: str, *, stream: bool = False) -> None:
    assert request.method == "POST" and request.target == _INVOCATIONS
    assert request.headers["authorization"] == f"Bearer {bearer}"
    assert _body(request) == {
        "input": [{"role": "user", "content": _PROMPT}],
        **({"stream": True} if stream else {}),
    }


def _assert_token_request(request: Request) -> None:
    assert request.method == "POST"
    assert request.headers["authorization"] == _basic(_CLIENT_ID, _CLIENT_SECRET)
    assert parse_qs(request.body.decode()) == {"grant_type": ["client_credentials"], "scope": ["all-apis"]}


def test_chat_on_a_pat_agent_reaches_the_serving_endpoint_and_bills_the_agent_usage(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _chat(gateway, name)
        assert response.status_code == 200, response.text
        payload: Final = response.json()
        assert payload["choices"][0]["message"] == {"content": _REPLY, "role": "assistant"}
        assert payload["choices"][0]["finish_reason"] == "stop"
        assert payload["usage"]["prompt_tokens"] == 11 and payload["usage"]["completion_tokens"] == 3
        received: Final = wire.drain()
        assert len(received) == 1, received
        _assert_invocation(received[0], _PAT)
        assert _spend_status(payload["id"]) == "success"


def test_async_chat_stream_on_a_pat_agent_sends_the_reply_once(gateway: Gateway) -> None:
    async def stream(name: str) -> str:
        client: Final = openai.AsyncOpenAI(base_url=_sdk_base_url(gateway), api_key=gateway.key, max_retries=0)
        chunks: Final = await client.chat.completions.create(
            model=f"a2a/{name}", messages=[{"role": "user", "content": _PROMPT}], stream=True
        )
        return "".join([chunk.choices[0].delta.content or "" async for chunk in chunks if chunk.choices])

    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        assert asyncio.run(stream(name)) == _REPLY
        received: Final = wire.drain()
        assert len(received) == 1, received
        _assert_invocation(received[0], _PAT, stream=True)


def test_sync_sdk_chat_on_a_pat_agent_returns_the_agent_reply(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        client: Final = openai.OpenAI(base_url=_sdk_base_url(gateway), api_key=gateway.key, max_retries=0)
        completion: Final = client.chat.completions.create(
            model=f"a2a/{name}", messages=[{"role": "user", "content": _PROMPT}]
        )
        assert completion.choices[0].message.content == _REPLY
        assert len(_calls_to(wire.drain(), _INVOCATIONS)) == 1


def test_responses_on_a_pat_agent_returns_the_reply_in_output(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = gateway.request("POST", "/v1/responses", {"model": f"a2a/{name}", "input": _PROMPT})
        assert response.status_code == 200, response.text
        texts: Final = tuple(
            part["text"]
            for item in response.json()["output"]
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
        assert texts == (_REPLY,)
        received: Final = wire.drain()
        assert len(received) == 1, received
        _assert_invocation(received[0], _PAT)


@pytest.mark.parametrize("drop_params", (False, True), ids=("strict", "drop-params"))
def test_messages_on_a_pat_agent_sends_max_tokens_as_max_output_tokens(gateway: Gateway, drop_params: bool) -> None:
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": f"a2a/{name}",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _PROMPT}],
                **({"drop_params": True} if drop_params else {}),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == [{"type": "text", "text": _REPLY}]
        invocations: Final = _calls_to(wire.drain(), _INVOCATIONS)
        assert len(invocations) == 1, invocations
        assert _body(invocations[0]) == {"input": [{"role": "user", "content": _PROMPT}], "max_output_tokens": 64}


def test_chat_sampling_params_reach_the_agent_in_its_own_shape(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _chat(gateway, name, max_completion_tokens=32, temperature=0.2, top_p=0.9, user="u-1")
        assert response.status_code == 200, response.text
        invocations: Final = _calls_to(wire.drain(), _INVOCATIONS)
        assert len(invocations) == 1, invocations
        assert _body(invocations[0]) == {
            "input": [{"role": "user", "content": _PROMPT}],
            "max_output_tokens": 32,
            "temperature": 0.2,
            "top_p": 0.9,
            "user": "u-1",
        }


def test_a_reply_with_several_message_items_reads_the_same_streamed_or_not(gateway: Gateway) -> None:
    first, second = "Let me look at the export queue.", "The export job is waiting on a lock."

    def message(item_id: str, text: str) -> dict[str, JsonValue]:
        return {"type": "message", "id": item_id, "content": [{"type": "output_text", "text": text}]}

    def reply(request: Request) -> Reply:
        if _body(request).get("stream") is not True:
            output: Final = (message("msg-1", first), {"type": "function_call", "id": "fc-1"}, message("msg-2", second))
            return Reply(body=json.dumps({"output": output, "usage": _USAGE}).encode())
        events: Final = (
            {"type": "response.output_text.delta", "item_id": "msg-1", "delta": first},
            {"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc-1"}},
            {"type": "response.output_text.delta", "item_id": "msg-2", "delta": second},
            {"type": "response.output_item.done", "item": message("msg-2", second)},
        )
        return Reply(
            body="".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(),
            content_type="text/event-stream",
        )

    with _agent(gateway, _Databricks(endpoint_reply=reply), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        client: Final = openai.OpenAI(base_url=_sdk_base_url(gateway), api_key=gateway.key, max_retries=0)
        messages: Final = [{"role": "user", "content": _PROMPT}]
        whole: Final = client.chat.completions.create(model=f"a2a/{name}", messages=messages)
        streamed: Final = client.chat.completions.create(model=f"a2a/{name}", messages=messages, stream=True)
        expected: Final = f"{first}\n\n{second}"
        assert whole.choices[0].message.content == expected
        assert "".join(chunk.choices[0].delta.content or "" for chunk in streamed if chunk.choices) == expected
        assert len(_calls_to(wire.drain(), _INVOCATIONS)) == 2


def test_identical_chat_requests_on_an_agent_each_reach_the_agent(gateway: Gateway) -> None:
    replies: Final = iter(("first answer", "second answer"))
    databricks: Final = _Databricks(endpoint_reply=lambda _: Reply(body=_agent_reply(next(replies))))
    with _agent(gateway, databricks, model=_ENDPOINT, api_key=_PAT) as (wire, name):
        answers: Final = tuple(_chat(gateway, name) for _ in range(2))
        assert [answer.json()["choices"][0]["message"]["content"] for answer in answers] == [
            "first answer",
            "second answer",
        ], [answer.text for answer in answers]
        assert len(_calls_to(wire.drain(), _INVOCATIONS)) == 2


def test_static_headers_sit_between_client_and_agent_extra_headers_on_chat(gateway: Gateway) -> None:
    with _agent(
        gateway,
        _Databricks(),
        static_headers={"X-Static": "static", "X-Shared": "from-static", "X-Both": "from-static"},
        model=_ENDPOINT,
        api_key=_PAT,
        extra_headers={"x-both": "from-agent"},
    ) as (wire, name):
        response: Final = _chat(gateway, name, extra_headers={"x-shared": "from-client", "X-Trace": "trace-1"})
        assert response.status_code == 200, response.text
        invocations: Final = _calls_to(wire.drain(), _INVOCATIONS)
        assert len(invocations) == 1, invocations
        headers: Final = invocations[0].headers
        assert (headers["x-static"], headers["x-shared"], headers["x-both"]) == ("static", "from-static", "from-agent")
        assert headers["x-trace"] == "trace-1"
        _assert_invocation(invocations[0], _PAT)


@pytest.mark.parametrize("drop_params", (False, True), ids=("strict", "drop-params"))
def test_a2a_message_metadata_reaches_the_agent_only_as_string_values(gateway: Gateway, drop_params: bool) -> None:
    params: Final[dict[str, JsonValue]] = {"drop_params": True} if drop_params else {}
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT, **params) as (wire, name):
        strings: Final = _a2a(gateway, name, "message/send", f"meta-{uuid.uuid4().hex}", {"session": "s-1"})
        assert strings.status_code == 200 and "error" not in strings.json(), strings.text
        assert _body(_calls_to(wire.drain(), _INVOCATIONS)[0])["metadata"] == {"session": "s-1"}
        mixed: Final = _a2a(gateway, name, "message/send", f"meta-{uuid.uuid4().hex}", {"session": "s-1", "turn": 1})
        sent: Final = _calls_to(wire.drain(), _INVOCATIONS)
        if drop_params:
            assert mixed.status_code == 200 and "error" not in mixed.json(), mixed.text
            assert [_body(request)["metadata"] for request in sent] == [{"session": "s-1"}]
        else:
            assert "string values" in mixed.text, mixed.text
            assert sent == ()


def test_a2a_message_send_on_a_pat_agent_answers_with_the_agent_text(gateway: Gateway) -> None:
    marker: Final = f"send-{uuid.uuid4().hex}"
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _a2a(gateway, name, "message/send", marker)
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["id"] == marker and "error" not in body, body
        assert body["result"]["parts"] == [{"kind": "text", "text": _REPLY}]
        received: Final = wire.drain()
        assert len(received) == 1, received
        _assert_invocation(received[0], _PAT)


def test_a2a_message_stream_on_a_pat_agent_streams_the_text_and_completes(gateway: Gateway) -> None:
    marker: Final = f"stream-{uuid.uuid4().hex}"
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _a2a(gateway, name, "message/stream", marker)
        assert response.status_code == 200, response.text
        events: Final = tuple(event["result"] for event in _sse(response.text))
        artifacts: Final = tuple(
            part["text"]
            for event in events
            if event["kind"] == "artifact-update"
            for part in event["artifact"]["parts"]
        )
        assert artifacts == (_REPLY,)
        assert events[-1]["kind"] == "status-update" and events[-1]["final"] is True
        assert events[-1]["status"]["state"] == "completed"
        received: Final = wire.drain()
        assert len(received) == 1, received
        _assert_invocation(received[0], _PAT, stream=True)


def test_the_unified_responses_url_names_the_endpoint_in_the_body(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), url_path=_UNIFIED, model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _chat(gateway, name)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == _REPLY
        received: Final = wire.drain()
        assert [(request.method, request.target) for request in received] == [("POST", _UNIFIED)]
        assert received[0].headers["authorization"] == f"Bearer {_PAT}"
        assert _body(received[0]) == {"model": _ENDPOINT, "input": [{"role": "user", "content": _PROMPT}]}


def test_an_app_agent_mints_its_token_at_the_workspace_and_calls_the_app(gateway: Gateway) -> None:
    with (
        wire_server(_Databricks()) as workspace,
        _agent(
            gateway,
            _Databricks(),
            url_path=_APP,
            client_id=_CLIENT_ID,
            client_secret=_CLIENT_SECRET,
            workspace_url=workspace.url,
        ) as (app, name),
    ):
        response: Final = _chat(gateway, name)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == _REPLY
        minted: Final = workspace.drain()
        assert [request.target for request in minted] == [_TOKEN]
        _assert_token_request(minted[0])
        called: Final = app.drain()
        assert [(request.method, request.target) for request in called] == [("POST", _APP)]
        assert called[0].headers["authorization"] == f"Bearer {_MINTED}"
        assert _body(called[0]) == {"input": [{"role": "user", "content": _PROMPT}]}


def test_a_nested_oauth_block_mints_once_for_chat_and_a2a_calls_on_one_agent(gateway: Gateway) -> None:
    def oauth(url: str) -> dict[str, JsonValue]:
        return {"client_id": _CLIENT_ID, "client_secret": _CLIENT_SECRET, "workspace_url": url}

    with wire_server(_Databricks()) as wire, gateway.scenario() as scenario:
        name: Final = _register(
            gateway,
            scenario.cleanups.callback,
            wire.url,
            model=_ENDPOINT,
            api_base=wire.url,
            databricks_oauth=oauth(wire.url),
        )
        for _ in range(3):
            response = _chat(gateway, name)
            assert response.status_code == 200, response.text
        sent: Final = _a2a(gateway, name, "message/send", f"send-{uuid.uuid4().hex}")
        assert sent.status_code == 200 and "error" not in sent.json(), sent.text
        received: Final = wire.drain()
        tokens: Final = _calls_to(received, _TOKEN)
        assert len(tokens) <= 2, [request.target for request in received]
        _assert_token_request(tokens[0])
        invocations: Final = _calls_to(received, _INVOCATIONS)
        assert len(invocations) == 4
        for invocation in invocations:
            _assert_invocation(invocation, _MINTED)


def test_the_minted_token_outranks_the_agent_pat_and_a_client_authorization(gateway: Gateway) -> None:
    with _agent(
        gateway,
        _Databricks(),
        model=_ENDPOINT,
        api_key=_PAT,
        client_id=_CLIENT_ID,
        client_secret=_CLIENT_SECRET,
    ) as (wire, name):
        response: Final = _chat(gateway, name, extra_headers={"Authorization": "Bearer client-supplied"})
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
        assert [request.target for request in received] == [_TOKEN, _INVOCATIONS]
        _assert_invocation(received[1], _MINTED)


def test_the_agent_pat_outranks_a_client_authorization(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _chat(gateway, name, extra_headers={"Authorization": "Bearer client-supplied"})
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
        assert len(received) == 1, received
        _assert_invocation(received[0], _PAT)


def test_empty_oauth_fields_beside_a_pat_keep_the_pat_path(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT, client_id="", client_secret="") as (wire, name):
        response: Final = _chat(gateway, name)
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
        assert [request.target for request in received] == [_INVOCATIONS]
        _assert_invocation(received[0], _PAT)


def test_custom_inputs_reach_the_agent_beside_the_input(gateway: Gateway) -> None:
    custom_inputs: Final[dict[str, JsonValue]] = {"session_id": "s-1", "tier": "gold"}
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _chat(gateway, name, custom_inputs=custom_inputs)
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
        assert len(received) == 1, received
        assert _body(received[0]) == {"input": [{"role": "user", "content": _PROMPT}], "custom_inputs": custom_inputs}


def test_two_agents_on_one_endpoint_name_never_answer_from_each_other(gateway: Gateway) -> None:
    def answering(text: str) -> _Databricks:
        return _Databricks(endpoint_reply=lambda _: Reply(body=_agent_reply(text)))

    with (
        _agent(gateway, answering("from workspace one"), model=_ENDPOINT, api_key=_PAT) as (first_wire, first),
        _agent(gateway, answering("from workspace two"), model=_ENDPOINT, api_key=_PAT) as (second_wire, second),
    ):
        responses: Final = tuple(_chat(gateway, name) for name in (first, second, first))
        assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
        answers: Final = tuple(response.json()["choices"][0]["message"]["content"] for response in responses)
        assert answers == ("from workspace one", "from workspace two", "from workspace one")
        assert len(_calls_to(first_wire.drain(), _INVOCATIONS)) == 2
        assert len(_calls_to(second_wire.drain(), _INVOCATIONS)) == 1


def test_a_workspace_agent_with_no_endpoint_is_refused_before_any_call(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), api_key=_PAT) as (wire, name):
        response: Final = _chat(gateway, name)
        assert response.status_code == 400, response.text
        assert "model must name the Databricks serving endpoint" in response.text
        assert wire.drain() == ()


def test_an_app_agent_without_a_workspace_url_is_refused_before_its_secret_leaves(gateway: Gateway) -> None:
    with _agent(gateway, _Databricks(), url_path=_APP, client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET) as (
        wire,
        name,
    ):
        response: Final = _chat(gateway, name)
        assert response.status_code == 400, response.text
        assert "workspace_url" in response.text
        assert wire.drain() == ()


@pytest.mark.parametrize(("drop_params", "status"), ((False, 400), (True, 200)), ids=("strict", "drop-params"))
def test_malformed_client_extra_headers_follow_drop_params(gateway: Gateway, drop_params: bool, status: int) -> None:
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _chat(
            gateway, name, extra_headers=["Authorization", "x"], **({"drop_params": True} if drop_params else {})
        )
        assert response.status_code == status, response.text
        if drop_params:
            received: Final = wire.drain()
            assert len(received) == 1, received
            _assert_invocation(received[0], _PAT)
        else:
            assert "extra_headers must be an object" in response.text
            assert wire.drain() == ()


@pytest.mark.parametrize(("drop_params", "status"), ((False, 400), (True, 200)), ids=("strict", "drop-params"))
def test_an_image_part_follows_drop_params(gateway: Gateway, drop_params: bool, status: int) -> None:
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": _PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ]
    with _agent(gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": f"a2a/{name}",
                "messages": [{"role": "user", "content": content}],
                **({"drop_params": True} if drop_params else {}),
            },
        )
        assert response.status_code == status, response.text
        if drop_params:
            received: Final = wire.drain()
            assert len(received) == 1, received
            _assert_invocation(received[0], _PAT)
        else:
            assert "image_url" in response.text
            assert wire.drain() == ()


@pytest.mark.parametrize(
    ("reply", "status"),
    (
        (Reply(status=401, body=b'{"error_code":"UNAUTHENTICATED","message":"Invalid token"}'), 401),
        (Reply(status=500, body=b'{"error_code":"INTERNAL_ERROR","message":"agent crashed"}'), 500),
        (Reply(body=b"{}"), 502),
    ),
    ids=("upstream-401", "upstream-500", "not-a-responses-agent"),
)
def test_a_failing_agent_keeps_its_status_and_unrelated_models_keep_serving(
    gateway: Gateway, reply: Reply, status: int
) -> None:
    with (
        _agent(gateway, _Databricks(endpoint_reply=lambda _: reply), model=_ENDPOINT, api_key=_PAT) as (wire, name),
        gateway.scenario() as scenario,
    ):
        response: Final = _chat(gateway, name)
        assert response.status_code == status, response.text
        assert len(_calls_to(wire.drain(), _INVOCATIONS)) >= 1
        unrelated: Final = gateway.chat(scenario.model())
        assert unrelated["choices"], unrelated


def test_a_refused_token_request_answers_an_error_without_calling_the_agent(gateway: Gateway) -> None:
    with _agent(
        gateway,
        _Databricks(token_status=lambda: 400),
        model=_ENDPOINT,
        client_id=_CLIENT_ID,
        client_secret=_CLIENT_SECRET,
    ) as (wire, name):
        response: Final = _chat(gateway, name)
        assert response.status_code >= 400, response.text
        received: Final = wire.drain()
        assert [request.target for request in received] == [_TOKEN]


def test_reading_an_agent_back_redacts_its_pat_and_client_secret(gateway: Gateway) -> None:
    with _agent(
        gateway, _Databricks(), model=_ENDPOINT, api_key=_PAT, client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET
    ) as (_, name):
        listed: Final = gateway.request("GET", "/v1/agents")
        assert listed.status_code == 200, listed.text
        text: Final = listed.text
        assert name in text
        assert _PAT not in text and _CLIENT_SECRET not in text


@pytest.mark.parametrize(
    "oauth_fields",
    (None, (), ("client_id",), ("client_id", "client_secret", "workspace_url")),
    ids=(
        "plain",
        "with-empty-databricks-oauth",
        "with-partial-databricks-oauth",
        "with-complete-databricks-oauth",
    ),
)
def test_a_url_agent_keeps_its_a2a_call_to_the_card_url(gateway: Gateway, oauth_fields: tuple[str, ...] | None) -> None:
    marker: Final = f"url-agent-{uuid.uuid4().hex}"

    def peer(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(status=404)
        outbound: Final = _body(request)
        return Reply(
            body=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": outbound["id"],
                    "result": {
                        "kind": "message",
                        "role": "agent",
                        "messageId": marker,
                        "parts": [{"kind": "text", "text": _REPLY}],
                    },
                }
            ).encode()
        )

    with wire_server(peer) as wire, gateway.scenario() as scenario:
        complete_block: Final = {"client_id": _CLIENT_ID, "client_secret": _CLIENT_SECRET, "workspace_url": wire.url}
        oauth_block: Final = None if oauth_fields is None else {field: complete_block[field] for field in oauth_fields}
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {
                "agent_name": marker,
                "agent_card_params": {
                    "protocolVersion": "0.3",
                    "name": marker,
                    "description": "URL agent",
                    "url": wire.url + "/",
                    "version": "1.0.0",
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "capabilities": {"streaming": False},
                    "skills": [],
                },
                **({"litellm_params": {"databricks_oauth": oauth_block}} if oauth_block is not None else {}),
            },
        )
        assert created.status_code == 200, created.text
        agent_id: Final = created.json()["agent_id"]
        scenario.cleanups.callback(lambda: gateway.request("DELETE", f"/v1/agents/{agent_id}"))
        response: Final = _chat(gateway, marker, extra_headers={"Authorization": "Bearer client-token"})
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == _REPLY
        posts: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert [request.target for request in posts] == ["/"]
        assert _body(posts[0])["method"] == "message/send"
        assert posts[0].headers["authorization"] == "Bearer client-token"


def test_a_url_agent_with_an_incomplete_databricks_oauth_answers_a2a_with_an_internal_error(gateway: Gateway) -> None:
    marker: Final = f"url-agent-{uuid.uuid4().hex}"
    with wire_server(lambda _: Reply(status=404)) as wire, gateway.scenario() as scenario:
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {
                "agent_name": marker,
                "agent_card_params": {
                    "protocolVersion": "0.3",
                    "name": marker,
                    "description": "URL agent",
                    "url": wire.url + "/",
                    "version": "1.0.0",
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "capabilities": {"streaming": False},
                    "skills": [],
                },
                "litellm_params": {"databricks_oauth": {"client_id": _CLIENT_ID}},
            },
        )
        assert created.status_code == 200, created.text
        agent_id: Final = created.json()["agent_id"]
        scenario.cleanups.callback(lambda: gateway.request("DELETE", f"/v1/agents/{agent_id}"))
        response: Final = _a2a(gateway, marker, "message/send", marker)
        assert response.status_code == 500, response.text
        assert response.json()["error"]["code"] == -32603
        assert "missing required field" in response.json()["error"]["message"]
        assert tuple(request for request in wire.drain() if request.method == "POST") == ()


@pytest.mark.parametrize(
    "pinned",
    ({"model": "another-endpoint"}, {"input": [{"role": "user", "content": "injected"}]}),
    ids=("model", "input"),
)
@pytest.mark.parametrize(("drop_params", "status"), ((False, 400), (True, 200)), ids=("strict", "drop-params"))
def test_extra_body_never_replaces_the_agent_endpoint_or_input(
    gateway: Gateway, pinned: dict[str, JsonValue], drop_params: bool, status: int
) -> None:
    custom_inputs: Final[dict[str, JsonValue]] = {"tenant": "t-1"}
    with _agent(gateway, _Databricks(), url_path=_UNIFIED, model=_ENDPOINT, api_key=_PAT) as (wire, name):
        response: Final = _chat(
            gateway,
            name,
            extra_body={**pinned, "custom_inputs": custom_inputs},
            **({"drop_params": True} if drop_params else {}),
        )
        assert response.status_code == status, response.text
        received: Final = wire.drain()
        if drop_params:
            assert [(request.method, request.target) for request in received] == [("POST", _UNIFIED)]
            assert _body(received[0]) == {
                "model": _ENDPOINT,
                "input": [{"role": "user", "content": _PROMPT}],
                "custom_inputs": custom_inputs,
            }
        else:
            assert f"cannot set {next(iter(pinned))}" in response.text
            assert received == ()


def test_a_token_outage_mid_burst_fails_its_calls_and_one_mint_serves_the_recovery(gateway: Gateway) -> None:
    outage: Final = threading.Event()
    outage.set()

    def token_status() -> int:
        return 503 if outage.is_set() else 200

    async def burst(name: str, size: int) -> tuple[int, ...]:
        async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
            responses: Final = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json={"model": f"a2a/{name}", "messages": [{"role": "user", "content": _PROMPT}]},
                        headers={"Authorization": f"Bearer {gateway.key}"},
                    )
                    for _ in range(size)
                )
            )
            return tuple(response.status_code for response in responses)

    with _agent(
        gateway,
        _Databricks(token_status=token_status),
        model=_ENDPOINT,
        client_id=_CLIENT_ID,
        client_secret=_CLIENT_SECRET,
    ) as (wire, name):
        failed: Final = asyncio.run(burst(name, 12))
        assert all(status >= 400 for status in failed), failed
        assert _calls_to(wire.drain(), _INVOCATIONS) == ()
        outage.clear()
        recovered: Final = asyncio.run(burst(name, 12))
        assert recovered == (200,) * 12, recovered
        received: Final = wire.drain()
        assert len(_calls_to(received, _TOKEN)) <= 2
        for invocation in _calls_to(received, _INVOCATIONS):
            _assert_invocation(invocation, _MINTED)
        assert len(_calls_to(received, _INVOCATIONS)) == 12


def _owned_config(
    directory: Path, wire: Wire, api_key: str | None = _PAT, *, allow_client_side_credentials: bool = False
) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    general_settings: Final = _JSON_OBJECT.validate_python(base.get("general_settings") or {})
    name: Final = "dbx-owned-agent"
    config: Final[dict[str, JsonValue]] = {
        **base,
        "general_settings": {
            **general_settings,
            **({"allow_client_side_credentials": True} if allow_client_side_credentials else {}),
        },
        "agents": [
            {
                "agent_name": name,
                "agent_card_params": {
                    "protocolVersion": "0.3",
                    "name": name,
                    "description": "Databricks Agent agent",
                    "url": wire.url,
                    "version": "1.0.0",
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "capabilities": {"streaming": True},
                    "skills": [],
                },
                "litellm_params": {
                    "custom_llm_provider": "databricks_agent",
                    "model": _ENDPOINT,
                    "api_base": wire.url,
                    **({"api_key": api_key} if api_key is not None else {}),
                },
            }
        ],
    }
    path: Final = directory / "databricks-agent-owned.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.timeout(_OWNED_PROXY_BUDGET)
def test_a_client_api_base_or_base_url_never_takes_the_agent_pat_off_its_card_url(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_Databricks()) as wire, wire_server(lambda _: Reply(body=_agent_reply())) as collector:
        config: Final = _owned_config(tmp_path, wire, allow_client_side_credentials=True)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            rejected: Final = _chat(owned.gateway, "dbx-owned-agent", api_base=collector.url)
            served: Final = _chat(owned.gateway, "dbx-owned-agent", base_url=collector.url)
            assert rejected.status_code == 400, rejected.text
            assert served.status_code == 200, served.text
            assert collector.drain() == ()
            invocations: Final = _calls_to(wire.drain(), _INVOCATIONS)
            assert len(invocations) == 1
            _assert_invocation(invocations[0], _PAT)


@pytest.mark.timeout(_OWNED_PROXY_BUDGET)
def test_a_killed_worker_leaves_the_sibling_serving_the_databricks_agent(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_Databricks()) as wire:
        with owned_proxy_process(gateway, tmp_path, {}, config=_owned_config(tmp_path, wire), workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=graceful_stop_seconds() + 60,
            )
            first: Final = _chat(candidate, "dbx-owned-agent")
            assert first.status_code == 200, first.text
            replacement: Final = owned.kill_worker(workers[0])
            assert replacement not in workers
            served: Final = tuple(_chat(candidate, "dbx-owned-agent") for _ in range(6))
            assert all(response.status_code == 200 for response in served), [r.text for r in served]
            assert {response.json()["choices"][0]["message"]["content"] for response in served} == {_REPLY}
            invocations: Final = _calls_to(wire.drain(), _INVOCATIONS)
            assert len(invocations) == 7
            for invocation in invocations:
                _assert_invocation(invocation, _PAT)


@pytest.mark.timeout(_OWNED_PROXY_BUDGET)
def test_the_env_service_principal_mints_once_and_env_references_resolve_on_an_api_created_agent(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_Databricks()) as wire:
        environment: Final = {
            "DATABRICKS_CLIENT_ID": _CLIENT_ID,
            "DATABRICKS_CLIENT_SECRET": _CLIENT_SECRET,
            "DATABRICKS_HOST": wire.url,
            "DBX_AGENT_HOST": wire.url,
            "DBX_AGENT_PAT": _PAT,
        }
        with owned_proxy_process(gateway, tmp_path, environment, config=_owned_config(tmp_path, wire, None)) as owned:
            candidate: Final = owned.gateway
            served: Final = tuple(_chat(candidate, "dbx-owned-agent") for _ in range(3))
            assert all(response.status_code == 200 for response in served), [r.text for r in served]
            received: Final = wire.drain()
            assert len(_calls_to(received, _TOKEN)) == 1, received
            _assert_token_request(_calls_to(received, _TOKEN)[0])
            for invocation in _calls_to(received, _INVOCATIONS):
                _assert_invocation(invocation, _MINTED)
            assert len(_calls_to(received, _INVOCATIONS)) == 3

            with candidate.scenario() as scenario:
                name: Final = _register(
                    candidate,
                    scenario.cleanups.callback,
                    wire.url,
                    model=_ENDPOINT,
                    api_base="os.environ/DBX_AGENT_HOST",
                    api_key="os.environ/DBX_AGENT_PAT",
                )
                response: Final = _chat(candidate, name)
                assert response.status_code == 200, response.text
                invocations: Final = _calls_to(wire.drain(), _INVOCATIONS)
                assert len(invocations) == 1, invocations
                _assert_invocation(invocations[0], _PAT)
