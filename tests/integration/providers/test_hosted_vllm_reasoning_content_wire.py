import json
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Final

import openai
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "qwen3-reasoning"
_FALLBACK_BACKEND: Final = "qwen3-reasoning-fallback"
_API_KEY: Final = "synthetic-hosted-vllm-key"
_REASONING: Final = "I compared the two invoices and the totals differ by 42."
_ANSWER_REASONING: Final = "The user wants the difference, which is 42."
_TOOL_CALL_ID: Final = "call_reasoning_wire_1"
_NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])


def _completion(identity: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "reasoning_content": _ANSWER_REASONING},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
        }
    ).encode()


def _streamed_completion(identity: str, content: str) -> Reply:
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": _BACKEND}
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": _ANSWER_REASONING}}]},
        {**chunk, "choices": [{"index": 0, "delta": {"content": content}}]},
        {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
        },
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), b"data: [DONE]\n\n"),
    )


def _replayed_conversation(reasoning: JsonValue, marker: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": f"Compare these invoices {marker}."},
        {"role": "assistant", "content": "Checking the totals.", "reasoning_content": reasoning},
        {"role": "user", "content": "What is the difference?"},
    ]


def _sent_messages(request: Request) -> list[dict[str, JsonValue]]:
    return _MESSAGES.validate_python(_JSON_OBJECT.validate_json(request.body)["messages"])


_DISCOVERY_PROBE: Final = ("GET", "/v1/models")


def _is_discovery_probe(request: Request) -> bool:
    return (request.method, request.target) == _DISCOVERY_PROBE


@contextmanager
def _vllm_server(respond: Callable[[Request], Reply]) -> Iterator[Wire]:
    with wire_server(
        lambda request: Reply(body=b'{"object":"list","data":[]}') if _is_discovery_probe(request) else respond(request)
    ) as wire:
        yield wire


def _provider_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if not _is_discovery_probe(request))


def _only_request(wire: Wire) -> Request:
    received: Final = _provider_calls(wire)
    assert [(request.method, request.target) for request in received] == [("POST", "/v1/chat/completions")]
    return received[0]


def _spend_row(identity: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_group, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (identity,),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _model_spend_statuses(model: str) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return [row["status"] for row in rows]


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _post_chat(gateway: Gateway, model: str, messages: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": list(messages), "cache": _NO_CACHE}
    )
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def test_hosted_vllm_assistant_reasoning_content_reaches_the_wire(gateway: Gateway) -> None:
    identity: Final = f"hosted-vllm-reasoning-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [
            {"role": "user", "content": "Compare these invoices."},
            {
                "role": "assistant",
                "content": "Checking the totals.",
                "reasoning_content": _REASONING,
                "tool_calls": [
                    {
                        "id": _TOOL_CALL_ID,
                        "type": "function",
                        "function": {"name": "lookup_invoice", "arguments": json.dumps({"id": "inv-7"})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "content": "invoice total is 1042"},
        ], body["messages"]
        return Reply(body=_completion(identity, "The totals differ by 42."))

    with _vllm_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Compare these invoices."},
                    {
                        "role": "assistant",
                        "content": "Checking the totals.",
                        "reasoning_content": _REASONING,
                        "tool_calls": [
                            {
                                "id": _TOOL_CALL_ID,
                                "type": "function",
                                "function": {"name": "lookup_invoice", "arguments": json.dumps({"id": "inv-7"})},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "content": "invoice total is 1042"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        _only_request(wire)


def test_openai_sdk_replayed_reasoning_reaches_hosted_vllm_and_is_billed_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"chatcmpl-sdk-{marker}"
    with _vllm_server(lambda _: Reply(body=_completion(identity, "They differ by 42."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            completion: Final = _openai_client(gateway).chat.completions.create(
                model=model,
                messages=_replayed_conversation(_REASONING, marker),  # pyright: ignore[reportArgumentType]  # reasoning_content is a provider extension the SDK types omit
            )
            assert completion.id == identity
            assert completion.choices[0].message.content == "They differ by 42."
            assert (completion.choices[0].message.model_extra or {})["reasoning_content"] == _ANSWER_REASONING
            assert _sent_messages(_only_request(wire)) == _replayed_conversation(_REASONING, marker)
            assert _spend_row(identity) == {
                "model_group": model,
                "status": "success",
                "prompt_tokens": 30,
                "completion_tokens": 5,
            }


async def test_async_openai_sdk_stream_forwards_replayed_reasoning_to_hosted_vllm(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"chatcmpl-stream-{marker}"
    with _vllm_server(lambda _: _streamed_completion(identity, "They differ by 42.")) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            stream: Final = await _async_openai_client(gateway).chat.completions.create(
                model=model,
                messages=_replayed_conversation(_REASONING, marker),  # pyright: ignore[reportArgumentType]  # reasoning_content is a provider extension the SDK types omit
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks: Final = [chunk async for chunk in stream]
            assert {chunk.id for chunk in chunks} == {identity}
            assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == (
                "They differ by 42."
            )
            sent: Final = _only_request(wire)
            assert _JSON_OBJECT.validate_json(sent.body)["stream"] is True
            assert _sent_messages(sent) == _replayed_conversation(_REASONING, marker)
            assert _spend_row(identity)["status"] == "success"


def test_each_replayed_turn_keeps_its_own_reasoning_in_order(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    conversation: Final[list[dict[str, JsonValue]]] = [
        {"role": "user", "content": f"Plan the migration {marker}."},
        {"role": "assistant", "content": "Step one.", "reasoning_content": f"first thought {marker}"},
        {"role": "user", "content": "Continue."},
        {"role": "assistant", "content": "Step two.", "reasoning_content": f"second thought {marker}"},
        {"role": "user", "content": "Summarize."},
    ]
    with _vllm_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}", "Done."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            assert _post_chat(gateway, model, conversation)["id"] == f"chatcmpl-{marker}"
            assert _sent_messages(_only_request(wire)) == conversation


@pytest.mark.parametrize(
    ("reasoning", "forwarded"),
    [
        pytest.param("", "", id="empty-string-forwarded"),
        pytest.param("x" * 5120, "x" * 5120, id="5kb-string-forwarded-intact"),
        pytest.param(None, None, id="null-dropped"),
        pytest.param(42, None, id="int-dropped"),
        pytest.param(["step one", "step two"], None, id="list-dropped"),
        pytest.param({"text": "step one"}, None, id="object-dropped"),
    ],
)
def test_only_string_reasoning_content_is_forwarded_to_hosted_vllm(
    gateway: Gateway, reasoning: JsonValue, forwarded: str | None
) -> None:
    marker: Final = uuid.uuid4().hex
    with _vllm_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}", "Done."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            assert _post_chat(gateway, model, _replayed_conversation(reasoning, marker))["id"] == f"chatcmpl-{marker}"
            sent_assistant: Final = _sent_messages(_only_request(wire))[1]
            expected_assistant: Final[dict[str, JsonValue]] = {"role": "assistant", "content": "Checking the totals."}
            assert sent_assistant == (
                expected_assistant if forwarded is None else {**expected_assistant, "reasoning_content": forwarded}
            )


def test_assistant_turn_without_reasoning_gets_no_reasoning_key(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    conversation: Final[list[dict[str, JsonValue]]] = [
        {"role": "user", "content": f"Hello {marker}"},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "Again"},
    ]
    with _vllm_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}", "Hello again."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            _post_chat(gateway, model, conversation)
            assert _sent_messages(_only_request(wire)) == conversation


def test_same_reasoning_on_two_turns_is_forwarded_on_both(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    reasoning: Final = f"repeated thought {marker}"
    conversation: Final[list[dict[str, JsonValue]]] = [
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "First.", "reasoning_content": reasoning},
        {"role": "user", "content": "Two"},
        {"role": "assistant", "content": "Second.", "reasoning_content": reasoning},
        {"role": "user", "content": "Three"},
    ]
    with _vllm_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}", "Third."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            _post_chat(gateway, model, conversation)
            assert _sent_messages(_only_request(wire)) == conversation


def test_thinking_blocks_are_stripped_while_reasoning_content_is_kept(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with _vllm_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}", "Done."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            _post_chat(
                gateway,
                model,
                [
                    {"role": "user", "content": f"Hello {marker}"},
                    {
                        "role": "assistant",
                        "content": "Hi.",
                        "reasoning_content": _REASONING,
                        "thinking_blocks": [{"type": "thinking", "thinking": _REASONING, "signature": "sig"}],
                    },
                    {"role": "user", "content": "Again"},
                ],
            )
            assert _sent_messages(_only_request(wire))[1] == {
                "role": "assistant",
                "content": "Hi.",
                "reasoning_content": _REASONING,
            }


def test_list_content_is_flattened_while_reasoning_content_is_kept(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with _vllm_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}", "Done."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            _post_chat(
                gateway,
                model,
                [
                    {"role": "user", "content": f"Hello {marker}"},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Part one."}, {"type": "text", "text": "Part two."}],
                        "reasoning_content": _REASONING,
                    },
                    {"role": "user", "content": "Again"},
                ],
            )
            assert _sent_messages(_only_request(wire))[1] == {
                "role": "assistant",
                "content": "Part one.\nPart two.",
                "reasoning_content": _REASONING,
            }


def test_unauthenticated_replay_is_rejected_before_hosted_vllm(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with _vllm_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}", "Done."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": _replayed_conversation(_REASONING, marker)},
                key=f"sk-not-a-key-{marker}",
            )
            assert response.status_code == 401, response.text
            assert _provider_calls(wire) == ()


def test_hosted_vllm_auth_error_reaches_the_caller_after_one_attempt_with_reasoning(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    error_message: Final = f"invalid api key for deployment {marker}"
    reply: Final = Reply(
        status=401,
        body=json.dumps({"error": {"message": error_message, "type": "authentication_error"}}).encode(),
    )
    with _vllm_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _replayed_conversation(_REASONING, marker), "cache": _NO_CACHE},
        )
        assert response.status_code == 401, response.text
        assert error_message in response.text, response.text
        assert _sent_messages(_only_request(wire)) == _replayed_conversation(_REASONING, marker)


def test_fallback_attempt_replays_reasoning_to_the_second_deployment(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if _JSON_OBJECT.validate_json(request.body)["model"] == _BACKEND:
            return Reply(status=500, body=b'{"error": {"message": "primary deployment is down"}}')
        return Reply(body=_completion(f"chatcmpl-fallback-{marker}", "Recovered."))

    with _vllm_server(respond) as wire, gateway.scenario() as scenario:
        primary: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        fallback: Final = scenario.model(
            model=f"hosted_vllm/{_FALLBACK_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": primary,
                "messages": _replayed_conversation(_REASONING, marker),
                "fallbacks": [fallback],
                "num_retries": 0,
                "cache": _NO_CACHE,
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"] == f"chatcmpl-fallback-{marker}"
        attempts: Final = _provider_calls(wire)
        assert [_JSON_OBJECT.validate_json(attempt.body)["model"] for attempt in attempts] == [
            _BACKEND,
            _FALLBACK_BACKEND,
        ]
        assert [_sent_messages(attempt) for attempt in attempts] == [
            _replayed_conversation(_REASONING, marker),
            _replayed_conversation(_REASONING, marker),
        ]


def test_identical_uncached_replays_are_each_forwarded_and_billed_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identities: Final = iter((f"chatcmpl-first-{marker}", f"chatcmpl-second-{marker}"))
    with _vllm_server(lambda _: Reply(body=_completion(next(identities), "Done."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            first: Final = _post_chat(gateway, model, _replayed_conversation(_REASONING, marker))
            second: Final = _post_chat(gateway, model, _replayed_conversation(_REASONING, marker))
            assert (first["id"], second["id"]) == (f"chatcmpl-first-{marker}", f"chatcmpl-second-{marker}")
            assert [_sent_messages(request) for request in _provider_calls(wire)] == [
                _replayed_conversation(_REASONING, marker),
                _replayed_conversation(_REASONING, marker),
            ]
            assert _spend_row(f"chatcmpl-first-{marker}")["status"] == "success"
            assert _spend_row(f"chatcmpl-second-{marker}")["status"] == "success"


def test_cached_replay_hits_only_for_the_same_reasoning(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identities: Final = iter((f"chatcmpl-cached-{marker}", f"chatcmpl-other-{marker}"))
    with _vllm_server(lambda _: Reply(body=_completion(next(identities), "Done."))) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)

            def ask(reasoning: str) -> dict[str, JsonValue]:
                response: Final = gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": _replayed_conversation(reasoning, marker)},
                )
                assert response.status_code == 200, response.text
                return _JSON_OBJECT.validate_json(response.content)

            assert ask(_REASONING)["id"] == f"chatcmpl-cached-{marker}"
            assert ask(_REASONING)["id"] == f"chatcmpl-cached-{marker}"
            assert ask(f"a different thought {marker}")["id"] == f"chatcmpl-other-{marker}"
            assert [_sent_messages(request)[1].get("reasoning_content") for request in _provider_calls(wire)] == [
                _REASONING,
                f"a different thought {marker}",
            ]


def _responses_input(marker: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": f"Compare these invoices {marker}."},
        {
            "id": f"rs_{marker}",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": _REASONING}],
        },
        {
            "id": f"msg_prior_{marker}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Checking the totals.", "annotations": []}],
        },
        {"role": "user", "content": "What is the difference?"},
    ]


def _responses_reply(identity: str, stream: bool) -> Reply:
    response: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": _BACKEND,
        "output": [
            {
                "id": "msg_" + identity,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "They differ by 42.", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": "They differ by 42.",
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _only_responses_body(wire: Wire) -> dict[str, JsonValue]:
    received: Final = _provider_calls(wire)
    assert [(request.method, request.target) for request in received] == [("POST", "/v1/responses")]
    return _JSON_OBJECT.validate_json(received[0].body)


def test_openai_sdk_responses_replay_reaches_hosted_vllm_with_its_reasoning_item(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with _vllm_server(lambda _: _responses_reply(f"resp_upstream_{marker}", stream=False)) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            response: Final = _openai_client(gateway).responses.create(
                model=model,
                input=_responses_input(marker),  # pyright: ignore[reportArgumentType]  # plain JSON input items
            )
            assert response.output_text == "They differ by 42."
            assert _only_responses_body(wire)["input"] == _responses_input(marker)
            assert _spend_row(response.id) == {
                "model_group": model,
                "status": "success",
                "prompt_tokens": 30,
                "completion_tokens": 5,
            }


async def test_async_openai_sdk_responses_stream_reaches_hosted_vllm_with_its_reasoning_item(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with _vllm_server(lambda _: _responses_reply(f"resp_upstream_{marker}", stream=True)) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
            stream: Final = await _async_openai_client(gateway).responses.create(
                model=model,
                input=_responses_input(marker),  # pyright: ignore[reportArgumentType]  # plain JSON input items
                stream=True,
            )
            events: Final = [event async for event in stream]
            assert [event.type for event in events] == [
                "response.created",
                "response.output_text.delta",
                "response.completed",
            ]
            completed: Final = events[-1]
            assert completed.type == "response.completed"
            body: Final = _only_responses_body(wire)
            assert body["stream"] is True
            assert body["input"] == _responses_input(marker)
            assert completed.response.output_text == "They differ by 42."
            assert _model_spend_statuses(model) == ["success"]
