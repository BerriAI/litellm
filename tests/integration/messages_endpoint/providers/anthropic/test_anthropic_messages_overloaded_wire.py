import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Final, Literal
from urllib.parse import unquote

import anthropic
import httpx
import openai
import pytest
from integration._support.anthropic_sse import (
    LIFECYCLE,
    error_frame,
    event_type,
    event_types,
    message_id,
    message_json,
    message_stream,
    parse_sse,
    status_reply,
    stream_reply,
    user_prompt,
)
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    JsonValue,
    Scenario,
    eventually,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.openai_wire import openai_error
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from redis import Redis

_MODEL: Final = "claude-sonnet-4-5-20250929"
_API_KEY: Final = "synthetic-anthropic-key"
_BEDROCK_MODEL: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
_BEDROCK_503: Final = b'{"message": "Bedrock is unable to process your request."}'
_CONVERSE_MODEL: Final = f"bedrock/converse/{_BEDROCK_MODEL}"
_CONVERSE_STREAM_TARGET: Final = f"/model/{_BEDROCK_MODEL}/converse-stream"
_OPENAI_BACKEND: Final = "gpt-4o-mini"
_CHAT_ONLY_BACKEND: Final = "audit-chat-only"
_CODEX_BACKEND: Final = "gpt-5.3-codex"
_TEXT: Final = "Hello"
_KEEPALIVE_SECONDS: Final = 0.2
_LATE_ANSWER_SECONDS: Final = 1.0
_OVERLOAD_CLASS: Final = "503_PROVIDER_OVERLOADED"

Shape = Literal["stream", "non-stream"]


def _prompt() -> str:
    return "overloaded-" + uuid.uuid4().hex


def _always(reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request
        return reply

    return respond


def _late(reply: Reply, seconds: float) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request
        time.sleep(seconds)
        return reply

    return respond


def _anthropic_sdk(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60)


def _async_anthropic_sdk(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60
    )


def _openai_sdk(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60)


def _async_openai_sdk(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60)


def _messages_body(model: str, prompt: str, *, stream: bool) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
        "num_retries": 0,
    }


def _anthropic_error(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.headers["content-type"].startswith("application/json"), dict(response.headers)
    payload: Final = object_value(json.loads(response.content))
    assert payload["type"] == "error", response.text
    return object_value(payload["error"])


def _openai_error(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(object_value(json.loads(response.content))["error"])


def _assert_overloaded(response: httpx.Response, *, message: str) -> None:
    assert response.status_code == 529, response.text
    error: Final = _anthropic_error(response)
    assert error["type"] == "overloaded_error", response.text
    assert message in string_value(error["message"]), response.text
    assert "content_block_delta" not in response.text, response.text


def _assert_internal_server_error(response: httpx.Response) -> None:
    assert response.status_code == 500, response.text
    error: Final = _openai_error(response)
    assert error["code"] == "500", response.text
    assert "InternalServerError" in string_value(error["message"]), response.text


def _one_post(wire: Wire, target: str) -> None:
    assert [(request.method, unquote(request.target)) for request in wire.drain()] == [("POST", target)]


def _spend_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            "SELECT status, metadata->'error_information'->>'error_code' AS code, "
            "metadata->'error_information'->>'error_class' AS error_class, "
            "metadata->'error_information'->>'normalized_error' AS normalized_error "
            'FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
            (model,),
        ),
        lambda rows: len(rows) >= expected,
        seconds=70,
    )


def _assert_logged_failure(model: str, *, rows: int, error_class: str, code: str = "529") -> None:
    assert (
        _spend_rows(model, rows)
        == [{"status": "failure", "code": code, "error_class": error_class, "normalized_error": _OVERLOAD_CLASS}] * rows
    )


def test_messages_529_answers_overloaded_error_to_the_sync_sdk_and_logs_each_call_once(gateway: Gateway) -> None:
    with wire_server(_always(status_reply(529))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        sdk: Final = _anthropic_sdk(gateway)
        prompt: Final = _prompt()
        for _ in range(2):
            with pytest.raises(anthropic.APIStatusError) as caught:
                sdk.messages.create(
                    model=model,
                    max_tokens=16,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"num_retries": 0},
                )
            assert caught.value.status_code == 529, caught.value.response.text
            _assert_overloaded(caught.value.response, message="scripted 529")
        assert [unquote(request.target) for request in wire.drain()] == ["/v1/messages"] * 2
        _assert_logged_failure(model, rows=2, error_class="InternalServerError")


async def test_messages_529_answers_overloaded_error_to_the_async_sdk(gateway: Gateway) -> None:
    with wire_server(_always(status_reply(529))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        with pytest.raises(anthropic.APIStatusError) as caught:
            await _async_anthropic_sdk(gateway).messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": _prompt()}],
                extra_body={"num_retries": 0},
            )
        assert caught.value.status_code == 529, caught.value.response.text
        _assert_overloaded(caught.value.response, message="scripted 529")
        _one_post(wire, "/v1/messages")


@pytest.mark.parametrize(
    ("upstream_status", "error_class"), ((529, "InternalServerError"), (503, "ServiceUnavailableError"))
)
def test_messages_stream_rejected_before_its_first_byte_answers_529_json(
    gateway: Gateway, upstream_status: int, error_class: str
) -> None:
    with wire_server(_always(status_reply(upstream_status))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", _messages_body(model, _prompt(), stream=True))
        _assert_overloaded(response, message=f"scripted {upstream_status}")
        _one_post(wire, "/v1/messages")
        _assert_logged_failure(model, rows=1, error_class=error_class)


def test_messages_stream_529_raises_529_in_the_sync_sdk(gateway: Gateway) -> None:
    with wire_server(_always(status_reply(529))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        with pytest.raises(anthropic.APIStatusError) as caught:
            with _anthropic_sdk(gateway).messages.stream(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": _prompt()}],
                extra_body={"num_retries": 0},
            ) as stream:
                stream.get_final_message()
        assert caught.value.status_code == 529, caught.value.response.text
        _assert_overloaded(caught.value.response, message="scripted 529")
        _one_post(wire, "/v1/messages")


def _error_frames(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        object_value(json.loads(line.removeprefix("data:").strip()))
        for line in text.splitlines()
        if line.startswith("data:") and '"error"' in line
    )


def test_messages_529_after_the_keepalive_opened_the_response_ends_with_an_overloaded_error_event(
    gateway: Gateway,
) -> None:
    with (
        wire_server(_late(status_reply(529), _LATE_ANSWER_SECONDS)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, keepalive_seconds=_KEEPALIVE_SECONDS
        )
        response: Final = gateway.request("POST", "/v1/messages", _messages_body(model, _prompt(), stream=True))
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), dict(response.headers)
        assert response.text.startswith(":"), response.text
        (failure,) = (event for event in parse_sse(response.text) if event.data and event.event != "ping")
        assert failure.event == "error", response.text
        assert object_value(failure.data["error"])["type"] == "overloaded_error", response.text
        assert "[DONE]" not in response.text, response.text
        _one_post(wire, "/v1/messages")


def _late_bedrock_deployment(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"bedrock/{_BEDROCK_MODEL}",
        api_key=_BEDROCK_TOKEN,
        aws_region_name="us-east-1",
        api_base=wire.url,
        keepalive_seconds=_KEEPALIVE_SECONDS,
    )


def _one_bedrock_post(wire: Wire) -> None:
    (request,) = wire.drain()
    assert unquote(request.target).startswith(f"/model/{_BEDROCK_MODEL}/"), request


def test_messages_bedrock_503_after_the_keepalive_opened_the_response_ends_with_an_overloaded_error_event(
    gateway: Gateway,
) -> None:
    with (
        wire_server(_late(Reply(status=503, body=_BEDROCK_503), _LATE_ANSWER_SECONDS)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _late_bedrock_deployment(scenario, wire)
        response: Final = gateway.request("POST", "/v1/messages", _messages_body(model, _prompt(), stream=True))
        assert response.status_code == 200, response.text
        assert response.text.startswith(":"), response.text
        assert "[DONE]" not in response.text, response.text
        (failure,) = (event for event in parse_sse(response.text) if event.data and event.event != "ping")
        assert failure.event == "error", response.text
        error: Final = object_value(failure.data["error"])
        assert error["type"] == "overloaded_error", response.text
        assert "Bedrock is unable to process your request." in string_value(error["message"]), response.text
        _one_bedrock_post(wire)
        _assert_logged_failure(model, rows=1, error_class="ServiceUnavailableError")


def test_chat_completions_bedrock_503_after_the_keepalive_opened_the_response_keeps_the_openai_error_frame(
    gateway: Gateway,
) -> None:
    with (
        wire_server(_late(Reply(status=503, body=_BEDROCK_503), _LATE_ANSWER_SECONDS)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _late_bedrock_deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _prompt()}], "stream": True, "num_retries": 0},
        )
        assert response.status_code == 200, response.text
        assert response.text.startswith(":"), response.text
        assert "event: error" not in response.text, response.text
        (frame,) = _error_frames(response.text)
        assert object_value(frame["error"])["code"] == "503", response.text
        assert response.text.rstrip().endswith("data: [DONE]"), response.text
        _one_bedrock_post(wire)
        _assert_logged_failure(model, rows=1, error_class="ServiceUnavailableError", code="503")


def test_chat_completions_overloaded_frame_after_content_ends_with_a_500_error_frame(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    upstream: Final = (*message_stream(prompt, _MODEL, _TEXT)[:3], error_frame(529, "Overloaded"))
    with wire_server(_always(stream_reply(upstream))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True, "num_retries": 0},
        )
        assert response.status_code == 200, response.text
        assert _TEXT in response.text, response.text
        (frame,) = _error_frames(response.text)
        error: Final = object_value(frame["error"])
        assert error["code"] == "500", response.text
        _one_post(wire, "/v1/messages")


def _overloaded_before_content(prompt: str) -> Reply:
    return stream_reply((*message_stream(prompt, _MODEL, _TEXT)[:1], error_frame(529, "Overloaded")))


def test_messages_overloaded_frame_before_content_without_a_retry_budget_is_relayed_as_an_overloaded_event(
    gateway: Gateway,
) -> None:
    prompt: Final = _prompt()
    with wire_server(_always(_overloaded_before_content(prompt))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", _messages_body(model, prompt, stream=True))
        assert response.status_code == 200, response.text
        events: Final = parse_sse(response.text)
        assert event_types(events) == ("message_start", "error"), response.text
        assert message_id(events) == prompt, response.text
        assert events[1].data == {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        _one_post(wire, "/v1/messages")
        assert _spend_rows(model, 1) == [
            {"status": "success", "code": None, "error_class": None, "normalized_error": None}
        ]


def test_chat_completions_overloaded_frame_before_content_answers_500(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    with wire_server(_always(_overloaded_before_content(prompt))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True, "num_retries": 0},
        )
        _assert_internal_server_error(response)
        _one_post(wire, "/v1/messages")


def test_chat_completions_529_answers_500_to_the_sync_sdk(gateway: Gateway) -> None:
    with wire_server(_always(status_reply(529))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        with pytest.raises(openai.APIStatusError) as caught:
            _openai_sdk(gateway).chat.completions.create(
                model=model, messages=[{"role": "user", "content": _prompt()}], extra_body={"num_retries": 0}
            )
        assert caught.value.status_code == 500, caught.value.response.text
        _assert_internal_server_error(caught.value.response)
        _one_post(wire, "/v1/messages")
        _assert_logged_failure(model, rows=1, error_class="InternalServerError", code="500")


async def test_chat_completions_stream_529_answers_500_to_the_async_sdk(gateway: Gateway) -> None:
    with wire_server(_always(status_reply(529))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        with pytest.raises(openai.APIStatusError) as caught:
            await _async_openai_sdk(gateway).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt()}],
                stream=True,
                extra_body={"num_retries": 0},
            )
        assert caught.value.status_code == 500, caught.value.response.text
        _assert_internal_server_error(caught.value.response)
        _one_post(wire, "/v1/messages")


@pytest.mark.parametrize("shape", ("non-stream", "stream"))
def test_responses_529_answers_500(gateway: Gateway, shape: Shape) -> None:
    with wire_server(_always(status_reply(529))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": _prompt(), "stream": shape == "stream", "num_retries": 0}
        )
        _assert_internal_server_error(response)
        _one_post(wire, "/v1/messages")


@pytest.mark.parametrize("shape", ("non-stream", "stream"))
def test_messages_bedrock_503_answers_529_overloaded_error(gateway: Gateway, shape: Shape) -> None:
    with wire_server(_always(Reply(status=503, body=_BEDROCK_503))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{_BEDROCK_MODEL}", api_key=_BEDROCK_TOKEN, aws_region_name="us-east-1", api_base=wire.url
        )
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, _prompt(), stream=shape == "stream")
        )
        _assert_overloaded(response, message="Bedrock is unable to process your request.")
        (request,) = wire.drain()
        assert unquote(request.target).startswith(f"/model/{_BEDROCK_MODEL}/"), request
        _assert_logged_failure(model, rows=1, error_class="ServiceUnavailableError")


def _converse_event(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return aws_event_stream_frame(
        {":event-type": event, ":content-type": "application/json", ":message-type": "event"},
        json.dumps(payload).encode(),
    )


def _converse_503_after_text() -> Reply:
    unavailable: Final = aws_event_stream_frame(
        {
            ":exception-type": "serviceUnavailableException",
            ":content-type": "application/json",
            ":message-type": "exception",
        },
        _BEDROCK_503,
    )
    frames: Final = (
        _converse_event("messageStart", {"role": "assistant"}),
        _converse_event("contentBlockDelta", {"delta": {"text": _TEXT}, "contentBlockIndex": 0}),
        unavailable,
    )
    return Reply(body=b"".join(frames), content_type="application/vnd.amazon.eventstream")


def _converse_deployment(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model=_CONVERSE_MODEL, api_key=_BEDROCK_TOKEN, aws_region_name="us-east-1", api_base=wire.url)


def test_messages_bedrock_stream_503_after_text_ends_with_an_overloaded_error_event(gateway: Gateway) -> None:
    with wire_server(_always(_converse_503_after_text())) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        response: Final = gateway.request("POST", "/v1/messages", _messages_body(model, _prompt(), stream=True))
        assert response.status_code == 200, response.text
        events: Final = parse_sse(response.text)
        types: Final = event_types(events)
        assert "message_stop" not in types, response.text
        assert types.index("content_block_delta") < types.index("error"), response.text
        assert _TEXT in response.text, response.text
        (failure,) = (event for event in events if event_type(event) == "error")
        error: Final = object_value(failure.data["error"])
        assert error["type"] == "overloaded_error", response.text
        assert "Bedrock is unable to process your request." in string_value(error["message"]), response.text
        _one_post(wire, _CONVERSE_STREAM_TARGET)
        _assert_logged_failure(model, rows=1, error_class="ServiceUnavailableError")


def test_chat_completions_bedrock_stream_503_after_text_ends_with_a_503_error_frame(gateway: Gateway) -> None:
    with wire_server(_always(_converse_503_after_text())) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _prompt()}], "stream": True, "num_retries": 0},
        )
        assert response.status_code == 200, response.text
        assert _TEXT in response.text, response.text
        (frame,) = _error_frames(response.text)
        error: Final = object_value(frame["error"])
        assert error["code"] == "503", response.text
        _one_post(wire, _CONVERSE_STREAM_TARGET)
        _assert_logged_failure(model, rows=1, error_class="ServiceUnavailableError", code="503")


@pytest.mark.parametrize("shape", ("non-stream", "stream"))
def test_chat_bridge_503_answers_529_overloaded_error_on_messages(gateway: Gateway, shape: Shape) -> None:
    with wire_server(_always(openai_error(503))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"hosted_vllm/{_CHAT_ONLY_BACKEND}", api_base=wire.url + "/v1", api_key="synthetic-vllm-key"
        )
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, _prompt(), stream=shape == "stream")
        )
        _assert_overloaded(response, message="scripted 503")
        _one_post(wire, "/v1/chat/completions")


@pytest.mark.parametrize("shape", ("non-stream", "stream"))
def test_responses_bridge_503_answers_529_overloaded_error_on_messages(gateway: Gateway, shape: Shape) -> None:
    with wire_server(_always(openai_error(503))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{_CODEX_BACKEND}", api_base=wire.url + "/v1", api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, _prompt(), stream=shape == "stream")
        )
        _assert_overloaded(response, message="scripted 503")
        _one_post(wire, "/v1/responses")


_SETTLED_ANSWERS: Final = 4


def _messages_to(gateway: Gateway, model: str) -> tuple[tuple[str, httpx.Response], ...]:
    prompts: Final = tuple(_prompt() for _ in range(_SETTLED_ANSWERS))
    return tuple(
        (prompt, gateway.request("POST", "/v1/messages", _messages_body(model, prompt, stream=False)))
        for prompt in prompts
    )


def _failure_rows_of(models: tuple[str, str], expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            "SELECT metadata->'error_information'->>'error_code' AS code, "
            "metadata->'error_information'->>'error_class' AS error_class, "
            "metadata->'error_information'->>'normalized_error' AS normalized_error "
            'FROM "LiteLLM_SpendLogs" WHERE status = %s AND model_group IN (%s, %s)',
            ("failure", *models),
        ),
        lambda rows: len(rows) >= expected,
        seconds=70,
    )


@contextmanager
def _paused_deployment(gateway: Gateway) -> Generator[tuple[Wire, str, str]]:
    unpaused: Final = Reply(body=message_json("msg_unpaused", _MODEL, _TEXT))
    with wire_server(_always(unpaused)) as wire, gateway.scenario() as scenario:
        identity: Final = f"audit-paused-{uuid.uuid4().hex}"
        name: Final = scenario.model(
            model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, model_info={"id": identity}
        )
        paused: Final = gateway.request("PATCH", f"/model/{identity}/update", {"blocked": True})
        assert paused.status_code == 200, paused.text
        yield wire, identity, name


def _assert_refused_without_an_upstream_call(
    gateway: Gateway, wire: Wire, target: str, *, status: int, error_type: str, message: str
) -> None:
    answers: Final = eventually(
        lambda: _messages_to(gateway, target),
        lambda settled: all(response.status_code != 200 for _, response in settled),
        seconds=90,
    )
    texts: Final = [response.text for _, response in answers]
    assert [response.status_code for _, response in answers] == [status] * _SETTLED_ANSWERS, texts
    errors: Final = [_anthropic_error(response) for _, response in answers]
    assert [error["type"] for error in errors] == [error_type] * _SETTLED_ANSWERS, texts
    assert all(message in string_value(error["message"]) for error in errors), texts
    reached: Final = {user_prompt(JSON_OBJECT.validate_json(request.body)) for request in wire.drain()}
    assert reached.isdisjoint(prompt for prompt, _ in answers), reached


def test_messages_to_a_paused_deployment_named_by_its_id_answer_529_overloaded_error(gateway: Gateway) -> None:
    with _paused_deployment(gateway) as (wire, identity, name):
        _assert_refused_without_an_upstream_call(
            gateway, wire, identity, status=529, error_type="overloaded_error", message="is currently paused"
        )
        rows: Final = _failure_rows_of((identity, name), _SETTLED_ANSWERS)
        assert {(row["code"], row["error_class"], row["normalized_error"]) for row in rows} == {
            ("529", "ServiceUnavailableError", _OVERLOAD_CLASS)
        }, rows


def test_messages_to_a_paused_model_group_named_by_its_name_answer_403_permission_error(gateway: Gateway) -> None:
    with _paused_deployment(gateway) as (wire, _, name):
        _assert_refused_without_an_upstream_call(
            gateway, wire, name, status=403, error_type="permission_error", message="Model is blocked"
        )


_MALFORMED_529_BODIES: Final = {
    "empty": b"",
    "text-5k": b"x" * 5000,
    "html": b"<html><body><h1>529</h1></body></html>",
    "api_error-type": b'{"type": "error", "error": {"type": "api_error", "message": "capacity"}}',
}


def _overloaded_for_sad_prompts(body: bytes) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        prompt: Final = user_prompt(object_value(json.loads(request.body)))
        if prompt.startswith("sad:"):
            return Reply(status=529, body=body, content_type="text/plain" if body[:1] != b"{" else "application/json")
        return Reply(body=message_json(f"msg_{prompt}", _MODEL, _TEXT))

    return respond


@pytest.mark.parametrize("body", _MALFORMED_529_BODIES.keys())
def test_messages_529_with_a_malformed_body_still_answers_overloaded_error_and_unrelated_calls_keep_working(
    gateway: Gateway, body: str
) -> None:
    with wire_server(_overloaded_for_sad_prompts(_MALFORMED_529_BODIES[body])) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        sad: Final = gateway.request("POST", "/v1/messages", _messages_body(model, "sad:" + _prompt(), stream=False))
        assert sad.status_code == 529, sad.text
        assert _anthropic_error(sad)["type"] == "overloaded_error", sad.text
        prompt: Final = _prompt()
        healthy: Final = gateway.request("POST", "/v1/messages", _messages_body(model, prompt, stream=False))
        assert healthy.status_code == 200, healthy.text
        assert object_value(json.loads(healthy.content))["id"] == f"msg_{prompt}", healthy.text
        assert [unquote(request.target) for request in wire.drain()] == ["/v1/messages"] * 2
        assert sorted(_spend_rows(model, 2), key=lambda row: str(row["status"])) == [
            {
                "status": "failure",
                "code": "529",
                "error_class": "InternalServerError",
                "normalized_error": _OVERLOAD_CLASS,
            },
            {"status": "success", "code": None, "error_class": None, "normalized_error": None},
        ]


@pytest.mark.parametrize(("upstream_status", "error_type"), ((500, "api_error"), (429, "rate_limit_error")))
def test_messages_other_upstream_statuses_keep_their_status_and_type(
    gateway: Gateway, upstream_status: int, error_type: str
) -> None:
    with wire_server(_always(status_reply(upstream_status))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", _messages_body(model, _prompt(), stream=True))
        assert response.status_code == upstream_status, response.text
        assert _anthropic_error(response)["type"] == error_type, response.text
        _one_post(wire, "/v1/messages")


def _anthropic_rejection(status: int, error_type: str, message: str) -> Reply:
    return Reply(
        status=status, body=json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()
    )


_REJECTIONS_SAYING_OVERLOADED: Final = {
    "400-tool-name": _anthropic_rejection(
        400, "invalid_request_error", "tools.0.name: 'Overloaded' is not a valid tool name"
    ),
    "404-model": _anthropic_rejection(404, "not_found_error", "model: claude-Overloaded"),
}


@pytest.mark.parametrize("shape", ("non-stream", "stream"))
@pytest.mark.parametrize("rejection", _REJECTIONS_SAYING_OVERLOADED.keys())
def test_messages_4xx_whose_text_says_overloaded_is_not_answered_as_an_overload(
    gateway: Gateway, rejection: str, shape: Shape
) -> None:
    with wire_server(_always(_REJECTIONS_SAYING_OVERLOADED[rejection])) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, _prompt(), stream=shape == "stream")
        )
        assert response.status_code == 500, response.text
        error: Final = _anthropic_error(response)
        assert error["type"] == "api_error", response.text
        assert "Overloaded" in string_value(error["message"]), response.text
        _one_post(wire, "/v1/messages")
        assert _spend_rows(model, 1) == [
            {
                "status": "failure",
                "code": "500",
                "error_class": "InternalServerError",
                "normalized_error": _OVERLOAD_CLASS,
            }
        ]


def _string_values(cache: Redis) -> tuple[bytes, ...]:
    keys: Final = tuple(key for key in cache.scan_iter(count=1000) if cache.type(key) == b"string")
    return tuple(value for value in cache.mget(keys) if value is not None) if keys else ()


def _cached_somewhere(served_id: str) -> bool:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        return any(served_id.encode() in value for value in _string_values(cache))


def _healthy_until(overloaded: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if overloaded.is_set():
            return status_reply(529)
        prompt: Final = user_prompt(object_value(json.loads(request.body)))
        return stream_reply(message_stream(f"msg_{prompt}", _MODEL, _TEXT))

    return respond


def test_messages_cache_hit_is_served_while_the_upstream_is_overloaded(gateway: Gateway) -> None:
    overloaded: Final = threading.Event()
    with wire_server(_healthy_until(overloaded)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        prompt: Final = _prompt()
        body: Final = _messages_body(model, prompt, stream=True)
        first: Final = gateway.request("POST", "/v1/messages", body)
        assert first.status_code == 200, first.text
        assert event_types(parse_sse(first.text)) == LIFECYCLE, first.text
        assert message_id(parse_sse(first.text)) == f"msg_{prompt}", first.text
        eventually(lambda: _cached_somewhere(f"msg_{prompt}"), bool, seconds=30)
        overloaded.set()
        replay: Final = gateway.request("POST", "/v1/messages", body)
        assert replay.status_code == 200, replay.text
        assert message_id(parse_sse(replay.text)) == f"msg_{prompt}", replay.text
        fresh: Final = gateway.request("POST", "/v1/messages", _messages_body(model, _prompt(), stream=True))
        _assert_overloaded(fresh, message="scripted 529")
        assert [unquote(request.target) for request in wire.drain()] == ["/v1/messages"] * 2
