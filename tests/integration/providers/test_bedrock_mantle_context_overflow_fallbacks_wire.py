import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120))

_BACKEND: Final = "openai.gpt-5.6-luna"
_MANTLE_MODEL: Final = f"bedrock_mantle/{_BACKEND}"
_MANTLE_KEY: Final = "synthetic-mantle-bearer"
_OPENAI_KEY: Final = "synthetic-openai-key"
_RESPONSES_PATH: Final = "/openai/v1/responses"
_GENERIC: Final = "prompt is too long: your prompt exceeds the model's context window"
_UPSTREAM_MESSAGE: Final = (
    "Your input exceeds the context window of this model. Please adjust your input and try again."
)
_OPENAI_OVERFLOW_MESSAGE: Final = (
    "This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens."
)
_OPENAI_OVERFLOW_MARK: Final = "maximum context length is 128000 tokens"
_INVALID_INPUT_MESSAGE: Final = "Invalid 'input': expected a string or array"
_INVALID_PROMPT_MESSAGE: Final = "Invalid prompt: your prompt was flagged as potentially violating our usage policy."
_FALLBACK_TEXT: Final = "fallback answered"
_GPT_OSS_OVERFLOW_TEXT: Final = "prompt is too long: 165068 tokens > 131072 maximum"
_FAILED_INPUT_TOKENS: Final = 165068
_SERVER_ERROR: Final[dict[str, JsonValue]] = {"code": "server_error", "message": "The model had an internal error"}
_NO_ERROR_MESSAGE: Final = "The provider reported a failed response"
_FAILED_REQUEST_ID: Final = "req-mantle-failed-body"
_GPT_OSS_MODEL: Final = "bedrock_mantle/openai.gpt-oss-20b"
_GPT_OSS_RESPONSES_PATH: Final = "/v1/responses"
_GPT_OSS_OVERFLOW_ERROR: Final[dict[str, JsonValue]] = {
    "code": "invalid_prompt",
    "message": (
        'ErrorEvent { error: APIError { type: "invalid_request_error", code: Some(400), message: "The engine prompt '
        'length 165068 exceeds the max_model_len 131072. Please reduce prompt.", param: Some("input") } }'
    ),
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OVERFLOW_ENVELOPE: Final[dict[str, JsonValue]] = {
    "error": {
        "code": "context_length_exceeded",
        "message": _UPSTREAM_MESSAGE,
        "param": "input",
        "type": "invalid_request_error",
    }
}
_OVERFLOW_BODY: Final = json.dumps(_OVERFLOW_ENVELOPE).encode()
_BAD_INPUT_BODY: Final = json.dumps(
    {"error": {"code": None, "message": _INVALID_INPUT_MESSAGE, "param": "input", "type": "invalid_request_error"}}
).encode()
_OPENAI_OVERFLOW_ERROR: Final[dict[str, JsonValue]] = {
    "message": _OPENAI_OVERFLOW_MESSAGE,
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded",
}
_OPENAI_OVERFLOW_BODY: Final = json.dumps({"error": _OPENAI_OVERFLOW_ERROR}).encode()


def _sse(events: tuple[dict[str, JsonValue], ...]) -> tuple[bytes, ...]:
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _response_object(identity: str, status: str, model: str) -> dict[str, JsonValue]:
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1789788253,
        "status": status,
        "model": model,
        "output": [],
    }


def _failed_frames(identity: str, error: JsonValue, model: str = _BACKEND) -> tuple[bytes, ...]:
    return _sse(
        (
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": _response_object(identity, "in_progress", model),
            },
            {
                "type": "response.failed",
                "sequence_number": 1,
                "response": {**_response_object(identity, "failed", model), "error": error},
            },
        )
    )


_PASSTHROUGH_FAILED_FRAMES: Final = _failed_frames("passthrough", _OVERFLOW_ENVELOPE["error"])


def _streaming(request: Request) -> bool:
    return _JSON_OBJECT.validate_json(request.body).get("stream") is True


def _overflow_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == _RESPONSES_PATH, request.target
    if not _streaming(request):
        return Reply(status=400, body=_OVERFLOW_BODY)
    return Reply(content_type="text/event-stream", chunks=_failed_frames(uuid4().hex, _OVERFLOW_ENVELOPE["error"]))


def _body_peer(status: str, fields: Mapping[str, JsonValue], path: str = _RESPONSES_PATH) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        assert request.method == "POST" and request.target == path, request.target
        assert not _streaming(request), request.body
        body: Final = {
            **_response_object(uuid4().hex, status, _BACKEND),
            "usage": {"input_tokens": _FAILED_INPUT_TOKENS, "output_tokens": 0, "total_tokens": _FAILED_INPUT_TOKENS},
            **fields,
        }
        return Reply(body=json.dumps(body).encode(), headers={"x-request-id": _FAILED_REQUEST_ID})

    return reply


def _passthrough_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == _RESPONSES_PATH, request.target
    if not _streaming(request):
        return Reply(status=400, body=_OVERFLOW_BODY)
    return Reply(content_type="text/event-stream", chunks=_PASSTHROUGH_FAILED_FRAMES)


def _bad_input_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == _RESPONSES_PATH, request.target
    if not _streaming(request):
        return Reply(status=400, body=_BAD_INPUT_BODY)
    error: Final[dict[str, JsonValue]] = {"code": "invalid_prompt", "message": _INVALID_PROMPT_MESSAGE}
    return Reply(content_type="text/event-stream", chunks=_failed_frames(uuid4().hex, error))


_EMPTY_MODEL_LIST: Final = Reply(body=b'{"object":"list","data":[]}')


def _openai_overflow_peer(request: Request) -> Reply:
    if request.method == "GET":
        return _EMPTY_MODEL_LIST
    if request.target == "/v1/responses" and _streaming(request):
        return Reply(
            content_type="text/event-stream",
            chunks=_failed_frames(uuid4().hex, _OPENAI_OVERFLOW_ERROR, "gpt-4o-mini"),
        )
    assert request.target in ("/v1/chat/completions", "/v1/responses"), request.target
    return Reply(status=400, body=_OPENAI_OVERFLOW_BODY)


def _chat_reply(identity: str, stream: bool) -> Reply:
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{identity}",
                    "object": "chat.completion",
                    "created": 1789788253,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": _FALLBACK_TEXT},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
                }
            ).encode()
        )
    chunk: Final[dict[str, JsonValue]] = {
        "id": f"chatcmpl-{identity}",
        "object": "chat.completion.chunk",
        "created": 1789788253,
        "model": "gpt-4o-mini",
    }
    frames: Final = (
        {
            **chunk,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": _FALLBACK_TEXT}, "finish_reason": None}],
        },
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames) + (b"data: [DONE]\n\n",),
    )


def _responses_reply(identity: str, stream: bool) -> Reply:
    completed: Final[dict[str, JsonValue]] = {
        **_response_object(identity, "completed", "gpt-4o-mini"),
        "output": [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _FALLBACK_TEXT, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
    }
    if not stream:
        return Reply(body=json.dumps(completed).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(
            (
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**completed, "status": "in_progress", "output": []},
                },
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": f"msg_{identity}",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": _FALLBACK_TEXT,
                },
                {"type": "response.completed", "sequence_number": 2, "response": completed},
            )
        ),
    )


def _fallback_peer(request: Request) -> Reply:
    if request.method == "GET":
        return _EMPTY_MODEL_LIST
    assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}", dict(request.headers)
    identity: Final = uuid4().hex
    if request.target == "/v1/responses":
        return _responses_reply(identity, _streaming(request))
    assert request.target == "/v1/chat/completions", request.target
    return _chat_reply(identity, _streaming(request))


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    owned: OwnedProxy
    fallback: Wire
    peers: Mapping[str, Wire]
    models: Mapping[str, str]


def _mantle_params(api_base: str, model: str = _MANTLE_MODEL) -> dict[str, JsonValue]:
    return {"model": model, "api_base": api_base, "api_key": _MANTLE_KEY}


def _openai_params(api_base: str) -> dict[str, JsonValue]:
    return {"model": "openai/gpt-4o-mini", "api_base": api_base + "/v1", "api_key": _OPENAI_KEY}


def _write_config(
    root: Path,
    name: str,
    model_list: list[dict[str, JsonValue]],
    router_settings: dict[str, JsonValue],
    pass_through_target: str | None,
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = model_list
    config["router_settings"] = {"disable_cooldowns": True, "num_retries": 0, **router_settings}
    if pass_through_target is not None:
        config["general_settings"]["pass_through_endpoints"] = [
            {
                "path": "/mantle-passthrough",
                "target": pass_through_target,
                "headers": {"Authorization": f"Bearer {_MANTLE_KEY}"},
                "auth": True,
            }
        ]
    path: Final = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def cwf_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("mantle-cwf")
    rig_id: Final = uuid4().hex[:8]
    names: Final = {
        "cwf": f"mantle-cwf-{rig_id}",
        "cwf_failed": f"mantle-cwf-failed-{rig_id}",
        "failed_alone": f"mantle-failed-alone-{rig_id}",
        "cwf_failed_fake_stream": f"mantle-cwf-failed-fake-stream-{rig_id}",
        "failed_server_error": f"mantle-failed-server-error-{rig_id}",
        "fb_failed_server_error": f"mantle-fb-failed-server-error-{rig_id}",
        "failed_no_error": f"mantle-failed-no-error-{rig_id}",
        "incomplete": f"mantle-incomplete-{rig_id}",
    }
    fallback_name: Final = f"fallback-{rig_id}"
    with (
        gateway_from_environment() as gateway,
        wire_server(_overflow_peer) as overflow,
        wire_server(_body_peer("failed", {"error": _GPT_OSS_OVERFLOW_ERROR})) as failed,
        wire_server(_body_peer("failed", {"error": _GPT_OSS_OVERFLOW_ERROR}, _GPT_OSS_RESPONSES_PATH)) as failed_oss,
        wire_server(_body_peer("failed", {"error": _SERVER_ERROR})) as server_error,
        wire_server(_body_peer("failed", {"error": None})) as no_error,
        wire_server(_body_peer("incomplete", {"incomplete_details": {"reason": "max_output_tokens"}})) as incomplete,
        wire_server(_fallback_peer) as fallback,
    ):
        config: Final = _write_config(
            root,
            "cwf",
            [
                {"model_name": names["cwf"], "litellm_params": _mantle_params(overflow.url)},
                {"model_name": names["cwf_failed"], "litellm_params": _mantle_params(failed.url)},
                {"model_name": names["failed_alone"], "litellm_params": _mantle_params(failed.url)},
                {
                    "model_name": names["cwf_failed_fake_stream"],
                    "litellm_params": _mantle_params(failed_oss.url, _GPT_OSS_MODEL),
                    "model_info": {"supports_native_streaming": False},
                },
                {"model_name": names["failed_server_error"], "litellm_params": _mantle_params(server_error.url)},
                {"model_name": names["fb_failed_server_error"], "litellm_params": _mantle_params(server_error.url)},
                {"model_name": names["failed_no_error"], "litellm_params": _mantle_params(no_error.url)},
                {"model_name": names["incomplete"], "litellm_params": _mantle_params(incomplete.url)},
                {"model_name": fallback_name, "litellm_params": _openai_params(fallback.url)},
            ],
            {
                "context_window_fallbacks": [
                    {name: [fallback_name]}
                    for name in (names["cwf"], names["cwf_failed"], names["cwf_failed_fake_stream"])
                ],
                "fallbacks": [{names["fb_failed_server_error"]: [fallback_name]}],
            },
            None,
        )
        with owned_proxy_process(gateway, root, {}, config=config, workers=2) as owned:
            peers: Final = {
                "overflow": overflow,
                "failed": failed,
                "failed_oss": failed_oss,
                "server_error": server_error,
                "no_error": no_error,
                "incomplete": incomplete,
            }
            yield Rig(owned.gateway, owned, fallback, peers, names)


@pytest.fixture(scope="module")
def fb_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("mantle-fb")
    rig_id: Final = uuid4().hex[:8]
    names: Final = {
        "fb": f"mantle-fb-{rig_id}",
        "bad": f"mantle-bad-{rig_id}",
        "openai_fb": f"openai-fb-{rig_id}",
    }
    fallback_name: Final = f"fallback-{rig_id}"
    with (
        gateway_from_environment() as gateway,
        wire_server(_overflow_peer) as overflow,
        wire_server(_passthrough_peer) as passthrough,
        wire_server(_bad_input_peer) as bad,
        wire_server(_openai_overflow_peer) as openai_overflow,
        wire_server(_fallback_peer) as fallback,
    ):
        config: Final = _write_config(
            root,
            "fb",
            [
                {"model_name": names["fb"], "litellm_params": _mantle_params(overflow.url)},
                {"model_name": names["bad"], "litellm_params": _mantle_params(bad.url)},
                {"model_name": names["openai_fb"], "litellm_params": _openai_params(openai_overflow.url)},
                {"model_name": fallback_name, "litellm_params": _openai_params(fallback.url)},
            ],
            {"fallbacks": [{name: [fallback_name]} for name in names.values()]},
            passthrough.url + _RESPONSES_PATH,
        )
        with owned_proxy_process(gateway, root, {}, config=config, workers=2) as owned:
            yield Rig(
                owned.gateway,
                owned,
                fallback,
                {"overflow": overflow, "passthrough": passthrough, "bad": bad, "openai_overflow": openai_overflow},
                names,
            )


def _chat(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def _messages(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def _responses(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "input": prompt, "stream": stream}


def _consumed(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[httpx.Response, str]:
    with gateway.client.stream("POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}"}) as response:
        text: Final = b"".join(response.iter_bytes()).decode()
    return response, text


def _events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _only_error(text: str) -> dict[str, JsonValue]:
    errors: Final = tuple(event for event in _events(text) if event.get("type") == "error")
    assert len(errors) == 1, text
    return object_value(errors[0]["error"])


def _only_failed(text: str) -> dict[str, JsonValue]:
    failed: Final = tuple(event for event in _events(text) if event.get("type") == "response.failed")
    assert len(failed) == 1, text
    return object_value(object_value(failed[0]["response"])["error"])


def _calls_with(wire: Wire, prompt: str) -> int:
    return sum(1 for request in wire.drain() if prompt in request.body.decode())


def _prompt(label: str) -> str:
    return f"{label} {uuid4().hex}"


def _assert_fell_back(rig: Rig, peer: str, prompt: str, response_text: str) -> None:
    assert _FALLBACK_TEXT in response_text, response_text
    assert _calls_with(rig.peers[peer], prompt) == 1
    assert _calls_with(rig.fallback, prompt) == 1


def _assert_no_fallback(rig: Rig, peer: str, prompt: str) -> None:
    assert _calls_with(rig.peers[peer], prompt) == 1
    assert _calls_with(rig.fallback, prompt) == 0


def test_context_window_fallback_fires_for_the_overflow_on_chat_completions(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("cwf chat")
    response: Final = cwf_rig.gateway.request("POST", "/v1/chat/completions", _chat(cwf_rig.models["cwf"], prompt))
    assert response.status_code == 200, response.text
    _assert_fell_back(cwf_rig, "overflow", prompt, response.text)


def test_context_window_fallback_fires_for_the_overflow_on_messages(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("cwf messages")
    response: Final = cwf_rig.gateway.request("POST", "/v1/messages", _messages(cwf_rig.models["cwf"], prompt))
    assert response.status_code == 200, response.text
    _assert_fell_back(cwf_rig, "overflow", prompt, response.text)


def test_context_window_fallback_fires_for_the_overflow_on_responses(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("cwf responses")
    response: Final = cwf_rig.gateway.request("POST", "/v1/responses", _responses(cwf_rig.models["cwf"], prompt))
    assert response.status_code == 200, response.text
    _assert_fell_back(cwf_rig, "overflow", prompt, response.text)


def test_context_window_fallback_fires_when_the_provider_answers_200_status_failed_on_responses(
    cwf_rig: Rig,
) -> None:
    prompt: Final = _prompt("cwf responses failed body")
    response: Final = cwf_rig.gateway.request("POST", "/v1/responses", _responses(cwf_rig.models["cwf_failed"], prompt))
    assert response.status_code == 200, response.text
    _assert_fell_back(cwf_rig, "failed", prompt, response.text)


def test_a_200_status_failed_overflow_with_no_fallback_answers_400_on_responses(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("failed body alone")
    response: Final = cwf_rig.gateway.request(
        "POST", "/v1/responses", _responses(cwf_rig.models["failed_alone"], prompt)
    )
    assert response.status_code == 400, response.text
    assert _GPT_OSS_OVERFLOW_TEXT in str(object_value(_JSON_OBJECT.validate_json(response.content)["error"])["message"])
    _assert_no_fallback(cwf_rig, "failed", prompt)
    rows: Final = eventually(
        lambda: cwf_rig.gateway.rows(
            'SELECT status, prompt_tokens FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (response.headers["x-litellm-call-id"],),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert (rows[0]["status"], rows[0]["prompt_tokens"]) == ("failure", _FAILED_INPUT_TOKENS), rows


def test_context_window_fallback_fires_when_a_fake_streamed_provider_answers_200_status_failed_on_responses(
    cwf_rig: Rig,
) -> None:
    prompt: Final = _prompt("cwf responses failed body fake stream")
    response, text = _consumed(
        cwf_rig.gateway, "/v1/responses", _responses(cwf_rig.models["cwf_failed_fake_stream"], prompt, True)
    )
    assert response.status_code == 200, text
    _assert_fell_back(cwf_rig, "failed_oss", prompt, text)


def test_plain_fallback_fires_when_the_provider_answers_200_status_failed_with_a_server_error_on_responses(
    cwf_rig: Rig,
) -> None:
    prompt: Final = _prompt("fb failed body server error")
    response: Final = cwf_rig.gateway.request(
        "POST", "/v1/responses", _responses(cwf_rig.models["fb_failed_server_error"], prompt)
    )
    assert response.status_code == 200, response.text
    _assert_fell_back(cwf_rig, "server_error", prompt, response.text)


def test_a_200_status_failed_server_error_is_retried_and_answers_503_with_the_provider_request_id(
    cwf_rig: Rig,
) -> None:
    prompt: Final = _prompt("failed body server error")
    response: Final = cwf_rig.gateway.request(
        "POST", "/v1/responses", {**_responses(cwf_rig.models["failed_server_error"], prompt), "num_retries": 2}
    )
    assert response.status_code == 503, response.text
    assert str(_SERVER_ERROR["message"]) in response.text
    forwarded: Final = response.headers.get("llm_provider-x-request-id") or response.headers.get("x-request-id")
    assert forwarded == _FAILED_REQUEST_ID, response.headers
    assert _calls_with(cwf_rig.peers["server_error"], prompt) == 3


def test_a_200_status_failed_body_with_no_error_answers_503_on_responses(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("failed body no error")
    response: Final = cwf_rig.gateway.request(
        "POST", "/v1/responses", _responses(cwf_rig.models["failed_no_error"], prompt)
    )
    assert response.status_code == 503, response.text
    assert _NO_ERROR_MESSAGE in response.text
    assert _calls_with(cwf_rig.peers["no_error"], prompt) == 1


def test_an_incomplete_body_still_answers_200_on_responses(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("incomplete body")
    response: Final = cwf_rig.gateway.request("POST", "/v1/responses", _responses(cwf_rig.models["incomplete"], prompt))
    assert response.status_code == 200, response.text
    assert _JSON_OBJECT.validate_json(response.content)["status"] == "incomplete"
    assert _calls_with(cwf_rig.peers["incomplete"], prompt) == 1


def test_context_window_fallback_fires_for_the_overflow_on_streamed_chat_completions(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("cwf chat stream")
    response, text = _consumed(cwf_rig.gateway, "/v1/chat/completions", _chat(cwf_rig.models["cwf"], prompt, True))
    assert response.status_code == 200, text
    _assert_fell_back(cwf_rig, "overflow", prompt, text)


def test_context_window_fallback_is_not_consulted_on_streamed_messages(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("cwf messages stream")
    response, text = _consumed(cwf_rig.gateway, "/v1/messages", _messages(cwf_rig.models["cwf"], prompt, True))
    assert response.status_code == 200, text
    error: Final = _only_error(text)
    assert error["type"] == "invalid_request_error", text
    assert _GENERIC in str(error["message"]), text
    _assert_no_fallback(cwf_rig, "overflow", prompt)


def test_context_window_fallback_fires_on_streamed_responses(cwf_rig: Rig) -> None:
    prompt: Final = _prompt("cwf responses stream")
    response, text = _consumed(cwf_rig.gateway, "/v1/responses", _responses(cwf_rig.models["cwf"], prompt, True))
    assert response.status_code == 200, text
    _assert_fell_back(cwf_rig, "overflow", prompt, text)


def test_plain_fallback_fires_for_the_overflow_on_chat_completions(fb_rig: Rig) -> None:
    prompt: Final = _prompt("fb chat")
    response: Final = fb_rig.gateway.request("POST", "/v1/chat/completions", _chat(fb_rig.models["fb"], prompt))
    assert response.status_code == 200, response.text
    _assert_fell_back(fb_rig, "overflow", prompt, response.text)


def test_plain_fallback_fires_for_the_overflow_on_messages(fb_rig: Rig) -> None:
    prompt: Final = _prompt("fb messages")
    response: Final = fb_rig.gateway.request("POST", "/v1/messages", _messages(fb_rig.models["fb"], prompt))
    assert response.status_code == 200, response.text
    _assert_fell_back(fb_rig, "overflow", prompt, response.text)


def test_plain_fallback_fires_for_the_overflow_on_responses(fb_rig: Rig) -> None:
    prompt: Final = _prompt("fb responses")
    response: Final = fb_rig.gateway.request("POST", "/v1/responses", _responses(fb_rig.models["fb"], prompt))
    assert response.status_code == 200, response.text
    _assert_fell_back(fb_rig, "overflow", prompt, response.text)


def test_plain_fallback_fires_for_the_overflow_on_streamed_chat_completions(fb_rig: Rig) -> None:
    prompt: Final = _prompt("fb chat stream")
    response, text = _consumed(fb_rig.gateway, "/v1/chat/completions", _chat(fb_rig.models["fb"], prompt, True))
    assert response.status_code == 200, text
    _assert_fell_back(fb_rig, "overflow", prompt, text)


def test_streamed_messages_overflow_on_a_plain_fallback_deployment_returns_400_without_falling_back(
    fb_rig: Rig,
) -> None:
    prompt: Final = _prompt("fb messages stream")
    response, text = _consumed(fb_rig.gateway, "/v1/messages", _messages(fb_rig.models["fb"], prompt, True))
    assert response.status_code == 400, text
    error: Final = object_value(_JSON_OBJECT.validate_json(text)["error"])
    assert error["type"] == "invalid_request_error", text
    assert _GENERIC in str(error["message"]), text
    _assert_no_fallback(fb_rig, "overflow", prompt)


def test_plain_fallback_fires_on_streamed_responses(fb_rig: Rig) -> None:
    prompt: Final = _prompt("fb responses stream")
    response, text = _consumed(fb_rig.gateway, "/v1/responses", _responses(fb_rig.models["fb"], prompt, True))
    assert response.status_code == 200, text
    _assert_fell_back(fb_rig, "overflow", prompt, text)


def test_streamed_messages_overflow_on_an_openai_fallback_deployment_returns_the_error_event_without_falling_back(
    fb_rig: Rig,
) -> None:
    prompt: Final = _prompt("openai fb messages stream")
    response, text = _consumed(fb_rig.gateway, "/v1/messages", _messages(fb_rig.models["openai_fb"], prompt, True))
    assert response.status_code == 200, text
    error: Final = _only_error(text)
    assert error["type"] == "invalid_request_error", text
    assert _OPENAI_OVERFLOW_MARK in str(error["message"]), text
    _assert_no_fallback(fb_rig, "openai_overflow", prompt)


def test_streamed_messages_non_overflow_failure_returns_400_without_falling_back(fb_rig: Rig) -> None:
    prompt: Final = _prompt("bad messages stream")
    response, text = _consumed(fb_rig.gateway, "/v1/messages", _messages(fb_rig.models["bad"], prompt, True))
    assert response.status_code == 400, text
    error: Final = object_value(_JSON_OBJECT.validate_json(text)["error"])
    assert error["type"] == "invalid_request_error", text
    assert _INVALID_PROMPT_MESSAGE in str(error["message"]), text
    assert _GENERIC not in text, text
    _assert_no_fallback(fb_rig, "bad", prompt)


def test_pass_through_relays_the_overflow_envelope_verbatim(fb_rig: Rig) -> None:
    prompt: Final = _prompt("passthrough")
    response: Final = fb_rig.gateway.request("POST", "/mantle-passthrough", {"model": _BACKEND, "input": prompt})
    assert response.status_code == 400, response.text
    assert _JSON_OBJECT.validate_json(response.content) == _OVERFLOW_ENVELOPE, response.text
    assert _calls_with(fb_rig.peers["passthrough"], prompt) == 1


def test_pass_through_relays_the_failed_stream_frames_verbatim(fb_rig: Rig) -> None:
    prompt: Final = _prompt("passthrough stream")
    body: Final[dict[str, JsonValue]] = {"model": _BACKEND, "input": prompt, "stream": True}
    response, text = _consumed(fb_rig.gateway, "/mantle-passthrough", body)
    assert response.status_code == 200, text
    assert text.encode() == b"".join(_PASSTHROUGH_FAILED_FRAMES), text
    assert _calls_with(fb_rig.peers["passthrough"], prompt) == 1


def test_killing_one_worker_leaves_the_sibling_serving_the_fallback(fb_rig: Rig) -> None:
    workers: Final = eventually(fb_rig.owned.worker_pids, lambda pids: len(pids) >= 2, seconds=30)
    with fb_rig.owned.replacing_worker(min(workers)) as kill:
        kill()
        with httpx.Client(base_url=fb_rig.gateway.client.base_url, timeout=15, trust_env=False) as fresh:
            survivor: Final = Gateway(fresh, fb_rig.gateway.key, fb_rig.gateway.upstream_url)
            responses: Final = tuple(
                survivor.request("POST", "/v1/chat/completions", _chat(fb_rig.models["fb"], _prompt("after kill")))
                for _ in range(3)
            )
    for response in responses:
        assert response.status_code == 200, response.text
        assert _FALLBACK_TEXT in response.text, response.text
    assert fb_rig.gateway.request("GET", "/health/liveliness").status_code == 200
