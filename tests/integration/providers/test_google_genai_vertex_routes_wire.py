import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, held, object_value
from integration._support.database import read_rows
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROJECT: Final = "scripted-project"
_REPLY_TEXT: Final = "scripted vertex reply"
_PROMPT_TOKENS: Final = 11
_COMPLETION_TOKENS: Final = 7
_USAGE: Final = {
    "promptTokenCount": _PROMPT_TOKENS,
    "candidatesTokenCount": _COMPLETION_TOKENS,
    "totalTokenCount": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_NATIVE_COUNT: Final = 31
_SPEND_SELECT: Final = (
    "SELECT request_id, call_type, model_group, status, cache_hit, prompt_tokens, completion_tokens, spend"
    ' FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"'
)


@dataclass(frozen=True, slots=True)
class Route:
    model: str
    location: str
    adapter: bool
    endpoint_path: str = ""
    overrides: Mapping[str, JsonValue] | None = None


_ROUTES: Final = {
    "model_garden_publisher": Route("xai/grok-4.20-non-reasoning", "global", adapter=True),
    "model_garden_endpoint": Route("openai/1234567890123456789", "us-central1", adapter=True),
    "gemma": Route(
        "gemma/gemma-2-2b-it",
        "us-central1",
        adapter=True,
        endpoint_path=f"/v1/projects/{_PROJECT}/locations/us-central1/endpoints/4242:predict",
    ),
    "gemini": Route("gemini-2.5-flash", "us-central1", adapter=False),
    "legacy_non_gemini": Route("chat-bison", "us-central1", adapter=False),
    "tuned_gemini_endpoint": Route(
        "9876543210", "us-central1", adapter=False, overrides={"base_model": "vertex_ai/gemini-2.5-flash"}
    ),
}
_ADAPTER_ROUTES: Final = tuple(name for name, route in _ROUTES.items() if route.adapter)


def _openai_reply(model: str) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": _PROMPT_TOKENS,
            "completion_tokens": _COMPLETION_TOKENS,
            "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
        },
    }


def _openai_frames(model: str, include_usage: bool) -> tuple[bytes, ...]:
    base: Final = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": model,
    }

    def frame(chunk: Mapping[str, JsonValue]) -> bytes:
        return f"data: {json.dumps({**base, **chunk})}\n\n".encode()

    usage: Final = (frame({"choices": [], "usage": _openai_reply(model)["usage"]}),) if include_usage else ()
    return (
        frame(
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": None}]}
        ),
        frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        *usage,
        b"data: [DONE]\n\n",
    )


def _asks_for_usage(body: Mapping[str, JsonValue]) -> bool:
    options: Final = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


def _native_reply() -> dict[str, JsonValue]:
    return {
        "candidates": [
            {"content": {"parts": [{"text": _REPLY_TEXT}], "role": "model"}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": _USAGE,
    }


def _native_frames() -> tuple[bytes, ...]:
    return (
        b'data: {"candidates":[{"content":{"parts":[{"text":"'
        + _REPLY_TEXT.encode()
        + b'"}],"role":"model"},"index":0}]}\n\n',
        b'data: {"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":'
        + json.dumps(_USAGE).encode()
        + b"}\n\n",
    )


def _publisher_not_found(request: Request) -> Reply:
    return Reply(
        status=404,
        body=json.dumps(
            {
                "error": {
                    "code": 404,
                    "message": f"Publisher Model {request.target} was not found",
                    "status": "NOT_FOUND",
                }
            }
        ).encode(),
    )


def _google_native_target(target: str) -> bool:
    return any(
        name in target for name in ("/models/gemini-2.5-flash:", "/models/chat-bison:", "/endpoints/9876543210:")
    )


def _vertex_peer(status: int = 200, error: str = "scripted vertex rejection") -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if status != 200:
            return Reply(status=status, body=json.dumps({"error": {"message": error, "code": status}}).encode())
        if request.target.endswith("/chat/completions"):
            body: Final = _JSON_OBJECT.validate_json(request.body)
            model: Final = str(body.get("model", ""))
            if body.get("stream") is True:
                return Reply(content_type="text/event-stream", chunks=_openai_frames(model, _asks_for_usage(body)))
            return Reply(body=json.dumps(_openai_reply(model)).encode())
        if request.target.endswith(":predict"):
            return Reply(body=json.dumps({"predictions": _openai_reply("gemma-2-2b-it")}).encode())
        if not _google_native_target(request.target):
            return _publisher_not_found(request)
        if ":streamGenerateContent" in request.target:
            return Reply(content_type="text/event-stream", chunks=_native_frames())
        if request.target.endswith(":countTokens"):
            return Reply(body=json.dumps({"totalTokens": _NATIVE_COUNT}).encode())
        return Reply(body=json.dumps(_native_reply()).encode())

    return respond


def _deployment(gateway: Gateway, scenario: Scenario, wire: Wire, route: Route, **extra: JsonValue) -> str:
    return scenario.model(
        model=f"vertex_ai/{route.model}",
        api_base=f"{wire.url}{route.endpoint_path}",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=route.location,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        **dict(route.overrides or {}),
        **extra,
    )


def _contents(marker: str) -> list[JsonValue]:
    return [{"role": "user", "parts": [{"text": f"route probe {marker}"}]}]


def _generate(gateway: Gateway, model: str, key: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", f"/v1beta/models/{model}:generateContent", body, key=key)


def _stream(gateway: Gateway, model: str, key: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.client.post(
        f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
        json=dict(body),
        headers={"Authorization": f"Bearer {key}"},
    )


def _frames(response: httpx.Response) -> list[dict[str, JsonValue]]:
    parts: Final = [part for part in response.text.split("\n\n") if part.strip()]
    assert all(part.startswith("data: ") for part in parts), response.text
    return [_JSON_OBJECT.validate_json(part.removeprefix("data: ")) for part in parts]


def _finish_reasons(frames: Sequence[Mapping[str, JsonValue]]) -> list[JsonValue]:
    return [
        object_value(candidate).get("finishReason")
        for frame in frames
        for candidate in (frame.get("candidates") or [])  # comprehension-ok: two-level frame walk
        if isinstance(candidate, dict) and candidate.get("finishReason")
    ]


def _text(frames: Sequence[Mapping[str, JsonValue]]) -> str:
    return "".join(
        str(object_value(part).get("text", ""))
        for frame in frames
        for candidate in (frame.get("candidates") or [])  # comprehension-ok: three-level frame walk
        for part in object_value(object_value(candidate).get("content") or {}).get("parts") or []
    )


def _spend_rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: read_rows(_SPEND_SELECT, (digest,)), lambda values: len(values) >= count, seconds=70)
    return held(lambda: read_rows(_SPEND_SELECT, (digest,)), lambda values: len(values) == count, holding=3)


def _route_targets(wire: Wire) -> list[tuple[str, str]]:
    return [(request.method, request.target) for request in wire.drain()]


def _expected_adapter_target(route: Route) -> str:
    return route.endpoint_path if route.endpoint_path else "/chat/completions"


def _cost_map_prices(gateway: Gateway, model: str) -> tuple[float, float]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    info: Final = next(
        object_value(object_value(entry)["model_info"])
        for entry in entries
        if object_value(entry)["model_name"] == model
    )
    prompt: Final = info.get("input_cost_per_token")
    completion: Final = info.get("output_cost_per_token")
    assert isinstance(prompt, float) and isinstance(completion, float) and prompt > 0 and completion > 0, info
    return prompt, completion


@pytest.mark.parametrize("route_name", tuple(_ROUTES))
def test_generate_content_reaches_the_route_handler_chat_uses(gateway: Gateway, route_name: str) -> None:
    route: Final = _ROUTES[route_name]
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route)
        key: Final = scenario.key(models=[model])
        response: Final = _generate(gateway, model, key, {"contents": _contents(uuid.uuid4().hex)})
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert _text([body]) == _REPLY_TEXT, response.text
        assert _finish_reasons([body]) == ["STOP"], response.text
        assert body["usageMetadata"] == _USAGE, response.text
        targets: Final = _route_targets(wire)
        assert len(targets) == 1, targets
        if route.adapter:
            assert targets == [("POST", _expected_adapter_target(route))], targets
        else:
            assert targets[0][1].endswith(":generateContent"), targets
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["model_group"] == model, rows
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS), rows


@pytest.mark.parametrize("route_name", tuple(_ROUTES))
def test_stream_generate_content_ends_with_one_stop_carrying_upstream_usage(gateway: Gateway, route_name: str) -> None:
    route: Final = _ROUTES[route_name]
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route)
        key: Final = scenario.key(models=[model])
        response: Final = _stream(gateway, model, key, {"contents": _contents(uuid.uuid4().hex)})
        assert response.status_code == 200, response.text
        frames: Final = _frames(response)
        assert _text(frames) == _REPLY_TEXT, response.text
        assert _finish_reasons(frames) == ["STOP"], response.text
        assert _finish_reasons(frames[-1:]) == ["STOP"], response.text
        assert frames[-1]["usageMetadata"] == _USAGE, response.text
        received: Final = wire.drain()
        assert len(received) == 1, [request.target for request in received]
        if route.adapter and not route.endpoint_path:
            assert received[0].target == "/chat/completions", received[0].target
            assert _JSON_OBJECT.validate_json(received[0].body)["stream_options"] == {"include_usage": True}
        if not route.adapter:
            assert ":streamGenerateContent" in received[0].target, received[0].target
            assert "stream_options" not in _JSON_OBJECT.validate_json(received[0].body), received[0].body
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["model_group"] == model, rows
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS), rows


@pytest.mark.parametrize("route_name", _ADAPTER_ROUTES)
def test_count_tokens_counts_locally_on_adapter_routes(gateway: Gateway, route_name: str) -> None:
    route: Final = _ROUTES[route_name]
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", f"/v1beta/models/{model}:countTokens", {"contents": _contents("count")}, key=key
        )
        assert response.status_code == 200, response.text
        total: Final = _JSON_OBJECT.validate_json(response.content)["totalTokens"]
        assert isinstance(total, int) and 0 < total != _NATIVE_COUNT, response.text
        assert _route_targets(wire) == []


@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
def test_model_garden_spend_is_priced_from_the_cost_map(gateway: Gateway, stream: bool) -> None:
    route: Final = _ROUTES["model_garden_publisher"]
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route)
        key: Final = scenario.key(models=[model])
        prompt_price, completion_price = _cost_map_prices(gateway, model)
        call: Final = _stream if stream else _generate
        response: Final = call(gateway, model, key, {"contents": _contents(uuid.uuid4().hex)})
        assert response.status_code == 200, response.text
        expected: Final = _PROMPT_TOKENS * prompt_price + _COMPLETION_TOKENS * completion_price
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["call_type"] == "acompletion", rows
        assert float(str(rows[0]["spend"])) == pytest.approx(expected), rows
        assert len(wire.drain()) == 1


def test_model_garden_chat_and_generate_content_bill_the_same(gateway: Gateway) -> None:
    route: Final = _ROUTES["model_garden_publisher"]
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route)
        chat_key: Final = scenario.key(models=[model])
        genai_key: Final = scenario.key(models=[model])
        chat: Final = gateway.client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "route probe"}], "stream": True},
            headers={"Authorization": f"Bearer {chat_key}"},
        )
        assert chat.status_code == 200, chat.text
        genai: Final = _stream(gateway, model, genai_key, {"contents": _contents("route probe")})
        assert genai.status_code == 200, genai.text
        chat_rows: Final = _spend_rows(chat_key, 1)
        genai_rows: Final = _spend_rows(genai_key, 1)
        assert (chat_rows[0]["prompt_tokens"], chat_rows[0]["completion_tokens"]) == (
            _PROMPT_TOKENS,
            _COMPLETION_TOKENS,
        ), chat_rows
        assert float(str(genai_rows[0]["spend"])) == pytest.approx(float(str(chat_rows[0]["spend"]))), (
            chat_rows,
            genai_rows,
        )
        assert float(str(genai_rows[0]["spend"])) > 0, genai_rows
        assert len(wire.drain()) == 2


def _replayed_frames(response: httpx.Response, streamed: bool) -> list[dict[str, JsonValue]]:
    return _frames(response) if streamed else [_JSON_OBJECT.validate_json(response.content)]


@pytest.mark.parametrize("streamed", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize("route_name", _ADAPTER_ROUTES)
def test_adapter_route_cache_hit_skips_the_upstream_and_keeps_attribution(
    gateway: Gateway, route_name: str, streamed: bool
) -> None:
    route: Final = _ROUTES[route_name]
    call: Final = _stream if streamed else _generate
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route)
        key: Final = scenario.key(models=[model])
        body: Final = {"contents": _contents(uuid.uuid4().hex)}
        first: Final = call(gateway, model, key, body)
        assert first.status_code == 200, first.text
        second: Final = call(gateway, model, key, body)
        assert second.status_code == 200, second.text
        replayed: Final = _replayed_frames(second, streamed)
        assert _text(replayed) == _REPLY_TEXT, second.text
        assert replayed[-1]["usageMetadata"] == _replayed_frames(first, streamed)[-1]["usageMetadata"], second.text
        assert len(wire.drain()) == 1
        rows: Final = _spend_rows(key, 2)
        assert [row["cache_hit"] == "True" for row in rows] == [False, True], rows
        assert [row["model_group"] for row in rows] == [model, model], rows
        assert len({row["request_id"] for row in rows}) == 2, rows


@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize("status", [400, 401, 429])
def test_adapter_route_upstream_error_reaches_the_caller_with_its_status(
    gateway: Gateway, status: int, stream: bool
) -> None:
    route: Final = _ROUTES["model_garden_publisher"]
    message: Final = f"scripted vertex rejection {uuid.uuid4().hex}f4039"
    with wire_server(_vertex_peer(status=status, error=message)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route, num_retries=0)
        key: Final = scenario.key(models=[model])
        call: Final = _stream if stream else _generate
        response: Final = call(gateway, model, key, {"contents": _contents(uuid.uuid4().hex)})
        assert response.status_code == status, response.text
        assert message in response.text, response.text
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["status"] == "failure", rows
        assert len(wire.drain()) >= 1
        healthy: Final = gateway.request("GET", "/health/liveliness")
        assert healthy.status_code == 200, healthy.text


def test_adapter_route_failure_leaves_other_keys_serving(gateway: Gateway) -> None:
    route: Final = _ROUTES["model_garden_publisher"]
    with (
        wire_server(_vertex_peer(status=400)) as failing,
        wire_server(_vertex_peer()) as serving,
        gateway.scenario() as scenario,
    ):
        broken: Final = _deployment(gateway, scenario, failing, route, num_retries=0)
        working: Final = _deployment(gateway, scenario, serving, route)
        broken_key: Final = scenario.key(models=[broken])
        working_key: Final = scenario.key(models=[working])
        assert _generate(gateway, broken, broken_key, {"contents": _contents("a")}).status_code == 400
        served: Final = _generate(gateway, working, working_key, {"contents": _contents("b")})
        assert served.status_code == 200, served.text
        assert _spend_rows(working_key, 1)[0]["status"] == "success"


@pytest.mark.parametrize(
    ("base_model", "adapter"),
    [
        pytest.param("vertex_ai/gemini-2.5-flash", False, id="gemini_alias_forces_native"),
        pytest.param("", True, id="empty_string"),
        pytest.param(None, True, id="null"),
        pytest.param("vertex_ai/xai/grok-4.20-non-reasoning", True, id="non_gemini_alias"),
    ],
)
def test_base_model_steers_generate_content_like_chat(gateway: Gateway, base_model: str | None, adapter: bool) -> None:
    route: Final = _ROUTES["model_garden_publisher"]
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route, base_model=base_model)
        key: Final = scenario.key(models=[model])
        genai: Final = _generate(gateway, model, key, {"contents": _contents("alias")})
        genai_targets: Final = _route_targets(wire)
        chat: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "alias"}]},
            key=key,
        )
        chat_targets: Final = _route_targets(wire)
        assert len(genai_targets) == 1 and len(chat_targets) == 1, (genai_targets, chat_targets)
        assert (genai_targets[0][1] == "/chat/completions") is adapter, genai_targets
        assert (chat_targets[0][1] == "/chat/completions") is adapter, chat_targets
        assert genai.status_code == chat.status_code, (genai.text, chat.text)


@pytest.mark.parametrize(
    "stream_options",
    [
        pytest.param(7, id="int"),
        pytest.param(["include_usage"], id="list"),
        pytest.param("", id="empty_string"),
        pytest.param("x" * 5000, id="five_kb"),
    ],
)
def test_malformed_stream_options_fall_back_to_usage(gateway: Gateway, stream_options: JsonValue) -> None:
    route: Final = _ROUTES["model_garden_publisher"]
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route)
        key: Final = scenario.key(models=[model])
        response: Final = _stream(
            gateway, model, key, {"contents": _contents(uuid.uuid4().hex), "stream_options": stream_options}
        )
        assert response.status_code == 200, response.text
        frames: Final = _frames(response)
        assert _finish_reasons(frames) == ["STOP"], response.text
        assert frames[-1]["usageMetadata"] == _USAGE, response.text
        received: Final = wire.drain()
        assert len(received) == 1
        assert _JSON_OBJECT.validate_json(received[0].body)["stream_options"] == {"include_usage": True}


def test_duplicate_stream_options_key_keeps_the_last_value(gateway: Gateway) -> None:
    route: Final = _ROUTES["model_garden_publisher"]
    with wire_server(_vertex_peer()) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, route)
        key: Final = scenario.key(models=[model])
        contents: Final = json.dumps(_contents(uuid.uuid4().hex))
        raw: Final = (
            '{"contents": ' + contents + ', "stream_options": {"include_usage": true},'
            ' "stream_options": {"include_usage": false}}'
        )
        response: Final = gateway.client.post(
            f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
            content=raw.encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        assert response.status_code == 200, response.text
        frames: Final = _frames(response)
        assert _finish_reasons(frames) == ["STOP"], response.text
        assert frames[-1]["usageMetadata"] == {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
        received: Final = wire.drain()
        assert len(received) == 1
        assert "stream_options" not in _JSON_OBJECT.validate_json(received[0].body), received[0].body
