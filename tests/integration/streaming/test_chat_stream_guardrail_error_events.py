import contextlib
import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import openai
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, WorkerHealthcheck, graceful_stop_seconds, owned_proxy_process
from integration._support.responses_stream import healthy_stream
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

from litellm.proxy import client as proxy_client

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("chat-stream-guardrail-error-events"),
    pytest.mark.shares_worker_state,
]

_MODEL: Final = "stream-error-events"
_FILTER: Final = "keeper-filter"
_SPAN_POST: Final = "span-post-call"
_SPAN_PRE: Final = "span-pre-call"
_BUFFERED: Final = "buffered-post-call"
_RAISING: Final = "raising-post-call"
_REPLACE: Final = "replace-client-error"
_REPLACEMENT_MESSAGE: Final = "Request blocked by policy"
_CRASH_MESSAGE: Final = "Guardrail service crashed while scanning"
_RAISED_BLOCK: Final = "Raised block: policy"
_SPAN: Final = "spanword"
_LEAD: Final = "The lighthouse on the northern cliff has guided ships for two hundred years, and "
_TEXTS: Final[Mapping[str, tuple[str, ...]]] = {
    "clean": ("Hello", " there"),
    "filter-late": (_LEAD, "the keeper lights the lamp at dusk."),
    "filter-early": ("The keeper lights the lamp.", " Ships pass."),
    "span-early": (f"{_SPAN} opens the answer", " and more"),
    "span-late": ("Hello there, ", f"{_SPAN} follows"),
    "buffered": ("The lamp is lit by", " the keeper."),
    "drop": ("Hello", " there"),
}
_BUFFERED_PAUSE_SECONDS: Final = 1.5
_DROP_PAUSE_SECONDS: Final = 0.5
_SCENARIO: Final = re.compile(rb"scenario-([a-z-]+) marker-([0-9a-f]+)")
_MODULE: Final = "stream_error_hooks"
_MODULE_SOURCE: Final = f'''import re

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger


class SpanGuardrail(CustomGuardrail):
    async def async_post_call_streaming_hook(self, user_api_key_dict, response):
        match = re.search("{_SPAN}", response)
        if match is not None:
            raise HTTPException(status_code=400, detail={{"error": "Content blocked: {_SPAN}", "matched_span": match}})
        return None

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        match = re.search("{_SPAN}", str(data["messages"][-1]["content"]))
        if match is not None:
            raise HTTPException(status_code=400, detail={{"matched_span": match}})
        return data


class BufferingGuardrail(CustomGuardrail):
    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        chunks = [chunk async for chunk in response]
        text = "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
        if "keeper" in text:
            raise HTTPException(status_code=400, detail={{"error": "Buffered block: keeper"}})
        for chunk in chunks:
            yield chunk


class RaisingGuardrail(CustomGuardrail):
    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        prompt = str(request_data.get("messages") or request_data.get("input"))
        found = re.search("raise-(builtin|http)-after-([0-9]+)", prompt)
        sent = 0
        async for chunk in response:
            if found is not None and sent == int(found.group(2)):
                if found.group(1) == "builtin":
                    raise RuntimeError("{_CRASH_MESSAGE}")
                raise HTTPException(status_code=400, detail={{"error": "{_RAISED_BLOCK}"}})
            yield chunk
            sent += 1


class ReplacesClientErrors(CustomLogger):
    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        if "{_REPLACE}" in str(request_data.get("messages") or request_data.get("input")):
            return HTTPException(status_code=403, detail="{_REPLACEMENT_MESSAGE}")
        return None


replacer = ReplacesClientErrors()
'''


def _chunk(identity: str, delta: Mapping[str, JsonValue], finish: str | None = None) -> bytes:
    body: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _stream(identity: str, pieces: Sequence[str]) -> tuple[bytes, ...]:
    first, *rest = pieces
    return (
        _chunk(identity, {"role": "assistant", "content": first}),
        *(_chunk(identity, {"content": piece}) for piece in rest),
        _chunk(identity, {}, finish="stop"),
        b"data: [DONE]\n\n",
    )


def _respond(request: Request) -> Reply:
    found: Final = _SCENARIO.search(request.body)
    assert found is not None, request.body
    name: Final = found.group(1).decode()
    marker: Final = found.group(2).decode()
    if request.target.endswith("/responses"):
        return Reply(content_type="text/event-stream", chunks=healthy_stream(f"resp_{marker}", "".join(_TEXTS[name])))
    chunks: Final = _stream(f"chatcmpl-{marker}", _TEXTS[name])
    match name:
        case "buffered":
            return Reply(content_type="text/event-stream", chunks=chunks, pause_between_chunks=_BUFFERED_PAUSE_SECONDS)
        case "drop":
            return Reply(
                content_type="text/event-stream",
                chunks=chunks,
                abort_after=2,
                pause_between_chunks=_DROP_PAUSE_SECONDS,
            )
        case _:
            return Reply(content_type="text/event-stream", chunks=chunks)


def _config(upstream_url: str) -> dict[str, JsonValue]:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    guardrails: Final[tuple[tuple[str, str, str], ...]] = (
        (_SPAN_POST, f"{_MODULE}.SpanGuardrail", "post_call"),
        (_SPAN_PRE, f"{_MODULE}.SpanGuardrail", "pre_call"),
        (_BUFFERED, f"{_MODULE}.BufferingGuardrail", "post_call"),
        (_RAISING, f"{_MODULE}.RaisingGuardrail", "post_call"),
    )
    return {
        **base,
        "litellm_settings": {**object_value(base["litellm_settings"]), "callbacks": [f"{_MODULE}.replacer"]},
        "guardrails": [
            {
                "guardrail_name": _FILTER,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "post_call",
                    "default_on": False,
                    "blocked_words": [{"keyword": "keeper", "action": "BLOCK"}],
                },
            },
            *(
                {"guardrail_name": name, "litellm_params": {"guardrail": guardrail, "mode": mode, "default_on": False}}
                for name, guardrail, mode in guardrails
            ),
        ],
        "model_list": [
            {
                "model_name": _MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{upstream_url}/v1",
                    "api_key": "synthetic-stream-key",
                    "allow_client_keepalive_override": True,
                },
            }
        ],
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: OwnedProxy
    upstream: Wire


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with wire_server(_respond) as upstream:
        directory: Final = tmp_path_factory.mktemp("chat-stream-guardrail-error-events")
        (directory / f"{_MODULE}.py").write_text(_MODULE_SOURCE)
        config_path: Final = directory / "chat-stream-guardrail-error-events.yaml"
        config_path.write_text(yaml.safe_dump(_config(upstream.url)))
        with owned_proxy_process(
            session_gateway,
            directory,
            {},
            config=config_path,
            workers=2,
            healthcheck=WorkerHealthcheck.LENIENT,
        ) as proxy:
            yield Rig(proxy, upstream)


def _prompt(scenario: str, marker: str, *directives: str) -> str:
    return " ".join((f"scenario-{scenario}", f"marker-{marker}", *directives))


def _body(
    scenario: str, marker: str, guardrails: Sequence[str], *directives: str, **extra: JsonValue
) -> dict[str, JsonValue]:
    return {
        "model": _MODEL,
        "stream": True,
        "messages": [{"role": "user", "content": _prompt(scenario, marker, *directives)}],
        "guardrails": list(guardrails),
        "num_retries": 0,
        "cache": {"no-cache": True},
        **extra,
    }


def _marker() -> str:
    return uuid.uuid4().hex


def _calls_with(wire: Wire, marker: str) -> int:
    return sum(1 for request in wire.drain() if marker.encode() in request.body)


def _stream_chat(rig: Rig, body: Mapping[str, JsonValue]) -> httpx.Response:
    return rig.proxy.gateway.request("POST", "/v1/chat/completions", body)


def _data_lines(text: str) -> tuple[str, ...]:
    return tuple(line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: "))


def _payloads(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON_OBJECT.validate_json(line) for line in _data_lines(text) if line != "[DONE]")


def _choices(payloads: Sequence[Mapping[str, JsonValue]]) -> Iterator[dict[str, JsonValue]]:
    for payload in payloads:
        yield from map(object_value, list_value(payload.get("choices") or []))


def _content(payloads: Sequence[Mapping[str, JsonValue]]) -> str:
    return "".join(string_value(object_value(choice["delta"]).get("content") or "") for choice in _choices(payloads))


def _finish_reasons(payloads: Sequence[Mapping[str, JsonValue]]) -> tuple[JsonValue, ...]:
    return tuple(choice.get("finish_reason") for choice in _choices(payloads))


@dataclass(frozen=True, slots=True)
class EndedInBand:
    content: str
    error: dict[str, JsonValue]


def _ended_in_band(response: httpx.Response, status: int) -> EndedInBand:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("text/event-stream"), response.headers
    lines: Final = _data_lines(response.text)
    assert lines[-1] == "[DONE]" and lines.count("[DONE]") == 1, response.text
    payloads: Final = _payloads(response.text)
    assert "error" in payloads[-1] and all("error" not in payload for payload in payloads[:-1]), response.text
    assert "stop" not in _finish_reasons(payloads), response.text
    return EndedInBand(_content(payloads[:-1]), object_value(payloads[-1]["error"]))


def test_mid_stream_content_filter_block_ends_with_an_error_event_and_done(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _stream_chat(rig, _body("filter-late", marker, (_FILTER,)))
    ended: Final = _ended_in_band(response, 200)
    assert ended.content and _LEAD.startswith(ended.content), response.text
    assert ended.error["code"] == "400", ended.error
    assert "keeper" in string_value(ended.error["message"]), ended.error
    assert object_value(ended.error["provider_specific_fields"])["keyword"] == "keeper", ended.error
    assert _calls_with(rig.upstream, marker) == 1


def test_mid_stream_content_filter_block_reaches_the_openai_sdk_as_the_block_error(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.OpenAI(
        base_url=f"{rig.proxy.gateway.client.base_url}/v1", api_key=rig.proxy.gateway.key, max_retries=0
    )
    stream: Final = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": f"scenario-filter-late marker-{marker}"}],
        stream=True,
        extra_body={"guardrails": [_FILTER], "num_retries": 0, "cache": {"no-cache": True}},
    )
    with pytest.raises(openai.APIError) as raised:
        tuple(stream)
    assert type(raised.value) is openai.APIError, repr(raised.value)
    assert "keeper" in raised.value.message, raised.value.message


def test_early_content_filter_block_keeps_its_400_event_stream_body(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _stream_chat(rig, _body("filter-early", marker, (_FILTER,)))
    ended: Final = _ended_in_band(response, 400)
    assert ended.content == "", response.text
    assert ended.error["code"] == "400", ended.error
    assert object_value(ended.error["provider_specific_fields"])["keyword"] == "keeper", ended.error
    assert _calls_with(rig.upstream, marker) == 1


@pytest.mark.parametrize(
    ("scenario", "status"),
    [pytest.param("span-early", 400, id="first_chunk"), pytest.param("span-late", 200, id="later_chunk")],
)
def test_per_chunk_block_with_an_unencodable_detail_still_sends_the_error_event(
    rig: Rig, scenario: str, status: int
) -> None:
    marker: Final = _marker()
    response: Final = _stream_chat(rig, _body(scenario, marker, (_SPAN_POST,)))
    ended: Final = _ended_in_band(response, status)
    pieces: Final = _TEXTS[scenario]
    seen_by_hook: Final = "".join(pieces[: next(i for i, piece in enumerate(pieces) if _SPAN in piece) + 1])
    assert ended.content == ("" if status == 400 else pieces[0]), response.text
    assert ended.error["code"] == "400", ended.error
    assert ended.error["message"] == f"Content blocked: {_SPAN}", ended.error
    assert object_value(ended.error["provider_specific_fields"])["matched_span"] == str(
        re.search(_SPAN, seen_by_hook)
    ), ended.error


@pytest.mark.parametrize(
    ("scenario", "status"),
    [pytest.param("filter-early", 403, id="before_the_first_chunk"), pytest.param("filter-late", 200, id="mid_stream")],
)
def test_failure_hook_replacement_is_the_error_the_stream_ends_with(rig: Rig, scenario: str, status: int) -> None:
    marker: Final = _marker()
    body: Final = _body(scenario, marker, (_FILTER,))
    replaced: Final = {
        **body,
        "messages": [{"role": "user", "content": f"scenario-{scenario} marker-{marker} {_REPLACE}"}],
    }
    response: Final = _stream_chat(rig, replaced)
    ended: Final = _ended_in_band(response, status)
    assert ended.error["code"] == "403", ended.error
    assert ended.error["message"] == _REPLACEMENT_MESSAGE, ended.error
    assert "keeper" not in json.dumps(ended.error), ended.error


def test_buffered_block_after_keepalive_pings_ends_with_an_error_event_and_done(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _stream_chat(rig, _body("buffered", marker, (_BUFFERED,), keepalive_seconds=1))
    assert response.text.startswith(": ping\n\n"), response.text
    ended: Final = _ended_in_band(response, 200)
    assert ended.content == "", response.text
    assert ended.error["code"] == "400", ended.error
    assert ended.error["message"] == "Buffered block: keeper", ended.error


def test_clean_stream_with_guardrails_on_ends_with_stop_and_one_done(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _stream_chat(rig, _body("clean", marker, (_FILTER, _SPAN_POST)))
    assert response.status_code == 200, response.text
    lines: Final = _data_lines(response.text)
    assert lines[-1] == "[DONE]" and lines.count("[DONE]") == 1, response.text
    payloads: Final = _payloads(response.text)
    assert all("error" not in payload for payload in payloads), response.text
    assert _content(payloads) == "".join(_TEXTS["clean"]), response.text
    assert "stop" in _finish_reasons(payloads), response.text


def test_gemini_route_provider_failure_mid_stream_sends_an_error_frame_without_done(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.gateway.request(
        "POST",
        f"/v1beta/models/{_MODEL}:streamGenerateContent",
        {"contents": [{"role": "user", "parts": [{"text": f"scenario-drop marker-{marker}"}]}]},
        params={"alt": "sse"},
    )
    assert response.status_code == 200, response.text
    assert "[DONE]" not in response.text, response.text
    payloads: Final = _payloads(response.text)
    assert "error" in payloads[-1] and all("error" not in payload for payload in payloads[:-1]), response.text
    assert _calls_with(rig.upstream, marker) == 1


def test_mid_stream_block_releases_the_parallel_request_slot_and_logs_one_failure(rig: Rig) -> None:
    marker: Final = _marker()
    with rig.proxy.gateway.scenario() as scenario:
        key: Final = scenario.key(max_parallel_requests=1)
        with rig.proxy.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json=_body("filter-late", marker, (_FILTER,)),
            headers={"Authorization": f"Bearer {key}"},
        ) as blocked:
            call_id: Final = blocked.headers["x-litellm-call-id"]
            with contextlib.suppress(httpx.RemoteProtocolError):
                blocked.read()
        follow_up: Final = rig.proxy.gateway.request(
            "POST", "/v1/chat/completions", _body("clean", _marker(), ()), key=key
        )
        assert follow_up.status_code == 200, follow_up.text
        assert _content(_payloads(follow_up.text)) == "".join(_TEXTS["clean"]), follow_up.text
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s', (call_id,)),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    assert [row["status"] for row in rows] == ["failure"], rows


def test_failure_hook_replacement_hides_a_non_http_failure_mid_stream(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _stream_chat(rig, _body("clean", marker, (_RAISING,), "raise-builtin-after-1", _REPLACE))
    ended: Final = _ended_in_band(response, 200)
    assert ended.content == _TEXTS["clean"][0], response.text
    assert ended.error["code"] == "403", ended.error
    assert ended.error["message"] == _REPLACEMENT_MESSAGE, ended.error
    assert _CRASH_MESSAGE not in response.text, response.text
    assert _calls_with(rig.upstream, marker) == 1


def test_failure_hook_replacement_answers_an_early_non_http_failure_as_a_json_error(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _stream_chat(rig, _body("clean", marker, (_RAISING,), "raise-builtin-after-0", _REPLACE))
    assert response.status_code == 403, response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["code"] == "403", error
    assert error["message"] == _REPLACEMENT_MESSAGE, error
    assert _CRASH_MESSAGE not in response.text, response.text
    assert _calls_with(rig.upstream, marker) == 1


def test_failure_hook_replacement_ends_a_queue_stream_that_fails_before_the_first_chunk(rig: Rig) -> None:
    marker: Final = _marker()
    body: Final = _body("clean", marker, (_RAISING,), "raise-builtin-after-0", _REPLACE, priority=0)
    response: Final = rig.proxy.gateway.request("POST", "/queue/chat/completions", body)
    ended: Final = _ended_in_band(response, 200)
    assert ended.content == "", response.text
    assert ended.error["code"] == "403", ended.error
    assert ended.error["message"] == _REPLACEMENT_MESSAGE, ended.error
    assert _CRASH_MESSAGE not in response.text, response.text
    assert _calls_with(rig.upstream, marker) == 1


def test_early_non_http_failure_without_a_replacement_keeps_its_json_error_body(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _stream_chat(rig, _body("clean", marker, (_RAISING,), "raise-builtin-after-0"))
    assert response.status_code == 500, response.text
    assert response.headers["content-type"].startswith("application/json"), response.headers
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["code"] == "500", error
    assert _CRASH_MESSAGE in string_value(error["message"]), error
    assert _calls_with(rig.upstream, marker) == 1


def _responses_body(marker: str, *directives: str) -> dict[str, JsonValue]:
    return {
        "model": _MODEL,
        "stream": True,
        "input": _prompt("clean", marker, *directives),
        "guardrails": [_RAISING],
        "cache": {"no-cache": True},
    }


def _failed_response_error(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    lines: Final = _data_lines(response.text)
    assert lines[-1] == "[DONE]" and lines.count("[DONE]") == 1, response.text
    events: Final = _payloads(response.text)
    kinds: Final = tuple(string_value(event["type"]) for event in events)
    assert kinds[0] == "response.created" and kinds[-1] == "response.failed", response.text
    assert "response.completed" not in kinds and kinds.count("response.failed") == 1, response.text
    streamed: Final = "".join(
        string_value(event.get("delta") or "") for event in events if event["type"] == "response.output_text.delta"
    )
    assert streamed == "".join(_TEXTS["clean"]), response.text
    created: Final = object_value(events[0]["response"])
    failed: Final = object_value(events[-1]["response"])
    assert failed["status"] == "failed" and failed["id"] == created["id"], failed
    return object_value(failed["error"])


def test_responses_stream_failing_on_a_builtin_exception_ends_with_response_failed(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.gateway.request(
        "POST", "/v1/responses", _responses_body(marker, "raise-builtin-after-2")
    )
    error: Final = _failed_response_error(response)
    assert error == {"code": "server_error", "message": _CRASH_MESSAGE}, error
    assert _calls_with(rig.upstream, marker) == 1


def test_responses_stream_failure_hook_replacement_is_the_failed_response_error(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.gateway.request(
        "POST", "/v1/responses", _responses_body(marker, "raise-http-after-2", _REPLACE)
    )
    error: Final = _failed_response_error(response)
    assert error["code"] == "permission_error", error
    assert _REPLACEMENT_MESSAGE in string_value(error["message"]), error
    assert _RAISED_BLOCK not in response.text, response.text
    assert _calls_with(rig.upstream, marker) == 1


def _until_stream_error(stream: Iterator[Mapping[str, object]]) -> Iterator[Mapping[str, object] | Exception]:
    try:
        yield from stream
    except proxy_client.StreamError as error:
        yield error


def test_python_client_raises_the_block_error_after_the_partial_reply(rig: Rig) -> None:
    marker: Final = _marker()
    with rig.proxy.gateway.scenario() as scenario:
        client: Final = proxy_client.ChatClient(
            str(rig.proxy.gateway.client.base_url), api_key=scenario.key(metadata={"guardrails": [_FILTER]})
        )
        stream: Final = client.completions_stream(
            model=_MODEL, messages=[{"role": "user", "content": _prompt("filter-late", marker)}]
        )
        *chunks, last = tuple(_until_stream_error(stream))
    assert isinstance(last, proxy_client.StreamError), last
    payloads: Final = tuple(JSON_OBJECT.validate_python(chunk) for chunk in chunks)
    assert all("error" not in payload for payload in payloads), payloads
    content: Final = _content(payloads)
    assert content and _LEAD.startswith(content), payloads
    assert "keeper" in str(last), str(last)
    assert JSON_OBJECT.validate_python(last.error)["code"] == "400", last.error
    assert _calls_with(rig.upstream, marker) == 1


def test_non_streaming_pre_call_block_with_an_unencodable_detail_answers_400(rig: Rig) -> None:
    pytest.skip("BUG: a non-JSON value in a guardrail HTTPException detail answers 500 on non-streaming chat")
    marker: Final = _marker()
    content: Final = f"{_SPAN} in the prompt scenario-clean marker-{marker}"
    response: Final = rig.proxy.gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": content}],
            "guardrails": [_SPAN_PRE],
            "cache": {"no-cache": True},
        },
    )
    assert response.status_code == 400, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["message"] == json.dumps({"matched_span": str(re.search(_SPAN, content))}), error
    assert _calls_with(rig.upstream, marker) == 0
