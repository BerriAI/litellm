import asyncio
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final, Literal
from urllib.parse import unquote

import httpx
import pytest
from integration._support.client import Gateway, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.timing import assert_within_factor_of_control
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm

Route = Literal["/v1/systemone", "/systemone", "/v1/decisions", "/decisions"]

_ROUTES: Final[tuple[Route, ...]] = ("/v1/systemone", "/systemone", "/v1/decisions", "/decisions")
_API_KEY: Final = "synthetic-cloudflare-key"
_CANONICAL: Final = "cloudflare/@cf/cloudflare/clef-omni"
_ALIAS: Final = "cloudflare/clef-omni"
_COST_MAP_KEY: Final = "cloudflare/@cf/cloudflare/clef-omni"
_UPSTREAM_PATH: Final = "/ai/run/@cf/cloudflare/clef-omni"
_WAV: Final = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
_MP4: Final = "data:video/mp4;base64,AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQ=="
_WAV_OBJECT: Final[dict[str, JsonValue]] = {"content_type": "audio/wav", "base64": "UklGRiQAAABXQVZF"}
_MP4_OBJECT: Final[dict[str, JsonValue]] = {"content_type": "video/mp4", "base64": "AAAAHGZ0eXBpc29t"}
_PARAMETERIZED_WAV_OBJECT: Final[dict[str, JsonValue]] = {"content_type": "audio/wav;rate=16000", "base64": "UklGRg=="}
_INPUT_TOKENS: Final = 537
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_SYSTEM_ONE_QUESTIONS: Final[dict[str, JsonValue]] = {
    "pineapple": {"type": "noul", "instructions": "Does the audio say pineapple?"},
    "red": {"type": "noul", "instructions": "Is the video red?"},
}
_OPENAI_QUESTIONS: Final[list[JsonValue]] = [
    {"type": "predicate", "name": "pineapple", "instructions": "Does the audio say pineapple?"},
    {"type": "predicate", "name": "red", "instructions": "Is the video red?"},
]
_RED: Final = 0.97
_SPEND_QUERY: Final = (
    'SELECT spend, status, call_type, model_group, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
    "WHERE request_id = %s"
)
_WORST_CASE_LENGTH: Final = 1_048_576


@dataclass(frozen=True, slots=True)
class _Call:
    route: Route
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body[:200]
    return found.group(1)


def _state(marker: str) -> str:
    return f"Does the clip say pineapple? marker-{marker}"


def _probability(marker: str) -> float:
    return round(int(marker[:6], 16) / 0x1000000, 7)


def _upstream_reply(marker: str) -> dict[str, JsonValue]:
    return {
        "result": {
            "model": "clef-omni",
            "answers": {
                "pineapple": {"type": "noul", "noul": _probability(marker)},
                "red": {"type": "noul", "noul": _RED},
            },
            "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
        },
        "success": True,
        "errors": [],
        "messages": [],
    }


def _openai_decider_reply(marker: str) -> dict[str, JsonValue]:
    return {
        "model": "gpt-6-luna",
        "answers": [
            {"type": "predicate", "name": "pineapple", "probability": _probability(marker)},
            {"type": "predicate", "name": "red", "probability": _RED},
        ],
        "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0, "total_tokens": _INPUT_TOKENS},
    }


def _echo(request: Request) -> Reply:
    marker: Final = _marker_of(request)
    if request.target.endswith("/v1/decisions"):
        return Reply(body=json.dumps(_openai_decider_reply(marker)).encode())
    return Reply(body=json.dumps(_upstream_reply(marker)).encode())


def _speaks_system_one(route: Route) -> bool:
    return route.endswith("systemone")


def _body(route: Route, model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    if _speaks_system_one(route):
        return {"model": model, "state": _state(marker), "questions": _SYSTEM_ONE_QUESTIONS, **extra}
    return {"model": model, "input": _state(marker), "questions": _OPENAI_QUESTIONS, **extra}


def _expected_response(route: Route, marker: str) -> dict[str, JsonValue]:
    if _speaks_system_one(route):
        return {
            "model": "clef-omni",
            "answers": {
                "pineapple": {"type": "noul", "noul": _probability(marker)},
                "red": {"type": "noul", "noul": _RED},
            },
            "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
        }
    return {
        "model": "clef-omni",
        "answers": [
            {"type": "predicate", "name": "pineapple", "probability": _probability(marker)},
            {"type": "predicate", "name": "red", "probability": _RED},
        ],
        "usage": {
            "input_tokens": _INPUT_TOKENS,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": _INPUT_TOKENS,
        },
    }


def _expected_upstream_body(marker: str, **media: JsonValue) -> dict[str, JsonValue]:
    return {"model": "clef-omni", "state": _state(marker), **media, "questions": _SYSTEM_ONE_QUESTIONS}


def _upstream_body(request: Request) -> dict[str, JsonValue]:
    assert request.method == "POST", request.method
    assert unquote(request.target) == _UPSTREAM_PATH, request.target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}", request.headers
    return _JSON_OBJECT.validate_json(request.body)


def _decide(gateway: Gateway, route: Route, model: str, marker: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", route, _body(route, model, marker, **extra))


def _answered_probability(route: Route, text: str) -> JsonValue:
    answers: Final = _JSON_OBJECT.validate_json(text)["answers"]
    if _speaks_system_one(route):
        return object_value(object_value(answers)["pineapple"])["noul"]
    return object_value(list_value(answers)[0])["probability"]


def _price() -> dict[str, JsonValue]:
    cost_map: Final = _JSON_OBJECT.validate_json(Path("model_prices_and_context_window.json").read_bytes())
    priced: Final = cost_map.get(_COST_MAP_KEY)
    assert priced is not None, f"{_COST_MAP_KEY} has no cost map entry"
    return object_value(priced)


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def _expected_spend() -> float:
    return _INPUT_TOKENS * _number(_price()["input_cost_per_token"])


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _spend_rows_for(model: str, expected: int) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: read_rows('SELECT litellm_call_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    assert len({row["litellm_call_id"] for row in rows}) == len(rows) == expected, rows
    return rows


def _call_type(route: Route) -> str:
    return "asystemone" if route.endswith("systemone") else "adecisions"


def _assert_billed(gateway: Gateway, response: httpx.Response, model: str, route: Route) -> None:
    expected: Final = _expected_spend()
    assert math.isclose(float(response.headers["x-litellm-response-cost"]), expected, rel_tol=1e-9), response.headers
    call_id: Final = response.headers["x-litellm-call-id"]
    row: Final = _spend_row(call_id)
    assert (row["status"], row["call_type"], row["model_group"], row["prompt_tokens"], row["completion_tokens"]) == (
        "success",
        _call_type(route),
        model,
        _INPUT_TOKENS,
        0,
    ), row
    assert math.isclose(_number(row["spend"]), expected, rel_tol=1e-9), row
    listed: Final = gateway.request("GET", "/spend/logs", params={"request_id": call_id})
    assert listed.status_code == 200, listed.text
    (reported,) = _JSON_ROWS.validate_json(listed.content)
    assert reported["request_id"] == call_id, reported
    assert (reported["model_group"], reported["prompt_tokens"], reported["completion_tokens"]) == (
        model,
        _INPUT_TOKENS,
        0,
    ), reported
    assert math.isclose(_number(reported["spend"]), expected, rel_tol=1e-9), reported


def _assert_served_with_media(response: httpx.Response, route: Route, marker: str, request: Request) -> None:
    assert response.status_code == 200, response.text
    assert response.json() == _expected_response(route, marker)
    assert _upstream_body(request) == _expected_upstream_body(marker, audio=[_WAV], videos=[_MP4])


@pytest.mark.parametrize("route", _ROUTES)
def test_data_url_clips_reach_clef_omni_on_every_route_and_are_billed(gateway: Gateway, route: Route) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        response: Final = _decide(gateway, route, model, marker, audio=[_WAV], videos=[_MP4])
        (request,) = wire.drain()
        _assert_served_with_media(response, route, marker, request)
        _assert_billed(gateway, response, model, route)


async def test_an_async_caller_on_the_unversioned_decisions_route_gets_its_media_forwarded(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
            response: Final = await client.post(
                "/decisions",
                json=_body("/decisions", model, marker, audio=[_WAV], videos=[_MP4]),
                headers={"Authorization": f"Bearer {gateway.key}"},
            )
        (request,) = wire.drain()
        _assert_served_with_media(response, "/decisions", marker, request)
        _assert_billed(gateway, response, model, "/decisions")


@pytest.mark.parametrize(
    "audio_clip", (_WAV_OBJECT, _PARAMETERIZED_WAV_OBJECT), ids=("plain_content_type", "parameterized_content_type")
)
async def test_object_form_clips_are_forwarded_verbatim(gateway: Gateway, audio_clip: dict[str, JsonValue]) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
            response: Final = await client.post(
                "/v1/systemone",
                json=_body("/v1/systemone", model, marker, audio=[audio_clip], videos=[_MP4_OBJECT]),
                headers={"Authorization": f"Bearer {gateway.key}"},
            )
        assert response.status_code == 200, response.text
        assert response.json() == _expected_response("/v1/systemone", marker)
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body(marker, audio=[audio_clip], videos=[_MP4_OBJECT])


def test_both_clef_omni_spellings_forward_media_and_carry_the_cost_map_price(gateway: Gateway) -> None:
    price: Final = _price()
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        deployments: Final = tuple(
            scenario.model(model=spelling, api_base=wire.url, api_key=_API_KEY) for spelling in (_ALIAS, _CANONICAL)
        )
        for model in deployments:
            marker: Final = uuid.uuid4().hex
            response: Final = _decide(gateway, "/v1/systemone", model, marker, audio=[_WAV], videos=[_MP4])
            (request,) = wire.drain()
            _assert_served_with_media(response, "/v1/systemone", marker, request)
            _assert_billed(gateway, response, model, "/v1/systemone")
        listed: Final = list_value(gateway.get("/model/info")["data"])
        for model in deployments:
            entry: Final = next(object_value(item) for item in listed if object_value(item)["model_name"] == model)
            info: Final = object_value(entry["model_info"])
            assert (
                info.get("input_cost_per_token"),
                info.get("output_cost_per_token"),
                info.get("mode"),
                info.get("max_input_tokens"),
                info.get("supports_audio_input"),
                info.get("supports_video_input"),
            ) == (
                price["input_cost_per_token"],
                price["output_cost_per_token"],
                price["mode"],
                price["max_input_tokens"],
                price["supports_audio_input"],
                price["supports_video_input"],
            ), (model, info)


def test_the_maxima_of_four_audio_and_two_video_clips_are_forwarded(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    audio: Final[list[JsonValue]] = [_WAV, _WAV_OBJECT, _WAV, _WAV_OBJECT]
    videos: Final[list[JsonValue]] = [_MP4, _MP4_OBJECT]
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        response: Final = _decide(gateway, "/v1/systemone", model, marker, audio=audio, videos=videos)
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body(marker, audio=audio, videos=videos)


def test_missing_empty_and_null_media_send_the_same_text_only_body(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        missing: Final = _decide(gateway, "/v1/systemone", model, marker)
        empty: Final = _decide(gateway, "/v1/systemone", model, marker, audio=[], videos=[])
        null: Final = _decide(gateway, "/v1/systemone", model, marker, audio=None, videos=None)
        for response in (missing, empty, null):
            assert response.status_code == 200, response.text
            assert response.json() == _expected_response("/v1/systemone", marker)
        bodies: Final = tuple(_upstream_body(request) for request in wire.drain())
        assert bodies == (_expected_upstream_body(marker),) * 3, bodies


@pytest.mark.parametrize("route", _ROUTES)
def test_null_clips_mean_no_clips_of_that_kind_on_every_route(gateway: Gateway, route: Route) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        both_null: Final = _decide(gateway, route, model, marker, audio=None, videos=None)
        audio_only: Final = _decide(gateway, route, model, marker, audio=[_WAV], videos=None)
        video_only: Final = _decide(gateway, route, model, marker, audio=None, videos=[_MP4])
        for response in (both_null, audio_only, video_only):
            assert response.status_code == 200, response.text
            assert response.json() == _expected_response(route, marker)
        bodies: Final = tuple(_upstream_body(request) for request in wire.drain())
        assert bodies == (
            _expected_upstream_body(marker),
            _expected_upstream_body(marker, audio=[_WAV]),
            _expected_upstream_body(marker, videos=[_MP4]),
        ), bodies


_SDK_QUESTIONS: Final = TypeAdapter(dict[str, dict[str, object]]).validate_python(_SYSTEM_ONE_QUESTIONS)


async def test_positional_sdk_callers_keep_the_api_key_fifth_and_the_api_base_sixth() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire:
        synchronous: Final = litellm.decisions(
            _CANONICAL, None, _SDK_QUESTIONS, None, _API_KEY, wire.url, state=_state(marker), audio=[_WAV]
        )
        asynchronous: Final = await litellm.adecisions(
            _CANONICAL, None, _SDK_QUESTIONS, None, _API_KEY, wire.url, state=_state(marker), videos=[_MP4]
        )
        for response in (synchronous, asynchronous):
            assert response.model_dump(mode="json") == _expected_response("/v1/systemone", marker)
        bodies: Final = tuple(_upstream_body(request) for request in wire.drain())
        assert bodies == (
            _expected_upstream_body(marker, audio=[_WAV]),
            _expected_upstream_body(marker, videos=[_MP4]),
        ), bodies


_MALFORMED: Final[tuple[tuple[str, dict[str, JsonValue], str], ...]] = (
    ("audio_int", {"audio": 5}, "audio"),
    ("audio_list_of_ints", {"audio": [1, 2]}, "audio"),
    ("audio_empty_string", {"audio": ""}, "audio"),
    ("audio_5kb_garbage", {"audio": ["a" * 5000]}, "pattern"),
    ("videos_object_missing_base64", {"videos": [{"content_type": "video/mp4"}]}, "base64"),
    ("videos_object_extra_key", {"videos": [{**_MP4_OBJECT, "url": "https://example.invalid/clip.mp4"}]}, "url"),
    ("image_data_url_in_audio", {"audio": ["data:image/png;base64,iVBORw0KGgo="]}, "pattern"),
    (
        "parameterized_audio_data_url",
        {"audio": ["data:audio/webm;codecs=opus;base64,GkXfo0AgQoaBAUL3gQFC8oEEQvOBCA=="]},
        "pattern",
    ),
    ("five_audio_clips", {"audio": [_WAV] * 5}, "at most 4"),
    ("three_videos", {"videos": [_MP4] * 3}, "at most 2"),
    (
        "audio_object_with_image_content_type",
        {"audio": [{"content_type": "image/png", "base64": "iVBORw0KGgo="}]},
        "pattern",
    ),
    ("video_object_with_empty_base64", {"videos": [{"content_type": "video/mp4", "base64": ""}]}, "at least 1"),
    (
        "audio_object_content_type_with_a_space",
        {"audio": [{"content_type": "audio/wav x", "base64": "UklGRg=="}]},
        "pattern",
    ),
)


@pytest.mark.parametrize(("label", "extra", "fragment"), _MALFORMED, ids=[label for label, _, _ in _MALFORMED])
def test_malformed_clips_are_refused_before_any_upstream_call(
    gateway: Gateway, label: str, extra: dict[str, JsonValue], fragment: str
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        response: Final = _decide(gateway, "/v1/systemone", model, marker, **extra)
        assert response.status_code == 400, (label, response.text)
        assert "Invalid Decisions request" in response.text and fragment in response.text, (label, response.text)
        assert wire.drain() == (), label


def _raw_system_one(gateway: Gateway, model: str, marker: str, audio_fragment: str) -> httpx.Response:
    raw: Final = (
        f'{{"model": {json.dumps(model)}, "state": {json.dumps(_state(marker))}, '
        f'"questions": {json.dumps(_SYSTEM_ONE_QUESTIONS)}, {audio_fragment}}}'
    )
    return gateway.client.post(
        "/v1/systemone",
        content=raw.encode(),
        headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
    )


def test_a_duplicated_audio_key_is_judged_by_its_last_value(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    image: Final = '"data:image/png;base64,iVBORw0KGgo="'
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        refused: Final = _raw_system_one(gateway, model, marker, f'"audio": [{json.dumps(_WAV)}], "audio": [{image}]')
        assert refused.status_code == 400, refused.text
        assert "Invalid Decisions request" in refused.text and "pattern" in refused.text, refused.text
        assert wire.drain() == ()
        accepted: Final = _raw_system_one(gateway, model, marker, f'"audio": [{image}], "audio": [{json.dumps(_WAV)}]')
        assert accepted.status_code == 200, accepted.text
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body(marker, audio=[_WAV])


_UNSUPPORTED: Final[tuple[tuple[str, Route], ...]] = (
    ("cloudflare/clef-flash", "/v1/systemone"),
    ("cloudflare/clef", "/v1/systemone"),
    ("cloudflare/@cf/cloudflare/clef-flash", "/v1/decisions"),
    ("openai/gpt-6-luna", "/v1/systemone"),
    ("openai/gpt-6-luna", "/v1/decisions"),
)


@pytest.mark.parametrize(("spelling", "route"), _UNSUPPORTED, ids=[f"{m}@{r}" for m, r in _UNSUPPORTED])
def test_media_on_a_model_without_support_is_refused_before_any_upstream_call(
    gateway: Gateway, spelling: str, route: Route
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=spelling, api_base=wire.url, api_key=_API_KEY)
        response: Final = _decide(gateway, route, model, marker, audio=[_WAV])
        assert response.status_code == 400, response.text
        assert "does not support parameters: ['audio']" in response.text, response.text
        assert "drop_params" in response.text, response.text
        assert wire.drain() == ()
        text_only: Final = _decide(gateway, route, model, marker)
        assert text_only.status_code == 200, text_only.text
        assert len(wire.drain()) == 1


@pytest.mark.parametrize("route", ("/v1/systemone", "/v1/decisions"))
def test_media_on_a_model_without_support_is_dropped_when_the_deployment_drops_params(
    gateway: Gateway, route: Route
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="cloudflare/clef-flash", api_base=wire.url, api_key=_API_KEY, drop_params=True
        )
        text_only: Final = _decide(gateway, route, model, marker)
        assert text_only.status_code == 200, text_only.text
        (text_only_request,) = wire.drain()
        with_clips: Final = _decide(gateway, route, model, marker, audio=[_WAV], videos=[_MP4])
        assert with_clips.status_code == 200, with_clips.text
        (dropped_request,) = wire.drain()
        text_only_body: Final = _JSON_OBJECT.validate_json(text_only_request.body)
        assert "audio" not in text_only_body and "videos" not in text_only_body, text_only_body
        assert _JSON_OBJECT.validate_json(dropped_request.body) == text_only_body
        assert with_clips.json() == text_only.json()


_MALFORMED_CLIPS: Final[dict[str, JsonValue]] = {
    "audio": ["https://example.invalid/clip.wav"],
    "videos": [{"content_type": "video/mp4"}],
}
_NO_MEDIA_SPELLINGS: Final = ("cloudflare/clef-flash", "cloudflare/clef")


def _assert_refused_before_any_upstream_call(response: httpx.Response, fragment: str, wire_received: int) -> None:
    assert response.status_code == 400, response.text
    assert fragment in response.text, response.text
    assert wire_received == 0


@pytest.mark.parametrize("route", ("/v1/systemone", "/v1/decisions"))
def test_malformed_clips_on_a_model_without_support_are_refused_unless_the_deployment_drops_params(
    gateway: Gateway, route: Route
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model="cloudflare/clef-flash", api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(
            model="cloudflare/clef-flash", api_base=wire.url, api_key=_API_KEY, drop_params=True
        )
        refused: Final = _decide(gateway, route, strict, marker, **_MALFORMED_CLIPS)
        _assert_refused_before_any_upstream_call(refused, "Invalid Decisions request", len(wire.drain()))
        text_only: Final = _decide(gateway, route, dropping, marker)
        assert text_only.status_code == 200, text_only.text
        (text_only_request,) = wire.drain()
        dropped: Final = _decide(gateway, route, dropping, marker, **_MALFORMED_CLIPS)
        assert dropped.status_code == 200, dropped.text
        (dropped_request,) = wire.drain()
        assert _JSON_OBJECT.validate_json(dropped_request.body) == _JSON_OBJECT.validate_json(text_only_request.body)
        assert dropped.json() == text_only.json()


def test_a_malformed_clip_on_a_media_model_is_refused_even_when_the_deployment_drops_params(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        refused: Final = _decide(gateway, "/v1/systemone", model, marker, **_MALFORMED_CLIPS)
        _assert_refused_before_any_upstream_call(refused, "Invalid Decisions request", len(wire.drain()))
        served: Final = _decide(gateway, "/v1/systemone", model, marker, audio=[_WAV], videos=[_MP4])
        (request,) = wire.drain()
        _assert_served_with_media(served, "/v1/systemone", marker, request)


def _advertised_media_flags(gateway: Gateway, deployments: Mapping[str, str]) -> dict[str, tuple[JsonValue, JsonValue]]:
    listed: Final = list_value(gateway.get("/model/info")["data"])
    entries: Final = {
        string_value(object_value(item)["model_name"]): object_value(object_value(item)["model_info"])
        for item in listed
    }
    return {
        spelling: (entries[name].get("supports_audio_input"), entries[name].get("supports_video_input"))
        for spelling, name in deployments.items()
    }


def test_model_info_advertises_media_support_only_on_clef_omni_deployments(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        deployments: Final = {
            spelling: scenario.model(model=spelling, api_base=wire.url, api_key=_API_KEY)
            for spelling in (_ALIAS, _CANONICAL, *_NO_MEDIA_SPELLINGS)
        }
        flags: Final = _advertised_media_flags(gateway, deployments)
        assert flags[_ALIAS] == (True, True), flags
        assert flags[_CANONICAL] == (True, True), flags
        advertised_without_media: Final = tuple(any(flags[spelling]) for spelling in _NO_MEDIA_SPELLINGS)
        assert advertised_without_media == (False, False), flags


def _unique_backend(kind: str) -> str:
    return f"@cf/cloudflare/clef-{kind}-{uuid.uuid4().hex[:12]}"


def _body_sent_to(request: Request, backend: str) -> dict[str, JsonValue]:
    assert request.method == "POST", request.method
    assert unquote(request.target) == f"/ai/run/{backend}", request.target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}", request.headers
    return _JSON_OBJECT.validate_json(request.body)


def _expected_backend_body(backend: str, marker: str, **media: JsonValue) -> dict[str, JsonValue]:
    return {**_expected_upstream_body(marker, **media), "model": backend.rsplit("/", 1)[-1]}


def test_a_deployment_advertising_only_audio_forwards_audio_and_refuses_or_drops_video(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    backend: Final = _unique_backend("audio")
    advertised: Final[dict[str, JsonValue]] = {"mode": "evaluation", "supports_audio_input": True}
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(
            model=f"cloudflare/{backend}", api_base=wire.url, api_key=_API_KEY, model_info=advertised
        )
        dropping: Final = scenario.model(
            model=f"cloudflare/{backend}", api_base=wire.url, api_key=_API_KEY, drop_params=True, model_info=advertised
        )
        with_audio: Final = _decide(gateway, "/v1/systemone", strict, marker, audio=[_WAV])
        assert with_audio.status_code == 200, with_audio.text
        (audio_request,) = wire.drain()
        assert _body_sent_to(audio_request, backend) == _expected_backend_body(backend, marker, audio=[_WAV])
        with_video: Final = _decide(gateway, "/v1/systemone", strict, marker, videos=[_MP4])
        _assert_refused_before_any_upstream_call(
            with_video, "does not support parameters: ['videos']", len(wire.drain())
        )
        assert "drop_params" in with_video.text, with_video.text
        dropped: Final = _decide(gateway, "/v1/systemone", dropping, marker, audio=[_WAV], videos=[_MP4])
        assert dropped.status_code == 200, dropped.text
        (dropped_request,) = wire.drain()
        assert _body_sent_to(dropped_request, backend) == _expected_backend_body(backend, marker, audio=[_WAV])


def test_a_chat_mode_deployment_advertising_media_is_not_a_decisions_media_model(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    backend: Final = _unique_backend("chat")
    advertised: Final[dict[str, JsonValue]] = {
        "mode": "chat",
        "supports_audio_input": True,
        "supports_video_input": True,
    }
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(
            model=f"cloudflare/{backend}", api_base=wire.url, api_key=_API_KEY, model_info=advertised
        )
        dropping: Final = scenario.model(
            model=f"cloudflare/{backend}", api_base=wire.url, api_key=_API_KEY, drop_params=True, model_info=advertised
        )
        refused: Final = _decide(gateway, "/v1/systemone", strict, marker, audio=[_WAV], videos=[_MP4])
        _assert_refused_before_any_upstream_call(refused, "does not support parameters", len(wire.drain()))
        assert "'audio'" in refused.text and "'videos'" in refused.text, refused.text
        assert "drop_params" in refused.text, refused.text
        text_only: Final = _decide(gateway, "/v1/systemone", dropping, marker)
        assert text_only.status_code == 200, text_only.text
        (text_only_request,) = wire.drain()
        assert _body_sent_to(text_only_request, backend) == _expected_backend_body(backend, marker)
        dropped: Final = _decide(gateway, "/v1/systemone", dropping, marker, audio=[_WAV], videos=[_MP4])
        assert dropped.status_code == 200, dropped.text
        (dropped_request,) = wire.drain()
        assert _body_sent_to(dropped_request, backend) == _expected_backend_body(backend, marker)
        assert dropped.json() == text_only.json()


def test_clips_on_an_openai_chat_model_are_refused_or_dropped_never_forwarded(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model="openai/gpt-audio", api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(
            model="openai/gpt-audio", api_base=wire.url, api_key=_API_KEY, drop_params=True
        )
        refused: Final = _decide(gateway, "/v1/decisions", strict, marker, audio=[_WAV])
        _assert_refused_before_any_upstream_call(refused, "does not support parameters: ['audio']", len(wire.drain()))
        assert "drop_params" in refused.text, refused.text
        text_only: Final = _decide(gateway, "/v1/decisions", dropping, marker)
        assert text_only.status_code == 200, text_only.text
        (text_only_request,) = wire.drain()
        assert unquote(text_only_request.target) == "/v1/decisions", text_only_request.target
        text_only_body: Final = _JSON_OBJECT.validate_json(text_only_request.body)
        assert "audio" not in text_only_body and "videos" not in text_only_body, text_only_body
        dropped: Final = _decide(gateway, "/v1/decisions", dropping, marker, audio=[_WAV], videos=[_MP4])
        assert dropped.status_code == 200, dropped.text
        (dropped_request,) = wire.drain()
        assert _JSON_OBJECT.validate_json(dropped_request.body) == text_only_body
        assert dropped.json() == text_only.json()


_UPSTREAM_REJECTIONS: Final[tuple[tuple[int, str], ...]] = (
    (422, "Each audio entry must be a data URL or an object with content_type and base64"),
    (413, "Request entity too large"),
)


@pytest.mark.parametrize(("status", "message"), _UPSTREAM_REJECTIONS, ids=("422", "413"))
def test_an_upstream_rejection_of_a_media_request_reaches_the_caller_like_a_text_only_one(
    gateway: Gateway, status: int, message: str
) -> None:
    marker: Final = uuid.uuid4().hex
    rejection: Final = json.dumps(
        {"result": {}, "success": False, "errors": [{"code": 5006, "message": message}], "messages": []}
    ).encode()
    with wire_server(lambda _: Reply(status=status, body=rejection)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        text_only: Final = _decide(gateway, "/v1/systemone", model, marker, num_retries=0)
        with_media: Final = _decide(gateway, "/v1/systemone", model, marker, num_retries=0, audio=[_WAV], videos=[_MP4])
        for response in (text_only, with_media):
            assert response.status_code >= 400, response.text
            assert message in response.text, response.text
            assert "answers" not in response.text, response.text
        assert with_media.status_code == text_only.status_code, (text_only.status_code, with_media.status_code)
        first, second = (_upstream_body(request) for request in wire.drain())
        assert first == _expected_upstream_body(marker)
        assert second == _expected_upstream_body(marker, audio=[_WAV], videos=[_MP4])


def test_an_unauthenticated_media_request_is_refused_before_validation(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        for route in _ROUTES:
            response: Final = gateway.client.post(route, json=_body(route, model, marker, audio=[_WAV], videos=[_MP4]))
            assert response.status_code == 401, (route, response.text)
            assert "Authentication Error" in response.text, response.text
        assert wire.drain() == ()


def test_unrelated_chat_keeps_working_after_a_run_of_refused_media_requests(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        control: Final = scenario.model()
        for _, extra, _ in _MALFORMED:
            assert _decide(gateway, "/v1/systemone", model, marker, **extra).status_code == 400
        assert wire.drain() == ()
        assert gateway.chat(control)["choices"], "the control deployment stopped answering"
        accepted: Final = _decide(gateway, "/v1/systemone", model, marker, audio=[_WAV])
        assert accepted.status_code == 200, accepted.text
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body(marker, audio=[_WAV])


def test_repeated_and_clip_changed_requests_each_reach_the_upstream_and_are_each_billed(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        first: Final = _decide(gateway, "/v1/systemone", model, marker, audio=[_WAV], videos=[_MP4])
        second: Final = _decide(gateway, "/v1/systemone", model, marker, audio=[_WAV], videos=[_MP4])
        swapped: Final = _decide(gateway, "/v1/systemone", model, marker, audio=[_WAV_OBJECT], videos=[_MP4])
        responses: Final = (first, second, swapped)
        for response in responses:
            assert response.status_code == 200, response.text
            assert response.json() == _expected_response("/v1/systemone", marker)
        call_ids: Final = tuple(response.headers["x-litellm-call-id"] for response in responses)
        assert len(set(call_ids)) == 3, call_ids
        bodies: Final = tuple(_upstream_body(request) for request in wire.drain())
        assert bodies == (
            _expected_upstream_body(marker, audio=[_WAV], videos=[_MP4]),
            _expected_upstream_body(marker, audio=[_WAV], videos=[_MP4]),
            _expected_upstream_body(marker, audio=[_WAV_OBJECT], videos=[_MP4]),
        ), bodies
        for call_id in call_ids:
            assert _spend_row(call_id)["status"] == "success", call_id


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    response: Final = await client.post(
        call.route,
        json=_body(call.route, model, call.marker, audio=[_WAV], videos=[_MP4]),
        headers={"Authorization": f"Bearer {key}"},
    )
    return _Served(call=call, status=response.status_code, text=response.text)


async def _burst(base_url: str, key: str, model: str, calls: Sequence[_Call]) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, key, model, call) for call in calls)))


def _assert_answered_with_its_own_probability(served: _Served) -> None:
    assert served.status == 200, served.text
    assert _answered_probability(served.call.route, served.text) == _probability(served.call.marker), served.text


def _assert_each_landed_once_with_media(received: tuple[Request, ...], calls: Sequence[_Call]) -> None:
    bodies: Final = {_marker_of(request): _upstream_body(request) for request in received}
    assert len(bodies) == len(received) == len(calls), (len(bodies), len(received), len(calls))
    for call in calls:
        assert bodies[call.marker] == _expected_upstream_body(call.marker, audio=[_WAV], videos=[_MP4]), call


async def test_a_concurrent_burst_across_the_four_routes_lands_each_request_once_with_its_media(
    gateway: Gateway,
) -> None:
    calls: Final = tuple(_Call(route=_ROUTES[index % len(_ROUTES)], marker=uuid.uuid4().hex) for index in range(20))
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 20
        for item in served:
            _assert_answered_with_its_own_probability(item)
        _assert_each_landed_once_with_media(wire.drain(), calls)
        rows: Final = _spend_rows_for(model, 20)
        assert [row["status"] for row in rows] == ["success"] * 20, rows


def test_openapi_documents_the_media_fields_on_every_decisions_route(gateway: Gateway) -> None:
    paths: Final = object_value(gateway.get("/openapi.json")["paths"])
    for route in _ROUTES:
        post: Final = object_value(object_value(paths[route])["post"])
        content: Final = object_value(object_value(post["requestBody"])["content"])
        schema: Final = object_value(object_value(content["application/json"])["schema"])
        properties: Final = object_value(schema["properties"])
        assert "audio" in properties and "videos" in properties, (route, sorted(properties))
        assert object_value(properties["audio"])["maxItems"] == 4, (route, properties["audio"])
        assert object_value(properties["videos"])["maxItems"] == 2, (route, properties["videos"])


def _probe_liveliness(base_url: str, stop: threading.Event, latencies: SimpleQueue[float]) -> None:
    with httpx.Client(base_url=base_url, timeout=10, trust_env=False) as client:
        while not stop.is_set():
            started: Final = time.perf_counter()
            response: Final = client.get("/health/liveliness")
            latencies.put(time.perf_counter() - started)
            assert response.status_code == 200, response.text


def _timed_refusal_under_liveliness_probes(
    gateway: Gateway, model: str, clip: str
) -> tuple[httpx.Response, float, tuple[float, ...]]:
    stop: Final = threading.Event()
    latencies: Final[SimpleQueue[float]] = SimpleQueue()
    probe: Final = threading.Thread(
        target=_probe_liveliness, args=(str(gateway.client.base_url), stop, latencies), daemon=True
    )
    probe.start()
    started: Final = time.perf_counter()
    with httpx.Client(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        response: Final = client.post(
            "/v1/systemone",
            json=_body("/v1/systemone", model, uuid.uuid4().hex, audio=[clip]),
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
    elapsed: Final = time.perf_counter() - started
    stop.set()
    probe.join(timeout=15)
    return response, elapsed, tuple(latencies.get_nowait() for _ in range(latencies.qsize()))


def test_the_data_url_pattern_worst_case_is_refused_in_bounded_time_while_liveliness_answers(
    gateway: Gateway,
) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=wire.url, api_key=_API_KEY)
        tenth_response, tenth_elapsed, _ = _timed_refusal_under_liveliness_probes(
            gateway, model, "data:audio/" + "a" * (_WORST_CASE_LENGTH // 10)
        )
        response, elapsed, observed = _timed_refusal_under_liveliness_probes(
            gateway, model, "data:audio/" + "a" * _WORST_CASE_LENGTH
        )
        timing: Final = {
            "tenth_seconds": round(tenth_elapsed, 3),
            "classifier_seconds": round(elapsed, 3),
            "slowest_liveliness_seconds": round(max(observed), 3),
        }
        assert tenth_response.status_code == 400, tenth_response.text[:300]
        assert_within_factor_of_control(elapsed, control=tenth_elapsed, factor=25)
        assert observed and max(observed) < max(1.0, elapsed / 2), timing
        assert response.status_code == 400, (timing, response.text[:300])
        assert "pattern" in response.text, response.text[:300]
        assert wire.drain() == (), timing
