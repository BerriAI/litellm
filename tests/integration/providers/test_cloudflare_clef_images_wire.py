import base64
import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Literal
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
import pytest
import yaml
from integration._support.bedrock_runtime_peer import PNG_1X1
from integration._support.client import Gateway, eventually, gateway_from_environment, list_value, object_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.timing import assert_within_factor_of_control, timed
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

Route = Literal["/v1/systemone", "/systemone", "/v1/decisions", "/decisions"]

_ROUTES: Final[tuple[Route, ...]] = ("/v1/systemone", "/systemone", "/v1/decisions", "/decisions")
_API_KEY: Final = "synthetic-cloudflare-key"
_CLEF: Final = "cloudflare/clef"
_ALLOWLISTED_MODEL: Final = "clef-images-allowlisted"
_INPUT_TOKENS: Final = 412
_PNG_BASE64: Final = base64.b64encode(PNG_1X1).decode()
_PNG_DATA_URL: Final = f"data:image/png;base64,{_PNG_BASE64}"
_JPEG_DATA_URL: Final = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="
_PNG_OBJECT: Final[dict[str, JsonValue]] = {"content_type": "image/png", "base64": _PNG_BASE64}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_SYSTEM_ONE_QUESTIONS: Final[dict[str, JsonValue]] = {"red": {"type": "noul", "instructions": "Is the screenshot red?"}}
_OPENAI_QUESTIONS: Final[list[JsonValue]] = [
    {"type": "predicate", "name": "red", "instructions": "Is the screenshot red?"}
]
_SPEND_QUERY: Final = (
    'SELECT spend, status, call_type, model_group, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
    "WHERE request_id = %s"
)
_OWNED_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
_COLD_OPENAPI_SECONDS: Final = 120
_SLOW_IMAGE_RELEASE: Final = threading.Event()
_IMAGE_INPUT_RECORDER: Final = """
import json
import os

import httpx

from litellm.integrations.custom_logger import CustomLogger


class ImageInputRecorder(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload = {"call_id": kwargs.get("litellm_call_id"), "input": kwargs.get("input")}
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            await client.post(os.environ["IMAGE_INPUT_SINK"], content=json.dumps(payload, default=str))


instance = ImageInputRecorder()
"""


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body[:200]
    return found.group(1)


def _state(marker: str) -> str:
    return f"Is the checkout screenshot red? marker-{marker}"


def _probability(marker: str) -> float:
    return round(int(marker[:6], 16) / 0x1000000, 7)


def _clef_name(request: Request) -> str:
    return unquote(request.target).rsplit("/", 1)[-1]


def _clef_reply(request: Request) -> Reply:
    reply: Final[dict[str, JsonValue]] = {
        "result": {
            "model": _clef_name(request),
            "answers": {"red": {"type": "noul", "noul": _probability(_marker_of(request))}},
            "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
        },
        "success": True,
        "errors": [],
        "messages": [],
    }
    return Reply(body=json.dumps(reply).encode())


def _speaks_system_one(route: Route) -> bool:
    return route.endswith("systemone")


def _openai_input(text: str, images: Sequence[JsonValue]) -> list[JsonValue]:
    text_parts: Final[tuple[JsonValue, ...]] = ({"type": "input_text", "text": text},) if text else ()
    image_parts: Final = tuple({"type": "input_image", "image_url": image} for image in images)
    return [{"role": "user", "content": [*text_parts, *image_parts]}]


def _body(
    route: Route, model: str, marker: str, images: Sequence[JsonValue], **extra: JsonValue
) -> dict[str, JsonValue]:
    if _speaks_system_one(route):
        return {
            "model": model,
            "state": _state(marker),
            "questions": _SYSTEM_ONE_QUESTIONS,
            "images": list(images),
            **extra,
        }
    return {"model": model, "input": _openai_input(_state(marker), images), "questions": _OPENAI_QUESTIONS, **extra}


def _text_only_body(route: Route, model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    if _speaks_system_one(route):
        return {"model": model, "state": _state(marker), "questions": _SYSTEM_ONE_QUESTIONS, **extra}
    return {"model": model, "input": _state(marker), "questions": _OPENAI_QUESTIONS, **extra}


def _expected_response(route: Route, clef_name: str, marker: str) -> dict[str, JsonValue]:
    if _speaks_system_one(route):
        return {
            "model": clef_name,
            "answers": {"red": {"type": "noul", "noul": _probability(marker)}},
            "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
        }
    return {
        "model": clef_name,
        "answers": [{"type": "predicate", "name": "red", "probability": _probability(marker)}],
        "usage": {
            "input_tokens": _INPUT_TOKENS,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": _INPUT_TOKENS,
        },
    }


def _expected_upstream_body(clef_name: str, state: str, images: Sequence[JsonValue] = ()) -> dict[str, JsonValue]:
    return {
        "model": clef_name,
        "state": state,
        "questions": _SYSTEM_ONE_QUESTIONS,
        **({"images": list(images)} if images else {}),
    }


def _upstream_body(request: Request, clef_name: str = "clef") -> dict[str, JsonValue]:
    assert request.method == "POST", (request.method, request.target)
    assert unquote(request.target) == f"/ai/run/@cf/cloudflare/{clef_name}", request.target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}", request.headers
    return _JSON_OBJECT.validate_json(request.body)


def _decide(
    gateway: Gateway, route: Route, model: str, marker: str, images: Sequence[JsonValue], **extra: JsonValue
) -> httpx.Response:
    return gateway.request("POST", route, _body(route, model, marker, images, **extra))


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def _expected_spend(cost_map_key: str) -> float:
    cost_map: Final = _JSON_OBJECT.validate_json(Path("model_prices_and_context_window.json").read_bytes())
    priced: Final = cost_map.get(cost_map_key)
    assert priced is not None, f"{cost_map_key} has no cost map entry"
    return _INPUT_TOKENS * _number(object_value(priced)["input_cost_per_token"])


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _call_type(route: Route) -> str:
    return "asystemone" if _speaks_system_one(route) else "adecisions"


def _assert_billed(gateway: Gateway, response: httpx.Response, model: str, route: Route, cost_map_key: str) -> None:
    expected: Final = _expected_spend(cost_map_key)
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
    assert (reported["request_id"], reported["model_group"], reported["prompt_tokens"]) == (
        call_id,
        model,
        _INPUT_TOKENS,
    ), reported
    assert math.isclose(_number(reported["spend"]), expected, rel_tol=1e-9), reported


def _images_for(route: Route) -> list[JsonValue]:
    return [_PNG_DATA_URL, _PNG_OBJECT] if _speaks_system_one(route) else [_PNG_DATA_URL, _JPEG_DATA_URL]


@pytest.mark.parametrize("route", _ROUTES)
def test_images_reach_clef_in_its_images_field_on_every_route_and_are_billed(gateway: Gateway, route: Route) -> None:
    marker: Final = uuid.uuid4().hex
    images: Final = _images_for(route)
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        response: Final = _decide(gateway, route, model, marker, images)
        assert response.status_code == 200, response.text
        assert response.json() == _expected_response(route, "clef", marker)
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), images)
        _assert_billed(gateway, response, model, route, "cloudflare/@cf/cloudflare/clef")


@pytest.mark.parametrize(
    ("spelling", "clef_name"),
    (("cloudflare/clef-flash", "clef-flash"), ("cloudflare/@cf/cloudflare/clef-omni", "clef-omni")),
    ids=("clef-flash", "clef-omni"),
)
def test_clef_flash_and_clef_omni_take_images_and_carry_their_own_price(
    gateway: Gateway, spelling: str, clef_name: str
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=spelling, api_base=wire.url, api_key=_API_KEY)
        response: Final = _decide(gateway, "/v1/systemone", model, marker, [_PNG_DATA_URL])
        assert response.status_code == 200, response.text
        assert response.json() == _expected_response("/v1/systemone", clef_name, marker)
        (request,) = wire.drain()
        assert _upstream_body(request, clef_name) == _expected_upstream_body(clef_name, _state(marker), [_PNG_DATA_URL])
        _assert_billed(gateway, response, model, "/v1/systemone", spelling)


def test_an_image_only_message_reaches_clef_with_an_empty_state(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    reply: Final = json.dumps(
        {
            "result": {
                "model": "clef",
                "answers": {"red": {"type": "noul", "noul": _probability(marker)}},
                "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
            },
            "success": True,
        }
    ).encode()
    with wire_server(lambda _: Reply(body=reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/decisions",
            {"model": model, "input": _openai_input("", [_PNG_DATA_URL]), "questions": _OPENAI_QUESTIONS},
        )
        assert response.status_code == 200, response.text
        assert response.json() == _expected_response("/v1/decisions", "clef", marker)
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", "", [_PNG_DATA_URL])


def _luna_reply(request: Request) -> Reply:
    reply: Final[dict[str, JsonValue]] = {
        "model": "gpt-6-luna",
        "answers": [{"type": "predicate", "name": "red", "probability": _probability(_marker_of(request))}],
        "usage": {
            "input_tokens": _INPUT_TOKENS,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": _INPUT_TOKENS,
        },
    }
    return Reply(body=json.dumps(reply).encode())


def test_an_openai_decision_model_gets_system_one_images_as_input_image_parts(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_luna_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6-luna", api_base=wire.url, api_key=_API_KEY)
        response: Final = _decide(gateway, "/v1/systemone", model, marker, [_PNG_DATA_URL, _PNG_OBJECT])
        assert response.status_code == 200, response.text
        assert object_value(object_value(response.json()["answers"])["red"])["noul"] == _probability(marker)
        (request,) = wire.drain()
        assert unquote(request.target) == "/v1/decisions", request.target
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "gpt-6-luna",
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {"type": "input_image", "image_url": _PNG_DATA_URL},
                        {"type": "input_image", "image_url": _PNG_DATA_URL},
                        {"type": "input_text", "text": _state(marker)},
                    ],
                }
            ],
            "questions": _OPENAI_QUESTIONS,
        }


@pytest.mark.parametrize("route", ("/v1/systemone", "/v1/decisions"))
def test_more_images_than_clef_takes_are_refused_before_any_download_or_upstream_call(
    gateway: Gateway, route: Route
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        remote: Final = [f"{wire.url}/image.png?n={index}&id=marker-{marker}" for index in range(5)]
        for model in (strict, dropping):
            refused = _decide(gateway, route, model, marker, remote)
            assert refused.status_code == 400, refused.text
            assert "at most 4 images per request, got 5" in refused.text, refused.text
        assert wire.drain() == ()
        four: Final = [_PNG_DATA_URL, _JPEG_DATA_URL, _PNG_DATA_URL, _JPEG_DATA_URL]
        accepted: Final = _decide(gateway, route, strict, marker, four)
        assert accepted.status_code == 200, accepted.text
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), four)


_MALFORMED: Final[tuple[tuple[str, JsonValue], ...]] = (
    ("bare_string", _PNG_DATA_URL),
    ("number", 5),
    ("object_not_list", _PNG_OBJECT),
    ("url_object", [{"url": _PNG_DATA_URL}]),
    ("chat_image_url", [{"type": "image_url", "image_url": {"url": _PNG_DATA_URL}}]),
    ("extra_key", [{**_PNG_OBJECT, "detail": "high"}]),
    ("missing_base64", [{"content_type": "image/png"}]),
    ("non_image_type", [{"content_type": "text/plain", "base64": "AA=="}]),
    ("empty_base64", [{"content_type": "image/png", "base64": ""}]),
    ("not_a_url", ["a" * 5000]),
    ("non_image_data_url", ["data:text/plain;base64,AA=="]),
)


@pytest.mark.parametrize(("label", "images"), _MALFORMED, ids=[label for label, _ in _MALFORMED])
def test_malformed_images_are_refused_unless_the_deployment_drops_params(
    gateway: Gateway, label: str, images: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        body: Final = {**_text_only_body("/v1/systemone", strict, marker), "images": images}
        refused: Final = gateway.request("POST", "/v1/systemone", body)
        assert refused.status_code == 400, (label, refused.text)
        assert "Invalid Decisions request" in refused.text and "images" in refused.text, (label, refused.text)
        assert wire.drain() == (), label
        dropped: Final = gateway.request("POST", "/v1/systemone", {**body, "model": dropping})
        assert dropped.status_code == 200, (label, dropped.text)
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker)), label


def test_a_malformed_image_under_drop_params_keeps_the_well_formed_ones(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        dropping: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        images: Final[list[JsonValue]] = [
            {"content_type": "text/plain", "base64": "AA=="},
            _PNG_DATA_URL,
            {"url": _PNG_DATA_URL},
            _PNG_OBJECT,
        ]
        response: Final = _decide(gateway, "/v1/systemone", dropping, marker, images)
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), [_PNG_DATA_URL, _PNG_OBJECT])


_MALFORMED_IMAGE_URLS: Final = (
    ("not_a_url", "not-an-image"),
    ("pdf_data_url", "data:application/pdf;base64,AA=="),
    ("ftp_url", "ftp://images.example/a.png"),
)


@pytest.mark.parametrize(
    ("label", "image_url"), _MALFORMED_IMAGE_URLS, ids=[label for label, _ in _MALFORMED_IMAGE_URLS]
)
def test_a_malformed_input_image_is_refused_unless_the_deployment_drops_params(
    gateway: Gateway, label: str, image_url: str
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        refused: Final = _decide(gateway, "/v1/decisions", strict, marker, [image_url, _PNG_DATA_URL])
        assert refused.status_code == 400, (label, refused.text)
        assert "Invalid Decisions request: 1 input_image" in refused.text, (label, refused.text)
        assert wire.drain() == (), label
        dropped: Final = _decide(gateway, "/v1/decisions", dropping, marker, [image_url, _PNG_DATA_URL])
        assert dropped.status_code == 200, (label, dropped.text)
        assert dropped.json() == _expected_response("/v1/decisions", "clef", marker), label
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), [_PNG_DATA_URL]), label


_GIF_BASE64: Final = "R0lGODlhAQABAAAAACw="
_PNG_DATA_URL_WITH_NAME: Final = f"data:IMAGE/PNG;name=screen.png;base64,{_PNG_BASE64}"
_WEBP_DATA_URL: Final = "data:image/webp;base64,UklGRhoAAABXRUJQ"


@pytest.mark.parametrize("route", ("/v1/systemone", "/v1/decisions"))
def test_clef_gets_png_jpeg_and_webp_images_without_media_type_parameters(gateway: Gateway, route: Route) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        images: Final = [_PNG_DATA_URL_WITH_NAME, _JPEG_DATA_URL, _WEBP_DATA_URL]
        response: Final = _decide(gateway, route, model, marker, images)
        assert response.status_code == 200, response.text
        assert response.json() == _expected_response(route, "clef", marker)
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body(
            "clef", _state(marker), [_PNG_DATA_URL, _JPEG_DATA_URL, _WEBP_DATA_URL]
        )


def test_an_openai_decision_model_gets_a_data_url_with_media_type_parameters_unchanged(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_luna_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6-luna", api_base=wire.url, api_key=_API_KEY)
        response: Final = _decide(gateway, "/v1/decisions", model, marker, [_PNG_DATA_URL_WITH_NAME])
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
        sent: Final = list_value(
            object_value(list_value(_JSON_OBJECT.validate_json(request.body)["input"])[0])["content"]
        )
        assert sent == [
            {"type": "input_text", "text": _state(marker)},
            {"type": "input_image", "image_url": _PNG_DATA_URL_WITH_NAME},
        ]


_REFUSED_TYPES: Final[tuple[tuple[str, Route, JsonValue], ...]] = (
    ("gif_data_url", "/v1/decisions", f"data:image/gif;base64,{_GIF_BASE64}"),
    ("gif_object", "/v1/systemone", {"content_type": "image/gif", "base64": _GIF_BASE64}),
)


@pytest.mark.parametrize(("label", "route", "image"), _REFUSED_TYPES, ids=[label for label, _, _ in _REFUSED_TYPES])
def test_an_image_type_clef_refuses_is_refused_unless_the_deployment_drops_params(
    gateway: Gateway, label: str, route: Route, image: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        refused: Final = _decide(gateway, route, strict, marker, [image, _PNG_DATA_URL])
        assert refused.status_code == 400, (label, refused.text)
        assert (
            "cloudflare takes only image/jpeg, image/jpg, image/png, image/webp images, got image/gif" in refused.text
        ), (label, refused.text)
        assert wire.drain() == (), label
        dropped: Final = _decide(gateway, route, dropping, marker, [image, _PNG_DATA_URL])
        assert dropped.status_code == 200, (label, dropped.text)
        assert dropped.json() == _expected_response(route, "clef", marker), label
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), [_PNG_DATA_URL]), label


@pytest.mark.parametrize("route", ("/v1/systemone", "/v1/decisions"))
def test_an_image_clef_refuses_does_not_count_toward_its_limit_when_the_deployment_drops_params(
    gateway: Gateway, route: Route
) -> None:
    marker: Final = uuid.uuid4().hex
    four: Final = [_PNG_DATA_URL, _JPEG_DATA_URL, _PNG_DATA_URL, _JPEG_DATA_URL]
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        dropping: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        accepted: Final = _decide(gateway, route, dropping, marker, [f"data:image/gif;base64,{_GIF_BASE64}", *four])
        assert accepted.status_code == 200, accepted.text
        assert accepted.json() == _expected_response(route, "clef", marker)
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), four)


_MIB: Final = 1024 * 1024


def _png_data_url(decoded_bytes: int) -> str:
    return f"data:image/png;base64,{base64.b64encode(bytes(decoded_bytes)).decode()}"


def _png_object(decoded_bytes: int) -> dict[str, JsonValue]:
    return {"content_type": "image/png", "base64": base64.b64encode(bytes(decoded_bytes)).decode()}


_OVERSIZED: Final[tuple[tuple[str, Route, tuple[JsonValue, ...], str], ...]] = (
    ("one_over_4mib", "/v1/systemone", (_png_object(4 * _MIB + 1),), "got one of 4,194,305 bytes"),
    ("one_over_4mib", "/v1/decisions", (_png_data_url(4 * _MIB + 1),), "got one of 4,194,305 bytes"),
    (
        "total_over_8mib",
        "/v1/systemone",
        (_png_data_url(4 * _MIB), _png_object(4 * _MIB), _png_data_url(1)),
        "got 8,388,609 bytes",
    ),
)


@pytest.mark.parametrize(
    ("label", "route", "images", "refusal"), _OVERSIZED, ids=[f"{label}{route}" for label, route, _, _ in _OVERSIZED]
)
def test_images_over_clefs_size_limits_are_refused_before_the_upstream_call(
    gateway: Gateway, label: str, route: Route, images: tuple[JsonValue, ...], refusal: str
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        for model in (strict, dropping):
            refused = _decide(gateway, route, model, marker, images)
            assert refused.status_code == 400, (label, refused.text[:500])
            assert refusal in refused.text, (label, refused.text[:500])
        assert wire.drain() == (), label
        at_limits: Final[list[JsonValue]] = [_png_data_url(4 * _MIB), _png_object(4 * _MIB)]
        accepted: Final = _decide(gateway, "/v1/systemone", strict, marker, at_limits)
        assert accepted.status_code == 200, accepted.text[:500]
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), at_limits), label


def _raw_system_one(gateway: Gateway, model: str, marker: str, images_fragment: str) -> httpx.Response:
    raw: Final = (
        f'{{"model": {json.dumps(model)}, "state": {json.dumps(_state(marker))}, '
        f'"questions": {json.dumps(_SYSTEM_ONE_QUESTIONS)}, {images_fragment}}}'
    )
    return gateway.client.post(
        "/v1/systemone",
        content=raw.encode(),
        headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
    )


def test_a_duplicated_images_key_is_judged_by_its_last_value(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    malformed: Final = json.dumps([{"content_type": "text/plain", "base64": "AA=="}])
    good: Final = json.dumps([_PNG_DATA_URL])
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        refused: Final = _raw_system_one(gateway, model, marker, f'"images": {good}, "images": {malformed}')
        assert refused.status_code == 400, refused.text
        assert "Invalid Decisions request" in refused.text, refused.text
        assert wire.drain() == ()
        accepted: Final = _raw_system_one(gateway, model, marker, f'"images": {malformed}, "images": {good}')
        assert accepted.status_code == 200, accepted.text
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), [_PNG_DATA_URL])


def test_images_next_to_input_are_refused_unless_params_are_dropped(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        strict: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        dropping: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY, drop_params=True)
        refused: Final = gateway.request(
            "POST", "/v1/decisions", _text_only_body("/v1/decisions", strict, marker, images=[_PNG_DATA_URL])
        )
        assert refused.status_code == 400, refused.text
        assert "images belongs to the System One format" in refused.text, refused.text
        assert wire.drain() == ()
        for accepted in (
            gateway.request(
                "POST", "/v1/decisions", _text_only_body("/v1/decisions", dropping, marker, images=[_PNG_DATA_URL])
            ),
            gateway.request(
                "POST",
                "/v1/decisions",
                _text_only_body("/v1/decisions", strict, marker, images=[_PNG_DATA_URL], drop_params=True),
            ),
        ):
            assert accepted.status_code == 200, accepted.text
            assert accepted.json() == _expected_response("/v1/decisions", "clef", marker)
        assert [_upstream_body(request) for request in wire.drain()] == [
            _expected_upstream_body("clef", _state(marker))
        ] * 2


def test_a_remote_image_on_the_shared_proxy_is_refused_before_any_fetch(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        for route in ("/v1/systemone", "/v1/decisions"):
            refused = _decide(gateway, route, model, marker, [f"{wire.url}/image.png?id=marker-{marker}"])
            assert refused.status_code == 400, (route, refused.text)
            assert "Unable to fetch image from URL" in refused.text, (route, refused.text)
            assert "user_url_allowed_hosts" in refused.text, (route, refused.text)
        assert wire.drain() == ()


def test_a_clef_rejection_of_an_image_reaches_the_caller_like_a_text_only_one(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    message: Final = "Image could not be decoded"
    rejection: Final = json.dumps(
        {"result": {}, "success": False, "errors": [{"code": 5006, "message": message}], "messages": []}
    ).encode()
    garbage: Final = "data:image/png;base64," + "a" * 5000
    with wire_server(lambda _: Reply(status=422, body=rejection)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        text_only: Final = gateway.request(
            "POST", "/v1/systemone", _text_only_body("/v1/systemone", model, marker, num_retries=0)
        )
        with_image: Final = _decide(gateway, "/v1/systemone", model, marker, [garbage], num_retries=0)
        for response in (text_only, with_image):
            assert response.status_code >= 400, response.text
            assert message in response.text and "answers" not in response.text, response.text
        assert with_image.status_code == text_only.status_code, (text_only.status_code, with_image.status_code)
        first, second = (_upstream_body(request) for request in wire.drain())
        assert first == _expected_upstream_body("clef", _state(marker))
        assert second == _expected_upstream_body("clef", _state(marker), [garbage])


def test_an_unauthenticated_image_request_is_refused_on_every_route(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        for route in _ROUTES:
            response = gateway.client.post(route, json=_body(route, model, marker, [_PNG_DATA_URL]))
            assert response.status_code == 401, (route, response.text)
            assert "Authentication Error" in response.text, response.text
        assert wire.drain() == ()


def test_unrelated_chat_keeps_working_after_a_run_of_refused_image_requests(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        control: Final = scenario.model()
        for _, images in _MALFORMED:
            refused = gateway.request(
                "POST", "/v1/systemone", {**_text_only_body("/v1/systemone", model, marker), "images": images}
            )
            assert refused.status_code == 400, refused.text
        assert _decide(gateway, "/v1/systemone", model, marker, [_PNG_DATA_URL] * 5).status_code == 400
        assert wire.drain() == ()
        assert gateway.chat(control)["choices"], "the control deployment stopped answering"
        accepted: Final = _decide(gateway, "/v1/systemone", model, marker, [_PNG_DATA_URL])
        assert accepted.status_code == 200, accepted.text
        (request,) = wire.drain()
        assert _upstream_body(request) == _expected_upstream_body("clef", _state(marker), [_PNG_DATA_URL])


def _timed_liveliness(gateway: Gateway) -> tuple[int, float]:
    started: Final = time.monotonic()
    response: Final = gateway.request("GET", "/health/liveliness")
    return response.status_code, time.monotonic() - started


_WORST_CASE_IMAGES: Final = (
    ("unterminated_data_url", "DATA:IMAGE/" + "x" * 2_000_000),
    ("url_with_trailing_space", "HTTPS://" + "x" * 2_000_000 + " "),
    ("parameters_without_base64", "DATA:IMAGE/PNG" + ";x" * 1_000_000),
)


@pytest.mark.parametrize(("label", "image"), _WORST_CASE_IMAGES, ids=[label for label, _ in _WORST_CASE_IMAGES])
def test_the_worst_case_image_string_is_refused_in_time_while_health_keeps_answering(
    gateway: Gateway, label: str, image: str
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_clef_reply) as wire, gateway.scenario() as scenario, ThreadPoolExecutor(max_workers=1) as pool:
        model: Final = scenario.model(model=_CLEF, api_base=wire.url, api_key=_API_KEY)
        control: Final = timed(lambda: _decide(gateway, "/v1/systemone", model, marker, ["x" * len(image)]))
        assert control.value.status_code == 400, control.value.text
        liveliness: Final = pool.submit(_timed_liveliness, gateway)
        refused: Final = timed(lambda: _decide(gateway, "/v1/systemone", model, marker, [image]))
        health_status, health_elapsed = liveliness.result()
        assert refused.value.status_code == 400, (label, refused.value.text[:500])
        assert "Invalid Decisions request" in refused.value.text, (label, refused.value.text[:500])
        assert_within_factor_of_control(refused.seconds, control=control.seconds, factor=5)
        assert health_status == 200, (health_status, health_elapsed)
        assert_within_factor_of_control(health_elapsed, control=control.seconds, factor=5)
        assert wire.drain() == ()


def _image_object_schema(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    images: Final = object_value(object_value(schema["properties"])["images"])
    array: Final = next(
        object_value(branch) for branch in list_value(images["anyOf"]) if object_value(branch).get("type") == "array"
    )
    entries: Final = list_value(object_value(array["items"])["anyOf"])
    return next(object_value(entry) for entry in entries if object_value(entry).get("type") == "object")


def test_openapi_documents_what_an_image_object_must_carry(gateway: Gateway) -> None:
    response: Final = gateway.client.get(
        "/openapi.json", headers={"Authorization": f"Bearer {gateway.key}"}, timeout=_COLD_OPENAPI_SECONDS
    )
    assert response.status_code == 200, response.text
    paths: Final = object_value(_JSON_OBJECT.validate_json(response.content)["paths"])
    for route in ("/v1/systemone", "/systemone"):
        post: Final = object_value(object_value(paths[route])["post"])
        content: Final = object_value(object_value(post["requestBody"])["content"])
        schema: Final = object_value(object_value(content["application/json"])["schema"])
        properties: Final = object_value(_image_object_schema(schema)["properties"])
        assert object_value(properties["content_type"])["pattern"] == r"^image/\S+$", (route, properties)
        assert object_value(properties["base64"])["minLength"] == 1, (route, properties)


def _pooled_database_url(url: str) -> str:
    parts: Final = urlsplit(url)
    query: Final = "&".join(part for part in (parts.query, "connection_limit=5") if part)
    return urlunsplit(parts._replace(query=query))


def _image_host_and_clef(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if request.method == "GET" and path == "/image.png":
        return Reply(body=PNG_1X1, content_type="image/png")
    if request.method == "GET" and path == "/page.html":
        return Reply(body=b"<html></html>", content_type="text/html; charset=utf-8")
    if request.method == "GET" and path == "/slow.png":
        _SLOW_IMAGE_RELEASE.wait(timeout=30)
        return Reply(body=PNG_1X1, content_type="image/png")
    if request.method == "GET":
        return Reply(status=404, body=b"not found", content_type="text/plain")
    return _clef_reply(request)


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(_image_host_and_clef) as served:
        yield served
        _SLOW_IMAGE_RELEASE.set()


@pytest.fixture(scope="module")
def sink() -> Iterator[Wire]:
    with wire_server(lambda _: Reply()) as served:
        yield served


@pytest.fixture(scope="module")
def allowlisted(rig: Gateway, wire: Wire, sink: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("clef-images-allowlisted")
    (directory / "image_input_recorder.py").write_text(_IMAGE_INPUT_RECORDER)
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    path: Final = directory / "clef-images-allowlisted.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": _ALLOWLISTED_MODEL,
                        "litellm_params": {"model": _CLEF, "api_base": wire.url, "api_key": _API_KEY},
                    }
                ],
                "litellm_settings": {
                    **object_value(config["litellm_settings"]),
                    "callbacks": ["image_input_recorder.instance"],
                },
                "general_settings": {
                    **object_value(config["general_settings"]),
                    "user_url_allowed_hosts": ["127.0.0.1"],
                },
            }
        )
    )
    overrides: Final = {
        "DATABASE_URL": _pooled_database_url(os.environ["DATABASE_URL"]),
        "IMAGE_INPUT_SINK": sink.url,
    }
    with owned_proxy(rig, directory, overrides, config=path) as owned:
        yield owned


@pytest.fixture(autouse=True)
def _drained(request: pytest.FixtureRequest) -> None:
    for name in ("wire", "sink"):
        if name in request.fixturenames:
            request.getfixturevalue(name).drain()


def _routes(received: Sequence[Request]) -> list[tuple[str, str]]:
    return [(request.method, urlsplit(unquote(request.target)).path) for request in received]


def _logged_inputs(sink: Wire, call_ids: Sequence[str]) -> dict[str, JsonValue]:
    eventually(sink.received.qsize, lambda pending: pending >= len(call_ids), seconds=30)
    events: Final = tuple(_JSON_OBJECT.validate_json(request.body) for request in sink.drain())
    return {str(event["call_id"]): event["input"] for event in events}


@pytest.mark.timeout(_OWNED_TIMEOUT)
@pytest.mark.parametrize("route", ("/v1/systemone", "/v1/decisions"))
def test_an_allowlisted_remote_image_reaches_clef_as_bytes_while_callbacks_log_the_url_the_caller_sent(
    allowlisted: Gateway, wire: Wire, sink: Wire, route: Route
) -> None:
    marker: Final = uuid.uuid4().hex
    image_url: Final = f"{wire.url}/image.png?id=marker-{marker}"
    responses: Final = tuple(_decide(allowlisted, route, _ALLOWLISTED_MODEL, marker, [image_url]) for _ in range(2))
    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json() == _expected_response(route, "clef", marker)
    received: Final = wire.drain()
    posts: Final = tuple(request for request in received if request.method == "POST")
    assert {target for method, target in _routes(received) if method == "GET"} == {"/image.png"}, _routes(received)
    assert [_upstream_body(request) for request in posts] == [
        _expected_upstream_body("clef", _state(marker), [_PNG_DATA_URL])
    ] * 2
    call_ids: Final = tuple(response.headers["x-litellm-call-id"] for response in responses)
    assert len(set(call_ids)) == 2, call_ids
    assert _logged_inputs(sink, call_ids) == {
        call_id: _expected_upstream_body("clef", _state(marker), [image_url]) for call_id in call_ids
    }


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_a_missing_remote_image_is_refused_without_a_call_to_clef(allowlisted: Gateway, wire: Wire) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _decide(
        allowlisted, "/v1/systemone", _ALLOWLISTED_MODEL, marker, [f"{wire.url}/missing.png?id=marker-{marker}"]
    )
    assert response.status_code == 400, response.text
    assert "Unable to fetch image from URL. Status code: 404" in response.text, response.text
    assert _routes(wire.drain()) == [("GET", "/missing.png")]


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_a_remote_page_that_is_not_an_image_is_refused_without_a_call_to_clef(allowlisted: Gateway, wire: Wire) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _decide(
        allowlisted, "/v1/decisions", _ALLOWLISTED_MODEL, marker, [f"{wire.url}/page.html?id=marker-{marker}"]
    )
    assert response.status_code == 400, response.text
    assert "images, got text/html" in response.text, response.text
    assert _routes(wire.drain()) == [("GET", "/page.html")]


@pytest.mark.timeout(_OWNED_TIMEOUT)
def test_a_hanging_image_host_gets_one_attempt_within_the_callers_timeout(allowlisted: Gateway, wire: Wire) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _decide(
        allowlisted,
        "/v1/systemone",
        _ALLOWLISTED_MODEL,
        marker,
        [f"{wire.url}/slow.png?id=marker-{marker}"],
        timeout=2,
        num_retries=0,
    )
    received: Final = wire.drain()
    _SLOW_IMAGE_RELEASE.set()
    assert response.status_code == 400, response.text
    assert "Unable to fetch image from URL after 3 attempts" in response.text, response.text
    assert _routes(received) == [("GET", "/slow.png")]
