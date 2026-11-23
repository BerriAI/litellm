from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import HTTP
from enum import StrEnum
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_PROMPT: Final = "draw a red circle as vector art"
_PNG_BYTES: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff"
    b"\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)
_SVG_B64: Final = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4="
_IMAGE_PROVIDER_MODEL: Final = "gpt-image-2.6-flare"
_AUTHOR_PROVIDER_MODEL: Final = "gpt-5-nano"
_RESTRICTED_PROVIDER_MODEL: Final = "gpt-5.5"
_UNLISTED_PROVIDER_MODEL: Final = "gpt-5-mini"
_AUTHOR_INPUT_PRICE: Final = 0.0007
_AUTHOR_OUTPUT_PRICE: Final = 0.0013
_SVG_INPUT_TOKENS: Final = 300
_SVG_OUTPUT_TOKENS: Final = 500
_FORM_BYTES: Final = TypeAdapter(bytes)
_SPEND: Final = TypeAdapter(float)
_IMAGE_USAGE: Final = {
    "input_tokens": 40,
    "input_tokens_details": {"image_tokens": 0, "text_tokens": 40},
    "output_tokens": 1000,
    "total_tokens": 1040,
}


class _Shape(StrEnum):
    GENERATIONS_TOP_LEVEL = "generations_top_level"
    GENERATIONS_EXTRA_BODY = "generations_extra_body"
    EDITS_MULTIPART = "edits_multipart"


_SHAPES: Final = tuple(_Shape)


@dataclass(frozen=True, slots=True)
class _Deployments:
    image: str
    author: str


def _form_fields(request: Request) -> dict[str, str]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return {
        str(part.get_param("name", header="content-disposition")): _FORM_BYTES.validate_python(
            part.get_payload(decode=True)
        ).decode()
        for part in parsed.iter_parts()
        if part.get_filename() is None
    }


def _model_of(svg: JsonValue) -> str | None:
    if not isinstance(svg, dict):
        return None
    model: Final = svg.get("model")
    return model if isinstance(model, str) else None


def _svg_author_sent(request: Request) -> str | None:
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        fields: Final = _form_fields(request)
        if "svg[model]" in fields:
            return fields["svg[model]"]
        return _model_of(JSON_OBJECT.validate_json(fields["svg"])) if "svg" in fields else None
    return _model_of(JSON_OBJECT.validate_json(request.body).get("svg"))


def _image_reply(request: Request) -> Reply:
    author: Final = _svg_author_sent(request)
    svg_part: Final = (
        {}
        if author is None
        else {
            "output_format": "svg",
            "svg_usage": {
                "model": author,
                "input_tokens": _SVG_INPUT_TOKENS,
                "output_tokens": _SVG_OUTPUT_TOKENS,
                "total_tokens": _SVG_INPUT_TOKENS + _SVG_OUTPUT_TOKENS,
            },
        }
    )
    return Reply(
        body=json.dumps(
            {"created": 1700000000, "data": [{"b64_json": _SVG_B64}], "usage": _IMAGE_USAGE, **svg_part}
        ).encode()
    )


def _deployments(scenario: Scenario, wire: Wire) -> _Deployments:
    image: Final = scenario.model(
        name=f"svg-image-{uuid.uuid4().hex}",
        model=f"openai/{_IMAGE_PROVIDER_MODEL}",
        api_base=f"{wire.url}/v1",
        api_key="synthetic-openai-key",
        model_info={"mode": "image_generation"},
    )
    author: Final = scenario.model(
        name=f"svg-author-{uuid.uuid4().hex}",
        model=f"openai/{_AUTHOR_PROVIDER_MODEL}",
        api_base=f"{wire.url}/v1",
        api_key="synthetic-openai-key",
        model_info={"mode": "chat"},
        input_cost_per_token=_AUTHOR_INPUT_PRICE,
        output_cost_per_token=_AUTHOR_OUTPUT_PRICE,
    )
    return _Deployments(image=image, author=author)


def _restricted_chat(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        name=f"restricted-chat-{uuid.uuid4().hex}",
        model=f"openai/{_RESTRICTED_PROVIDER_MODEL}",
        api_base=f"{wire.url}/v1",
        api_key="synthetic-openai-key",
        model_info={"mode": "chat"},
    )


def _generation_svg_fields(shape: _Shape, author: str | None) -> dict[str, JsonValue]:
    if author is None:
        return {}
    match shape:
        case _Shape.GENERATIONS_EXTRA_BODY:
            return {"output_format": "svg", "extra_body": {"svg": {"model": author}}}
        case _:
            return {"output_format": "svg", "svg": {"model": author}}


def _image_request(gateway: Gateway, shape: _Shape, image_model: str, author: str | None, key: str) -> httpx.Response:
    if shape is _Shape.EDITS_MULTIPART:
        svg_fields: Final = {} if author is None else {"output_format": "svg", "svg[model]": author}
        return gateway.request_multipart(
            "/v1/images/edits",
            {"model": image_model, "prompt": _PROMPT, **svg_fields},
            {"image": ("source.png", _PNG_BYTES, "image/png")},
            key=key,
        )
    return gateway.request(
        "POST",
        "/v1/images/generations",
        {"model": image_model, "prompt": _PROMPT, **_generation_svg_fields(shape, author)},
        key=key,
    )


def _upstream_path(shape: _Shape) -> str:
    return "/v1/images/edits" if shape is _Shape.EDITS_MULTIPART else "/v1/images/generations"


def _logged_spend(gateway: Gateway, response: httpx.Response) -> float:
    rows: Final = eventually(
        lambda: gateway.rows(
            'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response.headers["x-litellm-call-id"],)
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return _SPEND.validate_python(rows[0]["spend"])


def _assert_denied_without_upstream_call(
    gateway: Gateway, shape: _Shape, image_model: str, author: str, key: str, error_type: str, wire: Wire
) -> None:
    denied: Final = _image_request(gateway, shape, image_model, author, key)
    allowed: Final = _image_request(gateway, shape, image_model, None, key)
    reached: Final = tuple((request.target, _svg_author_sent(request)) for request in wire.drain())

    assert denied.status_code == 403, denied.text
    assert denied.json()["error"]["type"] == error_type, denied.text
    assert allowed.status_code == 200, allowed.text
    assert reached == ((_upstream_path(shape), None),), f"the denied svg author reached the provider: {reached}"


@pytest.mark.parametrize("author_kind", ("proxy_model_name", "provider_model_name"))
@pytest.mark.parametrize("shape", _SHAPES)
def test_key_without_author_model_access_is_denied_before_upstream(
    gateway: Gateway, shape: _Shape, author_kind: str
) -> None:
    """regression: a key granted only the image model gets 403 key_model_access_denied for an svg author
    it was not granted, by proxy name or by provider name, and the provider is never called with it."""
    with wire_server(_image_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)
        restricted: Final = _restricted_chat(scenario, wire)
        key: Final = scenario.key(models=[deployments.image])
        author: Final = restricted if author_kind == "proxy_model_name" else _RESTRICTED_PROVIDER_MODEL

        _assert_denied_without_upstream_call(
            gateway, shape, deployments.image, author, key, "key_model_access_denied", wire
        )


@pytest.mark.parametrize("shape", _SHAPES)
def test_team_without_author_model_access_is_denied_before_upstream(gateway: Gateway, shape: _Shape) -> None:
    """regression: a key on a team whose models exclude the svg author gets 403 team_model_access_denied
    and the provider is never called with that author."""
    with wire_server(_image_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)
        restricted: Final = _restricted_chat(scenario, wire)
        team: Final = scenario.team(models=[deployments.image])
        key: Final = scenario.key(team_id=team)

        _assert_denied_without_upstream_call(
            gateway, shape, deployments.image, restricted, key, "team_model_access_denied", wire
        )


@pytest.mark.parametrize("shape", _SHAPES)
def test_teamless_user_without_author_model_access_is_denied_before_upstream(gateway: Gateway, shape: _Shape) -> None:
    """regression: a key with no team and no models of its own, owned by a user whose models exclude the svg
    author, gets 403 user_model_access_denied and the provider is never called with that author."""
    with wire_server(_image_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)
        restricted: Final = _restricted_chat(scenario, wire)
        user: Final = scenario.user(models=[deployments.image, deployments.author])
        key: Final = scenario.key(user_id=user)

        _assert_denied_without_upstream_call(
            gateway, shape, deployments.image, restricted, key, "user_model_access_denied", wire
        )


@pytest.mark.parametrize("shape", _SHAPES)
def test_svg_author_proxy_name_reaches_provider_as_its_deployment_model(gateway: Gateway, shape: _Shape) -> None:
    """regression: naming the proxy model my-nano as svg author sends the deployment's provider model
    name to OpenAI instead of the proxy alias OpenAI does not know."""
    with wire_server(_image_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)
        key: Final = scenario.key(models=[deployments.image, deployments.author])

        response: Final = _image_request(gateway, shape, deployments.image, deployments.author, key)
        reached: Final = tuple((request.target, _svg_author_sent(request)) for request in wire.drain())

        assert response.status_code == 200, response.text
        assert reached == ((_upstream_path(shape), _AUTHOR_PROVIDER_MODEL),), response.text


@pytest.mark.parametrize("shape", _SHAPES)
def test_svg_authoring_tokens_are_billed_at_the_author_deployment_price(gateway: Gateway, shape: _Shape) -> None:
    """regression: the svg authoring tokens are charged at my-nano's configured prices, so an svg request
    costs exactly that much more than the same image request without svg, in the header and the spend log."""
    with wire_server(_image_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)

        plain: Final = _image_request(gateway, shape, deployments.image, None, gateway.key)
        authored: Final = _image_request(gateway, shape, deployments.image, deployments.author, gateway.key)

        assert plain.status_code == 200, plain.text
        assert authored.status_code == 200, authored.text
        expected_authoring_cost: Final = (
            _SVG_INPUT_TOKENS * _AUTHOR_INPUT_PRICE + _SVG_OUTPUT_TOKENS * _AUTHOR_OUTPUT_PRICE
        )
        header_difference: Final = float(authored.headers["x-litellm-response-cost"]) - float(
            plain.headers["x-litellm-response-cost"]
        )
        assert header_difference == pytest.approx(expected_authoring_cost, rel=1e-6), (
            f"plain={plain.headers['x-litellm-response-cost']} authored={authored.headers['x-litellm-response-cost']}"
        )
        spend_difference: Final = _logged_spend(gateway, authored) - _logged_spend(gateway, plain)
        assert spend_difference == pytest.approx(expected_authoring_cost, rel=1e-6), authored.text


@pytest.mark.parametrize("shape", _SHAPES)
def test_key_granted_image_and_author_models_gets_svg(gateway: Gateway, shape: _Shape) -> None:
    """guard: a key granted both the image model and my-nano gets a 200 svg response and exactly one
    provider call that carries an svg author."""
    with wire_server(_image_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)
        key: Final = scenario.key(models=[deployments.image, deployments.author])

        response: Final = _image_request(gateway, shape, deployments.image, deployments.author, key)
        reached: Final = tuple((request.target, _svg_author_sent(request) is not None) for request in wire.drain())

        assert response.status_code == 200, response.text
        assert reached == ((_upstream_path(shape), True),), response.text


@pytest.mark.parametrize("shape", (_Shape.GENERATIONS_TOP_LEVEL, _Shape.EDITS_MULTIPART))
def test_image_request_without_svg_sends_no_svg_and_logs_its_charged_cost(gateway: Gateway, shape: _Shape) -> None:
    """guard: an image request with no svg reaches the provider with no svg field, and the spend log
    records the same nonzero cost the caller was charged in the header."""
    with wire_server(_image_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)
        key: Final = scenario.key(models=[deployments.image])

        response: Final = _image_request(gateway, shape, deployments.image, None, key)
        requests: Final = wire.drain()

        assert response.status_code == 200, response.text
        assert tuple(request.target for request in requests) == (_upstream_path(shape),), response.text
        sent_fields: Final = (
            frozenset(_form_fields(requests[0]))
            if shape is _Shape.EDITS_MULTIPART
            else frozenset(JSON_OBJECT.validate_json(requests[0].body))
        )
        assert not any(field.startswith("svg") for field in sent_fields), sorted(sent_fields)
        charged: Final = float(response.headers["x-litellm-response-cost"])
        assert charged > 0, response.text
        assert _logged_spend(gateway, response) == pytest.approx(charged, rel=1e-6), response.text


@pytest.mark.parametrize("shape", _SHAPES)
def test_unrestricted_caller_provider_author_name_passes_through_unchanged(gateway: Gateway, shape: _Shape) -> None:
    """guard: the master key naming a provider model that is not a proxy model as svg author gets a 200
    and the provider receives that name unchanged."""
    with wire_server(_image_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)

        response: Final = _image_request(gateway, shape, deployments.image, _UNLISTED_PROVIDER_MODEL, gateway.key)
        reached: Final = tuple((request.target, _svg_author_sent(request)) for request in wire.drain())

        assert response.status_code == 200, response.text
        assert reached == ((_upstream_path(shape), _UNLISTED_PROVIDER_MODEL),), response.text


def test_author_deployment_bills_chat_at_its_configured_price(gateway: Gateway) -> None:
    """guard: a chat call to my-nano is billed at the prices in its model_info, the premise the svg
    pricing contract relies on."""

    def chat_reply(_: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-svg-author",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": _AUTHOR_PROVIDER_MODEL,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": _SVG_INPUT_TOKENS,
                        "completion_tokens": _SVG_OUTPUT_TOKENS,
                        "total_tokens": _SVG_INPUT_TOKENS + _SVG_OUTPUT_TOKENS,
                    },
                }
            ).encode()
        )

    with wire_server(chat_reply) as wire, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, wire)

        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": deployments.author, "messages": [{"role": "user", "content": "price control"}]},
        )

        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(
            _SVG_INPUT_TOKENS * _AUTHOR_INPUT_PRICE + _SVG_OUTPUT_TOKENS * _AUTHOR_OUTPUT_PRICE, rel=1e-6
        ), response.text
        assert tuple(request.target for request in wire.drain()) == ("/v1/chat/completions",), response.text
