import base64
import json
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gemini-2.5-flash-image"
_API_KEY: Final = "synthetic-gemini-key"
_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ONLY_MODALITIES: Final[dict[str, JsonValue]] = {"response_modalities": ["IMAGE", "TEXT"]}
_FULL_USAGE: Final[dict[str, JsonValue]] = {
    "promptTokenCount": 263,
    "candidatesTokenCount": 1290,
    "totalTokenCount": 1553,
    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 5}, {"modality": "IMAGE", "tokenCount": 258}],
    "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1290}],
}
_FULL_USAGE_REPLY: Final[dict[str, JsonValue]] = {
    "total_tokens": 1553,
    "input_tokens": 263,
    "input_tokens_details": {"image_tokens": 258, "text_tokens": 5},
    "output_tokens": 1290,
    "output_tokens_details": {"image_tokens": 1290, "text_tokens": 0},
}
_COUNTS_ONLY_REPLY: Final[dict[str, JsonValue]] = {
    "total_tokens": 30,
    "input_tokens": 10,
    "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
    "output_tokens": 20,
    "output_tokens_details": {"image_tokens": 20, "text_tokens": 0},
}


def _image_reply(usage: JsonValue, include_usage: bool = True) -> bytes:
    candidates: Final = [
        {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(_PNG).decode()}}]}}
    ]
    return json.dumps({"candidates": candidates, **({"usageMetadata": usage} if include_usage else {})}).encode()


def _peer(reply: bytes):
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request.method
        assert request.target.split("?")[0] == f"/models/{_BACKEND}:generateContent", request.target.split("?")[0]
        return Reply(body=reply)

    return respond


def _edit(gateway: Gateway, model: str, image_config: str | None) -> httpx.Response:
    fields: Final = {"model": model, "prompt": "synthetic edit request"}
    return gateway.request_multipart(
        "/v1/images/edits",
        fields if image_config is None else {**fields, "imageConfig": image_config},
        {"image": ("pixel.png", _PNG, "image/png")},
    )


def _generation_config(request: Request) -> JsonValue:
    return _JSON_OBJECT.validate_json(request.body)["generationConfig"]


@pytest.mark.parametrize(
    ("image_config", "expected"),
    (
        pytest.param(
            '{"aspectRatio": "1:1"}',
            {**_ONLY_MODALITIES, "imageConfig": {"aspectRatio": "1:1"}},
            id="json-object",
        ),
        pytest.param("null", _ONLY_MODALITIES, id="json-null"),
        pytest.param("[1, 2]", _ONLY_MODALITIES, id="json-array"),
        pytest.param('"1:1"', _ONLY_MODALITIES, id="json-string"),
        pytest.param("7", _ONLY_MODALITIES, id="json-number"),
        pytest.param(None, _ONLY_MODALITIES, id="field-absent"),
    ),
)
def test_gemini_image_edit_forwards_only_a_json_object_image_config(
    gateway: Gateway,
    image_config: str | None,
    expected: dict[str, JsonValue],
) -> None:
    with wire_server(_peer(_image_reply(_FULL_USAGE))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = _edit(gateway, model, image_config)
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert [image["b64_json"] for image in payload["data"]] == [base64.b64encode(_PNG).decode()], response.text
        assert payload["usage"] == _FULL_USAGE_REPLY, response.text
        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        assert _generation_config(requests[0]) == expected, requests[0].body


@pytest.mark.parametrize("image_config", (pytest.param("{not json", id="truncated-json"), pytest.param("", id="empty")))
def test_gemini_image_edit_rejects_an_image_config_string_that_is_not_json(
    gateway: Gateway,
    image_config: str,
) -> None:
    with wire_server(_peer(_image_reply(_FULL_USAGE))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = _edit(gateway, model, image_config)
        assert response.status_code == 400, response.text
        error: Final = _JSON_OBJECT.validate_json(response.content)["error"]
        assert isinstance(error, dict), response.text
        assert error["type"] == "invalid_request_error" and error["code"] == "400", response.text
        assert str(error["message"]).startswith(
            "litellm.UnsupportedParamsError: `imageConfig` must be valid JSON when provided as a string."
        ), response.text
        assert wire.drain() == (), "a rejected imageConfig must not reach the provider"


@pytest.mark.parametrize(
    ("usage", "include_usage", "expected"),
    (
        pytest.param(_FULL_USAGE, True, _FULL_USAGE_REPLY, id="counts-and-modality-details"),
        pytest.param(
            {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30},
            True,
            _COUNTS_ONLY_REPLY,
            id="counts-only",
        ),
        pytest.param(
            {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
                "promptTokensDetails": "none",
                "candidatesTokensDetails": {"modality": "IMAGE"},
            },
            True,
            _COUNTS_ONLY_REPLY,
            id="details-that-are-not-lists",
        ),
        pytest.param(
            {
                "promptTokenCount": 10,
                "candidatesTokenCount": 25,
                "totalTokenCount": 35,
                "candidatesTokensDetails": [
                    {"modality": "IMAGE", "tokenCount": 20},
                    {"modality": "TEXT", "tokenCount": 2},
                ],
            },
            True,
            {
                "total_tokens": 35,
                "input_tokens": 10,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 25,
                "output_tokens_details": {"image_tokens": 20, "text_tokens": 5},
            },
            id="output-text-is-the-remainder",
        ),
        pytest.param(
            {},
            True,
            {
                "total_tokens": 0,
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
            },
            id="empty-usage-object",
        ),
        pytest.param(
            None,
            False,
            {
                "total_tokens": 0,
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
            },
            id="usage-absent",
        ),
    ),
)
def test_gemini_image_edit_reports_the_provider_usage_metadata_as_image_usage(
    gateway: Gateway,
    usage: JsonValue,
    include_usage: bool,
    expected: dict[str, JsonValue],
) -> None:
    with wire_server(_peer(_image_reply(usage, include_usage))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = _edit(gateway, model, None)
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["usage"] == expected, response.text
        assert len(wire.drain()) == 1


@pytest.mark.parametrize(
    ("count", "detail"),
    (
        pytest.param(
            "ten",
            "Input should be a valid integer, unable to parse string as an integer [type=int_parsing",
            id="count-is-a-word",
        ),
        pytest.param(None, "Input should be a valid integer [type=int_type", id="count-is-null"),
    ),
)
def test_gemini_image_edit_answers_500_when_a_provider_token_count_is_not_a_number(
    gateway: Gateway,
    count: JsonValue,
    detail: str,
) -> None:
    usage: Final[dict[str, JsonValue]] = {"promptTokenCount": count, "candidatesTokenCount": 20, "totalTokenCount": 30}
    with wire_server(_peer(_image_reply(usage))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = _edit(gateway, model, None)
        assert response.status_code == 500, response.text
        error: Final = _JSON_OBJECT.validate_json(response.content)["error"]
        assert isinstance(error, dict), response.text
        assert error["type"] == "internal_server_error", response.text
        assert "1 validation error for ImageUsage\ninput_tokens\n" in str(error["message"]), response.text
        assert detail in str(error["message"]), response.text
        assert len(wire.drain()) >= 1


def test_gemini_image_generation_reports_the_provider_usage_metadata_as_image_usage(gateway: Gateway) -> None:
    with wire_server(_peer(_image_reply(_FULL_USAGE))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/images/generations", {"model": model, "prompt": "synthetic generation request"}
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert [image["b64_json"] for image in payload["data"]] == [base64.b64encode(_PNG).decode()], response.text
        assert payload["usage"] == _FULL_USAGE_REPLY, response.text
        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        assert _JSON_OBJECT.validate_json(requests[0].body)["contents"] == [
            {"parts": [{"text": "synthetic generation request"}]}
        ], requests[0].body
