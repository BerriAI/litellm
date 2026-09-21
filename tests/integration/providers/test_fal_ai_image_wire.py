import base64
import json
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_GPT_IMAGE_MODEL: Final = "openai/gpt-image-2.5/flare/text-to-image"
_FLUX_MODEL: Final = "fal-ai/flux/dev"
_EDIT_MODEL: Final = "openai/gpt-image-2.5/flare/edit"
_PNG_BYTES: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff"
    b"\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PROMPT: Final = "a red circle on a blue background"
_COST_MAP_PATH: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_COST_MAP: Final = TypeAdapter(dict[str, dict[str, object]])


def _catalog_cost(key: str) -> float:
    cost_map: Final = _COST_MAP.validate_json(_COST_MAP_PATH.read_bytes())
    cost_value: Final = cost_map[key]["output_cost_per_image"]
    assert isinstance(cost_value, (int, float))
    return float(cost_value)


def _image_response(urls: tuple[str, ...], prompt: str, width: int = 1024, height: int = 768) -> bytes:
    return json.dumps(
        {
            "images": [
                {
                    "url": url,
                    "content_type": "image/png",
                    "file_name": url.rsplit("/", 1)[-1],
                    "file_size": 123456,
                    "width": width,
                    "height": height,
                }
                for url in urls
            ],
            "timings": {"inference": 2.1},
            "seed": 1234567,
            "has_nsfw_concepts": [False],
            "prompt": prompt,
        }
    ).encode()


def _response_cost(response: httpx.Response) -> float:
    return float(response.headers["x-litellm-response-cost"])


def _approx(value: float) -> object:
    return pytest.approx(value, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]  # pytest lacks typed approx stubs


@pytest.mark.covers("other.provider_wire.fal_ai.gpt_image_generation_quality_size_wire_and_keyed_pricing")
def test_fal_gpt_image_25_generation_sends_quality_and_size_and_charges_keyed_row(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == "/openai/gpt-image-2.5/flare/text-to-image"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if body.get("quality") == "high":
            assert body == {"prompt": _PROMPT, "quality": "high", "image_size": {"width": 1024, "height": 1536}}
            return Reply(body=_image_response((f"{wire_url}/files/high.png",), _PROMPT, width=1024, height=1536))
        assert body == {"prompt": _PROMPT, "quality": "low"}
        return Reply(body=_image_response((f"{wire_url}/files/low.png",), _PROMPT))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model=f"fal_ai/{_GPT_IMAGE_MODEL}", api_base=wire.url, api_key="synthetic-fal-key"
        )
        high_response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT, "quality": "high", "size": "1024x1536"},
        )
        assert high_response.status_code == 200, high_response.text
        high_payload: Final = _JSON_OBJECT.validate_json(high_response.content)
        assert high_payload["data"] == [{"url": f"{wire.url}/files/high.png", "b64_json": None, "revised_prompt": None}]
        high_cost: Final = _response_cost(high_response)
        assert high_cost == _approx(_catalog_cost("fal_ai/high/1024-x-1536/openai/gpt-image-2.5/flare/text-to-image"))

        low_response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT, "quality": "low"},
        )
        assert low_response.status_code == 200, low_response.text
        low_payload: Final = _JSON_OBJECT.validate_json(low_response.content)
        assert low_payload["data"] == [{"url": f"{wire.url}/files/low.png", "b64_json": None, "revised_prompt": None}]
        low_cost: Final = _response_cost(low_response)
        assert low_cost == _approx(_catalog_cost("fal_ai/low/1024-x-768/openai/gpt-image-2.5/flare/text-to-image"))
        assert high_cost != low_cost
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/gpt-image-2.5/flare/text-to-image"),
            ("POST", "/openai/gpt-image-2.5/flare/text-to-image"),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.image_pricing_uses_response_dimensions")
def test_fal_gpt_image_25_charges_the_size_fal_returned_not_the_requested_size(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == "/openai/gpt-image-2.5/flare/text-to-image"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if body.get("image_size") is not None:
            assert body == {"prompt": _PROMPT, "quality": "low", "image_size": {"width": 1024, "height": 1024}}
            return Reply(body=_image_response((f"{wire_url}/files/rounded.png",), _PROMPT, width=1024, height=1536))
        assert body == {"prompt": _PROMPT, "quality": "low"}
        return Reply(body=_image_response((f"{wire_url}/files/default.png",), _PROMPT, width=1920, height=1080))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model=f"fal_ai/{_GPT_IMAGE_MODEL}", api_base=wire.url, api_key="synthetic-fal-key"
        )
        requested_size_response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT, "quality": "low", "size": "1024x1024"},
        )
        assert requested_size_response.status_code == 200, requested_size_response.text
        requested_size_cost: Final = _response_cost(requested_size_response)
        assert requested_size_cost == _approx(
            _catalog_cost("fal_ai/low/1024-x-1536/openai/gpt-image-2.5/flare/text-to-image")
        )
        assert requested_size_cost != _catalog_cost("fal_ai/low/1024-x-1024/openai/gpt-image-2.5/flare/text-to-image")

        default_size_response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT, "quality": "low"},
        )
        assert default_size_response.status_code == 200, default_size_response.text
        default_size_cost: Final = _response_cost(default_size_response)
        assert default_size_cost == _approx(
            _catalog_cost("fal_ai/low/1920-x-1080/openai/gpt-image-2.5/flare/text-to-image")
        )
        assert default_size_cost != _catalog_cost("fal_ai/low/1024-x-768/openai/gpt-image-2.5/flare/text-to-image")
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/gpt-image-2.5/flare/text-to-image"),
            ("POST", "/openai/gpt-image-2.5/flare/text-to-image"),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.flux_dev_endpoint_and_per_image_pricing")
def test_fal_flux_dev_generation_targets_dev_endpoint_and_charges_per_image(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == "/fal-ai/flux/dev"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "prompt": _PROMPT,
            "num_images": 2,
            "image_size": "square_hd",
        }
        return Reply(
            body=_image_response(
                (f"{wire_url}/files/flux-1.png", f"{wire_url}/files/flux-2.png"),
                _PROMPT,
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(model=f"fal_ai/{_FLUX_MODEL}", api_base=wire.url, api_key="synthetic-fal-key")
        response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT, "n": 2, "size": "1024x1024"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["data"] == [
            {"url": f"{wire.url}/files/flux-1.png", "b64_json": None, "revised_prompt": None},
            {"url": f"{wire.url}/files/flux-2.png", "b64_json": None, "revised_prompt": None},
        ]
        cost: Final = _response_cost(response)
        assert cost == _approx(2 * _catalog_cost("fal_ai/fal-ai/flux/dev"))
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/fal-ai/flux/dev")]


@pytest.mark.covers("other.provider_wire.fal_ai.image_edit_json_data_urls_and_keyed_pricing")
def test_fal_gpt_image_25_edit_inlines_upload_as_data_url_and_charges_keyed_row(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == "/openai/gpt-image-2.5/flare/edit"
        assert request.headers["content-type"] == "application/json"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "prompt": _PROMPT,
            "image_urls": ["data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode()],
            "quality": "low",
        }
        return Reply(body=_image_response((f"{wire_url}/files/edit.png",), _PROMPT))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(model=f"fal_ai/{_EDIT_MODEL}", api_base=wire.url, api_key="synthetic-fal-key")
        response: Final = gateway.client.post(
            "/v1/images/edits",
            data={"model": model, "prompt": _PROMPT, "quality": "low"},
            files={"image": ("red_circle.png", _PNG_BYTES, "image/png")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["data"] == [{"url": f"{wire.url}/files/edit.png", "b64_json": None, "revised_prompt": None}]
        cost: Final = _response_cost(response)
        assert cost == _approx(_catalog_cost("fal_ai/low/1024-x-768/openai/gpt-image-2.5/flare/edit"))
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/gpt-image-2.5/flare/edit")
        ]
