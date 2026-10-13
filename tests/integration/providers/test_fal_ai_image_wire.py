import base64
import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm

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


def _catalog_cost(key: str, field: str = "output_cost_per_image") -> float:
    cost_map: Final = _COST_MAP.validate_json(_COST_MAP_PATH.read_bytes())
    cost_value: Final = cost_map[key][field]
    assert isinstance(cost_value, (int, float))
    return float(cost_value)


def _image_response(images: tuple[tuple[str, int, int], ...], prompt: str) -> bytes:
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
                for url, width, height in images
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
            return Reply(body=_image_response(((f"{wire_url}/files/high.png", 1024, 1536),), _PROMPT))
        assert body == {"prompt": _PROMPT, "quality": "low"}
        return Reply(body=_image_response(((f"{wire_url}/files/low.png", 1024, 1536),), _PROMPT))

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
        assert high_payload["data"] == [
            {
                "url": f"{wire.url}/files/high.png",
                "b64_json": None,
                "revised_prompt": None,
                "provider_specific_fields": {"width": 1024, "height": 1536, "content_type": "image/png"},
            }
        ]
        high_cost: Final = _response_cost(high_response)
        assert high_cost == _approx(_catalog_cost("fal_ai/high/1024-x-1536/openai/gpt-image-2.5/flare/text-to-image"))

        low_response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT, "quality": "low"},
        )
        assert low_response.status_code == 200, low_response.text
        low_payload: Final = _JSON_OBJECT.validate_json(low_response.content)
        assert low_payload["data"] == [
            {
                "url": f"{wire.url}/files/low.png",
                "b64_json": None,
                "revised_prompt": None,
                "provider_specific_fields": {"width": 1024, "height": 1536, "content_type": "image/png"},
            }
        ]
        low_cost: Final = _response_cost(low_response)
        assert low_cost == _approx(_catalog_cost("fal_ai/low/1024-x-1536/openai/gpt-image-2.5/flare/text-to-image"))
        assert high_cost != low_cost
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/gpt-image-2.5/flare/text-to-image"),
            ("POST", "/openai/gpt-image-2.5/flare/text-to-image"),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.gpt_image_generation_noncanonical_size_uses_nearest_keyed_row")
def test_fal_gpt_image_25_generation_prices_non_canonical_size_from_nearest_row(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == "/openai/gpt-image-2.5/flare/text-to-image"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body == {"prompt": _PROMPT, "quality": "low", "image_size": {"width": 1536, "height": 1024}}
        return Reply(body=_image_response(((f"{wire_url}/files/noncanonical.png", 1536, 1024),), _PROMPT))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model=f"fal_ai/{_GPT_IMAGE_MODEL}", api_base=wire.url, api_key="synthetic-fal-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT, "quality": "low", "size": "1536x1024"},
        )
        assert response.status_code == 200, response.text
        cost: Final = _response_cost(response)
        assert cost == _approx(_catalog_cost("fal_ai/low/1024-x-1536/openai/gpt-image-2.5/flare/text-to-image"))
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/gpt-image-2.5/flare/text-to-image")
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.sdk_image_response_dump_options")
def test_fal_gpt_image_sdk_response_honors_dump_options() -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == "/openai/gpt-image-2.5/flare/text-to-image"
        assert _JSON_OBJECT.validate_json(request.body) == {"prompt": _PROMPT, "quality": "low"}
        return Reply(body=_image_response((("https://example.com/fal.png", 1024, 1536),), _PROMPT))

    with wire_server(respond) as wire:
        response: Final = litellm.image_generation(
            model=_GPT_IMAGE_MODEL,
            prompt=_PROMPT,
            quality="low",
            api_base=wire.url,
            api_key="synthetic-fal-key",
            custom_llm_provider="fal_ai",
        )
        assert response.model_dump(exclude_none=True)["data"] == [
            {
                "url": "https://example.com/fal.png",
                "provider_specific_fields": {"width": 1024, "height": 1536, "content_type": "image/png"},
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/gpt-image-2.5/flare/text-to-image")
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
                ((f"{wire_url}/files/flux-1.png", 1024, 1024), (f"{wire_url}/files/flux-2.png", 1920, 1080)),
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
            {
                "url": f"{wire.url}/files/flux-1.png",
                "b64_json": None,
                "revised_prompt": None,
                "provider_specific_fields": {"width": 1024, "height": 1024, "content_type": "image/png"},
            },
            {
                "url": f"{wire.url}/files/flux-2.png",
                "b64_json": None,
                "revised_prompt": None,
                "provider_specific_fields": {"width": 1920, "height": 1080, "content_type": "image/png"},
            },
        ]
        cost: Final = _response_cost(response)
        assert cost == _approx(3 * _catalog_cost("fal_ai/fal-ai/flux/dev", "output_cost_per_pixel") * 1_048_576)
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
        return Reply(body=_image_response(((f"{wire_url}/files/edit.png", 1024, 1536),), _PROMPT))

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
        assert payload["data"] == [
            {
                "url": f"{wire.url}/files/edit.png",
                "b64_json": None,
                "revised_prompt": None,
                "provider_specific_fields": {"width": 1024, "height": 1536, "content_type": "image/png"},
            }
        ]
        cost: Final = _response_cost(response)
        assert cost == _approx(_catalog_cost("fal_ai/low/1024-x-1536/openai/gpt-image-2.5/flare/edit"))
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/gpt-image-2.5/flare/edit")
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.flux_lora_depth_edit_single_image_url_and_flat_pricing")
def test_fal_flux_lora_depth_edit_sends_single_image_url_and_charges_flat_row(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == "/fal-ai/flux-lora-depth"
        assert request.headers["content-type"] == "application/json"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "prompt": _PROMPT,
            "image_url": "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode(),
        }
        return Reply(body=_image_response(((f"{wire_url}/files/depth.png", 1024, 1024),), _PROMPT))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model="fal_ai/fal-ai/flux-lora-depth", api_base=wire.url, api_key="synthetic-fal-key"
        )
        response: Final = gateway.client.post(
            "/v1/images/edits",
            data={"model": model, "prompt": _PROMPT},
            files={"image": ("red_circle.png", _PNG_BYTES, "image/png")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["data"] == [
            {
                "url": f"{wire.url}/files/depth.png",
                "b64_json": None,
                "revised_prompt": None,
                "provider_specific_fields": {"width": 1024, "height": 1024, "content_type": "image/png"},
            }
        ]
        cost: Final = _response_cost(response)
        assert cost == _approx(_catalog_cost("fal_ai/fal-ai/flux-lora-depth"))
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/fal-ai/flux-lora-depth")]


@pytest.mark.covers("other.provider_wire.fal_ai.global_api_base_routes_image_generation")
def test_fal_flux_dev_generation_without_deployment_api_base_uses_global_api_base(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == "/fal-ai/flux/dev"
        assert _JSON_OBJECT.validate_json(request.body) == {"prompt": _PROMPT, "num_images": 1}
        return Reply(body=_image_response(((f"{wire_url}/files/global.png", 1024, 1024),), _PROMPT))

    with wire_server(respond) as wire:
        wire_url: Final = wire.url
        configuration: Final = _JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        configuration["litellm_settings"] = {
            **object_value(configuration["litellm_settings"]),
            "api_base": wire.url,
        }
        path: Final = tmp_path / "global-api-base.yaml"
        path.write_text(yaml.safe_dump(configuration))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(model=f"fal_ai/{_FLUX_MODEL}", api_key="synthetic-fal-key", api_base=None)
            response: Final = candidate.request(
                "POST", "/v1/images/generations", {"model": model, "prompt": _PROMPT, "n": 1}
            )
            assert response.status_code == 200, response.text
            payload: Final = _JSON_OBJECT.validate_json(response.content)
            assert payload["data"] == [
                {
                    "url": f"{wire.url}/files/global.png",
                    "b64_json": None,
                    "revised_prompt": None,
                    "provider_specific_fields": {"width": 1024, "height": 1024, "content_type": "image/png"},
                }
            ]
            assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/fal-ai/flux/dev")]
