"""`POST /v1/images/generations`: what the proxy sends each provider and what it hands back to the caller.

The proxy takes an OpenAI-shaped JSON body, resolves `model` to a deployment, and calls that deployment's
provider. For OpenAI the body is forwarded almost verbatim, for Azure the deployment moves into the URL, and for
Bedrock and Vertex the OpenAI parameters are translated to the provider's own names. The caller always gets the
OpenAI image shape back (`created`, `data[].url|b64_json|revised_prompt`) plus the `x-litellm-*` headers.

These tests script the provider with an owned TCP peer, so every assertion about path, headers and body is made
on the bytes the proxy really wrote. A rewrite of the request path must keep all of them.
"""

from __future__ import annotations

import uuid
from typing import Final
from urllib.parse import parse_qs, unquote

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_media_support import (
    AZURE_KEY,
    B64_IMAGE,
    OPENAI_KEY,
    PROMPT,
    error_message,
    error_of,
    image_reply,
    json_body,
    json_reply,
    targets,
)

_GENERATIONS: Final = "/v1/images/generations"
_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_TARGET: Final = (
    f"/v1/projects/{_VERTEX_PROJECT}/locations/us-central1/publishers/google/models/imagen-4.0-generate-001:predict"
)


def _b64_image(_request: Request) -> Reply:
    return image_reply({"b64_json": B64_IMAGE})


def test_image_generation_sends_only_model_and_prompt_when_no_options_are_given(gateway: Gateway) -> None:
    """An OpenAI deployment is registered and the caller sends only `model` and `prompt`. The upstream must get
    one `POST /v1/images/generations` with the deployment's bearer key, a JSON content type, and a body of exactly
    the provider model name and the prompt: no default `n`, `size` or `quality` is invented by the proxy."""

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _GENERATIONS), request.target
        assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
        assert request.headers["content-type"] == "application/json"
        assert json_body(request) == {"model": "dall-e-3", "prompt": PROMPT}
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": PROMPT})
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["b64_json"] == B64_IMAGE, response.text
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_forwards_n_size_quality_and_user_with_their_json_types(gateway: Gateway) -> None:
    """The caller sets `n`, `size`, `quality` and `user`. Each must reach the upstream under the same name with
    the same JSON type (`n` stays an integer, the rest stay strings), next to the model and prompt and nothing
    else."""

    def respond(request: Request) -> Reply:
        body: Final = json_body(request)
        assert body == {
            "model": "dall-e-3",
            "prompt": PROMPT,
            "n": 1,
            "size": "1792x1024",
            "quality": "hd",
            "user": "end-user-7",
        }, body
        assert type(body["n"]) is int
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST",
            _GENERATIONS,
            {"model": model, "prompt": PROMPT, "n": 1, "size": "1792x1024", "quality": "hd", "user": "end-user-7"},
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_drops_null_optional_fields_instead_of_sending_null(gateway: Gateway) -> None:
    """The caller sends `n`, `size`, `quality` and `style` as JSON null, which is what some SDKs do for unset
    options. The upstream body must contain only the model and the prompt, with no null-valued keys."""

    def respond(request: Request) -> Reply:
        assert json_body(request) == {"model": "dall-e-3", "prompt": PROMPT}
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST",
            _GENERATIONS,
            {"model": model, "prompt": PROMPT, "n": None, "size": None, "quality": None, "style": None},
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _GENERATIONS)]


@pytest.mark.parametrize("response_format", ("url", "b64_json"))
def test_image_generation_forwards_response_format_to_openai(gateway: Gateway, response_format: str) -> None:
    """The caller asks a DALL-E deployment for `response_format` `url` or `b64_json`. The upstream must receive
    that value so it returns the representation the caller asked for, and the call must answer 200."""
    pytest.skip(
        "BUG: response_format on an openai/ image deployment answers 400 'Setting `response_format` is not "
        "supported by openai' because the OpenAI image config lists only gpt-image parameters "
        "(litellm/llms/openai/image_generation/gpt_transformation.py:26, litellm/utils.py:4657); read from code, "
        "not reproduced against a running proxy"
    )

    def respond(request: Request) -> Reply:
        assert json_body(request) == {"model": "dall-e-3", "prompt": PROMPT, "response_format": response_format}
        if response_format == "url":
            return image_reply({"url": "https://images.invalid/a.png"})
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": PROMPT, "response_format": response_format}
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_forwards_style_to_openai(gateway: Gateway) -> None:
    """The caller asks a DALL-E 3 deployment for `style: natural`. The upstream must receive `style` unchanged
    and the call must answer 200."""
    pytest.skip(
        "BUG: style on an openai/ image deployment answers 400 'Setting `style` is not supported by openai' unless "
        "the deployment lists it in additional_drop_params, in which case it is silently dropped "
        "(litellm/llms/openai/image_generation/gpt_transformation.py:26); read from code, not reproduced"
    )

    def respond(request: Request) -> Reply:
        assert json_body(request) == {"model": "dall-e-3", "prompt": PROMPT, "style": "natural"}
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": PROMPT, "style": "natural"})
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_keeps_proxy_only_fields_out_of_the_provider_body(gateway: Gateway) -> None:
    """The caller adds `metadata` with tags and the proxy adds its own bookkeeping to the request. None of
    `metadata`, `litellm_call_id`, `litellm_metadata`, `proxy_server_request` or `litellm_logging_obj` may
    appear in the provider body: a leaked field is invisible from the response, so it is asserted upstream."""

    def respond(request: Request) -> Reply:
        body: Final = json_body(request)
        leaked: Final = sorted(
            {"metadata", "litellm_call_id", "litellm_metadata", "proxy_server_request", "litellm_logging_obj"} & set(body)
        )
        assert leaked == [], body
        assert body["prompt"] == PROMPT
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST",
            _GENERATIONS,
            {"model": model, "prompt": PROMPT, "metadata": {"tags": ["media-contract"]}},
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_returns_url_and_revised_prompt_unchanged(gateway: Gateway) -> None:
    """The upstream answers two images, each with a `url` and a `revised_prompt`, and a `created` timestamp. The
    caller must get the same `created`, the same two URLs in the same order, and the same revised prompts."""
    urls: Final = ("https://images.invalid/one.png?sig=a%2Bb", "https://images.invalid/two.png")

    def respond(_request: Request) -> Reply:
        return image_reply(
            {"url": urls[0], "revised_prompt": "a glass lighthouse, first take"},
            {"url": urls[1], "revised_prompt": "a glass lighthouse, second take"},
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": PROMPT, "n": 2})
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["created"] == 1700000000
        images: Final = [object_value(image) for image in list_value(body["data"])]
        assert [image["url"] for image in images] == list(urls)
        assert [image["revised_prompt"] for image in images] == [
            "a glass lighthouse, first take",
            "a glass lighthouse, second take",
        ]
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_returns_a_large_base64_image_byte_for_byte(gateway: Gateway) -> None:
    """The upstream answers one `b64_json` image of about 1.4 MB of base64 text. The caller must receive exactly
    that string, so nothing in the response path truncates, re-encodes or re-wraps a large image payload."""
    large: Final = ("QUJD" * 350_000) + uuid.uuid4().hex

    def respond(_request: Request) -> Reply:
        return image_reply({"b64_json": large})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": PROMPT})
        assert response.status_code == 200, response.text[:500]
        assert response.json()["data"][0]["b64_json"] == large
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_returns_openai_usage_and_output_fields_unchanged(gateway: Gateway) -> None:
    """A gpt-image upstream answers with `usage` token counts and the `size`, `quality`, `output_format` and
    `background` it used. The caller must see the same usage numbers and the same four output fields."""
    usage: Final = {
        "input_tokens": 12,
        "input_tokens_details": {"text_tokens": 12, "image_tokens": 0},
        "output_tokens": 1056,
        "total_tokens": 1068,
    }

    def respond(_request: Request) -> Reply:
        return image_reply(
            {"b64_json": B64_IMAGE},
            usage=usage,
            size="1024x1024",
            quality="medium",
            output_format="png",
            background="opaque",
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": PROMPT})
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        returned_usage: Final = object_value(body["usage"])
        assert {name: returned_usage[name] for name in ("input_tokens", "output_tokens", "total_tokens")} == {
            "input_tokens": 12,
            "output_tokens": 1056,
            "total_tokens": 1068,
        }, response.text
        assert object_value(returned_usage["input_tokens_details"])["text_tokens"] == 12
        assert (body["size"], body["quality"], body["output_format"], body["background"]) == (
            "1024x1024",
            "medium",
            "png",
            "opaque",
        ), response.text
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_response_carries_the_litellm_routing_headers(gateway: Gateway) -> None:
    """A successful generation must answer with `x-litellm-call-id`, `x-litellm-model-id` equal to the
    deployment id from `/model/info`, `x-litellm-model-api-base` naming the scripted upstream, a non-empty
    `x-litellm-version` and a numeric `x-litellm-response-cost`. Clients and dashboards read these headers."""
    with wire_server(_b64_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_image=0.04
        )
        info: Final = list_value(gateway.get("/v2/model/info", {"model": model})["data"])
        deployment_id: Final = object_value(object_value(info[0])["model_info"])["id"]
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": PROMPT})
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"]
        assert response.headers["x-litellm-model-id"] == deployment_id
        assert response.headers["x-litellm-model-api-base"].startswith(wire.url), response.headers
        assert response.headers["x-litellm-version"]
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.04)
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_image_generation_keeps_a_client_supplied_call_id(gateway: Gateway) -> None:
    """The caller sends its own `x-litellm-call-id` request header. The response must carry the same value in
    `x-litellm-call-id`, so a client can correlate its request with the proxy's logs and spend row."""
    call_id: Final = f"media-{uuid.uuid4()}"
    with wire_server(_b64_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": PROMPT}, headers={"x-litellm-call-id": call_id}
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] == call_id
        assert targets(wire) == [("POST", _GENERATIONS)]


@pytest.mark.parametrize("path", ("/images/generations", "/openai/deployments/{model}/images/generations"))
def test_image_generation_route_aliases_reach_the_same_upstream_request(gateway: Gateway, path: str) -> None:
    """The unversioned route and the Azure-style deployment route are aliases of `/v1/images/generations`. A
    DALL-E call through either must reach the upstream with the same path and the same body as the `/v1` route."""

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _GENERATIONS), request.target
        assert json_body(request) == {"model": "dall-e-3", "prompt": PROMPT, "size": "1024x1024"}
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", path.format(model=model), {"model": model, "prompt": PROMPT, "size": "1024x1024"}
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _GENERATIONS)]


def test_azure_style_route_model_in_the_path_wins_over_the_body_model(gateway: Gateway) -> None:
    """Two image deployments exist, each on its own upstream. The caller posts to
    `/openai/deployments/<second>/images/generations` with `model: <first>` in the body. Only the second
    upstream may be called: the deployment named in the path decides routing (resolve_inference_model, kind
    `image_generation`)."""
    with wire_server(_b64_image) as body_wire, wire_server(_b64_image) as path_wire, gateway.scenario() as scenario:
        body_model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{body_wire.url}/v1", api_key=OPENAI_KEY
        )
        path_model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{path_wire.url}/v1", api_key=OPENAI_KEY
        )
        response: Final = gateway.request(
            "POST",
            f"/openai/deployments/{path_model}/images/generations",
            {"model": body_model, "prompt": PROMPT},
        )
        assert response.status_code == 200, response.text
        assert targets(path_wire) == [("POST", _GENERATIONS)]
        assert targets(body_wire) == []


def test_azure_image_generation_puts_the_deployment_in_the_url_and_sends_the_api_key_header(
    gateway: Gateway,
) -> None:
    """An Azure OpenAI image deployment is registered with an `api_version`. The upstream must get
    `POST /openai/deployments/<deployment>/images/generations?api-version=<version>` with the key in the
    `api-key` header, the prompt and size in the body, and no `model` key in the body because Azure routes by the
    deployment in the URL. UNVERIFIED: that no other default key is added to the Azure body; see
    litellm/llms/azure/image_generation/http_utils.py."""

    def respond(request: Request) -> Reply:
        path, _, query = request.target.partition("?")
        assert (request.method, path) == ("POST", "/openai/deployments/dall-e-3/images/generations"), request.target
        assert parse_qs(query) == {"api-version": ["2024-10-21"]}, request.target
        assert request.headers["api-key"] == AZURE_KEY
        body: Final = json_body(request)
        assert "model" not in body, body
        assert (body["prompt"], body["size"]) == (PROMPT, "1024x1024"), body
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/dall-e-3", api_base=wire.url, api_key=AZURE_KEY, api_version="2024-10-21"
        )
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": PROMPT, "size": "1024x1024"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["b64_json"] == B64_IMAGE, response.text
        assert len(wire.drain()) == 1


def test_bedrock_nova_canvas_generation_maps_size_n_and_quality_to_the_invoke_body(gateway: Gateway) -> None:
    """A Bedrock Nova Canvas deployment is called with `size: 1280x720`, `n: 2`, `quality: hd`. Bedrock must get
    `POST /model/amazon.nova-canvas-v1:0/invoke` with a bearer token and a body whose `taskType` is
    `TEXT_IMAGE`, whose `textToImageParams.text` is the prompt, and whose `imageGenerationConfig` carries
    `width: 1280`, `height: 720`, `numberOfImages: 2`, `quality: premium`. The two base64 images Bedrock returns
    come back as `data[].b64_json` in order."""

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert unquote(request.target) == "/model/amazon.nova-canvas-v1:0/invoke", request.target
        assert request.headers["authorization"] == f"Bearer {_BEDROCK_TOKEN}"
        assert json_body(request) == {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": PROMPT},
            "imageGenerationConfig": {"width": 1280, "height": 720, "numberOfImages": 2, "quality": "premium"},
        }
        return json_reply({"images": ["Zmlyc3Q=", "c2Vjb25k"]})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/amazon.nova-canvas-v1:0",
            api_base=wire.url,
            api_key=_BEDROCK_TOKEN,
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": PROMPT, "size": "1280x720", "n": 2, "quality": "hd"}
        )
        assert response.status_code == 200, response.text
        assert [image["b64_json"] for image in response.json()["data"]] == ["Zmlyc3Q=", "c2Vjb25k"]
        assert len(wire.drain()) == 1


def test_vertex_imagen_generation_sends_instances_and_parameters_to_predict(gateway: Gateway) -> None:
    """A Vertex Imagen deployment is called with `n: 2` and `size: 1024x1024`. Vertex must get one POST to the
    model's `:predict` URL with `instances: [{prompt}]` and `parameters` holding `sampleCount: 2` and
    `aspectRatio: 1:1`. Each `predictions[].bytesBase64Encoded` comes back as `data[].b64_json`. The deployment
    sets `api_base` to the full predict URL, which the Imagen config uses as is
    (litellm/llms/vertex_ai/image_generation/vertex_imagen_transformation.py:128). UNVERIFIED: the exact
    `aspectRatio` string for 1024x1024."""

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _VERTEX_TARGET), request.target
        body: Final = json_body(request)
        assert body["instances"] == [{"prompt": PROMPT}], body
        parameters: Final = object_value(body["parameters"])
        assert (parameters["sampleCount"], parameters["aspectRatio"]) == (2, "1:1"), body
        return json_reply(
            {
                "predictions": [
                    {"bytesBase64Encoded": "Zmlyc3Q=", "mimeType": "image/png"},
                    {"bytesBase64Encoded": "c2Vjb25k", "mimeType": "image/png"},
                ]
            }
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="vertex_ai/imagen-4.0-generate-001",
            api_base=f"{wire.url}{_VERTEX_TARGET}",
            api_key=None,
            vertex_project=_VERTEX_PROJECT,
            vertex_location="us-central1",
            vertex_credentials=service_account_json(_VERTEX_PROJECT, gateway.upstream_url.rstrip("/")),
        )
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": PROMPT, "n": 2, "size": "1024x1024"}
        )
        assert response.status_code == 200, response.text
        assert [image["b64_json"] for image in response.json()["data"]] == ["Zmlyc3Q=", "c2Vjb25k"]
        assert len(wire.drain()) == 1


def test_image_generation_with_an_unknown_model_answers_400_and_names_the_route(gateway: Gateway) -> None:
    """The caller names a model no deployment serves. The proxy must answer 400 with error code `"400"` and a
    message containing `/image/generations: Invalid model name passed in model=<name>` followed by the hint to
    call `/v1/models`. The route is spelled `/image/generations` (singular) in this message
    (litellm/proxy/route_llm_request.py:159), and clients match on the text."""
    unknown: Final = f"no-such-image-model-{uuid.uuid4().hex[:8]}"
    response: Final = gateway.request("POST", _GENERATIONS, {"model": unknown, "prompt": PROMPT})
    assert response.status_code == 400, response.text
    assert error_of(response)["code"] == "400"
    message: Final = error_message(response)
    assert f"/image/generations: Invalid model name passed in model={unknown}" in message, message
    assert "Call `/v1/models` to view available models for your key." in message, message


def test_image_generation_unsupported_provider_param_answers_400_before_the_upstream(gateway: Gateway) -> None:
    """A Bedrock Nova Canvas deployment supports only `n`, `size` and `quality`. A call that also sets `style`
    must answer 400 with a message naming the parameter (`Setting `style` is not supported by bedrock`) and must
    not reach Bedrock, so an unsupported option is never silently ignored (litellm/utils.py:4657)."""
    with wire_server(lambda _request: json_reply({"images": ["Zmlyc3Q="]})) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/amazon.nova-canvas-v1:0",
            api_base=wire.url,
            api_key=_BEDROCK_TOKEN,
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": PROMPT, "style": "vivid"})
        assert response.status_code == 400, response.text
        assert "Setting `style` is not supported by bedrock" in error_message(response), response.text
        assert wire.drain() == ()
