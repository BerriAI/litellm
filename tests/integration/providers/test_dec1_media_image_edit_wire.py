"""`POST /v1/images/edits`: multipart uploads through the proxy to the provider.

The caller uploads one or more images (and optionally a mask) with text form fields. The proxy re-encodes the
request as a new multipart body for the provider: text fields keep their names and values, image files are sent
under `image[]` with the uploaded bytes unchanged and a content type sniffed from those bytes, and the mask is
sent under `mask`. `model` is a form field, or the deployment segment of the Azure-style route. The existing
`test_openai_image_edit_wire.py` covers the bracketed and repeated field spellings; this file covers parameter
passthrough, byte integrity, model resolution, access, cost and the Azure wire.
"""

from __future__ import annotations

import uuid
from hashlib import sha256
from typing import Final
from urllib.parse import parse_qs

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_media_support import (
    AZURE_KEY,
    B64_IMAGE,
    OPENAI_KEY,
    FilePart,
    error_of,
    file_parts,
    image_reply,
    logged_rows,
    png,
    response_cost,
    targets,
    text_fields,
)

_EDITS: Final = "/v1/images/edits"
_PROMPT: Final = "turn the red circle green"
_RED: Final = png((255, 0, 0, 255))
_JPEG: Final = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + bytes(range(256)) + b"\xff\xd9"


def _edited(_request: Request) -> Reply:
    return image_reply({"b64_json": B64_IMAGE})


def test_image_edit_forwards_every_openai_option_as_a_text_part(gateway: Gateway) -> None:
    """The caller uploads one image with `n`, `size`, `quality`, `background`, `input_fidelity`,
    `response_format` and `user` as form fields. The upstream multipart body must contain each of them as a text
    part with the same value (`n` arrives as the text `2`), plus the provider model name and the prompt, and no
    other text part."""

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _EDITS), request.target
        assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        assert text_fields(request) == (
            ("background", "transparent"),
            ("input_fidelity", "high"),
            ("model", "gpt-image-1"),
            ("n", "2"),
            ("prompt", _PROMPT),
            ("quality", "high"),
            ("response_format", "b64_json"),
            ("size", "1024x1536"),
            ("user", "end-user-7"),
        )
        return image_reply({"b64_json": B64_IMAGE}, {"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request_multipart(
            _EDITS,
            {
                "model": model,
                "prompt": _PROMPT,
                "n": "2",
                "size": "1024x1536",
                "quality": "high",
                "background": "transparent",
                "input_fidelity": "high",
                "response_format": "b64_json",
                "user": "end-user-7",
            },
            {"image": ("source.png", _RED, "image/png")},
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["data"]) == 2, response.text
        assert targets(wire) == [("POST", _EDITS)]


def test_image_edit_sends_the_uploaded_png_bytes_unchanged_under_image_array(gateway: Gateway) -> None:
    """The caller uploads `photo-of-a-circle.png`. The upstream must receive exactly one file part named
    `image[]` whose bytes equal the upload and whose content type is `image/png`. The provider-side filename is
    `image.png`, not the caller's filename (litellm/llms/openai/image_edit/transformation.py:67)."""

    def respond(request: Request) -> Reply:
        assert file_parts(request) == (FilePart("image[]", "image.png", "image/png", _RED),)
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request_multipart(
            _EDITS, {"model": model, "prompt": _PROMPT}, {"image": ("photo-of-a-circle.png", _RED, "image/png")}
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _EDITS)]


def test_image_edit_labels_a_jpeg_upload_by_its_bytes_not_by_the_client_content_type(gateway: Gateway) -> None:
    """The caller uploads JPEG bytes but labels the part `application/octet-stream`. The upstream file part must
    carry the same bytes with content type `image/jpeg`, because the proxy sniffs the type from the first bytes
    of the upload (litellm/images/utils.py:100)."""

    def respond(request: Request) -> Reply:
        parts: Final = file_parts(request)
        assert [(part.name, part.content_type, part.content) for part in parts] == [
            ("image[]", "image/jpeg", _JPEG)
        ], [(part.name, part.content_type, len(part.content)) for part in parts]
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request_multipart(
            _EDITS, {"model": model, "prompt": _PROMPT}, {"image": ("scan.bin", _JPEG, "application/octet-stream")}
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _EDITS)]


def test_image_edit_passes_a_four_megabyte_upload_through_intact(gateway: Gateway) -> None:
    """The caller uploads a 4 MB image. The upstream file part must have the same length and SHA-256 as the
    upload, so buffering in the proxy neither truncates nor corrupts a large image."""
    large: Final = _RED + uuid.uuid4().bytes * (4 * 1024 * 1024 // 16)
    digest: Final = sha256(large).hexdigest()

    def respond(request: Request) -> Reply:
        (part,) = file_parts(request)
        assert (len(part.content), sha256(part.content).hexdigest()) == (len(large), digest)
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request_multipart(
            _EDITS, {"model": model, "prompt": _PROMPT}, {"image": ("large.png", large, "image/png")}
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _EDITS)]


def test_image_edit_forwards_a_multiline_unicode_prompt_unchanged(gateway: Gateway) -> None:
    """The prompt contains a newline, an accented letter and a double quote. The `prompt` text part the upstream
    receives must be the identical string, so multipart re-encoding does not alter prompt text."""
    prompt: Final = 'línea uno\nsecond line with "quotes" and 漢字'

    def respond(request: Request) -> Reply:
        assert dict(text_fields(request))["prompt"] == prompt
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request_multipart(
            _EDITS, {"model": model, "prompt": prompt}, {"image": ("source.png", _RED, "image/png")}
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _EDITS)]


@pytest.mark.parametrize("field", ("image", "mask"))
def test_image_edit_refuses_an_image_or_mask_sent_as_a_text_field(gateway: Gateway, field: str) -> None:
    """The caller sends `image` (or `mask`) as a plain text form field, for example a URL, instead of a file
    part. The proxy must answer 422 with a `detail` key and must not call the upstream. UNVERIFIED: whether the
    detail is the handler's own text (`'<field>' must be provided as a multipart file upload, not a string.`,
    litellm/proxy/image_endpoints/endpoints.py:330) or the framework's upload validation list, which runs
    first; both are 422 and only the status and the absence of an upstream call are asserted."""
    with wire_server(_edited) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        files: Final = {} if field == "image" else {"image": ("source.png", _RED, "image/png")}
        response: Final = gateway.request_multipart(
            _EDITS,
            {"model": model, "prompt": _PROMPT, field: "https://images.invalid/source.png"},
            files or {"unused": ("unused.txt", b"x", "text/plain")},
        )
        assert response.status_code == 422, response.text
        assert "detail" in response.json(), response.text
        assert wire.drain() == ()


def test_image_edit_returns_the_provider_images_and_usage_unchanged(gateway: Gateway) -> None:
    """The upstream answers an edit with one base64 image and token usage. The caller must get the same
    `b64_json` and the same `input_tokens`, `output_tokens` and `total_tokens`."""

    def respond(_request: Request) -> Reply:
        return image_reply(
            {"b64_json": B64_IMAGE},
            usage={
                "input_tokens": 340,
                "input_tokens_details": {"text_tokens": 17, "image_tokens": 323},
                "output_tokens": 1056,
                "total_tokens": 1396,
            },
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request_multipart(
            _EDITS, {"model": model, "prompt": _PROMPT}, {"image": ("source.png", _RED, "image/png")}
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert response.json()["data"][0]["b64_json"] == B64_IMAGE
        usage: Final = object_value(body["usage"])
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (340, 1056, 1396)
        assert targets(wire) == [("POST", _EDITS)]


def test_image_edit_resolves_the_model_from_the_azure_style_path_without_a_model_field(gateway: Gateway) -> None:
    """The caller posts to `/openai/deployments/<model>/images/edits` and sends no `model` form field, as the
    Azure SDK does. The proxy must route to that deployment and send the provider model name upstream."""

    def respond(request: Request) -> Reply:
        assert dict(text_fields(request))["model"] == "gpt-image-1"
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request_multipart(
            f"/openai/deployments/{model}/images/edits",
            {"prompt": _PROMPT},
            {"image": ("source.png", _RED, "image/png")},
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", _EDITS)]


def test_image_edit_enforces_key_model_access_from_the_multipart_model_field(gateway: Gateway) -> None:
    """A key is limited to one image model. An edit whose multipart `model` field names another model must
    answer 403 with error type `key_model_access_denied` before any upstream call: the auth layer has to read
    `model` out of a multipart body, not only out of JSON."""
    with wire_server(_edited) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        other: Final = scenario.model(model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        key: Final = scenario.key(models=[allowed])
        upload: Final = {"image": ("source.png", _RED, "image/png")}
        denied: Final = gateway.request_multipart(_EDITS, {"model": other, "prompt": _PROMPT}, upload, key=key)
        assert denied.status_code == 403, denied.text
        assert error_of(denied)["type"] == "key_model_access_denied", denied.text
        assert wire.drain() == ()
        granted: Final = gateway.request_multipart(_EDITS, {"model": allowed, "prompt": _PROMPT}, upload, key=key)
        assert granted.status_code == 200, granted.text
        assert len(wire.drain()) == 1


def test_image_edit_bills_the_per_image_price_and_logs_an_aimage_edit_row(gateway: Gateway) -> None:
    """A DALL-E 2 edit deployment costs 0.02 per image and the edit returns two images for `n: 2`. The response
    cost header must be 0.04 and the key's spend row must have `call_type` `aimage_edit`, `status` `success` and
    `spend` 0.04."""

    def respond(_request: Request) -> Reply:
        return image_reply({"b64_json": B64_IMAGE}, {"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_image=0.02
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request_multipart(
            _EDITS,
            {"model": model, "prompt": f"{_PROMPT} {uuid.uuid4().hex}", "n": "2"},
            {"image": ("source.png", _RED, "image/png")},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(2 * 0.02)
        row: Final = logged_rows(key)[0]
        assert (row["call_type"], row["status"]) == ("aimage_edit", "success"), row
        assert float(str(row["spend"])) == pytest.approx(2 * 0.02)
        assert len(wire.drain()) == 1


def test_azure_image_edit_uses_the_deployment_path_api_version_and_api_key_header(gateway: Gateway) -> None:
    """An Azure OpenAI image deployment receives an edit. Azure must get one POST to
    `/openai/deployments/<deployment>/images/edits?api-version=<version>` with the key in `api-key`, the prompt
    as a text part and the uploaded bytes as a file part. UNVERIFIED: the exact text and file part names on the
    Azure wire; see litellm/llms/azure/image_edit/transformation.py."""

    def respond(request: Request) -> Reply:
        path, _, query = request.target.partition("?")
        assert (request.method, path) == ("POST", "/openai/deployments/gpt-image-edit/images/edits"), request.target
        assert parse_qs(query) == {"api-version": ["2025-04-01-preview"]}, request.target
        assert request.headers["api-key"] == AZURE_KEY
        assert dict(text_fields(request))["prompt"] == _PROMPT
        assert [part.content for part in file_parts(request)] == [_RED]
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-image-edit", api_base=wire.url, api_key=AZURE_KEY, api_version="2025-04-01-preview"
        )
        response: Final = gateway.request_multipart(
            _EDITS, {"model": model, "prompt": _PROMPT}, {"image": ("source.png", _RED, "image/png")}
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["b64_json"] == B64_IMAGE
        assert len(wire.drain()) == 1


def test_image_variations_route_exists_and_forwards_the_upload(gateway: Gateway) -> None:
    """The OpenAI SDK's `images.create_variation` posts a multipart image to `/v1/images/variations`. The proxy
    should route it to the deployment's `/v1/images/variations` with the image bytes unchanged."""
    pytest.skip(
        "BUG: the proxy registers no /v1/images/variations route (only the billable-route list and the public "
        "endpoint catalogue mention it), so the call answers 404 or 405 and never reaches the provider"
    )

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/images/variations"), request.target
        assert [part.content for part in file_parts(request)] == [_RED]
        return image_reply({"url": "https://images.invalid/variation.png"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request_multipart(
            "/v1/images/variations", {"model": model}, {"image": ("source.png", _RED, "image/png")}
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", "/v1/images/variations")]
