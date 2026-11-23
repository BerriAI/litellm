"""
Regression tests for https://github.com/BerriAI/litellm/issues/36493

/v1/images/edits on the openai path silently dropped unknown provider params
(e.g. seed) and the extra_body escape hatch, unlike /v1/images/generations.
"""

from collections.abc import Mapping
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final, cast

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepng"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


def _capture_image_edit_request(captured):
    def respond(request):
        captured["content_type"] = request.headers.get("content-type")
        captured["body"] = request.content
        return httpx.Response(200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})

    return respond


def _multipart_text_fields(content_type: str, body: bytes) -> dict:
    boundary = content_type.split("boundary=")[1].encode()
    return {
        part.split(b'name="')[1].split(b'"')[0].decode(): part.partition(b"\r\n\r\n")[2].rstrip(b"\r\n-").decode()
        for part in body.split(b"--" + boundary)
        if b'name="' in part and b"filename=" not in part
    }


def test_image_edit_forwards_provider_params_and_extra_body():
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured))))

    response = litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        seed=42,
        extra_body={"quality_level": "high"},
    )

    assert captured["content_type"].startswith("multipart/form-data")
    fields = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["seed"] == "42"
    assert fields["quality_level"] == "high"
    assert "extra_body" not in fields
    assert fields["model"] == "gpt-image-1"
    assert fields["prompt"] == "add a hat"
    assert b'name="image[]"' in captured["body"]
    assert response.data


def test_image_edit_keeps_an_internal_prefixed_kwarg_out_of_the_provider_request():
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured))))

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        seed=42,
        _litellm_undeclared_sentinel="internal",
    )

    fields = _multipart_text_fields(captured["content_type"], captured["body"])
    assert "_litellm_undeclared_sentinel" not in fields
    assert fields["seed"] == "42"


def test_image_edit_extra_body_takes_precedence_over_kwargs():
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured))))

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        seed=42,
        extra_body={"seed": 7},
    )

    assert _multipart_text_fields(captured["content_type"], captured["body"])["seed"] == "7"


def test_image_edit_flattens_nested_provider_params():
    """A nested value in extra_body (or a nested unknown kwarg) must be
    serialized as OpenAI-SDK bracket form fields (key[subkey]) rather than
    handed to the httpx multipart encoder, which raises 'Invalid type for
    value. Expected primitive type' on a dict and 500s the request."""
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured))))

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        extra_body={"generation_config": {"steps": 30, "guidance": True}},
    )

    fields = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["generation_config[steps]"] == "30"
    assert fields["generation_config[guidance]"] == "true"
    assert "generation_config" not in fields


def test_image_edit_forwards_bracketed_image_config_fields_to_a_provider_without_image_config() -> None:
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured)))
    )

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        **{"imageConfig[aspectRatio]": "9:16"},
    )

    fields: Final = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["imageConfig[aspectRatio]"] == "9:16"
    assert "imageConfig" not in fields


def test_image_edit_sends_no_drop_controls_or_dropped_params_to_the_provider() -> None:
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured)))
    )

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        drop_params="true",
        additional_drop_params=["seed"],
        imageConfig={"aspectRatio": "9:16"},
        seed=42,
        style_preset="noir",
        **{"imageConfig[imageSize]": "2K"},
    )

    fields: Final = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["style_preset"] == "noir"
    assert not {
        "drop_params",
        "additional_drop_params",
        "imageConfig[aspectRatio]",
        "imageConfig[imageSize]",
        "seed",
    } & fields.keys()


def test_image_edit_leaves_out_bracket_fields_whose_root_additional_drop_params_lists() -> None:
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured)))
    )

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        additional_drop_params=["imageConfig"],
        style_preset="noir",
        **{"imageConfig[aspectRatio]": "9:16"},
    )

    fields: Final = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["style_preset"] == "noir"
    assert "imageConfig[aspectRatio]" not in fields


def test_image_edit_leaves_out_a_bracket_field_additional_drop_params_lists_by_name() -> None:
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured)))
    )

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        additional_drop_params=["imageConfig[aspectRatio]"],
        **{"imageConfig[aspectRatio]": "9:16", "imageConfig[imageSize]": "2K"},
    )

    fields: Final = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["imageConfig[imageSize]"] == "2K"
    assert "imageConfig[aspectRatio]" not in fields


def test_image_edit_forwards_drop_controls_and_bracket_fields_to_a_litellm_proxy_hop() -> None:
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured)))
    )

    litellm.image_edit(
        model="litellm_proxy/gemini-3.1-flash-image",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://proxy.example/v1",
        client=client,
        drop_params=True,
        **{"imageConfig[aspectRatio]": "9:16"},
    )

    fields: Final = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["imageConfig[aspectRatio]"] == "9:16"
    assert fields["drop_params"] == "true"


def test_image_edit_forwards_scalar_array_as_repeated_fields():
    """A list-valued provider param must reach the backend as one repeated part
    per element, not collapse to its last element under dict.update."""
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured))))

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        loras=["style_a", "style_b", "style_c"],
    )

    body = captured["body"]
    assert body.count(b'name="loras"') == 3
    assert b"style_a" in body and b"style_b" in body and b"style_c" in body


@pytest.mark.asyncio
async def test_aimage_edit_forwards_extra_body():
    """aimage_edit used to drop extra_headers/extra_query/extra_body when
    building its partial, so they never reached image_edit."""
    captured = {}
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(_capture_image_edit_request(captured)))

    response = await litellm.aimage_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        seed=42,
        extra_body={"quality_level": "high"},
    )

    fields = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["seed"] == "42"
    assert fields["quality_level"] == "high"
    assert "extra_body" not in fields
    assert response.data


_IMAGE_REFERENCES: Final = ({"image_url": "https://images.example/tree.png"}, {"file_id": "file-abc123"})
_DATA_URL_REFERENCE: Final = ({"image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="},)
_MASK_REFERENCE: Final = {"image_url": "https://images.example/mask.png"}
_MARKUP_REFERENCE: Final = {"image_url": "data:image/png;base64,bWFya3Vw"}


def _sent_json_body(captured: Mapping[str, object]) -> dict[str, object]:
    content_type: Final = cast(str, captured["content_type"])
    assert content_type.split(";")[0].strip() == "application/json", (content_type, captured["body"])
    return _JSON_OBJECT.validate_json(cast(bytes, captured["body"]))


def _multipart_file_parts(content_type: str, body: bytes) -> tuple[tuple[str, bytes], ...]:
    parsed: Final = BytesParser(policy=HTTP).parsebytes(f"content-type: {content_type}\r\n\r\n".encode() + body)
    return tuple(
        (cast(str, part.get_param("name", header="content-disposition")), cast(bytes, part.get_payload(decode=True)))
        for part in parsed.iter_parts()
        if part.get_filename() is not None
    )


@pytest.mark.asyncio
async def test_aimage_edit_sends_image_and_mask_references_to_openai_unchanged() -> None:
    """regression: image and mask references with no uploaded file reach OpenAI as the same application/json body."""
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_capture_image_edit_request(captured)))

    response: Final = await litellm.aimage_edit(
        image=None,
        images=[dict(reference) for reference in _IMAGE_REFERENCES],
        mask=dict(_MASK_REFERENCE),
        model="openai/gpt-image-1",
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
    )

    assert _sent_json_body(captured) == {
        "model": "gpt-image-1",
        "prompt": "add a hat",
        "images": list(_IMAGE_REFERENCES),
        "mask": _MASK_REFERENCE,
    }
    assert [image.b64_json for image in response.data] == ["aW1n"]


@pytest.mark.asyncio
async def test_aimage_edit_sends_markup_and_output_fields_to_openai_as_json_values() -> None:
    """regression: markup and the other edit fields keep their JSON shape and types in a reference edit."""
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_capture_image_edit_request(captured)))

    await litellm.aimage_edit(
        image=None,
        images=[dict(reference) for reference in _DATA_URL_REFERENCE],
        markup=dict(_MARKUP_REFERENCE),
        model="openai/gpt-image-1",
        prompt="add a hat",
        n=2,
        quality="high",
        size="1024x1024",
        output_format="webp",
        background="transparent",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
    )

    body: Final = _sent_json_body(captured)
    expected: Final[dict[str, object]] = {
        "images": list(_DATA_URL_REFERENCE),
        "markup": _MARKUP_REFERENCE,
        "n": 2,
        "quality": "high",
        "size": "1024x1024",
        "output_format": "webp",
        "background": "transparent",
    }
    assert {key: body.get(key) for key in expected} == expected


def test_image_edit_sends_image_references_to_openai_as_a_json_body() -> None:
    """regression: the sync SDK call sends image references to OpenAI as application/json, the list unchanged."""
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured)))
    )

    response: Final = litellm.image_edit(
        images=[dict(reference) for reference in _DATA_URL_REFERENCE],
        model="openai/gpt-image-1",
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
    )

    body: Final = _sent_json_body(captured)
    assert body["images"] == list(_DATA_URL_REFERENCE)
    assert body["prompt"] == "add a hat"
    assert [image.b64_json for image in response.data] == ["aW1n"]


@pytest.mark.asyncio
async def test_aimage_edit_sends_image_references_to_an_azure_deployment_as_a_json_body() -> None:
    """regression: an Azure deployment gets image references and a mask reference as application/json."""
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_capture_image_edit_request(captured)))

    response: Final = await litellm.aimage_edit(
        image=None,
        images=[dict(reference) for reference in _IMAGE_REFERENCES],
        mask=dict(_MASK_REFERENCE),
        model="azure/image-edit-deployment",
        prompt="add a hat",
        api_key="azure-test-key",
        api_base="https://resource.example.openai.azure.com",
        api_version="2025-04-01-preview",
        client=client,
    )

    body: Final = _sent_json_body(captured)
    assert body["images"] == list(_IMAGE_REFERENCES)
    assert body["mask"] == _MASK_REFERENCE
    assert body["prompt"] == "add a hat"
    assert [image.b64_json for image in response.data] == ["aW1n"]


@pytest.mark.asyncio
async def test_aimage_edit_with_an_uploaded_file_still_reaches_openai_as_multipart() -> None:
    """guard: an uploaded image keeps going to OpenAI as multipart with its bytes in an image[] part."""
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_capture_image_edit_request(captured)))

    await litellm.aimage_edit(
        image=PNG_BYTES,
        model="openai/gpt-image-1",
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
    )

    content_type: Final = cast(str, captured["content_type"])
    assert content_type.startswith("multipart/form-data"), content_type
    assert _multipart_file_parts(content_type, cast(bytes, captured["body"])) == (("image[]", PNG_BYTES),)
    assert _multipart_text_fields(content_type, cast(bytes, captured["body"]))["prompt"] == "add a hat"
