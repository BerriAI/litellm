"""
Regression tests for https://github.com/BerriAI/litellm/issues/36493

/v1/images/edits on the openai path silently dropped unknown provider params
(e.g. seed) and the extra_body escape hatch, unlike /v1/images/generations.
"""

from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepng"


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


@pytest.mark.parametrize("model", ["openai/dall-e-2", "recraft/recraftv3"])
def test_image_edit_ignores_markup_for_a_model_that_does_not_support_it(model: str) -> None:
    captured: Final[dict[str, object]] = {}  # mutable-ok: the MockTransport callback records the request
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured)))
    )

    response: Final = litellm.image_edit(
        model=model,
        image=PNG_BYTES,
        prompt="add a hat",
        markup=b"\x89PNG\r\n\x1a\nmarkup",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
    )

    assert response.data
    assert isinstance(captured["body"], bytes)
    assert b"markup" not in captured["body"]


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
