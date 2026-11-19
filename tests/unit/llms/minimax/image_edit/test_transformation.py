import base64
import io
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.minimax.image_edit.transformation import MinimaxImageEditConfig

EDITS_URL: Final = "https://api.minimax.io/v1/images/edits"
PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
PROMPT_TEXT_TOKENS: Final = 2950
INPUT_IMAGE_TOKENS: Final = 1024
OUTPUT_IMAGE_TOKENS: Final = 4096
SECOND_PNG_BYTES: Final = PNG_BYTES + b"second"


def _multipart_parts(request: httpx.Request) -> tuple[tuple[str, bool, bytes], ...]:
    boundary: Final = request.headers["content-type"].split("boundary=", 1)[1].encode()
    sections: Final = tuple(part.split(b"\r\n\r\n", 1) for part in request.read().split(b"--" + boundary))
    return tuple(
        (head.split(b'name="', 1)[1].split(b'"', 1)[0].decode(), b"filename=" in head, body.removesuffix(b"\r\n"))
        for head, body in (section for section in sections if len(section) == 2)
    )


def _form_fields(request: httpx.Request) -> dict[str, str]:
    return {name: value.decode() for name, is_file, value in _multipart_parts(request) if not is_file}


def _uploaded_images(request: httpx.Request) -> tuple[bytes, ...]:
    return tuple(value for name, is_file, value in _multipart_parts(request) if is_file and name == "image[]")


def _read_from(handle: io.BytesIO, size: int) -> io.BytesIO:
    handle.read(size)
    return handle


def _edit_response(model: str) -> dict[str, object]:
    return {
        "created": 1794341959,
        "data": [{"url": "https://example.com/h3-image/edit-0.jpg", "size": "1024x1024", "revised_prompt": "night"}],
        "model": model,
        "output_format": "png",
        "usage": {
            "generated_images": 1,
            "total_tokens": PROMPT_TEXT_TOKENS + INPUT_IMAGE_TOKENS + OUTPUT_IMAGE_TOKENS,
            "input_tokens": PROMPT_TEXT_TOKENS + INPUT_IMAGE_TOKENS,
            "output_tokens": OUTPUT_IMAGE_TOKENS,
            "input_tokens_details": {"text_tokens": PROMPT_TEXT_TOKENS, "image_tokens": INPUT_IMAGE_TOKENS},
            "output_tokens_details": {"image_tokens": OUTPUT_IMAGE_TOKENS},
        },
    }


def test_minimax_image_edit_uploads_every_image_as_a_multipart_file():
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=[PNG_BYTES, SECOND_PNG_BYTES],
            prompt="put the cat from picture 2 into picture 1",
            api_key="sk-fake",
            aspect_ratio="1:1",
            area="1K",
        )

    request: Final = route.calls.last.request
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert _uploaded_images(request) == (PNG_BYTES, SECOND_PNG_BYTES)
    assert _form_fields(request) == {
        "model": "MiniMax-H3-Image",
        "prompt": "put the cat from picture 2 into picture 1",
        "aspect_ratio": "1:1",
        "area": "1K",
    }


@pytest.mark.parametrize(
    ("build_image", "expected_uploads"),
    [
        (lambda path: ("dog.png", PNG_BYTES), (PNG_BYTES,)),
        (lambda path: ("dog.png", PNG_BYTES, "image/png"), (PNG_BYTES,)),
        (lambda path: path, (SECOND_PNG_BYTES,)),
        (lambda path: [("dog.png", io.BytesIO(PNG_BYTES)), path], (PNG_BYTES, SECOND_PNG_BYTES)),
        (lambda path: _read_from(io.BytesIO(PNG_BYTES), 8), (PNG_BYTES,)),
        (lambda path: ("dog.png", _read_from(io.BytesIO(PNG_BYTES), -1)), (PNG_BYTES,)),
    ],
    ids=[
        "filename_tuple",
        "filename_tuple_with_content_type",
        "path",
        "list_of_tuple_and_path",
        "partly_read_handle",
        "tuple_with_fully_read_handle",
    ],
)
def test_minimax_image_edit_uploads_the_whole_file_from_tuples_paths_and_handles(
    tmp_path: Path, build_image: Callable[[Path], object], expected_uploads: tuple[bytes, ...]
):
    image_path: Final = tmp_path / "cat.png"
    image_path.write_bytes(SECOND_PNG_BYTES)
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=build_image(image_path),
            prompt="put the cat into the picture",
            api_key="sk-fake",
        )

    assert _uploaded_images(route.calls.last.request) == expected_uploads


@pytest.mark.parametrize("model", ["MiniMax-H3-Image", "MiniMax-H3-Image-Highspeed"])
def test_minimax_image_edit_bills_its_token_rates(model: str):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response(model))
        response: Final = litellm.image_edit(
            model=f"minimax/{model}",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            aspect_ratio="1:1",
            area="1K",
            output_format="png",
        )

    request: Final = route.calls.last.request
    row: Final = litellm.model_cost[f"minimax/{model}"]
    expected_cost: Final = (
        PROMPT_TEXT_TOKENS * row["input_cost_per_token"]
        + INPUT_IMAGE_TOKENS * row["input_cost_per_image_token"]
        + OUTPUT_IMAGE_TOKENS * row["output_cost_per_image_token"]
    )
    assert _uploaded_images(request) == (PNG_BYTES,)
    assert _form_fields(request) == {
        "model": model,
        "prompt": "make it night",
        "aspect_ratio": "1:1",
        "area": "1K",
        "output_format": "png",
    }
    assert response.data[0].url == "https://example.com/h3-image/edit-0.jpg"
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("params", "expected_fields"),
    [
        ({"extra_body": {"area": "1K"}}, {"area": "1K"}),
        ({"area": "2K", "extra_body": {"area": "1K"}}, {"area": "1K"}),
        ({"extra_body": {"model": "MiniMax-H3-Image-Highspeed", "seed": 7}}, {"seed": "7"}),
        (
            {
                "extra_body": {
                    "prompt": "make it day",
                    "image": "https://example.com/day.png",
                    "seed": 7,
                }
            },
            {"seed": "7"},
        ),
    ],
)
def test_minimax_image_edit_sends_extra_body_fields_as_form_fields(
    params: dict[str, object], expected_fields: dict[str, str]
):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            **params,
        )

    request: Final = route.calls.last.request
    assert _uploaded_images(request) == (PNG_BYTES,)
    assert _form_fields(request) == {"model": "MiniMax-H3-Image", "prompt": "make it night", **expected_fields}


def test_minimax_image_edit_sends_no_prompt_from_extra_body_without_a_top_level_prompt():
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            api_key="sk-fake",
            extra_body={"prompt": "make it night", "seed": 7},
        )

    request: Final = route.calls.last.request
    assert _uploaded_images(request) == (PNG_BYTES,)
    assert _form_fields(request) == {"model": "MiniMax-H3-Image", "seed": "7"}


def test_minimax_image_edit_reports_each_failed_image_and_the_size_minimax_made():
    body: Final = {
        **_edit_response("MiniMax-H3-Image"),
        "data": [
            {"url": "https://example.com/h3-image/edit-0.jpg", "size": "1360x768", "revised_prompt": "night"},
            {"error": {"code": "1027", "message": "generated image failed the content safety check"}},
        ],
    }
    with respx.mock(assert_all_called=True) as router:
        router.post(EDITS_URL).respond(200, json=body)
        response: Final = litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            n=2,
            aspect_ratio="16:9",
        )

    assert [image.url for image in response.data] == ["https://example.com/h3-image/edit-0.jpg", None]
    assert response.data[0].revised_prompt == "night"
    assert response.data[0].provider_specific_fields is None
    assert response.data[1].provider_specific_fields == {
        "error": {"code": "1027", "message": "generated image failed the content safety check"}
    }
    assert response.size == "1360x768"
    assert response.quality is None


@pytest.mark.parametrize("stream", [True, "true"])
def test_minimax_image_edit_refuses_stream_before_calling_minimax(stream: bool | str):
    with respx.mock(assert_all_called=False) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        with pytest.raises(UnsupportedParamsError) as error:
            litellm.image_edit(
                model="minimax/MiniMax-H3-Image",
                image=PNG_BYTES,
                prompt="make it night",
                api_key="sk-fake",
                stream=stream,
            )

    assert error.value.status_code == 400
    assert route.call_count == 0


@pytest.mark.parametrize("stream", [False, "false"])
def test_minimax_image_edit_accepts_stream_off(stream: bool | str):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            stream=stream,
        )

    assert route.call_count == 1
    assert "stream" not in _form_fields(route.calls.last.request)


def test_minimax_image_edit_drops_stream_under_drop_params():
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            stream=True,
            drop_params=True,
        )

    assert route.call_count == 1
    assert "stream" not in _form_fields(route.calls.last.request)


def test_minimax_image_edit_refuses_json_images_before_calling_minimax():
    with respx.mock(assert_all_called=False) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        with pytest.raises(UnsupportedParamsError) as error:
            litellm.image_edit(
                model="minimax/MiniMax-H3-Image",
                prompt="make it blue",
                api_key="sk-fake",
                images=[{"image_url": "https://example.com/cube.png"}],
            )

    assert error.value.status_code == 400
    assert "`images`" in error.value.message
    assert route.call_count == 0


def test_minimax_image_edit_drops_json_images_under_drop_params():
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it blue",
            api_key="sk-fake",
            images=[{"image_url": "https://example.com/cube.png"}],
            drop_params=True,
        )

    assert _uploaded_images(route.calls.last.request) == (PNG_BYTES,)
    assert "images" not in _form_fields(route.calls.last.request)


def test_minimax_image_edit_raises_content_policy_error_with_status_400_on_a_safety_reject():
    with respx.mock(assert_all_called=True) as router:
        router.post(EDITS_URL).respond(
            422, json={"type": "error", "error": {"type": "error", "message": "input new_sensitive (1026)"}}
        )
        with pytest.raises(litellm.ContentPolicyViolationError) as error:
            litellm.image_edit(model="minimax/MiniMax-H3-Image", image=PNG_BYTES, prompt="x", api_key="sk-fake")

    assert error.value.status_code == 400


@pytest.mark.parametrize(
    ("api_base", "url"),
    [
        ("https://api.minimax.io", EDITS_URL),
        ("https://api.minimax.io/v1/", EDITS_URL),
        ("https://api.minimax.io/anthropic", EDITS_URL),
        ("https://api.minimax.io/anthropic/v1/messages", EDITS_URL),
        (EDITS_URL, EDITS_URL),
        ("https://gateway.example.com/minimax/v1", "https://gateway.example.com/minimax/v1/images/edits"),
    ],
)
def test_minimax_image_edit_url_from_each_api_base_shape(api_base: str, url: str):
    assert MinimaxImageEditConfig().get_complete_url(model="MiniMax-H3-Image", api_base=api_base, litellm_params={}) == url
