from io import BufferedReader, BytesIO
from typing import Dict

import pytest

from litellm import image_edit
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.types.router import GenericLiteLLMParams


@pytest.fixture
def image_edit_config() -> OpenAIImageEditConfig:
    return OpenAIImageEditConfig()


def test_transform_image_edit_request_basic(image_edit_config: OpenAIImageEditConfig):
    """Test basic image edit request transformation with image and prompt"""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    image_edit_optional_request_params = {}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains model and prompt but not image
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains the image
    assert len(files) == 1
    assert files[0][0] == "image[]"  # field name
    assert files[0][1][0] == "image.png"  # filename
    assert files[0][1][1] == image  # image data
    assert "image/png" in files[0][1][2]  # content type


def test_transform_image_edit_request_with_mask(
    image_edit_config: OpenAIImageEditConfig,
):
    """Test transformation with mask parameter"""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    mask = b"fake_mask_data"
    image_edit_optional_request_params = {"mask": mask, "size": "1024x1024"}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains model, prompt, and size but not image or mask
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert data["size"] == "1024x1024"
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains both image and mask
    assert len(files) == 2

    # Find image and mask in files
    image_file = next(f for f in files if f[0] == "image[]")
    mask_file = next(f for f in files if f[0] == "mask")

    assert image_file[1][0] == "image.png"
    assert image_file[1][1] == image
    assert "image/png" in image_file[1][2]

    assert mask_file[1][0] == "mask.png"
    assert mask_file[1][1] == mask
    assert "image/png" in mask_file[1][2]


def test_transform_image_edit_request_with_buffered_reader(
    image_edit_config: OpenAIImageEditConfig,
):
    """Test transformation with BufferedReader as image input"""
    import os
    import tempfile

    model = "dall-e-2"
    prompt = "Make the background blue"

    # Create a real file to get a proper BufferedReader
    image_data = b"fake_image_data"
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(image_data)
        temp_file_path = temp_file.name

    try:
        # Open the file as BufferedReader
        with open(temp_file_path, "rb") as image_buffer:
            image_edit_optional_request_params = {}
            litellm_params = GenericLiteLLMParams()
            headers = {}

            data, files = image_edit_config.transform_image_edit_request(
                model=model,
                prompt=prompt,
                image=image_buffer,
                image_edit_optional_request_params=image_edit_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

            # Check that data contains model and prompt but not image
            assert data["model"] == model
            assert data["prompt"] == prompt
            assert "image" not in data

            # Check that files contains the image with the original filename
            assert len(files) == 1
            assert files[0][0] == "image[]"
            # Should use the buffer's name (full path from the BufferedReader.name)
            assert files[0][1][0] == temp_file_path  # Uses full path from buffer.name
            assert files[0][1][1] == image_buffer  # Should be the buffer object
            # Content type detection defaults to PNG for fake data without image headers
            assert files[0][1][2].startswith("image/")  # Should detect some image type
    finally:
        # Clean up the temp file
        os.unlink(temp_file_path)


def test_transform_image_edit_request_with_optional_params(
    image_edit_config: OpenAIImageEditConfig,
):
    """Test transformation with optional parameters like size, quality, etc."""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    image_edit_optional_request_params = {
        "size": "512x512",
        "response_format": "b64_json",
        "n": 2,
        "user": "test_user",
    }
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains all the optional parameters
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert data["size"] == "512x512"
    assert data["response_format"] == "b64_json"
    assert data["n"] == 2
    assert data["user"] == "test_user"
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains only the image
    assert len(files) == 1
    assert files[0][0] == "image[]"
    assert files[0][1][1] == image


def test_transform_image_edit_request_with_multiple_images(
    image_edit_config: OpenAIImageEditConfig,
):
    """Test transformation with multiple images and no mask"""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image1 = b"fake_image_data_1"
    image2 = b"fake_image_data_2"
    image3 = b"fake_image_data_3"
    images = [image1, image2, image3]
    image_edit_optional_request_params = {"size": "1024x1024", "n": 1}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=images,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains model, prompt, and optional params but not image or mask
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert data["size"] == "1024x1024"
    assert data["n"] == 1
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains all three images and no mask
    assert len(files) == 3

    # All files should be image entries with image[] key
    image_files = [f for f in files if f[0] == "image[]"]
    assert len(image_files) == 3

    # Check that all image data is present
    image_data_in_files = [f[1][1] for f in image_files]
    assert image1 in image_data_in_files
    assert image2 in image_data_in_files
    assert image3 in image_data_in_files

    # Check that all files have proper content type
    for file_entry in image_files:
        assert file_entry[1][0] == "image.png"  # filename
        assert file_entry[1][2].startswith("image/")  # content type


def test_transform_image_edit_request_with_mask_list(
    image_edit_config: OpenAIImageEditConfig,
):
    """Test transformation with mask as list (should take first element)"""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    mask1 = b"fake_mask_data_1"
    mask2 = b"fake_mask_data_2"
    image_edit_optional_request_params = {"mask": [mask1, mask2]}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains model and prompt but not image or mask
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains image and only the first mask
    assert len(files) == 2

    mask_file = next(f for f in files if f[0] == "mask")
    assert mask_file[1][1] == mask1  # Should be the first mask, not the second


def test_transform_image_edit_request_with_input_fidelity(
    image_edit_config: OpenAIImageEditConfig,
):
    """Test that input_fidelity is included in the data dict when provided"""
    model = "gpt-image-1"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    image_edit_optional_request_params = {"input_fidelity": "high"}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    assert data["input_fidelity"] == "high"
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert "image" not in data


def test_get_supported_openai_params_includes_input_fidelity(
    image_edit_config: OpenAIImageEditConfig,
):
    """Test that input_fidelity is in the supported params list"""
    supported = image_edit_config.get_supported_openai_params(model="gpt-image-1")
    assert "input_fidelity" in supported


def test_input_fidelity_passes_through_optional_param_filter():
    """Test that input_fidelity is not dropped by get_requested_image_edit_optional_param"""
    params = {
        "input_fidelity": "low",
        "quality": "high",
        "unknown_param": "should_be_dropped",
    }
    filtered = ImageEditRequestUtils.get_requested_image_edit_optional_param(params)
    assert filtered["input_fidelity"] == "low"
    assert filtered["quality"] == "high"
    assert "unknown_param" not in filtered


def test_transform_image_edit_request_with_markup(
    image_edit_config: OpenAIImageEditConfig,
) -> None:
    """Markup upload must be sent as a multipart file part, not a text form field."""
    import io

    png_bytes: Final = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    markup: Final = io.BytesIO(png_bytes)
    markup.name = "markup.png"
    image: Final = io.BytesIO(png_bytes)
    image.name = "square.png"

    assert "markup" in image_edit_config.get_supported_openai_params(model="gpt-image-2.5-flare")

    data, files = image_edit_config.transform_image_edit_request(
        model="gpt-image-2.5-flare",
        prompt="Apply the marked changes",
        image=[image],
        image_edit_optional_request_params={"markup": markup},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "markup" not in data
    markup_part: Final = next((f for f in files if f[0] == "markup"), None)
    assert markup_part is not None
    assert markup_part[1] == ("markup.png", markup, "image/png")


@pytest.mark.parametrize(
    ("markup_input", "expected_part"),
    [
        (("overlay.png", b"\x89PNG\r\n\x1a\n"), ("overlay.png", b"\x89PNG\r\n\x1a\n", "image/png")),
        (
            ("overlay.png", b"\x89PNG\r\n\x1a\n", "image/webp"),
            ("overlay.png", b"\x89PNG\r\n\x1a\n", "image/webp"),
        ),
        (
            (None, b"\x89PNG\r\n\x1a\n", None),
            ("markup.png", b"\x89PNG\r\n\x1a\n", "image/png"),
        ),
        (
            ("overlay.png", b"\x89PNG\r\n\x1a\n", "image/png", {"X-Custom": "1"}),
            ("overlay.png", b"\x89PNG\r\n\x1a\n", "image/png", {"X-Custom": "1"}),
        ),
    ],
)
def test_transform_image_edit_request_unpacks_markup_tuples(
    image_edit_config: OpenAIImageEditConfig,
    markup_input: tuple[object, ...],
    expected_part: tuple[object, ...],
) -> None:
    """FileTypes tuples passed as markup are unpacked rather than double-wrapped."""
    data, files = image_edit_config.transform_image_edit_request(
        model="gpt-image-2.5-flare",
        prompt="Apply the marked changes",
        image=[b"\x89PNG\r\n\x1a\n"],
        image_edit_optional_request_params={"markup": markup_input},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "markup" not in data
    markup_part: Final = next((f for f in files if f[0] == "markup"), None)
    assert markup_part is not None
    assert markup_part[1] == expected_part



def test_transform_image_edit_request_unpacks_mask_tuple(
    image_edit_config: OpenAIImageEditConfig,
) -> None:
    """A FileTypes tuple passed as mask is sent as that file, not nested as the part's content."""
    _, files = image_edit_config.transform_image_edit_request(
        model="gpt-image-2.5-flare",
        prompt="Fill the masked area",
        image=[b"\x89PNG\r\n\x1a\n"],
        image_edit_optional_request_params={"mask": ("cutout", b"\x89PNG\r\n\x1a\n")},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    mask_part: Final = next((f for f in files if f[0] == "mask"), None)
    assert mask_part is not None
    assert mask_part[1] == ("cutout", b"\x89PNG\r\n\x1a\n", "image/png")

@pytest.mark.asyncio
async def test_aimage_edit_sends_markup_as_multipart_file_not_repr() -> None:
    """End-to-end SDK aimage_edit sends markup as a real file part, never str(BytesIO)."""
    import base64
    import io
    from queue import SimpleQueue

    import httpx

    import litellm
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    png: Final = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg==",
    )
    reply: Final = {
        "created": 1,
        "data": [{"b64_json": base64.b64encode(png).decode()}],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"image_tokens": 1, "text_tokens": 0},
        },
    }
    sent: Final[SimpleQueue[httpx.Request]] = SimpleQueue()

    def respond(request: httpx.Request) -> httpx.Response:
        request.read()
        sent.put(request)
        return httpx.Response(200, json=reply)

    img: Final = io.BytesIO(png)
    img.name = "square.png"
    markup: Final = io.BytesIO(png)
    markup.name = "markup.png"

    await litellm.aimage_edit(
        model="openai/gpt-image-2.5-flare",
        prompt="Apply the marked changes",
        image=[img],
        markup=markup,
        api_key="sk-test",
        api_base="https://openai.test/v1",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(respond)),
    )

    assert sent.qsize() == 1
    req: Final = sent.get_nowait()
    assert req.headers["content-type"].startswith("multipart/form-data")
    assert b'name="markup"; filename="markup.png"' in req.content
    assert b"BytesIO" not in req.content


@pytest.mark.asyncio
async def test_aimage_edit_sends_markup_tuple_upload() -> None:
    """End-to-end SDK aimage_edit unpacks FileTypes tuple markup uploads for httpx."""
    import base64
    from queue import SimpleQueue

    import httpx

    import litellm
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    png: Final = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg==",
    )
    reply: Final = {
        "created": 1,
        "data": [{"b64_json": base64.b64encode(png).decode()}],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"image_tokens": 1, "text_tokens": 0},
        },
    }
    sent: Final[SimpleQueue[httpx.Request]] = SimpleQueue()

    def respond(request: httpx.Request) -> httpx.Response:
        request.read()
        sent.put(request)
        return httpx.Response(200, json=reply)

    await litellm.aimage_edit(
        model="openai/gpt-image-2.5-flare",
        prompt="Apply the marked changes",
        image=[("square.png", png, "image/png")],
        markup=("overlay.png", png, "image/png"),
        api_key="sk-test",
        api_base="https://openai.test/v1",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(respond)),
    )

    assert sent.qsize() == 1
    req: Final = sent.get_nowait()
    assert req.headers["content-type"].startswith("multipart/form-data")
    assert b'name="markup"; filename="overlay.png"' in req.content
    assert b'name="image[]"; filename="square.png"' in req.content
    assert png in req.content
