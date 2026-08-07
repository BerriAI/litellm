from io import BytesIO
from typing import Any, cast

import httpx
import pytest

from litellm.llms.chatgpt.image_edit import ChatGPTImageEditConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from tests.test_litellm.llms.chatgpt.chatgpt_image_test_utils import mock_logging


@pytest.fixture(autouse=True)
def _chatgpt_token_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))


def test_chatgpt_image_edit_config_registered():
    config = ProviderConfigManager.get_provider_image_edit_config(
        model="gpt-image-2",
        provider=LlmProviders.CHATGPT,
    )

    assert isinstance(config, ChatGPTImageEditConfig)


def test_chatgpt_image_edit_transforms_request():
    config = ChatGPTImageEditConfig()
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    )

    request, files = config.transform_image_edit_request(
        model="gpt-image-2",
        prompt="replace the background with a warm sunset",
        image=png_bytes,
        image_edit_optional_request_params={"size": "1024x1024"},
        litellm_params=cast(Any, {"chatgpt_responses_model": "gpt-5.5"}),
        headers={},
    )

    assert files == []
    assert request["model"] == "gpt-5.5"
    assert request["input"][0]["content"][0] == {
        "type": "input_text",
        "text": "replace the background with a warm sunset",
    }
    assert request["input"][0]["content"][1]["type"] == "input_image"
    assert request["input"][0]["content"][1]["image_url"].startswith(
        "data:image/png;base64,"
    )
    assert request["tools"] == [
        {
            "type": "image_generation",
            "model": "gpt-image-2",
            "size": "1024x1024",
        }
    ]
    assert request["tool_choice"] == {"type": "image_generation"}


def test_chatgpt_image_edit_uses_json_requests():
    config = ChatGPTImageEditConfig()

    assert config.use_multipart_form_data() is False


def test_chatgpt_image_edit_requires_image():
    config = ChatGPTImageEditConfig()

    with pytest.raises(ValueError, match="requires at least one image"):
        config.transform_image_edit_request(
            model="gpt-image-2",
            prompt="edit this image",
            image=None,
            image_edit_optional_request_params={},
            litellm_params=cast(Any, {}),
            headers={},
        )


def test_chatgpt_image_edit_delegates_environment_and_url():
    config = ChatGPTImageEditConfig()

    class FakeImageGenerationConfig:
        def validate_environment(self, **kwargs):
            assert kwargs["messages"] == []
            assert kwargs["optional_params"] == {}
            assert kwargs["litellm_params"] == {"session_id": "session-123"}
            assert kwargs["api_key"] == "api-key"
            assert kwargs["api_base"] == "https://ignored.test"
            return {"Authorization": "Bearer token"}

        def get_complete_url(self, **kwargs):
            assert kwargs["api_key"] == "api-key"
            assert kwargs["optional_params"] == {}
            return "https://chatgpt.com/backend-api/codex/responses"

    config.image_generation_config = cast(Any, FakeImageGenerationConfig())

    assert config.validate_environment(
        headers={},
        model="gpt-image-2",
        api_key="api-key",
        litellm_params={"session_id": "session-123"},
        api_base="https://ignored.test",
    ) == {"Authorization": "Bearer token"}
    assert (
        config.get_complete_url(
            model="gpt-image-2",
            api_base="https://ignored.test",
            litellm_params={"api_key": "api-key"},
        )
        == "https://chatgpt.com/backend-api/codex/responses"
    )


def test_chatgpt_image_edit_supports_image_generation_params():
    config = ChatGPTImageEditConfig()

    assert config.get_supported_openai_params("gpt-image-2") == [
        "output_format",
        "size",
    ]
    assert config.map_openai_params(
        image_edit_optional_params={
            "output_format": "png",
            "size": "1024x1024",
        },
        model="gpt-image-2",
        drop_params=False,
    ) == {
        "output_format": "png",
        "size": "1024x1024",
    }


def test_chatgpt_image_edit_drops_unsupported_params_when_requested():
    config = ChatGPTImageEditConfig()

    assert config.map_openai_params(
        image_edit_optional_params={
            "output_format": "png",
            "size": "1024x1024",
            "quality": "high",
            "n": 2,
        },
        model="gpt-image-2",
        drop_params=True,
    ) == {
        "output_format": "png",
        "size": "1024x1024",
    }


@pytest.mark.parametrize("param", ["quality", "n"])
def test_chatgpt_image_edit_rejects_unsupported_params_by_default(param):
    config = ChatGPTImageEditConfig()

    with pytest.raises(ValueError, match=f"Parameter {param} is not supported"):
        config.map_openai_params(
            image_edit_optional_params={param: "unsupported"},
            model="gpt-image-2",
            drop_params=False,
        )


def test_chatgpt_image_edit_transform_response_and_error_class():
    config = ChatGPTImageEditConfig()

    raw_response = httpx.Response(
        status_code=200,
        json={
            "output": [
                {
                    "type": "image_generation",
                    "image": "edited-image-data",
                }
            ]
        },
    )

    response = config.transform_image_edit_response(
        model="gpt-image-2",
        raw_response=raw_response,
        logging_obj=mock_logging(),
    )
    error = config.get_error_class(
        error_message="bad edit",
        status_code=400,
        headers={"x-request-id": "req-456"},
    )

    assert response.data is not None
    assert response.data[0].b64_json == "edited-image-data"
    assert isinstance(error, OpenAIError)
    assert error.status_code == 400
    assert error.message == "bad edit"


def test_chatgpt_image_edit_prepare_input_images_handles_supported_file_types(
    tmp_path,
):
    config = ChatGPTImageEditConfig()
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"path-bytes")

    bytes_io = BytesIO(b"bytes-io-data")
    bytes_io.seek(5)
    with image_path.open("rb") as buffered_reader:
        buffered_reader.seek(2)
        input_images = config._prepare_input_images(
            [
                None,
                bytes_io,
                buffered_reader,
                ("image.png", b"tuple-bytes", "image/png"),
                image_path,
            ]
        )
        assert buffered_reader.tell() == 2

    assert bytes_io.tell() == 5
    assert [image["type"] for image in input_images] == ["input_image"] * 4
    assert input_images[0]["image_url"].startswith("data:image/png;base64,")
    assert input_images[1]["image_url"].startswith("data:image/png;base64,")
    assert input_images[2]["image_url"].startswith("data:image/png;base64,")
    assert input_images[3]["image_url"].startswith("data:image/png;base64,")

    with pytest.raises(ValueError, match="Unsupported image type"):
        config._read_image_bytes(cast(Any, object()))
