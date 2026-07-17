import base64
from io import BytesIO
from typing import Optional

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.together_ai.common_utils import TogetherAIException
from litellm.llms.together_ai.image_edit.transformation import TogetherAIImageEditConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

SAMPLE_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

TOGETHER_KEY_ENV_VARS = (
    "TOGETHER_API_KEY",
    "TOGETHER_AI_API_KEY",
    "TOGETHERAI_API_KEY",
    "TOGETHER_AI_TOKEN",
)


@pytest.fixture
def together_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in TOGETHER_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("TOGETHER_AI_API_BASE", raising=False)
    monkeypatch.setenv("TOGETHERAI_API_KEY", "test-together-key")
    monkeypatch.setattr(litellm, "api_key", None)


class RecordingHTTPHandler(HTTPHandler):
    def __init__(self, response_json: dict):
        super().__init__()
        self.response_json = response_json
        self.captured_url: Optional[str] = None
        self.captured_json: Optional[dict] = None
        self.captured_headers: Optional[dict] = None

    def post(self, url, data=None, json=None, params=None, headers=None, stream=False, timeout=None, files=None, content=None, logging_obj=None):
        self.captured_url = url
        self.captured_json = json
        self.captured_headers = headers
        return httpx.Response(
            status_code=200,
            json=self.response_json,
            request=httpx.Request("POST", url),
        )


def test_provider_config_manager_returns_together_ai_image_edit_config():
    config = ProviderConfigManager.get_provider_image_edit_config(
        model="ByteDance-Seed/Seedream-4.0",
        provider=LlmProviders.TOGETHER_AI,
    )
    assert isinstance(config, TogetherAIImageEditConfig)


def test_use_multipart_form_data_is_false():
    assert TogetherAIImageEditConfig().use_multipart_form_data() is False


def test_map_openai_params_maps_size_to_width_height():
    mapped = TogetherAIImageEditConfig().map_openai_params(
        image_edit_optional_params={"size": "1024x1024", "n": 1},
        model="ByteDance-Seed/Seedream-4.0",
        drop_params=False,
    )
    assert mapped == {"width": 1024, "height": 1024, "n": 1}


def test_get_complete_url_defaults_to_together_images_generations():
    url = TogetherAIImageEditConfig().get_complete_url(
        model="ByteDance-Seed/Seedream-4.0",
        api_base=None,
        litellm_params={},
    )
    assert url == "https://api.together.xyz/v1/images/generations"


def test_validate_environment_sets_bearer_auth_from_env(together_env):
    headers = TogetherAIImageEditConfig().validate_environment(
        headers={},
        model="ByteDance-Seed/Seedream-4.0",
    )
    assert headers["Authorization"] == "Bearer test-together-key"


def test_transform_image_edit_request_builds_json_body_with_data_url():
    body, files = TogetherAIImageEditConfig().transform_image_edit_request(
        model="ByteDance-Seed/Seedream-4.0",
        prompt="Make this look like a watercolor painting",
        image=[BytesIO(SAMPLE_PNG_BYTES)],
        image_edit_optional_request_params={"width": 1024, "height": 1024},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    expected_data_url = f"data:image/png;base64,{base64.b64encode(SAMPLE_PNG_BYTES).decode('utf-8')}"
    assert body == {
        "model": "ByteDance-Seed/Seedream-4.0",
        "prompt": "Make this look like a watercolor painting",
        "image_url": expected_data_url,
        "width": 1024,
        "height": 1024,
    }
    assert files == []


def test_transform_image_edit_request_rejects_multiple_images():
    with pytest.raises(TogetherAIException):
        TogetherAIImageEditConfig().transform_image_edit_request(
            model="ByteDance-Seed/Seedream-4.0",
            prompt="edit",
            image=[BytesIO(SAMPLE_PNG_BYTES), BytesIO(SAMPLE_PNG_BYTES)],
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_transform_image_edit_request_rejects_missing_image():
    with pytest.raises(TogetherAIException):
        TogetherAIImageEditConfig().transform_image_edit_request(
            model="ByteDance-Seed/Seedream-4.0",
            prompt="edit",
            image=None,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


@pytest.mark.parametrize(
    "data_item,expected_url,expected_b64",
    [
        ({"index": 0, "url": "https://api.together.ai/imgproxy/abc.png", "type": "url"}, "https://api.together.ai/imgproxy/abc.png", None),
        ({"index": 0, "b64_json": "aGVsbG8=", "type": "b64_json"}, None, "aGVsbG8="),
    ],
)
def test_transform_image_edit_response(data_item, expected_url, expected_b64):
    raw_response = httpx.Response(
        status_code=200,
        json={"id": "abc", "model": "ByteDance-Seed/Seedream-4.0", "object": "list", "data": [data_item]},
        request=httpx.Request("POST", "https://api.together.xyz/v1/images/generations"),
    )
    model_response = TogetherAIImageEditConfig().transform_image_edit_response(
        model="ByteDance-Seed/Seedream-4.0",
        raw_response=raw_response,
        logging_obj=None,
    )
    assert model_response.data is not None and len(model_response.data) == 1
    assert model_response.data[0].url == expected_url
    assert model_response.data[0].b64_json == expected_b64


def test_image_edit_no_longer_raises_not_supported(together_env):
    client = RecordingHTTPHandler(
        response_json={"id": "abc", "data": [{"index": 0, "url": "https://api.together.ai/imgproxy/out.png"}]}
    )
    response = litellm.image_edit(
        model="together_ai/ByteDance-Seed/Seedream-4.0",
        image=BytesIO(SAMPLE_PNG_BYTES),
        prompt="Make this look like a watercolor painting",
        size="1024x1024",
        client=client,
    )
    assert client.captured_url == "https://api.together.xyz/v1/images/generations"
    assert client.captured_json is not None
    expected_data_url = f"data:image/png;base64,{base64.b64encode(SAMPLE_PNG_BYTES).decode('utf-8')}"
    assert client.captured_json["image_url"] == expected_data_url
    assert client.captured_json["model"] == "ByteDance-Seed/Seedream-4.0"
    assert client.captured_json["prompt"] == "Make this look like a watercolor painting"
    assert client.captured_json["width"] == 1024
    assert client.captured_json["height"] == 1024
    assert "size" not in client.captured_json
    assert client.captured_headers is not None
    assert client.captured_headers["Authorization"] == "Bearer test-together-key"
    assert response.data is not None
    assert response.data[0].url == "https://api.together.ai/imgproxy/out.png"
