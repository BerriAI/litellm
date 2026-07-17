import json
from typing import Optional

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.together_ai.common_utils import TogetherAIException
from litellm.llms.together_ai.image_generation.transformation import (
    TogetherAIImageGenerationConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_optional_params_image_gen

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


def test_provider_config_manager_returns_together_ai_image_generation_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="ByteDance-Seed/Seedream-4.0",
        provider=LlmProviders.TOGETHER_AI,
    )
    assert isinstance(config, TogetherAIImageGenerationConfig)


def test_map_openai_params_maps_size_to_width_height():
    config = TogetherAIImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={"size": "1792x1024", "n": 2},
        optional_params={},
        model="ByteDance-Seed/Seedream-4.0",
        drop_params=False,
    )
    assert mapped == {"width": 1792, "height": 1024, "n": 2}


def test_map_openai_params_maps_b64_json_response_format_to_base64():
    config = TogetherAIImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={"response_format": "b64_json"},
        optional_params={},
        model="ByteDance-Seed/Seedream-4.0",
        drop_params=False,
    )
    assert mapped == {"response_format": "base64"}


def test_get_optional_params_image_gen_keeps_image_input_params():
    config = TogetherAIImageGenerationConfig()
    optional_params = get_optional_params_image_gen(
        model="ByteDance-Seed/Seedream-4.0",
        n=1,
        size="1024x1024",
        custom_llm_provider="together_ai",
        provider_config=config,
        image_url="https://example.com/yosemite.png",
        steps=28,
    )
    body = config.transform_image_generation_request(
        model="ByteDance-Seed/Seedream-4.0",
        prompt="Make this look like a watercolor painting",
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )
    assert body["image_url"] == "https://example.com/yosemite.png"
    assert body["steps"] == 28
    assert body["width"] == 1024
    assert body["height"] == 1024
    assert "size" not in body
    assert "extra_body" not in body


def test_get_complete_url_appends_images_generations():
    config = TogetherAIImageGenerationConfig()
    url = config.get_complete_url(
        api_base="https://api.together.xyz/v1",
        api_key=None,
        model="ByteDance-Seed/Seedream-4.0",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.together.xyz/v1/images/generations"


def test_validate_environment_sets_bearer_auth_from_env(together_env):
    config = TogetherAIImageGenerationConfig()
    headers = config.validate_environment(
        headers={},
        model="ByteDance-Seed/Seedream-4.0",
        messages=[],
        optional_params={},
        litellm_params={},
    )
    assert headers["Authorization"] == "Bearer test-together-key"


def test_validate_environment_raises_without_key(monkeypatch: pytest.MonkeyPatch):
    for var in TOGETHER_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    config = TogetherAIImageGenerationConfig()
    with pytest.raises(TogetherAIException):
        config.validate_environment(
            headers={},
            model="ByteDance-Seed/Seedream-4.0",
            messages=[],
            optional_params={},
            litellm_params={},
        )


@pytest.mark.parametrize(
    "data_item,expected_url,expected_b64",
    [
        ({"index": 0, "url": "https://api.together.ai/imgproxy/abc.png", "type": "url"}, "https://api.together.ai/imgproxy/abc.png", None),
        ({"index": 0, "b64_json": "aGVsbG8=", "type": "b64_json"}, None, "aGVsbG8="),
    ],
)
def test_transform_image_generation_response(data_item, expected_url, expected_b64):
    config = TogetherAIImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={"id": "abc", "model": "ByteDance-Seed/Seedream-4.0", "object": "list", "data": [data_item]},
        request=httpx.Request("POST", "https://api.together.xyz/v1/images/generations"),
    )
    model_response = config.transform_image_generation_response(
        model="ByteDance-Seed/Seedream-4.0",
        raw_response=raw_response,
        model_response=litellm.ImageResponse(),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert model_response.data is not None and len(model_response.data) == 1
    assert model_response.data[0].url == expected_url
    assert model_response.data[0].b64_json == expected_b64


def test_image_generation_sends_together_shaped_json_body(together_env):
    client = RecordingHTTPHandler(
        response_json={"id": "abc", "data": [{"index": 0, "url": "https://api.together.ai/imgproxy/out.png"}]}
    )
    response = litellm.image_generation(
        model="together_ai/ByteDance-Seed/Seedream-4.0",
        prompt="Make this look like a watercolor painting",
        n=1,
        size="1024x1024",
        image_url="https://example.com/yosemite.png",
        client=client,
    )
    assert client.captured_url == "https://api.together.xyz/v1/images/generations"
    assert client.captured_json == {
        "model": "ByteDance-Seed/Seedream-4.0",
        "prompt": "Make this look like a watercolor painting",
        "n": 1,
        "width": 1024,
        "height": 1024,
        "image_url": "https://example.com/yosemite.png",
    }
    assert client.captured_headers is not None
    assert client.captured_headers["Authorization"] == "Bearer test-together-key"
    assert response.data is not None
    assert response.data[0].url == "https://api.together.ai/imgproxy/out.png"
