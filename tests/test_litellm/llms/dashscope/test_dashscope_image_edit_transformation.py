"""
Unit tests for DashScope image edit support (qwen-image-edit-plus).
"""

import base64
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.dashscope.image_edit.transformation import (
    DashScopeImageEditConfig,
)
from litellm.llms.dashscope.image_generation.transformation import DEFAULT_API_BASE
from litellm.types.router import GenericLiteLLMParams


@pytest.fixture
def config() -> DashScopeImageEditConfig:
    return DashScopeImageEditConfig()


def test_provider_routing():
    from litellm.utils import ProviderConfigManager, get_llm_provider
    from litellm.types.utils import LlmProviders

    _, provider, _, _ = get_llm_provider("dashscope/qwen-image-edit-plus")
    assert provider == "dashscope"

    cfg = ProviderConfigManager.get_provider_image_edit_config(
        "qwen-image-edit-plus", LlmProviders.DASHSCOPE
    )
    assert isinstance(cfg, DashScopeImageEditConfig)


def test_uses_json_not_multipart(config: DashScopeImageEditConfig):
    assert config.use_multipart_form_data() is False


def test_get_complete_url_default(config: DashScopeImageEditConfig):
    assert config.get_complete_url("qwen-image-edit-plus", None, {}) == DEFAULT_API_BASE
    assert (
        config.get_complete_url("qwen-image-edit-plus", "https://custom/api", {})
        == "https://custom/api"
    )


def test_validate_environment_requires_key(config: DashScopeImageEditConfig):
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY is not set"):
        config.validate_environment(headers={}, model="qwen-image-edit-plus", api_key=None)

    headers = config.validate_environment(
        headers={}, model="qwen-image-edit-plus", api_key="sk-test"
    )
    assert headers["Authorization"] == "Bearer sk-test"


def test_map_openai_params_size_and_n(config: DashScopeImageEditConfig):
    mapped = config.map_openai_params(
        image_edit_optional_params={"size": "1024x1024", "n": 2, "unsupported": "x"},
        model="qwen-image-edit-plus",
        drop_params=True,
    )
    assert mapped == {"size": "1024*1024", "n": 2}


def test_transform_request_embeds_image_and_prompt(config: DashScopeImageEditConfig):
    image_bytes = b"\x89PNG\r\n\x1a\nfakepng"
    body, files = config.transform_image_edit_request(
        model="qwen-image-edit-plus",
        prompt="make it snow",
        image=image_bytes,
        image_edit_optional_request_params={"size": "1024*1024", "n": 1},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert body["model"] == "qwen-image-edit-plus"
    assert body["parameters"] == {"size": "1024*1024", "n": 1}
    content = body["input"]["messages"][0]["content"]
    # first item is the base64 data-url image, last item is the prompt
    assert content[0]["image"].startswith("data:")
    assert base64.b64encode(image_bytes).decode("utf-8") in content[0]["image"]
    assert content[-1] == {"text": "make it snow"}


def test_transform_request_multiple_images(config: DashScopeImageEditConfig):
    body, _ = config.transform_image_edit_request(
        model="qwen-image-edit-plus",
        prompt="fuse",
        image=[b"imgone", b"imgtwo"],
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    content = body["input"]["messages"][0]["content"]
    image_parts = [c for c in content if "image" in c]
    assert len(image_parts) == 2


def test_transform_request_requires_image(config: DashScopeImageEditConfig):
    with pytest.raises(ValueError, match="requires at least one image"):
        config.transform_image_edit_request(
            model="qwen-image-edit-plus",
            prompt="x",
            image=None,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_transform_response_parses_image_url(config: DashScopeImageEditConfig):
    raw = MagicMock(spec=httpx.Response)
    raw.status_code = 200
    raw.json.return_value = {
        "output": {
            "choices": [
                {"message": {"content": [{"image": "https://cdn/edited.png"}]}}
            ]
        }
    }
    resp = config.transform_image_edit_response(
        model="qwen-image-edit-plus", raw_response=raw, logging_obj=MagicMock()
    )
    assert resp.data is not None
    assert resp.data[0].url == "https://cdn/edited.png"


def test_transform_response_raises_on_api_error_in_200(config: DashScopeImageEditConfig):
    raw = MagicMock(spec=httpx.Response)
    raw.status_code = 200
    raw.headers = httpx.Headers({})
    raw.json.return_value = {"code": "InvalidParameter", "message": "bad size"}
    with pytest.raises(Exception, match="bad size"):
        config.transform_image_edit_response(
            model="qwen-image-edit-plus", raw_response=raw, logging_obj=MagicMock()
        )
