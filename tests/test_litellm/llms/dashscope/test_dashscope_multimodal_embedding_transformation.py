"""
Unit tests for DashScope multimodal embedding transformation.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.dashscope.common_utils import DashScopeError
from litellm.llms.dashscope.embed.transformation_multimodal import (
    DEFAULT_API_BASE,
    DashScopeMultimodalEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse


# === 1. 模型识别 ===


def test_is_multimodal_embedding_vision_flash():
    assert DashScopeMultimodalEmbeddingConfig.is_multimodal_embedding(
        "tongyi-embedding-vision-flash"
    ) is True


def test_is_multimodal_embedding_vision_plus():
    assert DashScopeMultimodalEmbeddingConfig.is_multimodal_embedding(
        "tongyi-embedding-vision-plus"
    ) is True


def test_is_multimodal_embedding_text():
    assert DashScopeMultimodalEmbeddingConfig.is_multimodal_embedding(
        "text-embedding-v4"
    ) is False


def test_is_multimodal_embedding_with_prefix():
    assert DashScopeMultimodalEmbeddingConfig.is_multimodal_embedding(
        "host/tongyi-embedding-vision-flash"
    ) is True


# === 2. URL 拼接 ===


def test_get_complete_url_default():
    config = DashScopeMultimodalEmbeddingConfig()
    with patch(
        "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
        return_value=None,
    ):
        url = config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="tongyi-embedding-vision-flash",
            optional_params={},
            litellm_params={},
        )
    assert url == DEFAULT_API_BASE


def test_get_complete_url_custom_base():
    config = DashScopeMultimodalEmbeddingConfig()
    with patch(
        "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
        return_value=None,
    ):
        url = config.get_complete_url(
            api_base="https://xxx.maas.aliyuncs.com/api/v1",
            api_key="sk-test",
            model="tongyi-embedding-vision-flash",
            optional_params={},
            litellm_params={},
        )
    assert url == "https://xxx.maas.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"


def test_get_complete_url_compatible_mode():
    config = DashScopeMultimodalEmbeddingConfig()
    with patch(
        "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
        return_value=None,
    ):
        url = config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-test",
            model="tongyi-embedding-vision-flash",
            optional_params={},
            litellm_params={},
        )
    assert url == DEFAULT_API_BASE


def test_get_complete_url_env_var():
    config = DashScopeMultimodalEmbeddingConfig()
    custom_url = "https://custom.endpoint.com/api/v1/multimodal"
    with patch(
        "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
        return_value=custom_url,
    ):
        url = config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="tongyi-embedding-vision-flash",
            optional_params={},
            litellm_params={},
        )
    assert url == custom_url


def test_get_complete_url_already_has_path():
    config = DashScopeMultimodalEmbeddingConfig()
    with patch(
        "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
        return_value=None,
    ):
        url = config.get_complete_url(
            api_base="https://xxx.maas.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
            api_key="sk-test",
            model="tongyi-embedding-vision-flash",
            optional_params={},
            litellm_params={},
        )
    assert url == "https://xxx.maas.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"


# === 3. 输入归一化 ===


def test_normalize_input_item_string():
    config = DashScopeMultimodalEmbeddingConfig()
    result = config._normalize_input_item("hello")
    assert result == {"text": "hello"}


def test_normalize_input_item_dict_text():
    config = DashScopeMultimodalEmbeddingConfig()
    result = config._normalize_input_item([{"type": "text", "text": "hi"}])
    assert result == {"text": "hi"}


def test_normalize_input_item_dict_image():
    config = DashScopeMultimodalEmbeddingConfig()
    result = config._normalize_input_item(
        [{"type": "image_url", "image_url": {"url": "https://img.jpg"}}]
    )
    assert result == {"image": "https://img.jpg"}


def test_normalize_input_item_mixed():
    config = DashScopeMultimodalEmbeddingConfig()
    result = config._normalize_input_item(
        [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "https://img.jpg"}},
        ]
    )
    assert result == {
        "content_list": [
            {"text": "describe this image"},
            {"image": "https://img.jpg"},
        ]
    }


# === 4. 请求转换 ===


def test_transform_embedding_request_string_input():
    config = DashScopeMultimodalEmbeddingConfig()
    result = config.transform_embedding_request(
        model="tongyi-embedding-vision-flash",
        input=["hello"],
        optional_params={},
        headers={},
    )
    assert result == {
        "model": "tongyi-embedding-vision-flash",
        "input": {"contents": [{"text": "hello"}]},
    }


def test_transform_embedding_request_mixed_input():
    config = DashScopeMultimodalEmbeddingConfig()
    result = config.transform_embedding_request(
        model="tongyi-embedding-vision-flash",
        input=[
            [
                {"type": "text", "text": "描述"},
                {"type": "image_url", "image_url": {"url": "https://img.jpg"}},
            ]
        ],
        optional_params={},
        headers={},
    )
    assert result == {
        "model": "tongyi-embedding-vision-flash",
        "input": {
            "contents": [
                {
                    "content_list": [
                        {"text": "描述"},
                        {"image": "https://img.jpg"},
                    ]
                }
            ]
        },
    }


def test_transform_embedding_request_with_dimension():
    config = DashScopeMultimodalEmbeddingConfig()
    result = config.transform_embedding_request(
        model="tongyi-embedding-vision-flash",
        input=["hello"],
        optional_params={"dimension": 512},
        headers={},
    )
    assert result["parameters"] == {"dimension": 512}


# === 5. 响应解析 ===


def test_transform_embedding_response_success():
    config = DashScopeMultimodalEmbeddingConfig()
    payload = {
        "output": {
            "embeddings": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3], "type": "text"}
            ]
        },
        "usage": {"input_tokens": 10, "total_tokens": 10},
        "request_id": "mock-request-id",
    }
    raw = httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("POST", "https://example.com"),
    )
    result = config.transform_embedding_response(
        model="tongyi-embedding-vision-flash",
        raw_response=raw,
        model_response=EmbeddingResponse(),
        logging_obj=MagicMock(),
        api_key="sk-test",
        request_data={"input": {"contents": [{"text": "hello"}]}},
        optional_params={},
        litellm_params={},
    )
    assert result.object == "list"
    assert len(result.data) == 1
    assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert result.data[0]["index"] == 0
    assert result.model == "tongyi-embedding-vision-flash"
    assert result.usage.prompt_tokens == 10
    assert result.usage.total_tokens == 10
    assert result.id == "mock-request-id"


def test_transform_embedding_response_error():
    config = DashScopeMultimodalEmbeddingConfig()
    payload = {
        "code": "InvalidParameter",
        "message": "dimension must be between 1 and 2048",
        "request_id": "mock-request-id",
    }
    raw = httpx.Response(
        status_code=400,
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("POST", "https://example.com"),
    )
    with pytest.raises(DashScopeError) as exc:
        config.transform_embedding_response(
            model="tongyi-embedding-vision-flash",
            raw_response=raw,
            model_response=EmbeddingResponse(),
            logging_obj=MagicMock(),
            api_key="sk-test",
            request_data={},
            optional_params={},
            litellm_params={},
        )
    assert exc.value.status_code == 400
    assert "dimension must be between 1 and 2048" in exc.value.message


def test_transform_embedding_response_non_200():
    config = DashScopeMultimodalEmbeddingConfig()
    payload = {
        "code": "Unauthorized",
        "message": "Invalid API key",
        "request_id": "mock-request-id",
    }
    raw = httpx.Response(
        status_code=401,
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("POST", "https://example.com"),
    )
    with pytest.raises(DashScopeError) as exc:
        config.transform_embedding_response(
            model="tongyi-embedding-vision-flash",
            raw_response=raw,
            model_response=EmbeddingResponse(),
            logging_obj=MagicMock(),
            api_key="sk-bad",
            request_data={},
            optional_params={},
            litellm_params={},
        )
    assert exc.value.status_code == 401


# === 6. 参数映射 ===


def test_map_openai_params_dimensions():
    config = DashScopeMultimodalEmbeddingConfig()
    result = config.map_openai_params(
        non_default_params={"dimensions": 512},
        optional_params={},
        model="tongyi-embedding-vision-flash",
    )
    assert result == {"dimension": 512}


# === 7. 环境验证 ===


def test_validate_environment():
    config = DashScopeMultimodalEmbeddingConfig()
    headers = config.validate_environment(
        headers={},
        model="tongyi-embedding-vision-flash",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
    )
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_missing_key():
    config = DashScopeMultimodalEmbeddingConfig()
    with patch(
        "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            config.validate_environment(
                headers={},
                model="tongyi-embedding-vision-flash",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )
