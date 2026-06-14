"""
Unit tests for DashScope image generation support (qwen-image-2.0, qwen-image-2.0-pro).

Run in docker: pytest tests/test_litellm/test_dashscope_image_generation.py -v
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.dashscope.image_generation.transformation import (
    DashScopeImageGenerationConfig,
    DEFAULT_API_BASE,
)
from litellm.types.utils import ImageObject, ImageResponse
from litellm.utils import get_llm_provider


# ---------------------------------------------------------------------------
# 1. Provider detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_string",
    [
        "dashscope/qwen-image-2.0",
        "dashscope/qwen-image-2.0-pro",
    ],
)
def test_get_llm_provider_returns_dashscope(model_string: str):
    model, provider, _, _ = get_llm_provider(model_string)
    assert provider == "dashscope", f"Expected 'dashscope', got '{provider}'"
    assert "qwen-image" in model


# ---------------------------------------------------------------------------
# 2. Model info: mode == "image_generation"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_string, custom_provider",
    [
        ("dashscope/qwen-image-2.0", "dashscope"),
        ("dashscope/qwen-image-2.0-pro", "dashscope"),
    ],
)
def test_get_model_info_mode_is_image_generation(
    model_string: str, custom_provider: str
):
    import os

    prev_env = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    prev_model_cost = litellm.model_cost
    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        info = litellm.get_model_info(
            model=model_string, custom_llm_provider=custom_provider
        )
        assert (
            info["mode"] == "image_generation"
        ), f"Expected mode='image_generation', got '{info['mode']}'"
    finally:
        if prev_env is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = prev_env
        litellm.model_cost = prev_model_cost


# ---------------------------------------------------------------------------
# 3. Request transformation
# ---------------------------------------------------------------------------


class TestDashScopeImageGenerationConfig:
    def setup_method(self):
        self.cfg = DashScopeImageGenerationConfig()

    def test_get_complete_url_default(self):
        url = self.cfg.get_complete_url(None, None, "qwen-image-2.0", {}, {})
        assert url == DEFAULT_API_BASE

    def test_get_complete_url_custom(self):
        custom = "https://custom.endpoint/generate"
        url = self.cfg.get_complete_url(custom, None, "qwen-image-2.0", {}, {})
        assert url == custom

    def test_validate_environment_sets_auth_header(self):
        headers = self.cfg.validate_environment(
            headers={},
            model="qwen-image-2.0",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test-key",
        )
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_raises_without_key(self):
        with patch(
            "litellm.llms.dashscope.image_generation.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
                self.cfg.validate_environment(
                    headers={},
                    model="qwen-image-2.0",
                    messages=[],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                )

    def test_transform_request_structure(self):
        req = self.cfg.transform_image_generation_request(
            model="qwen-image-2.0",
            prompt="a puppy on green grass",
            optional_params={"size": "1024*1024"},
            litellm_params={},
            headers={},
        )
        assert req["model"] == "qwen-image-2.0"
        messages = req["input"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == "a puppy on green grass"
        assert req["parameters"]["size"] == "1024*1024"

    def test_transform_request_empty_params(self):
        req = self.cfg.transform_image_generation_request(
            model="qwen-image-2.0-pro",
            prompt="sunset over the ocean",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert req["parameters"] == {}

    # ---------------------------------------------------------------------------
    # 4. Response transformation
    # ---------------------------------------------------------------------------

    def _make_mock_response(self, image_url: str) -> httpx.Response:
        body = {
            "status_code": 200,
            "request_id": "test-request-id",
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [{"image": image_url}],
                        },
                    }
                ]
            },
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "width": 1024,
                "height": 1024,
                "image_count": 1,
            },
        }
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = body
        return mock_resp

    def test_transform_response_extracts_url(self):
        image_url = "https://example.oss.aliyuncs.com/generated/test.png"
        mock_resp = self._make_mock_response(image_url)
        model_response = ImageResponse()
        result = self.cfg.transform_image_generation_response(
            model="qwen-image-2.0",
            raw_response=mock_resp,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.data is not None
        assert len(result.data) == 1
        assert result.data[0].url == image_url

    def test_transform_response_multiple_images(self):
        body = {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [{"image": "https://example.com/img1.png"}],
                        },
                    },
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [{"image": "https://example.com/img2.png"}],
                        },
                    },
                ]
            },
            "usage": {},
        }
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = body

        model_response = ImageResponse()
        result = self.cfg.transform_image_generation_response(
            model="qwen-image-2.0",
            raw_response=mock_resp,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/img1.png"
        assert result.data[1].url == "https://example.com/img2.png"

    def test_transform_response_raises_on_non_200_status(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.headers = {}
        mock_resp.text = '{"code":"InvalidParameter","message":"Size not supported"}'
        mock_resp.json.return_value = {
            "code": "InvalidParameter",
            "message": "Size not supported",
        }

        with pytest.raises(Exception):
            self.cfg.transform_image_generation_response(
                model="qwen-image-2.0",
                raw_response=mock_resp,
                model_response=ImageResponse(),
                logging_obj=MagicMock(),
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_transform_response_raises_on_api_error_body(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "code": "InvalidParameter",
            "message": "Size not supported",
        }

        with pytest.raises(Exception):
            self.cfg.transform_image_generation_response(
                model="qwen-image-2.0",
                raw_response=mock_resp,
                model_response=ImageResponse(),
                logging_obj=MagicMock(),
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    # ---------------------------------------------------------------------------
    # 5. OpenAI → DashScope parameter mapping
    # ---------------------------------------------------------------------------

    def test_map_openai_params_size_conversion(self):
        mapped = self.cfg.map_openai_params(
            non_default_params={"size": "1024x1024"},
            optional_params={},
            model="qwen-image-2.0",
            drop_params=False,
        )
        assert mapped["size"] == "1024*1024"

    def test_map_openai_params_n_to_image_count(self):
        mapped = self.cfg.map_openai_params(
            non_default_params={"n": 2},
            optional_params={},
            model="qwen-image-2.0",
            drop_params=False,
        )
        assert mapped["image_count"] == 2

    def test_map_openai_params_unknown_size_uses_asterisk(self):
        mapped = self.cfg.map_openai_params(
            non_default_params={"size": "768x768"},
            optional_params={},
            model="qwen-image-2.0",
            drop_params=False,
        )
        assert mapped["size"] == "768*768"

    @pytest.mark.parametrize(
        "openai_size, expected",
        [
            ("256x256", "256*256"),
            ("512x512", "512*512"),
            ("1024x1024", "1024*1024"),
            ("1792x1024", "1792*1024"),
            ("1024x1792", "1024*1792"),
            ("2048x2048", "2048*2048"),
        ],
    )
    def test_map_openai_params_size_table(self, openai_size: str, expected: str):
        mapped = self.cfg.map_openai_params(
            non_default_params={"size": openai_size},
            optional_params={},
            model="qwen-image-2.0",
            drop_params=False,
        )
        assert mapped["size"] == expected


# ---------------------------------------------------------------------------
# 6. End-to-end flow via litellm.image_generation (HTTP mocked)
# ---------------------------------------------------------------------------


def test_litellm_image_generation_dashscope_end_to_end():
    mock_response_body = {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "image": "https://dashscope-result.oss.aliyuncs.com/test.png"
                            }
                        ],
                    },
                }
            ]
        },
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "width": 1024,
            "height": 1024,
            "image_count": 1,
        },
    }

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_body
        mock_http_response.status_code = 200
        mock_http_response.headers = {}
        mock_post.return_value = mock_http_response

        response = litellm.image_generation(
            model="dashscope/qwen-image-2.0",
            prompt="a puppy playing on green grass",
            api_key="sk-test-key",
            size="1024x1024",
        )

        assert response is not None
        assert response.data is not None
        assert len(response.data) == 1
        assert (
            response.data[0].url == "https://dashscope-result.oss.aliyuncs.com/test.png"
        )

        # Verify the HTTP call was made to the DashScope endpoint
        call_args = mock_post.call_args
        called_url = (
            call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        )
        assert "dashscope" in called_url or "aliyuncs" in called_url

        # Verify request body contains DashScope format
        call_kwargs = call_args[1] if call_args[1] else {}
        if "json" in call_kwargs:
            body = call_kwargs["json"]
            assert "input" in body
            assert "messages" in body["input"]
