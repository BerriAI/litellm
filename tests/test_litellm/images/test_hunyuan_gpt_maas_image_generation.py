"""Unit tests for Tencent Hunyuan Maas image generation (text-to-image) provider."""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.hunyuan_maas.image_generation import (
    HunyuanMaasImageGeneration,
    HunyuanMaasImageGenerationConfig,
    get_hunyuan_maas_image_generation_config,
    hunyuan_maas_image_generation,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


# ---------------------------------------------------------------------------
# HunyuanMaasImageGenerationConfig (transformation only)
# ---------------------------------------------------------------------------


class TestHunyuanMaasImageGenerationConfig:
    def setup_method(self):
        self.cfg = HunyuanMaasImageGenerationConfig()

    def test_get_complete_url_default(self):
        url = self.cfg.get_complete_url(None, None, "gpt-image-2", {}, {})
        assert url == "https://tokenhub.tencentmaas.com/v1/aiart/gttext"

    def test_get_complete_url_custom_base(self):
        url = self.cfg.get_complete_url(
            "https://custom.api.com", None, "gpt-image-2", {}, {}
        )
        assert url == "https://custom.api.com/v1/aiart/gttext"

    def test_validate_environment_from_env(self):
        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test-123"
        headers = self.cfg.validate_environment(
            headers={},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={},
        )
        assert headers["Authorization"] == "Bearer sk-test-123"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_explicit_key(self):
        headers = self.cfg.validate_environment(
            headers={},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-explicit",
        )
        assert headers["Authorization"] == "Bearer sk-explicit"

    def test_validate_environment_missing_key(self):
        env_backup = os.environ.pop("HUNYUAN_GPT_MAAS_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="HUNYUAN_GPT_MAAS_API_KEY is not set"):
                self.cfg.validate_environment(
                    headers={},
                    model="gpt-image-2",
                    messages=[],
                    optional_params={},
                    litellm_params={},
                )
        finally:
            if env_backup:
                os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = env_backup

    def test_get_supported_openai_params(self):
        params = self.cfg.get_supported_openai_params("gpt-image-2")
        assert "n" in params
        assert "quality" in params
        assert "size" in params

    def test_map_openai_params(self):
        result = self.cfg.map_openai_params(
            non_default_params={"quality": "high", "size": "1024x1024"},
            optional_params={},
            model="gpt-image-2",
            drop_params=False,
        )
        assert result["quality"] == "high"
        assert result["size"] == "1024x1024"

    def test_map_openai_params_unsupported_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            self.cfg.map_openai_params(
                non_default_params={"unsupported_param": "value"},
                optional_params={},
                model="gpt-image-2",
                drop_params=False,
            )

    def test_map_openai_params_unsupported_dropped(self):
        result = self.cfg.map_openai_params(
            non_default_params={"unsupported_param": "value"},
            optional_params={},
            model="gpt-image-2",
            drop_params=True,
        )
        assert "unsupported_param" not in result

    def test_transform_image_generation_request(self):
        body = self.cfg.transform_image_generation_request(
            model="gpt-image-2",
            prompt="A dancing dog",
            optional_params={"quality": "high", "size": "1024x1024"},
            litellm_params={},
            headers={},
        )
        assert body["prompt"] == "A dancing dog"
        assert body["model"] == "gpt-image-2"
        assert body["quality"] == "high"
        assert body["size"] == "1024x1024"

    def test_transform_image_generation_request_default_model(self):
        body = self.cfg.transform_image_generation_request(
            model="",
            prompt="test",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["model"] == "gpt-image-2"

    def test_transform_image_generation_response_success(self):
        class MockResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {
                    "status": "completed",
                    "created_at": 1779635753,
                    "data": [{"url": "https://example.com/img.png"}],
                    "usage_metadata": {"input_tokens": 100, "output_tokens": 50},
                }

        model_response = ImageResponse()
        model_response.data = []
        result = self.cfg.transform_image_generation_response(
            model="gpt-image-2",
            raw_response=MockResponse(),
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/img.png"
        assert result.created == 1779635753

    def test_transform_image_generation_response_failed_status(self):
        class MockResponse:
            status_code = 400
            headers = {}

            def json(self):
                return {
                    "status": "failed",
                    "error_code": "FailedOperation.InnerError",
                    "error_message": "服务内部错误，请重试。",
                }

        model_response = ImageResponse()
        model_response.data = []
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        with pytest.raises(BaseLLMException, match="FailedOperation.InnerError"):
            self.cfg.transform_image_generation_response(
                model="gpt-image-2",
                raw_response=MockResponse(),
                model_response=model_response,
                logging_obj=MagicMock(),
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )


# ---------------------------------------------------------------------------
# HunyuanMaasImageGeneration handler (sync, mocked HTTP)
# ---------------------------------------------------------------------------


class TestHunyuanMaasImageGenerationHandler:
    def setup_method(self):
        self.handler = HunyuanMaasImageGeneration()

    def _make_success_response(self) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        r.text = '{"status":"completed","created_at":1779635753,"data":[{"url":"https://example.com/gen.png"}]}'
        r.json.return_value = {
            "status": "completed",
            "created_at": 1779635753,
            "data": [{"url": "https://example.com/gen.png"}],
        }
        r.raise_for_status = MagicMock()
        return r

    def test_image_generation_sync(self):
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_maas.image_generation.handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = self.handler.image_generation(
                model="gpt-image-2",
                prompt="A simple blue circle",
                model_response=ImageResponse(),
                optional_params={"size": "1024x1024"},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert isinstance(result, ImageResponse)
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/gen.png"
        mock_client.post.assert_called_once()

    def test_image_generation_sync_posts_to_correct_url(self):
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_maas.image_generation.handler._get_httpx_client",
            return_value=mock_client,
        ):
            self.handler.image_generation(
                model="gpt-image-2",
                prompt="test",
                model_response=ImageResponse(),
                optional_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["url"] == "https://tokenhub.tencentmaas.com/v1/aiart/gttext"
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test"

    def test_image_generation_post_call_invoked(self):
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_maas.image_generation.handler._get_httpx_client",
            return_value=mock_client,
        ):
            self.handler.image_generation(
                model="gpt-image-2",
                prompt="test prompt",
                model_response=ImageResponse(),
                optional_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        mock_logging.post_call.assert_called_once()
        call_kwargs = mock_logging.post_call.call_args[1]
        assert call_kwargs["input"] == "test prompt"
        assert call_kwargs["api_key"] == "sk-test"

    def test_image_generation_no_polling(self):
        """Maas API is synchronous: exactly one HTTP call should be made."""
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_maas.image_generation.handler._get_httpx_client",
            return_value=mock_client,
        ):
            self.handler.image_generation(
                model="gpt-image-2",
                prompt="test",
                model_response=ImageResponse(),
                optional_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert mock_client.post.call_count == 1


# ---------------------------------------------------------------------------
# Singleton & ProviderConfigManager
# ---------------------------------------------------------------------------


def test_hunyuan_maas_image_generation_singleton():
    assert isinstance(hunyuan_maas_image_generation, HunyuanMaasImageGeneration)


def test_provider_config_manager_returns_hunyuan_maas_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        "gpt-image-2", LlmProviders.HUNYUAN_MAAS
    )
    assert isinstance(config, HunyuanMaasImageGenerationConfig)


def test_hunyuan_maas_in_llm_providers():
    assert LlmProviders.HUNYUAN_MAAS == "hunyuan_maas"
    assert LlmProviders.HUNYUAN_MAAS.value == "hunyuan_maas"


def test_get_hunyuan_maas_image_generation_config_factory():
    config = get_hunyuan_maas_image_generation_config("gpt-image-2")
    assert isinstance(config, HunyuanMaasImageGenerationConfig)
