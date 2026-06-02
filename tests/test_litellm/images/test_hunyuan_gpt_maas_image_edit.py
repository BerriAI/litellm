"""Unit tests for Tencent Hunyuan GPT-Maas image edit provider."""

import io
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.hunyuan_gpt_maas.image_edit import (
    HunyuanGptMaasImageEdit,
    HunyuanGptMaasImageEditConfig,
    get_hunyuan_gpt_maas_image_edit_config,
    hunyuan_gpt_maas_image_edit,
)
from litellm.llms.hunyuan_gpt_maas.image_edit.transformation import (
    _image_to_param,
    _image_to_url,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


# ---------------------------------------------------------------------------
# HunyuanGptMaasImageEditConfig (transformation only)
# ---------------------------------------------------------------------------


class TestHunyuanGptMaasImageEditConfig:
    def setup_method(self):
        self.cfg = HunyuanGptMaasImageEditConfig()

    def test_use_multipart_form_data_returns_false(self):
        assert self.cfg.use_multipart_form_data() is False

    def test_get_complete_url_default(self):
        url = self.cfg.get_complete_url(
            model="custom-imagemodel-gt", api_base=None, litellm_params={}
        )
        assert url == "https://tokenhub.tencentmaas.com/v1/aiart/gtimage"

    def test_get_complete_url_custom_base(self):
        url = self.cfg.get_complete_url(
            model="custom-imagemodel-gt",
            api_base="https://custom.api.com",
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/aiart/gtimage"

    def test_validate_environment_from_env(self):
        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test-456"
        headers = self.cfg.validate_environment(
            headers={}, model="custom-imagemodel-gt"
        )
        assert headers["Authorization"] == "Bearer sk-test-456"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_explicit_key(self):
        headers = self.cfg.validate_environment(
            headers={}, model="custom-imagemodel-gt", api_key="sk-explicit"
        )
        assert headers["Authorization"] == "Bearer sk-explicit"

    def test_validate_environment_missing_key(self):
        env_backup = os.environ.pop("HUNYUAN_GPT_MAAS_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="HUNYUAN_GPT_MAAS_API_KEY is not set"):
                self.cfg.validate_environment(headers={}, model="custom-imagemodel-gt")
        finally:
            if env_backup:
                os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = env_backup

    def test_get_supported_openai_params(self):
        params = self.cfg.get_supported_openai_params("custom-imagemodel-gt")
        assert "quality" in params
        assert "size" in params
        assert "mask" in params
        assert "background" in params
        assert "output_format" in params

    def test_map_openai_params(self):
        result = self.cfg.map_openai_params(
            image_edit_optional_params={"quality": "high", "size": "1024x1024"},  # type: ignore
            model="custom-imagemodel-gt",
            drop_params=False,
        )
        assert result["quality"] == "high"
        assert result["size"] == "1024x1024"

    def test_transform_image_edit_request_with_url(self):
        data, files = self.cfg.transform_image_edit_request(
            model="custom-imagemodel-gt",
            prompt="将图片改为油画风格",
            image="https://example.com/source.png",
            image_edit_optional_request_params={"quality": "high"},
            litellm_params={},
            headers={},
        )
        assert data["prompt"] == "将图片改为油画风格"
        assert data["images"] == [{"image_url": "https://example.com/source.png"}]
        assert data["quality"] == "high"
        assert files == []

    def test_transform_image_edit_request_with_mask_url(self):
        data, _ = self.cfg.transform_image_edit_request(
            model="custom-imagemodel-gt",
            prompt="填充遮罩区域",
            image="https://example.com/source.png",
            image_edit_optional_request_params={
                "mask": "https://example.com/mask.png",
            },
            litellm_params={},
            headers={},
        )
        assert data["images"] == [{"image_url": "https://example.com/source.png"}]
        assert data["mask"] == {"image_url": "https://example.com/mask.png"}

    def test_transform_image_edit_request_no_image(self):
        data, _ = self.cfg.transform_image_edit_request(
            model="custom-imagemodel-gt",
            prompt="generate something",
            image=None,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )
        assert "images" not in data
        assert data["prompt"] == "generate something"

    def test_transform_image_edit_request_with_bytes(self):
        """Bytes are converted to a base64 data URL wrapped in {"image_url": ...}."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        data, files = self.cfg.transform_image_edit_request(
            model="custom-imagemodel-gt",
            prompt="edit",
            image=png_bytes,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )
        assert len(data["images"]) == 1
        assert isinstance(data["images"][0], dict)
        assert data["images"][0]["image_url"].startswith("data:image/png;base64,")
        assert files == []

    def test_transform_image_edit_request_with_file_object(self):
        """File-like objects are read and converted to a base64 data URL."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        file_obj = io.BytesIO(png_bytes)
        data, files = self.cfg.transform_image_edit_request(
            model="custom-imagemodel-gt",
            prompt="edit",
            image=file_obj,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )
        assert len(data["images"]) == 1
        assert data["images"][0]["image_url"].startswith("data:image/png;base64,")
        assert files == []

    def test_transform_image_edit_request_n_is_cast_to_int(self):
        data, _ = self.cfg.transform_image_edit_request(
            model="custom-imagemodel-gt",
            prompt="edit",
            image="https://example.com/source.png",
            image_edit_optional_request_params={"n": "2"},
            litellm_params={},
            headers={},
        )
        assert data["n"] == 2
        assert isinstance(data["n"], int)

    def test_transform_image_edit_request_rejects_local_path(self):
        with pytest.raises(ValueError, match="HTTP/HTTPS URL or data URL"):
            self.cfg.transform_image_edit_request(
                model="custom-imagemodel-gt",
                prompt="edit",
                image="/local/path/image.png",
                image_edit_optional_request_params={},
                litellm_params={},
                headers={},
            )

    def test_transform_image_edit_response_success(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "completed",
            "created_at": 1779635753,
            "data": [{"url": "https://example.com/result.png"}],
        }
        mock_logging = MagicMock()

        result = self.cfg.transform_image_edit_response(
            model="custom-imagemodel-gt",
            raw_response=mock_response,
            logging_obj=mock_logging,
        )

        assert isinstance(result, ImageResponse)
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/result.png"
        assert result.created == 1779635753

    def test_transform_image_edit_response_failed_status(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "status": "failed",
            "error_code": "FailedOperation.InnerError",
            "error_message": "服务内部错误，请重试。",
        }
        mock_logging = MagicMock()

        with pytest.raises(BaseLLMException, match="FailedOperation.InnerError"):
            self.cfg.transform_image_edit_response(
                model="custom-imagemodel-gt",
                raw_response=mock_response,
                logging_obj=mock_logging,
            )

    def test_transform_image_edit_response_multiple_images(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "completed",
            "data": [
                {"url": "https://example.com/result1.png"},
                {"url": "https://example.com/result2.png"},
            ],
        }
        mock_logging = MagicMock()

        result = self.cfg.transform_image_edit_response(
            model="custom-imagemodel-gt",
            raw_response=mock_response,
            logging_obj=mock_logging,
        )

        assert len(result.data) == 2


# ---------------------------------------------------------------------------
# _image_to_url helper
# ---------------------------------------------------------------------------


class TestImageToUrl:
    def test_http_url_passthrough(self):
        assert (
            _image_to_url("https://example.com/img.png")
            == "https://example.com/img.png"
        )

    def test_data_url_passthrough(self):
        data_url = "data:image/png;base64,abc123"
        assert _image_to_url(data_url) == data_url

    def test_bytes_converted_to_data_url(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_url(png)
        assert result.startswith("data:image/png;base64,")

    def test_file_like_object_converted_to_data_url(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_url(io.BytesIO(png))
        assert result.startswith("data:image/png;base64,")

    def test_local_path_raises(self):
        with pytest.raises(ValueError, match="HTTP/HTTPS URL or data URL"):
            _image_to_url("/local/path.png")

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="unsupported image type"):
            _image_to_url(12345)  # type: ignore


# ---------------------------------------------------------------------------
# _image_to_param helper
# ---------------------------------------------------------------------------


class TestImageToParam:
    def test_http_url_wrapped_in_dict(self):
        result = _image_to_param("https://example.com/img.png")
        assert result == {"image_url": "https://example.com/img.png"}

    def test_data_url_wrapped(self):
        data_url = "data:image/jpeg;base64,/9j/4AAQSk...."
        assert _image_to_param(data_url) == {"image_url": data_url}

    def test_bytes_produce_wrapped_dict(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_param(png)
        assert isinstance(result, dict)
        assert result["image_url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# HunyuanGptMaasImageEdit handler (sync, mocked HTTP)
# ---------------------------------------------------------------------------


class TestHunyuanGptMaasImageEditHandler:
    def setup_method(self):
        self.handler = HunyuanGptMaasImageEdit()

    def _make_success_response(self) -> MagicMock:
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.text = '{"status":"completed","created_at":1779635753,"data":[{"url":"https://example.com/edited.png"}]}'
        r.json.return_value = {
            "status": "completed",
            "created_at": 1779635753,
            "data": [{"url": "https://example.com/edited.png"}],
        }
        r.raise_for_status = MagicMock()
        return r

    def test_image_edit_sync(self):
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_gpt_maas.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = self.handler.image_edit(
                model="custom-imagemodel-gt",
                image="https://example.com/source.png",
                prompt="油画风格",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert isinstance(result, ImageResponse)
        assert result.data[0].url == "https://example.com/edited.png"
        mock_client.post.assert_called_once()

    def test_image_edit_sync_posts_to_correct_url(self):
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_gpt_maas.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            self.handler.image_edit(
                model="custom-imagemodel-gt",
                image="https://example.com/source.png",
                prompt="test",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["url"] == "https://tokenhub.tencentmaas.com/v1/aiart/gtimage"
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test"

    def test_image_edit_no_polling(self):
        """GPT-Maas API is synchronous: exactly one HTTP call should be made."""
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_gpt_maas.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            self.handler.image_edit(
                model="custom-imagemodel-gt",
                image="https://example.com/source.png",
                prompt="test",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert mock_client.post.call_count == 1

    def test_image_edit_post_call_invoked(self):
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_gpt_maas.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            self.handler.image_edit(
                model="custom-imagemodel-gt",
                image="https://example.com/source.png",
                prompt="edit me",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        mock_logging.post_call.assert_called_once()
        call_kwargs = mock_logging.post_call.call_args[1]
        assert call_kwargs["input"] == "edit me"
        assert call_kwargs["api_key"] == "sk-test"

    def test_image_edit_with_bytes(self):
        """Bytes image is converted to base64 data URL and sent to the API."""
        mock_client = MagicMock()
        mock_client.post.return_value = self._make_success_response()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_GPT_MAAS_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan_gpt_maas.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            self.handler.image_edit(
                model="custom-imagemodel-gt",
                image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
                prompt="edit",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        call_body = mock_client.post.call_args[1]["json"]
        assert call_body["images"][0]["image_url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# ProviderConfigManager integration
# ---------------------------------------------------------------------------


def test_provider_config_manager_returns_hunyuan_gpt_maas_config():
    config = ProviderConfigManager.get_provider_image_edit_config(
        model="custom-imagemodel-gt",
        provider=LlmProviders.HUNYUAN_GPT_MAAS,
    )
    assert isinstance(config, HunyuanGptMaasImageEditConfig)


def test_get_hunyuan_gpt_maas_image_edit_config_factory():
    config = get_hunyuan_gpt_maas_image_edit_config("custom-imagemodel-gt")
    assert isinstance(config, HunyuanGptMaasImageEditConfig)


def test_hunyuan_gpt_maas_image_edit_singleton():
    assert isinstance(hunyuan_gpt_maas_image_edit, HunyuanGptMaasImageEdit)
