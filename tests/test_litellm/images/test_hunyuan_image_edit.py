"""Unit tests for Tencent Hunyuan image edit provider."""

import io
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.hunyuan.image_edit import (
    HunyuanImageEdit,
    HunyuanImageEditConfig,
    get_hunyuan_image_edit_config,
    hunyuan_image_edit,
)
from litellm.llms.hunyuan.image_edit.transformation import (
    _image_to_param,
    _image_to_url,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


# ---------------------------------------------------------------------------
# HunyuanImageEditConfig (transformation only)
# ---------------------------------------------------------------------------


class TestHunyuanImageEditConfig:
    def setup_method(self):
        self.cfg = HunyuanImageEditConfig()

    def test_use_multipart_form_data_returns_false(self):
        assert self.cfg.use_multipart_form_data() is False

    def test_get_complete_url_default(self):
        url = self.cfg.get_complete_url(
            model="gpt-image-2", api_base=None, litellm_params={}
        )
        assert url == "https://api.cloudai.tencent.com/v1/aiart/openai/image/submit"

    def test_get_complete_url_custom_base(self):
        url = self.cfg.get_complete_url(
            model="gpt-image-2",
            api_base="https://custom.api.com",
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1/aiart/openai/image/submit"

    def test_validate_environment_from_env(self):
        os.environ["HUNYUAN_API_KEY"] = "sk-test-456"
        headers = self.cfg.validate_environment(headers={}, model="gpt-image-2")
        assert headers["Authorization"] == "sk-test-456"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_explicit_key(self):
        headers = self.cfg.validate_environment(
            headers={}, model="gpt-image-2", api_key="sk-explicit"
        )
        assert headers["Authorization"] == "sk-explicit"

    def test_validate_environment_missing_key(self):
        env_backup = os.environ.pop("HUNYUAN_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="HUNYUAN_API_KEY is not set"):
                self.cfg.validate_environment(headers={}, model="gpt-image-2")
        finally:
            if env_backup:
                os.environ["HUNYUAN_API_KEY"] = env_backup

    def test_get_supported_openai_params(self):
        params = self.cfg.get_supported_openai_params("gpt-image-2")
        assert "quality" in params
        assert "size" in params
        assert "mask" in params

    def test_map_openai_params(self):
        result = self.cfg.map_openai_params(
            image_edit_optional_params={"quality": "high", "size": "1024x1024"},  # type: ignore
            model="gpt-image-2",
            drop_params=False,
        )
        assert result["quality"] == "high"
        assert result["size"] == "1024x1024"

    def test_transform_image_edit_request_with_url(self):
        data, files = self.cfg.transform_image_edit_request(
            model="gpt-image-2",
            prompt="将图片改为油画风格",
            image="https://example.com/source.png",
            image_edit_optional_request_params={"quality": "high"},
            litellm_params={},
            headers={},
        )
        assert data["prompt"] == "将图片改为油画风格"
        assert data["images"] == ["https://example.com/source.png"]
        assert data["quality"] == "high"
        assert files == []

    def test_transform_image_edit_request_with_mask_url(self):
        data, _ = self.cfg.transform_image_edit_request(
            model="gpt-image-2",
            prompt="填充遮罩区域",
            image="https://example.com/source.png",
            image_edit_optional_request_params={
                "mask": "https://example.com/mask.png",
            },
            litellm_params={},
            headers={},
        )
        assert data["images"] == ["https://example.com/source.png"]
        assert data["mask"] == "https://example.com/mask.png"

    def test_transform_image_edit_request_no_image(self):
        data, _ = self.cfg.transform_image_edit_request(
            model="gpt-image-2",
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
            model="gpt-image-2",
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
        """File-like objects are read and converted to a base64 data URL wrapped in {"image_url": ...}."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        file_obj = io.BytesIO(png_bytes)
        data, files = self.cfg.transform_image_edit_request(
            model="gpt-image-2",
            prompt="edit",
            image=file_obj,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )
        assert len(data["images"]) == 1
        assert isinstance(data["images"][0], dict)
        assert data["images"][0]["image_url"].startswith("data:image/png;base64,")
        assert files == []

    def test_transform_image_edit_request_with_base64_url(self):
        """Base64 data URL strings are wrapped in {"image_url": ...}."""
        data_url = "data:image/jpeg;base64,/9j/4AAQSk..."
        data, files = self.cfg.transform_image_edit_request(
            model="gpt-image-2",
            prompt="edit",
            image=data_url,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )
        assert data["images"] == [{"image_url": data_url}]
        assert files == []

    def test_transform_image_edit_request_n_is_cast_to_int(self):
        """n must be sent as int regardless of the input type (str or float)."""
        data, _ = self.cfg.transform_image_edit_request(
            model="gpt-image-2",
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
                model="gpt-image-2",
                prompt="edit",
                image="/local/path/image.png",
                image_edit_optional_request_params={},
                litellm_params={},
                headers={},
            )

    def test_transform_image_edit_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "DONE",
            "data": [{"url": "https://example.com/result.png"}],
        }
        mock_logging = MagicMock()

        result = self.cfg.transform_image_edit_response(
            model="gpt-image-2",
            raw_response=mock_response,
            logging_obj=mock_logging,
        )

        assert isinstance(result, ImageResponse)
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/result.png"

    def test_transform_image_edit_response_multiple_images(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "DONE",
            "data": [
                {"url": "https://example.com/result1.png"},
                {"url": "https://example.com/result2.png"},
            ],
        }
        mock_logging = MagicMock()

        result = self.cfg.transform_image_edit_response(
            model="gpt-image-2",
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

    def test_http_url_passthrough_http_scheme(self):
        assert (
            _image_to_url("http://example.com/img.png") == "http://example.com/img.png"
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

    def test_tuple_converted_to_data_url(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_url(("image.png", png))
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
    def test_http_url_passthrough(self):
        assert (
            _image_to_param("https://example.com/img.png")
            == "https://example.com/img.png"
        )

    def test_data_url_wrapped(self):
        data_url = "data:image/jpeg;base64,/9j/4AAQSk.............."
        assert _image_to_param(data_url) == {"image_url": data_url}

    def test_png_data_url_wrapped(self):
        data_url = "data:image/png;base64,abc123"
        result = _image_to_param(data_url)
        assert result == {"image_url": data_url}

    def test_bytes_produce_wrapped_dict(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_param(png)
        assert isinstance(result, dict)
        assert result["image_url"].startswith("data:image/png;base64,")

    def test_file_like_produces_wrapped_dict(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_param(io.BytesIO(png))
        assert isinstance(result, dict)
        assert result["image_url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# HunyuanImageEdit handler (polling logic, mocked HTTP)
# ---------------------------------------------------------------------------


class TestHunyuanImageEditHandler:
    def setup_method(self):
        self.handler = HunyuanImageEdit()

    def _make_submit_response(self, job_id: str = "job-123") -> MagicMock:
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {"job_id": job_id, "request_id": "req-abc"}
        r.raise_for_status = MagicMock()
        return r

    def _make_poll_response(self, status: str = "DONE") -> MagicMock:
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.json.return_value = {
            "status": status,
            "data": [{"url": "https://example.com/edited.png"}],
        }
        r.raise_for_status = MagicMock()
        return r

    def test_extract_poll_context(self):
        submit_response = self._make_submit_response("job-xyz")
        job_id, poll_headers, query_url = self.handler._extract_poll_context(
            submit_response=submit_response,
            api_key="sk-test",
            litellm_params={},
        )
        assert job_id == "job-xyz"
        assert poll_headers["Authorization"] == "sk-test"
        assert "image/query" in query_url

    def test_extract_poll_context_none_api_key_falls_back_to_env(self, monkeypatch):
        """When api_key=None, poll headers must resolve key from HUNYUAN_API_KEY env var."""
        monkeypatch.setenv("HUNYUAN_API_KEY", "sk-edit-from-env")
        submit_response = self._make_submit_response("job-env-xyz")
        job_id, poll_headers, query_url = self.handler._extract_poll_context(
            submit_response=submit_response,
            api_key=None,
            litellm_params={},
        )
        assert job_id == "job-env-xyz"
        assert poll_headers["Authorization"] == "sk-edit-from-env"

    def test_extract_poll_context_missing_job_id(self):
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.headers = {}
        r.json.return_value = {}
        with pytest.raises(BaseLLMException, match="missing job_id"):
            self.handler._extract_poll_context(r, "sk-test", {})

    def test_extract_poll_context_api_error_in_response(self):
        """API errors in the response body are surfaced as BaseLLMException."""
        r = MagicMock(spec=httpx.Response)
        r.status_code = 200
        r.headers = {}
        r.json.return_value = {
            "job_id": "",
            "error": {
                "message": "URL格式不合法。",
                "code": "InvalidParameterValue.UrlIllegal",
            },
        }
        with pytest.raises(BaseLLMException, match="URL格式不合法"):
            self.handler._extract_poll_context(r, "sk-test", {})

    def test_poll_for_result_sync_done_immediately(self):
        submit_resp = self._make_submit_response()
        poll_resp = self._make_poll_response("DONE")
        mock_client = MagicMock()
        mock_client.post.return_value = poll_resp

        result = self.handler._poll_for_result_sync(
            submit_response=submit_resp,
            api_key="sk-test",
            litellm_params={},
            sync_client=mock_client,
        )

        assert result is poll_resp
        mock_client.post.assert_called_once()

    def test_poll_for_result_sync_fail_status(self):
        submit_resp = self._make_submit_response()
        fail_resp = self._make_poll_response("FAIL")
        fail_resp.json.return_value = {"status": "FAIL", "message": "quota exceeded"}
        mock_client = MagicMock()
        mock_client.post.return_value = fail_resp

        with pytest.raises(BaseLLMException, match="quota exceeded"):
            self.handler._poll_for_result_sync(
                submit_response=submit_resp,
                api_key="sk-test",
                litellm_params={},
                sync_client=mock_client,
            )

    def test_image_edit_sync(self):
        submit_resp = self._make_submit_response()
        poll_resp = self._make_poll_response("DONE")
        mock_client = MagicMock()
        mock_client.post.side_effect = [submit_resp, poll_resp]
        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = self.handler.image_edit(
                model="gpt-image-2",
                image="https://example.com/source.png",
                prompt="油画风格",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert isinstance(result, ImageResponse)
        assert result.data[0].url == "https://example.com/edited.png"

    def test_image_edit_sync_with_bytes(self):
        """Bytes are converted to base64 data URL and passed to the API."""
        submit_resp = self._make_submit_response()
        poll_resp = self._make_poll_response("DONE")
        mock_client = MagicMock()
        mock_client.post.side_effect = [submit_resp, poll_resp]
        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = self.handler.image_edit(
                model="gpt-image-2",
                image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
                prompt="油画风格",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert isinstance(result, ImageResponse)
        call_body = mock_client.post.call_args_list[0][1]["json"]
        assert isinstance(call_body["images"][0], dict)
        assert call_body["images"][0]["image_url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# ProviderConfigManager integration
# ---------------------------------------------------------------------------


def test_provider_config_manager_returns_hunyuan_config():
    config = ProviderConfigManager.get_provider_image_edit_config(
        model="gpt-image-2",
        provider=LlmProviders.HUNYUAN,
    )
    assert isinstance(config, HunyuanImageEditConfig)


def test_get_hunyuan_image_edit_config_factory():
    config = get_hunyuan_image_edit_config("gpt-image-2")
    assert isinstance(config, HunyuanImageEditConfig)


def test_hunyuan_image_edit_singleton():
    assert isinstance(hunyuan_image_edit, HunyuanImageEdit)


class TestHunyuanImageEditPostCall:
    """Verify logging_obj.post_call is invoked after successful polling."""

    def _make_mock_client(self, job_id: str = "job-123") -> MagicMock:
        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": job_id}
        submit_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.status_code = 200
        poll_resp.text = (
            '{"status":"DONE","data":[{"url":"https://example.com/out.png"}]}'
        )
        poll_resp.json.return_value = {
            "status": "DONE",
            "data": [{"url": "https://example.com/out.png"}],
        }
        poll_resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.post.side_effect = [submit_resp, poll_resp]
        return client

    def test_post_call_invoked_sync(self):
        handler = HunyuanImageEdit()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = handler.image_edit(
                model="gpt-image-2",
                image="https://example.com/src.png",
                prompt="edit me",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert isinstance(result, ImageResponse)
        mock_logging.post_call.assert_called_once()
        call_kwargs = mock_logging.post_call.call_args[1]
        assert call_kwargs["input"] == "edit me"
        assert call_kwargs["api_key"] == "sk-test"


# ---------------------------------------------------------------------------
# litellm_params direct params support
# ---------------------------------------------------------------------------


class TestHunyuanImageEditLitellmParams:
    """Verify provider-specific params in litellm_params.extra are merged into the image edit request body."""

    def _make_mock_client(self, job_id: str = "job-edit-extra") -> MagicMock:
        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": job_id}
        submit_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.status_code = 200
        poll_resp.text = (
            '{"status":"DONE","data":[{"url":"https://example.com/out.png"}]}'
        )
        poll_resp.json.return_value = {
            "status": "DONE",
            "data": [{"url": "https://example.com/out.png"}],
        }
        poll_resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.post.side_effect = [submit_resp, poll_resp]
        return client

    def test_litellm_params_extra_appends_new_params(self):
        """Keys in litellm_params.extra are appended to the request body."""
        handler = HunyuanImageEdit()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_edit(
                model="gpt-image-2",
                image="https://example.com/source.png",
                prompt="油画风格",
                image_edit_optional_request_params={},
                litellm_params={
                    "api_key": "sk-test",
                    "extra": {"seed": 99, "custom_option": "value"},
                },
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["seed"] == 99
        assert submit_call_body["custom_option"] == "value"

    def test_litellm_params_extra_overwrites_existing_params(self):
        """Keys in litellm_params.extra overwrite keys already in the request body."""
        handler = HunyuanImageEdit()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_edit(
                model="gpt-image-2",
                image="https://example.com/source.png",
                prompt="original",
                image_edit_optional_request_params={"quality": "standard"},
                litellm_params={
                    "api_key": "sk-test",
                    "extra": {"quality": "high", "extra_param": "extra_value"},
                },
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["quality"] == "high"
        assert submit_call_body["extra_param"] == "extra_value"

    def test_no_extra_key_does_not_affect_request(self):
        """When litellm_params has no 'extra' key, the request body is unchanged."""
        handler = HunyuanImageEdit()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_edit(
                model="gpt-image-2",
                image="https://example.com/source.png",
                prompt="test",
                image_edit_optional_request_params={"quality": "standard"},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["quality"] == "standard"
        assert "extra" not in submit_call_body

    def test_logo_add_default_is_zero(self):
        """logo_add is set to 0 by default when not provided."""
        handler = HunyuanImageEdit()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_edit(
                model="gpt-image-2",
                image="https://example.com/source.png",
                prompt="test",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["logo_add"] == 0

    def test_logo_add_overridable_via_extra(self):
        """logo_add default can be overridden via litellm_params.extra."""
        handler = HunyuanImageEdit()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_edit(
                model="gpt-image-2",
                image="https://example.com/source.png",
                prompt="test",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test", "extra": {"logo_add": 1}},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["logo_add"] == 1

    def test_multiple_provider_params_in_extra(self):
        """Multiple Hunyuan-specific params in litellm_params.extra are all forwarded."""
        handler = HunyuanImageEdit()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_edit(
                model="gpt-image-2",
                image="https://example.com/source.png",
                prompt="test",
                image_edit_optional_request_params={},
                litellm_params={
                    "api_key": "sk-test",
                    "extra": {"seed": 42, "logo_add": 1, "revise_prompt": 0},
                },
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["seed"] == 42
        assert submit_call_body["logo_add"] == 1
        assert submit_call_body["revise_prompt"] == 0
