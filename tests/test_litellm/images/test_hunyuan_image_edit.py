"""Unit tests for Tencent Hunyuan image edit provider."""

import base64
import io
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.llms.hunyuan.image_edit import (
    HunyuanImageEdit,
    HunyuanImageEditConfig,
    get_hunyuan_image_edit_config,
    hunyuan_image_edit,
)
from litellm.llms.hunyuan.image_edit.transformation import (
    _bytes_to_data_url,
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
        assert data["images"][0].startswith("data:image/png;base64,")
        assert files == []

    def test_transform_image_edit_request_with_file_object(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        file_obj = io.BytesIO(png_bytes)
        data, _ = self.cfg.transform_image_edit_request(
            model="gpt-image-2",
            prompt="edit",
            image=file_obj,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )
        assert data["images"][0].startswith("data:image/png;base64,")

    def test_transform_image_edit_request_rejects_local_path(self):
        with pytest.raises(ValueError, match="HTTP/HTTPS URL"):
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
# _bytes_to_data_url / _image_to_url helpers
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

    def test_bytes_png(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_url(png)
        assert result.startswith("data:image/png;base64,")
        encoded = result.split(",", 1)[1]
        assert base64.b64decode(encoded) == png

    def test_bytes_jpeg(self):
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 16
        result = _image_to_url(jpeg)
        assert result.startswith("data:image/jpeg;base64,")

    def test_bytes_webp(self):
        webp = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4
        result = _image_to_url(webp)
        assert result.startswith("data:image/webp;base64,")

    def test_bytes_unknown_defaults_to_png(self):
        unknown = b"\x00\x01\x02\x03" * 4
        result = _image_to_url(unknown)
        assert result.startswith("data:image/png;base64,")

    def test_file_like_object(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_url(io.BytesIO(png))
        assert result.startswith("data:image/png;base64,")

    def test_tuple_filename_bytes(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_url(("image.png", png))
        assert result.startswith("data:image/png;base64,")

    def test_tuple_filename_file_object(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        result = _image_to_url(("image.png", io.BytesIO(png)))
        assert result.startswith("data:image/png;base64,")

    def test_tuple_with_content_type(self):
        jpeg = b"\xff\xd8\xff" + b"\x00" * 16
        result = _image_to_url(("image.jpg", jpeg, "image/jpeg"))
        assert result.startswith("data:image/jpeg;base64,")

    def test_local_path_raises(self):
        with pytest.raises(ValueError, match="HTTP/HTTPS URL"):
            _image_to_url("/local/path.png")

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="unsupported image type"):
            _image_to_url(12345)  # type: ignore


class TestBytesToDataUrl:
    def test_png_mime(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        url = _bytes_to_data_url(png)
        assert url.startswith("data:image/png;base64,")

    def test_jpeg_mime(self):
        jpeg = b"\xff\xd8" + b"\x00" * 8
        url = _bytes_to_data_url(jpeg)
        assert url.startswith("data:image/jpeg;base64,")

    def test_webp_mime(self):
        webp = b"RIFF\x00\x00\x00\x00WEBP"
        url = _bytes_to_data_url(webp)
        assert url.startswith("data:image/webp;base64,")

    def test_unknown_defaults_png(self):
        url = _bytes_to_data_url(b"\xde\xad\xbe\xef")
        assert url.startswith("data:image/png;base64,")

    def test_base64_roundtrip(self):
        data = b"hello world"
        url = _bytes_to_data_url(data)
        encoded = url.split(",", 1)[1]
        assert base64.b64decode(encoded) == data


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
        r.json.return_value = {}
        with pytest.raises(ValueError, match="missing job_id"):
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

        from litellm.llms.base_llm.chat.transformation import BaseLLMException

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
        """Handler accepts raw bytes and converts them to base64 data URL internally."""
        submit_resp = self._make_submit_response()
        poll_resp = self._make_poll_response("DONE")
        mock_client = MagicMock()
        mock_client.post.side_effect = [submit_resp, poll_resp]
        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = self.handler.image_edit(
                model="gpt-image-2",
                image=png_bytes,
                prompt="油画风格",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert isinstance(result, ImageResponse)
        # Verify the submit payload used a base64 data URL, not raw bytes
        submit_call_kwargs = mock_client.post.call_args_list[0]
        sent_json = submit_call_kwargs[1].get("json") or submit_call_kwargs[0][1]
        images = sent_json.get("images", [])
        assert len(images) == 1
        assert images[0].startswith("data:image/png;base64,")


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

    def test_multi_image_file_objects_no_double_read(self):
        """File objects in a list must not be read twice (bug fix)."""
        handler = HunyuanImageEdit()

        png1 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        png2 = b"\x89PNG\r\n\x1a\n" + b"\xff" * 16
        file1 = io.BytesIO(png1)
        file2 = io.BytesIO(png2)

        submit_resp = MagicMock(spec=httpx.Response)
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": "job-multi"}
        submit_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.status_code = 200
        poll_resp.text = (
            '{"status":"DONE","data":[{"url":"https://example.com/multi.png"}]}'
        )
        poll_resp.json.return_value = {
            "status": "DONE",
            "data": [{"url": "https://example.com/multi.png"}],
        }
        poll_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.side_effect = [submit_resp, poll_resp]
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_edit.handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = handler.image_edit(
                model="gpt-image-2",
                image=[file1, file2],
                prompt="merge images",
                image_edit_optional_request_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert isinstance(result, ImageResponse)
        submit_call = mock_client.post.call_args_list[0]
        sent_json = submit_call[1].get("json") or submit_call[0][1]
        images = sent_json.get("images", [])
        assert len(images) == 2
        # Both images must be non-empty base64 data URLs
        for img_url in images:
            assert img_url.startswith("data:image/png;base64,")
            encoded = img_url.split(",", 1)[1]
            decoded = base64.b64decode(encoded)
            assert len(decoded) > 0, "File object was read twice (empty data)"
