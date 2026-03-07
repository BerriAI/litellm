"""
Tests for ModelsLab video generation transformation.
All tests are mocked — no real network calls.
"""
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from litellm.llms.modelslab.videos.transformation import ModelsLabVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject


class TestModelsLabVideoTransformation:
    """Test ModelsLabVideoConfig transformation class."""

    def setup_method(self):
        self.config = ModelsLabVideoConfig()
        self.config._api_key = "test-api-key"
        self.mock_logging_obj = Mock()

    # -------------------------------------------------------------------------
    # validate_environment
    # -------------------------------------------------------------------------

    def test_validate_environment_no_auth_header(self):
        """Key-in-body auth: only Content-Type in headers, no Authorization."""
        with patch(
            "litellm.llms.modelslab.videos.transformation.get_secret_str",
            return_value="test-key",
        ):
            headers = self.config.validate_environment(
                headers={}, model="i2vgen-xl"
            )
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_validate_environment_raises_without_key(self):
        """Raises ValueError when no API key is available."""
        with patch(
            "litellm.llms.modelslab.videos.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="MODELSLAB_API_KEY"):
                self.config.validate_environment(headers={}, model="i2vgen-xl")

    # -------------------------------------------------------------------------
    # get_supported_openai_params / map_openai_params
    # -------------------------------------------------------------------------

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params("i2vgen-xl")
        assert "prompt" in params
        assert "input_reference" in params
        assert "size" in params
        assert "seconds" in params

    def test_map_openai_params_size_parsing(self):
        """'512x768' → width=512, height=768."""
        result = self.config.map_openai_params(
            video_create_optional_params={"size": "512x768"},
            model="i2vgen-xl",
            drop_params=False,
        )
        assert result["width"] == 512
        assert result["height"] == 768

    def test_map_openai_params_input_reference(self):
        """input_reference maps to init_image."""
        result = self.config.map_openai_params(
            video_create_optional_params={"input_reference": "https://example.com/img.jpg"},
            model="stable-video-diffusion",
            drop_params=False,
        )
        assert result["init_image"] == "https://example.com/img.jpg"

    # -------------------------------------------------------------------------
    # transform_video_create_request
    # -------------------------------------------------------------------------

    def test_transform_video_create_request_text2video(self):
        """text2video: key in body, model_id, prompt; URL is text2video endpoint."""
        data, files, url = self.config.transform_video_create_request(
            model="i2vgen-xl",
            prompt="A cat playing with a ball",
            api_base="https://modelslab.com/api/v6/video/text2video",
            video_create_optional_request_params={"width": 512, "height": 512},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["key"] == "test-api-key"
        assert data["model_id"] == "i2vgen-xl"
        assert data["prompt"] == "A cat playing with a ball"
        assert data["width"] == 512
        assert data["height"] == 512
        assert files == []
        assert "text2video" in url

    def test_transform_video_create_request_img2video(self):
        """img2video: init_image in body; URL is img2video endpoint."""
        data, files, url = self.config.transform_video_create_request(
            model="stable-video-diffusion",
            prompt="Camera panning right",
            api_base="https://modelslab.com/api/v6/video/img2video",
            video_create_optional_request_params={"init_image": "https://example.com/frame.jpg"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["key"] == "test-api-key"
        assert data["init_image"] == "https://example.com/frame.jpg"
        assert "img2video" in url

    def test_transform_video_create_request_img2video_auto_route(self):
        """When init_image is present, endpoint is auto-routed to img2video."""
        # Start with text2video base URL (what get_complete_url returns by default)
        base_url = "https://modelslab.com/api/v6/video/text2video"
        
        data, files, url = self.config.transform_video_create_request(
            model="stable-video-diffusion",
            prompt="Camera panning right",
            api_base=base_url,
            video_create_optional_request_params={"init_image": "https://example.com/frame.jpg"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        
        assert data["key"] == "test-api-key"
        assert data["init_image"] == "https://example.com/frame.jpg"
        # Should auto-route to img2video
        assert "img2video" in url
        assert "text2video" not in url

    def test_transform_video_create_request_text2video_no_routing(self):
        """When no init_image, stays on text2video endpoint."""
        base_url = "https://modelslab.com/api/v6/video/text2video"
        
        data, files, url = self.config.transform_video_create_request(
            model="i2vgen-xl",
            prompt="A flowing river",
            api_base=base_url,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        
        assert data["key"] == "test-api-key"
        # Should stay on text2video
        assert "text2video" in url

    # -------------------------------------------------------------------------
    # transform_video_create_response
    # -------------------------------------------------------------------------

    def test_transform_video_create_response_success(self):
        """Immediate success response → completed VideoObject with output_url."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "success",
            "request_id": "req_123",
            "output": ["https://cdn.modelslab.com/output/video.mp4"],
        }

        result = self.config.transform_video_create_response(
            model="i2vgen-xl",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="modelslab",
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result._hidden_params.get("output_url") == "https://cdn.modelslab.com/output/video.mp4"

    def test_transform_video_create_response_processing_polls(self):
        """Processing status → _poll_sync() is called → returns completed VideoObject."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "processing",
            "request_id": "req_456",
            "eta": 10,
        }

        poll_result = {
            "status": "success",
            "request_id": "req_456",
            "output": ["https://cdn.modelslab.com/output/video2.mp4"],
        }

        with patch.object(self.config, "_poll_sync", return_value=poll_result) as mock_poll:
            result = self.config.transform_video_create_response(
                model="i2vgen-xl",
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
            )
            mock_poll.assert_called_once_with("req_456")

        assert result.status == "completed"
        assert result._hidden_params.get("output_url") == "https://cdn.modelslab.com/output/video2.mp4"

    # -------------------------------------------------------------------------
    # transform_video_status_retrieve_request
    # -------------------------------------------------------------------------

    def test_transform_video_status_retrieve_request(self):
        """Fetch URL includes request_id; body has 'key'."""
        from litellm.types.videos.utils import encode_video_id_with_provider

        video_id = encode_video_id_with_provider("req_789", "modelslab", "i2vgen-xl")

        url, body = self.config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base="https://modelslab.com/api/v6/video",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "req_789" in url
        assert "fetch" in url
        assert body["key"] == "test-api-key"

    def test_transform_video_create_response_poll_error_surfaces_message(self):
        """When _poll_sync returns status=error, the actual error message is raised."""
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "processing",
            "request_id": "req_err",
        }

        poll_result = {
            "status": "error",
            "message": "Insufficient credits",
        }

        with patch.object(self.config, "_poll_sync", return_value=poll_result):
            with pytest.raises(BaseLLMException) as exc_info:
                self.config.transform_video_create_response(
                    model="i2vgen-xl",
                    raw_response=mock_response,
                    logging_obj=self.mock_logging_obj,
                )

        assert "Insufficient credits" in str(exc_info.value)

    def test_poll_sync_wraps_http_error_as_base_llm_exception(self):
        """_poll_sync wraps httpx.HTTPStatusError in BaseLLMException."""
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        mock_http_resp = Mock(spec=httpx.Response)
        mock_http_resp.status_code = 403
        mock_http_resp.text = "Forbidden"
        mock_http_resp.headers = {}

        mock_client = Mock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "403 Forbidden",
            request=Mock(),
            response=mock_http_resp,
        )

        with patch(
            "litellm.llms.modelslab.videos.transformation._get_httpx_client",
            return_value=mock_client,
        ):
            with pytest.raises(BaseLLMException) as exc_info:
                self.config._poll_sync("req_403", timeout=10, interval=0)

        assert exc_info.value.status_code == 403
        assert "ModelsLab poll request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_transform_video_create_response_success(self):
        """async_transform_video_create_response returns completed VideoObject on immediate success."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "success",
            "request_id": "req_async_ok",
            "output": ["https://cdn.modelslab.com/output/async_video.mp4"],
        }

        result = await self.config.async_transform_video_create_response(
            model="i2vgen-xl",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="modelslab",
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result._hidden_params.get("output_url") == "https://cdn.modelslab.com/output/async_video.mp4"

    @pytest.mark.asyncio
    async def test_async_transform_video_create_response_polls_with_asyncio_sleep(self):
        """async_transform_video_create_response uses _poll_async (asyncio.sleep, not time.sleep)."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "processing",
            "request_id": "req_async_poll",
        }

        poll_result = {
            "status": "success",
            "request_id": "req_async_poll",
            "output": ["https://cdn.modelslab.com/output/polled.mp4"],
        }

        with patch.object(self.config, "_poll_async", new=AsyncMock(return_value=poll_result)) as mock_poll:
            result = await self.config.async_transform_video_create_response(
                model="i2vgen-xl",
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
            )
            mock_poll.assert_awaited_once_with("req_async_poll")

        assert result.status == "completed"
        assert result._hidden_params.get("output_url") == "https://cdn.modelslab.com/output/polled.mp4"

    @pytest.mark.asyncio
    async def test_async_transform_video_create_response_poll_error_surfaces_message(self):
        """async path: when _poll_async returns status=error, error message is raised."""
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "processing",
            "request_id": "req_async_err",
        }

        poll_result = {
            "status": "error",
            "message": "Model overloaded",
        }

        with patch.object(self.config, "_poll_async", new=AsyncMock(return_value=poll_result)):
            with pytest.raises(BaseLLMException) as exc_info:
                await self.config.async_transform_video_create_response(
                    model="i2vgen-xl",
                    raw_response=mock_response,
                    logging_obj=self.mock_logging_obj,
                )

        assert "Model overloaded" in str(exc_info.value)
