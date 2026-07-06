"""
Unit tests for ModelScope image generation configuration.

These tests validate the ModelScopeImageGenerationConfig class which handles
transformation between OpenAI-compatible format and ModelScope API format.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.modelscope.common_utils import ModelScopeError
from litellm.llms.modelscope.image_generation.transformation import (
    ModelScopeImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


class TestModelScopeImageGenerationTransformation:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = ModelScopeImageGenerationConfig()
        self.model = "modelscope/Qwen/Qwen-Image-Edit"
        self.logging_obj = MagicMock()

    def test_get_supported_openai_params(self):
        """Only size is a documented+honored OpenAI param for ModelScope."""
        supported_params = self.config.get_supported_openai_params(self.model)

        assert "size" in supported_params
        # n is silently ignored (n=2 returns 1 image); user/response_format are
        # not honored by ModelScope.
        assert "n" not in supported_params
        assert "user" not in supported_params
        assert "response_format" not in supported_params

    def test_map_openai_params(self):
        """Supported OpenAI param (size) passes through to optional_params."""
        result = self.config.map_openai_params(
            non_default_params={"size": "1024x1024"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["size"] == "1024x1024"

    def test_map_openai_params_drops_unsupported(self):
        """Unsupported OpenAI params (e.g. quality) are dropped when
        drop_params=True; ModelScope only honors `size`."""
        result = self.config.map_openai_params(
            non_default_params={"size": "1024x1024", "quality": "hd"},
            optional_params={},
            model=self.model,
            drop_params=True,
        )

        assert result["size"] == "1024x1024"
        assert "quality" not in result

    def test_get_complete_url_default(self):
        """Test that get_complete_url returns default ModelScope URL."""
        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://api-inference.modelscope.cn/v1/images/generations"

    def test_get_complete_url_with_custom_base(self):
        """Test that get_complete_url uses custom api_base."""
        custom_base = "https://custom.modelscope.cn/v1"

        result = self.config.get_complete_url(
            api_base=custom_base,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == f"{custom_base}/images/generations"

    def test_get_complete_url_with_trailing_slash(self):
        """Test that get_complete_url strips trailing slashes from base."""
        custom_base = "https://custom.modelscope.cn/v1/"

        result = self.config.get_complete_url(
            api_base=custom_base,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://custom.modelscope.cn/v1/images/generations"

    @patch("litellm.llms.modelscope.image_generation.transformation.get_secret_str")
    def test_validate_environment_with_api_key(self, mock_get_secret):
        """Test that validate_environment correctly sets authorization header."""
        headers = {}
        api_key = "test_api_key"

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=api_key,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.modelscope.image_generation.transformation.get_secret_str")
    def test_validate_environment_with_secret_key(self, mock_get_secret):
        """Test that validate_environment uses secret API key when api_key is None."""
        mock_get_secret.return_value = "secret_api_key"
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )

        assert result["Authorization"] == "Bearer secret_api_key"
        mock_get_secret.assert_called_once_with("MODELSCOPE_API_KEY")

    @patch("litellm.llms.modelscope.image_generation.transformation.get_secret_str")
    def test_validate_environment_no_api_key(self, mock_get_secret):
        """Test that validate_environment raises error when no API key is available."""
        mock_get_secret.return_value = None
        headers = {}

        with pytest.raises(ValueError) as exc_info:
            self.config.validate_environment(
                headers=headers,
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

        assert "MODELSCOPE_API_KEY is not set" in str(exc_info.value)

    def test_transform_image_generation_request_basic(self):
        """Test that transform_image_generation_request creates correct request body."""
        prompt = "A beautiful sunset over mountains"
        optional_params = {}

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["prompt"] == prompt

    def test_transform_image_generation_request_with_optional_params(self):
        """Supported OpenAI params (size) are included in the request body."""
        prompt = "A beautiful sunset"
        optional_params = {
            "size": "1024x1024",
        }

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["prompt"] == prompt
        assert result["size"] == "1024x1024"

    def test_transform_image_generation_request_ignores_internal_params(self):
        """Params starting with _ are dropped from the request body."""
        prompt = "A beautiful sunset"
        optional_params = {
            "size": "1024x1024",
            "_internal_param": "should_be_ignored",
        }

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["size"] == "1024x1024"
        assert "_internal_param" not in result

    def test_transform_image_generation_request_merges_extra_body(self):
        """extra_body fields (e.g. image_url, negative_prompt) are merged into
        the request body as top-level params, alongside supported OpenAI params
        like size. ModelScope-specific fields are not OpenAI params, so they
        arrive via extra_body and must be flattened here."""
        prompt = "add a birthday hat to the dog"
        optional_params = {
            "size": "1024x1024",
            "extra_body": {
                "image_url": ["https://example.com/dog.png"],
                "negative_prompt": "lowres, blurry",
            },
        }

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["prompt"] == prompt
        assert result["size"] == "1024x1024"
        # extra_body contents are merged in as top-level params; the extra_body
        # key itself is not
        assert result["image_url"] == ["https://example.com/dog.png"]
        assert result["negative_prompt"] == "lowres, blurry"
        assert "extra_body" not in result

    def test_transform_image_generation_request_excludes_internal_extra_body_keys(self):
        """extra_headers/extra_query are litellm control params, not ModelScope
        body fields; they must not leak into the request body even though
        get_optional_params_image_gen sweeps them into extra_body."""
        prompt = "add a birthday hat to the dog"
        optional_params = {
            "size": "1024x1024",
            "extra_body": {
                "image_url": ["https://example.com/dog.png"],
                "extra_headers": {"X-Custom-Auth": "secret-value"},
                "extra_query": {"ref": "abc"},
            },
        }

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["image_url"] == ["https://example.com/dog.png"]
        assert "extra_body" not in result
        assert "extra_headers" not in result
        assert "extra_query" not in result
        assert "secret-value" not in str(result)

    def test_transform_image_generation_response_with_url_images(self):
        """Test that transform_image_generation_response extracts output_images URLs."""
        response_data = {
            "task_status": "SUCCEED",
            "task_id": "abc-123",
            "output_images": [
                "https://example.com/image1.png",
                "https://example.com/image2.png",
            ],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/image1.png"
        assert result.data[1].url == "https://example.com/image2.png"

    def test_transform_image_generation_response_failed_status_raises(self):
        """Test that a FAILED task_status raises an error."""
        response_data = {
            "task_status": "FAILED",
            "task_id": "abc-123",
            "output_images": [],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        with pytest.raises(Exception) as exc_info:
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert "failed" in str(exc_info.value).lower()

    def test_transform_image_generation_response_empty_data(self):
        """SUCCEED with no output_images is an error, not a silent empty success."""
        response_data = {
            "task_status": "SUCCEED",
            "task_id": "abc-123",
            "output_images": [],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        with pytest.raises(ModelScopeError) as exc_info:
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert "no output_images" in str(exc_info.value).lower()

    def test_transform_image_generation_response_error_handling(self):
        """Test that transform_image_generation_response raises error on API error."""
        response_data = {
            "errors": {
                "message": "Invalid prompt provided",
                "type": "invalid_request_error",
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 400
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        with pytest.raises(Exception) as exc_info:
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert "ModelScope error" in str(exc_info.value)
        assert "Invalid prompt provided" in str(exc_info.value)

    def test_transform_image_generation_response_json_error(self):
        """Test that transform_image_generation_response raises error on invalid JSON."""
        import json

        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 500
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        with pytest.raises(Exception) as exc_info:
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert "Error parsing ModelScope response" in str(exc_info.value)

    def test_get_error_class_returns_modelscope_error(self):
        """get_error_class always returns ModelScopeError and preserves status_code."""
        for status_code in (400, 401, 404, 429, 500, 502):
            error = self.config.get_error_class(
                error_message=f"err {status_code}",
                status_code=status_code,
                headers={"Content-Type": "application/json"},
            )
            assert isinstance(error, ModelScopeError), status_code
            assert error.status_code == status_code
            assert f"err {status_code}" in error.message


class TestModelScopeImageGenerationHandler:
    """Tests for the async submit + poll flow in ModelScopeImageGeneration."""

    def setup_method(self):
        from litellm.llms.modelscope.image_generation.handler import (
            ModelScopeImageGeneration,
        )

        self.handler = ModelScopeImageGeneration()
        self.model = "Qwen/Qwen-Image-2512"
        self.prompt = "a cute baby sea otter"
        self.logging_obj = MagicMock()

    @staticmethod
    def _mock_response(json_data, status_code=200):
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.status_code = status_code
        mock_response.text = str(json_data)
        mock_response.headers = {}
        return mock_response

    def test_sync_submit_and_poll_succeed(self):
        """Submit returns task_id, poll returns SUCCEED with output_images."""
        from litellm.types.utils import ImageResponse

        submit_resp = self._mock_response({"task_id": "task-123"})
        poll_resp = self._mock_response(
            {
                "task_status": "SUCCEED",
                "task_id": "task-123",
                "output_images": ["https://example.com/otter.png"],
            }
        )

        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        result = self.handler.image_generation(
            model=self.model,
            prompt=self.prompt,
            model_response=ImageResponse(data=[]),
            optional_params={},
            litellm_params={"api_key": "ms-test", "api_base": None},
            logging_obj=self.logging_obj,
            timeout=30,
            client=mock_client,
            aimg_generation=False,
        )

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/otter.png"
        # Submit POST hits /images/generations, poll GET hits /tasks/{task_id}
        assert mock_client.post.call_count == 1
        assert "/images/generations" in mock_client.post.call_args.kwargs["url"]
        assert mock_client.get.call_count == 1
        assert "/tasks/task-123" in mock_client.get.call_args.kwargs["url"]

    def test_sync_submit_returns_error_status_raises(self):
        """A 4xx/5xx on the submit call raises immediately."""
        submit_resp = self._mock_response({"errors": {"message": "bad model"}}, status_code=400)
        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = submit_resp

        with pytest.raises(ModelScopeError):
            self.handler.image_generation(
                model=self.model,
                prompt=self.prompt,
                model_response=MagicMock(),
                optional_params={},
                litellm_params={"api_key": "ms-test", "api_base": None},
                logging_obj=self.logging_obj,
                timeout=30,
                client=mock_client,
                aimg_generation=False,
            )
        mock_client.get.assert_not_called()

    def test_sync_poll_failed_status_raises(self):
        """A FAILED task_status during polling raises."""
        submit_resp = self._mock_response({"task_id": "task-123"})
        poll_resp = self._mock_response({"task_status": "FAILED", "task_id": "task-123", "output_images": []})
        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with pytest.raises(ModelScopeError):
            self.handler.image_generation(
                model=self.model,
                prompt=self.prompt,
                model_response=MagicMock(),
                optional_params={},
                litellm_params={"api_key": "ms-test", "api_base": None},
                logging_obj=self.logging_obj,
                timeout=30,
                client=mock_client,
                aimg_generation=False,
            )

    def test_sync_submit_missing_task_id_raises(self):
        """A submit response without task_id raises (cannot poll)."""
        submit_resp = self._mock_response({"task_status": "SUCCEED"})
        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = submit_resp

        with pytest.raises(ModelScopeError):
            self.handler.image_generation(
                model=self.model,
                prompt=self.prompt,
                model_response=MagicMock(),
                optional_params={},
                litellm_params={"api_key": "ms-test", "api_base": None},
                logging_obj=self.logging_obj,
                timeout=30,
                client=mock_client,
                aimg_generation=False,
            )

    def test_submit_sends_async_mode_header(self):
        """The submit POST must include X-ModelScope-Async-Mode: true."""
        from litellm.types.utils import ImageResponse

        submit_resp = self._mock_response({"task_id": "task-123"})
        poll_resp = self._mock_response(
            {
                "task_status": "SUCCEED",
                "task_id": "task-123",
                "output_images": ["https://example.com/x.png"],
            }
        )
        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        self.handler.image_generation(
            model=self.model,
            prompt=self.prompt,
            model_response=ImageResponse(data=[]),
            optional_params={},
            litellm_params={"api_key": "ms-test", "api_base": None},
            logging_obj=self.logging_obj,
            timeout=30,
            client=mock_client,
            aimg_generation=False,
        )

        submit_headers = mock_client.post.call_args.kwargs["headers"]
        assert submit_headers.get("X-ModelScope-Async-Mode") == "true"
        # Polling must carry the task-type header, not the async-mode header
        poll_headers = mock_client.get.call_args.kwargs["headers"]
        assert poll_headers.get("X-ModelScope-Task-Type") == "image_generation"
        assert "X-ModelScope-Async-Mode" not in poll_headers

    def test_sync_poll_running_then_succeed(self):
        """The poll loop must keep polling while RUNNING and stop on SUCCEED."""
        submit_resp = self._mock_response({"task_id": "task-123"})
        running_resp = self._mock_response({"task_status": "RUNNING", "task_id": "task-123", "output_images": []})
        succeed_resp = self._mock_response(
            {
                "task_status": "SUCCEED",
                "task_id": "task-123",
                "output_images": ["https://example.com/otter.png"],
            }
        )

        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = submit_resp
        mock_client.get.side_effect = [running_resp, succeed_resp]

        result = self.handler.image_generation(
            model=self.model,
            prompt=self.prompt,
            model_response=ImageResponse(data=[]),
            optional_params={},
            litellm_params={"api_key": "ms-test", "api_base": None},
            logging_obj=self.logging_obj,
            timeout=30,
            client=mock_client,
            aimg_generation=False,
        )

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/otter.png"
        # RUNNING did not terminate the loop; SUCCEED did on the second GET.
        assert mock_client.get.call_count == 2

    def test_sync_poll_timeout_raises(self):
        """A task that never leaves RUNNING must time out with a 408."""
        submit_resp = self._mock_response({"task_id": "task-123"})
        running_resp = self._mock_response({"task_status": "RUNNING", "task_id": "task-123", "output_images": []})

        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.get.return_value = running_resp

        with pytest.raises(ModelScopeError) as exc_info:
            self.handler._poll_for_result_sync(
                initial_response=submit_resp,
                api_base=None,
                headers={"Authorization": "Bearer ms-test"},
                sync_client=mock_client,
                max_wait=0.05,
                interval=0.01,
                timeout=30,
            )

        assert exc_info.value.status_code == 408
        assert "timed out" in str(exc_info.value).lower()

    def test_sync_submit_200_with_errors_raises(self):
        """A 200 submit response carrying an `errors` key must raise."""
        submit_resp = self._mock_response({"errors": {"message": "rate limited"}}, status_code=200)
        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = submit_resp

        with pytest.raises(ModelScopeError) as exc_info:
            self.handler.image_generation(
                model=self.model,
                prompt=self.prompt,
                model_response=ImageResponse(data=[]),
                optional_params={},
                litellm_params={"api_key": "ms-test", "api_base": None},
                logging_obj=self.logging_obj,
                timeout=30,
                client=mock_client,
                aimg_generation=False,
            )

        assert "rate limited" in str(exc_info.value)
        mock_client.get.assert_not_called()

    def test_sync_poll_non_json_response_raises(self):
        """A non-JSON poll response (e.g. gateway HTML) raises ModelScopeError,
        not a raw JSONDecodeError that bypasses error normalization."""
        import json as _json

        submit_resp = self._mock_response({"task_id": "task-123"})
        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.text = "<html>502 Bad Gateway</html>"
        poll_resp.headers = {}
        poll_resp.json.side_effect = _json.JSONDecodeError("Expecting value", "<html>", 0)

        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with pytest.raises(ModelScopeError) as exc_info:
            self.handler.image_generation(
                model=self.model,
                prompt=self.prompt,
                model_response=ImageResponse(data=[]),
                optional_params={},
                litellm_params={"api_key": "ms-test", "api_base": None},
                logging_obj=self.logging_obj,
                timeout=30,
                client=mock_client,
                aimg_generation=False,
            )

        assert "Error parsing poll response" in str(exc_info.value)

    def test_async_submit_and_poll_succeed(self):
        """Async path: submit returns task_id, poll returns SUCCEED."""
        submit_resp = self._mock_response({"task_id": "task-123"})
        poll_resp = self._mock_response(
            {
                "task_status": "SUCCEED",
                "task_id": "task-123",
                "output_images": ["https://example.com/otter.png"],
            }
        )

        mock_client = MagicMock(spec=AsyncHTTPHandler)
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)

        result = asyncio.run(
            self.handler.async_image_generation(
                model=self.model,
                prompt=self.prompt,
                model_response=ImageResponse(data=[]),
                optional_params={},
                litellm_params={"api_key": "ms-test", "api_base": None},
                logging_obj=self.logging_obj,
                timeout=30,
                client=mock_client,
            )
        )

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/otter.png"
        assert mock_client.post.call_count == 1
        assert mock_client.get.call_count == 1

    def test_async_poll_failed_status_raises(self):
        """Async path: a FAILED task_status during polling raises."""
        submit_resp = self._mock_response({"task_id": "task-123"})
        poll_resp = self._mock_response({"task_status": "FAILED", "task_id": "task-123", "output_images": []})

        mock_client = MagicMock(spec=AsyncHTTPHandler)
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)

        with pytest.raises(ModelScopeError):
            asyncio.run(
                self.handler.async_image_generation(
                    model=self.model,
                    prompt=self.prompt,
                    model_response=ImageResponse(data=[]),
                    optional_params={},
                    litellm_params={"api_key": "ms-test", "api_base": None},
                    logging_obj=self.logging_obj,
                    timeout=30,
                    client=mock_client,
                )
            )

    def test_async_poll_timeout_raises(self):
        """Async path: a task that never leaves RUNNING must time out with 408."""
        submit_resp = self._mock_response({"task_id": "task-123"})
        running_resp = self._mock_response({"task_status": "RUNNING", "task_id": "task-123", "output_images": []})

        mock_client = MagicMock(spec=AsyncHTTPHandler)
        mock_client.get = AsyncMock(return_value=running_resp)

        with pytest.raises(ModelScopeError) as exc_info:
            asyncio.run(
                self.handler._poll_for_result_async(
                    initial_response=submit_resp,
                    api_base=None,
                    headers={"Authorization": "Bearer ms-test"},
                    async_client=mock_client,
                    max_wait=0.05,
                    interval=0.01,
                    timeout=30,
                )
            )

        assert exc_info.value.status_code == 408
