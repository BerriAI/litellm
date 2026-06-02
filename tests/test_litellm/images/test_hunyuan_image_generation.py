"""Unit tests for Tencent Hunyuan image generation provider."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.hunyuan.image_generation import (
    HunyuanImageGeneration,
    HunyuanImageGenerationConfig,
    get_hunyuan_image_generation_config,
    hunyuan_image_generation,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


# ---------------------------------------------------------------------------
# HunyuanImageGenerationConfig (transformation only)
# ---------------------------------------------------------------------------


class TestHunyuanImageGenerationConfig:
    def setup_method(self):
        self.cfg = HunyuanImageGenerationConfig()

    def test_get_complete_url_default(self):
        url = self.cfg.get_complete_url(None, None, "gpt-image-2", {}, {})
        assert url == "https://api.cloudai.tencent.com/v1/aiart/openai/image/submit"

    def test_get_complete_url_custom_base(self):
        url = self.cfg.get_complete_url(
            "https://custom.api.com", None, "gpt-image-2", {}, {}
        )
        assert url == "https://custom.api.com/v1/aiart/openai/image/submit"

    def test_validate_environment_from_env(self):
        os.environ["HUNYUAN_API_KEY"] = "sk-test-123"
        headers = self.cfg.validate_environment(
            headers={},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={},
        )
        assert headers["Authorization"] == "sk-test-123"
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
        assert headers["Authorization"] == "sk-explicit"

    def test_validate_environment_missing_key(self):
        env_backup = os.environ.pop("HUNYUAN_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="HUNYUAN_API_KEY is not set"):
                self.cfg.validate_environment(
                    headers={},
                    model="gpt-image-2",
                    messages=[],
                    optional_params={},
                    litellm_params={},
                )
        finally:
            if env_backup:
                os.environ["HUNYUAN_API_KEY"] = env_backup

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

    def test_transform_image_generation_request_with_b64_response_format(self):
        body = self.cfg.transform_image_generation_request(
            model="gpt-image-2",
            prompt="A dancing dog",
            optional_params={"response_format": "b64_json"},
            litellm_params={},
            headers={},
        )
        assert body["response_format"] == "b64_json"

    def test_transform_image_generation_response_pure_conversion(self):
        """transform_image_generation_response is now a pure format conversion."""

        class MockFinalResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {
                    "status": "DONE",
                    "data": [{"url": "https://example.com/img.png"}],
                }

        model_response = ImageResponse()
        model_response.data = []
        result = self.cfg.transform_image_generation_response(
            model="gpt-image-2",
            raw_response=MockFinalResponse(),
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/img.png"
        assert result.created is not None

    def test_check_task_status_done(self):
        assert self.cfg._check_task_status({"status": "DONE"}) == "done"

    def test_check_task_status_running(self):
        assert self.cfg._check_task_status({"status": "WAIT"}) == "running"
        assert self.cfg._check_task_status({"status": "RUN"}) == "running"

    def test_check_task_status_fail(self):
        with pytest.raises(ValueError, match="failed"):
            self.cfg._check_task_status({"status": "FAIL", "message": "error"})

    def test_transform_image_generation_request_default_extra_body(self):
        """When no extra_body in optional_params, logo_add=0 is injected inside extra_body."""
        body = self.cfg.transform_image_generation_request(
            model="gpt-image-2",
            prompt="A dancing dog",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["extra_body"] == {"logo_add": 0}

    def test_transform_image_generation_request_extra_body_preserves_logo_add(self):
        """When extra_body already contains logo_add, it is not overwritten."""
        body = self.cfg.transform_image_generation_request(
            model="gpt-image-2",
            prompt="A dancing dog",
            optional_params={"extra_body": {"logo_add": 1}},
            litellm_params={},
            headers={},
        )
        assert body["extra_body"]["logo_add"] == 1

    def test_transform_image_generation_request_extra_body_adds_default_logo_add(self):
        """When extra_body exists but lacks logo_add, logo_add=0 is added."""
        body = self.cfg.transform_image_generation_request(
            model="gpt-image-2",
            prompt="A dancing dog",
            optional_params={"extra_body": {"seed": 42}},
            litellm_params={},
            headers={},
        )
        assert body["extra_body"]["seed"] == 42
        assert body["extra_body"]["logo_add"] == 0


# ---------------------------------------------------------------------------
# HunyuanImageGeneration handler
# ---------------------------------------------------------------------------


class TestHunyuanImageGenerationHandler:
    def setup_method(self):
        self.handler = HunyuanImageGeneration()

    def test_extract_poll_context(self):
        class MockSubmitResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"request_id": "abc", "job_id": "j-123"}

        job_id, poll_headers, query_url = self.handler._extract_poll_context(
            MockSubmitResponse(),
            api_key="sk-test",
            litellm_params={"api_base": None},
        )
        assert job_id == "j-123"
        assert poll_headers["Authorization"] == "sk-test"
        assert "Content-Type" in poll_headers
        assert "v1/aiart/openai/image/query" in query_url

    def test_extract_poll_context_none_api_key_falls_back_to_env(self, monkeypatch):
        """When api_key=None, poll headers must resolve key from HUNYUAN_API_KEY env var."""
        monkeypatch.setenv("HUNYUAN_API_KEY", "sk-from-env")

        class MockSubmitResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"request_id": "abc", "job_id": "j-env-123"}

        job_id, poll_headers, query_url = self.handler._extract_poll_context(
            MockSubmitResponse(),
            api_key=None,
            litellm_params={"api_base": None},
        )
        assert job_id == "j-env-123"
        assert poll_headers["Authorization"] == "sk-from-env"

    def test_extract_poll_context_custom_base(self):
        class MockSubmitResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"job_id": "j-456"}

        _, _, query_url = self.handler._extract_poll_context(
            MockSubmitResponse(),
            api_key="sk-x",
            litellm_params={"api_base": "https://custom.example.com"},
        )
        assert query_url.startswith("https://custom.example.com")

    def test_extract_poll_context_missing_job_id(self):
        class MockSubmitResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"request_id": "abc"}

        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        with pytest.raises(BaseLLMException, match="missing job_id"):
            self.handler._extract_poll_context(
                MockSubmitResponse(),
                api_key="sk-test",
                litellm_params={},
            )

    def test_extract_poll_context_api_error_in_body(self):
        """API errors returned in the response body are surfaced as BaseLLMException."""
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        class MockErrorResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {
                    "job_id": "",
                    "error": {
                        "message": "URL格式不合法。",
                        "code": "InvalidParameterValue.UrlIllegal",
                    },
                }

        with pytest.raises(BaseLLMException, match="URL格式不合法"):
            self.handler._extract_poll_context(
                MockErrorResponse(),
                api_key="sk-test",
                litellm_params={},
            )

    def test_poll_for_result_sync_success(self):
        from unittest.mock import MagicMock

        class MockDoneResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {
                    "status": "DONE",
                    "data": [{"url": "https://example.com/img.png"}],
                }

            def raise_for_status(self):
                pass

        class MockSubmitResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"job_id": "j-abc"}

        mock_client = MagicMock()
        mock_client.post.return_value = MockDoneResponse()

        result = self.handler._poll_for_result_sync(
            submit_response=MockSubmitResponse(),
            api_key="sk-test",
            litellm_params={},
            sync_client=mock_client,
        )
        assert result.json()["status"] == "DONE"
        mock_client.post.assert_called_once()

    def test_poll_for_result_sync_timeout(self):
        import time

        from unittest.mock import MagicMock

        class MockRunningResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"status": "RUN"}

            def raise_for_status(self):
                pass

        class MockSubmitResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"job_id": "j-timeout"}

        mock_client = MagicMock()
        mock_client.post.return_value = MockRunningResponse()

        with pytest.raises(TimeoutError, match="timed out"):
            self.handler._poll_for_result_sync(
                submit_response=MockSubmitResponse(),
                api_key="sk-test",
                litellm_params={},
                sync_client=mock_client,
                max_wait=0.05,
                interval=0.01,
            )

    def test_poll_for_result_sync_fail_status(self):
        from unittest.mock import MagicMock

        class MockFailResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"status": "FAIL", "message": "generation failed"}

            def raise_for_status(self):
                pass

        class MockSubmitResponse:
            status_code = 200
            headers = {}

            def json(self):
                return {"job_id": "j-fail"}

        mock_client = MagicMock()
        mock_client.post.return_value = MockFailResponse()

        with pytest.raises(ValueError, match="generation failed"):
            self.handler._poll_for_result_sync(
                submit_response=MockSubmitResponse(),
                api_key="sk-test",
                litellm_params={},
                sync_client=mock_client,
            )


# ---------------------------------------------------------------------------
# Singleton & ProviderConfigManager
# ---------------------------------------------------------------------------


def test_hunyuan_image_generation_singleton():
    assert isinstance(hunyuan_image_generation, HunyuanImageGeneration)


def test_provider_config_manager_returns_hunyuan_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        "gpt-image-2", LlmProviders.HUNYUAN
    )
    assert isinstance(config, HunyuanImageGenerationConfig)


def test_hunyuan_in_llm_providers():
    assert LlmProviders.HUNYUAN == "hunyuan"
    assert LlmProviders.HUNYUAN.value == "hunyuan"


class TestHunyuanImageGenerationPostCall:
    """Verify logging_obj.post_call is invoked after successful polling."""

    def _make_mock_client(self, job_id: str = "job-gen") -> MagicMock:
        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": job_id}
        submit_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.text = (
            '{"status":"DONE","data":[{"url":"https://example.com/gen.png"}]}'
        )
        poll_resp.json.return_value = {
            "status": "DONE",
            "data": [{"url": "https://example.com/gen.png"}],
        }
        poll_resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.post.side_effect = [submit_resp, poll_resp]
        return client

    def test_post_call_invoked_sync(self):
        handler = HunyuanImageGeneration()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_generation.handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = handler.image_generation(
                model="gpt-image-2",
                prompt="一只跳舞的小狗",
                model_response=ImageResponse(),
                optional_params={"size": "1024x1024"},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        assert isinstance(result, ImageResponse)
        mock_logging.post_call.assert_called_once()
        call_kwargs = mock_logging.post_call.call_args[1]
        assert call_kwargs["input"] == "一只跳舞的小狗"
        assert call_kwargs["api_key"] == "sk-test"


# ---------------------------------------------------------------------------
# litellm_params direct params support
# ---------------------------------------------------------------------------


class TestHunyuanImageGenerationLitellmParams:
    """Verify extra_body params are forwarded correctly to the Hunyuan API."""

    def _make_mock_client(self, job_id: str = "job-extra") -> MagicMock:
        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {"job_id": job_id}
        submit_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.text = (
            '{"status":"DONE","data":[{"url":"https://example.com/gen.png"}]}'
        )
        poll_resp.json.return_value = {
            "status": "DONE",
            "data": [{"url": "https://example.com/gen.png"}],
        }
        poll_resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.post.side_effect = [submit_resp, poll_resp]
        return client

    def test_logo_add_default_is_zero_in_extra_body(self):
        """logo_add=0 is injected inside extra_body when not provided."""
        handler = HunyuanImageGeneration()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_generation.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_generation(
                model="gpt-image-2",
                prompt="一只跳舞的小狗",
                model_response=ImageResponse(),
                optional_params={},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["extra_body"]["logo_add"] == 0

    def test_logo_add_overridable_via_extra_body(self):
        """logo_add default can be overridden by passing extra_body in optional_params."""
        handler = HunyuanImageGeneration()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_generation.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_generation(
                model="gpt-image-2",
                prompt="一只跳舞的小狗",
                model_response=ImageResponse(),
                optional_params={"extra_body": {"logo_add": 1}},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["extra_body"]["logo_add"] == 1

    def test_extra_body_params_forwarded(self):
        """Provider-specific params passed in extra_body are forwarded to the API."""
        handler = HunyuanImageGeneration()
        mock_client = self._make_mock_client()
        mock_logging = MagicMock()

        os.environ["HUNYUAN_API_KEY"] = "sk-test"
        with patch(
            "litellm.llms.hunyuan.image_generation.handler._get_httpx_client",
            return_value=mock_client,
        ):
            handler.image_generation(
                model="gpt-image-2",
                prompt="一只跳舞的小狗",
                model_response=ImageResponse(),
                optional_params={"extra_body": {"seed": 42, "revise_prompt": 0}},
                litellm_params={"api_key": "sk-test"},
                logging_obj=mock_logging,
                timeout=30.0,
            )

        submit_call_body = mock_client.post.call_args_list[0][1]["json"]
        assert submit_call_body["extra_body"]["seed"] == 42
        assert submit_call_body["extra_body"]["revise_prompt"] == 0
        assert submit_call_body["extra_body"]["logo_add"] == 0
