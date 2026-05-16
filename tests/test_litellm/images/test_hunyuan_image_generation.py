"""Unit tests for Tencent Hunyuan image generation provider."""
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.hunyuan.image_generation import (
    HunyuanImageGenerationConfig,
    get_hunyuan_image_generation_config,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


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
            headers={}, model="gpt-image-2", messages=[], optional_params={}, litellm_params={}
        )
        assert headers["Authorization"] == "sk-test-123"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_missing_key(self):
        env_backup = os.environ.pop("HUNYUAN_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="HUNYUAN_API_KEY is not set"):
                self.cfg.validate_environment(
                    headers={}, model="gpt-image-2", messages=[], optional_params={}, litellm_params={}
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
        assert "response_format" not in body

    def test_transform_image_generation_response_success(self):
        class MockRequest:
            headers = {"Authorization": "sk-test", "Content-Type": "application/json"}

        class MockSubmitResponse:
            request = MockRequest()
            status_code = 200
            headers = {}

            def json(self):
                return {"request_id": "abc", "job_id": "12345"}

            def raise_for_status(self):
                pass

        def mock_poll(job_id, query_url, headers, timeout_secs=600):
            assert job_id == "12345"
            return {
                "status": "DONE",
                "data": [{"url": "https://example.com/img.png"}],
            }

        self.cfg._poll_task_sync = mock_poll

        model_response = ImageResponse()
        model_response.data = []
        result = self.cfg.transform_image_generation_response(
            model="gpt-image-2",
            raw_response=MockSubmitResponse(),
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={"api_base": None},
            encoding=None,
        )
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/img.png"

    def test_check_task_status_done(self):
        assert self.cfg._check_task_status({"status": "DONE"}) == "done"

    def test_check_task_status_running(self):
        assert self.cfg._check_task_status({"status": "WAIT"}) == "running"
        assert self.cfg._check_task_status({"status": "RUN"}) == "running"

    def test_check_task_status_fail(self):
        with pytest.raises(ValueError, match="failed"):
            self.cfg._check_task_status({"status": "FAIL", "message": "error"})

    def test_build_query_url(self):
        url = self.cfg._build_query_url("https://api.cloudai.tencent.com")
        assert url == "https://api.cloudai.tencent.com/v1/aiart/openai/image/query"


def test_provider_config_manager_returns_hunyuan_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        "gpt-image-2", LlmProviders.HUNYUAN
    )
    assert isinstance(config, HunyuanImageGenerationConfig)


def test_hunyuan_in_llm_providers():
    assert LlmProviders.HUNYUAN == "hunyuan"
    assert LlmProviders.HUNYUAN.value == "hunyuan"
