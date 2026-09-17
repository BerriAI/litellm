import pytest

from litellm.llms.cerebras.chat import CerebrasConfig


class TestCerebrasConfig:
    def test_get_supported_openai_params_includes_max_retries(self):
        config = CerebrasConfig()
        params = config.get_supported_openai_params(model="cerebras/llama-4-scout-17b-16e")
        assert "max_retries" in params

    def test_get_supported_openai_params_includes_extra_headers(self):
        config = CerebrasConfig()
        params = config.get_supported_openai_params(model="cerebras/llama-4-scout-17b-16e")
        assert "extra_headers" in params

    def test_map_openai_params_passes_through_max_retries(self):
        config = CerebrasConfig()
        result = config.map_openai_params(
            non_default_params={"max_retries": 0},
            optional_params={},
            model="cerebras/llama-4-scout-17b-16e",
            drop_params=False,
        )
        assert result["max_retries"] == 0

    def test_map_openai_params_passes_through_extra_headers(self):
        config = CerebrasConfig()
        headers = {"X-Custom": "value"}
        result = config.map_openai_params(
            non_default_params={"extra_headers": headers},
            optional_params={},
            model="cerebras/llama-4-scout-17b-16e",
            drop_params=False,
        )
        assert result["extra_headers"] == headers
