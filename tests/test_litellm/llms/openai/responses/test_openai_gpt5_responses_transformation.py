"""Tests for OpenAI GPT-5 Responses API configuration."""

import pytest

import litellm
from litellm.llms.openai.responses.gpt_5_transformation import (
    OpenAIGPT5ResponsesAPIConfig,
)


@pytest.fixture()
def gpt5_config() -> OpenAIGPT5ResponsesAPIConfig:
    return OpenAIGPT5ResponsesAPIConfig()


class TestGPT5ModelDetection:
    """Test GPT-5 model detection."""

    def test_gpt5_model_detection(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that GPT-5 models are correctly detected."""
        assert gpt5_config.is_model_gpt_5_model("gpt-5")
        assert gpt5_config.is_model_gpt_5_model("gpt-5-mini")
        assert gpt5_config.is_model_gpt_5_model("gpt-5-preview")

    def test_gpt5_codex_model_detection(
        self, gpt5_config: OpenAIGPT5ResponsesAPIConfig
    ):
        """Test that GPT-5-Codex models are correctly detected."""
        assert gpt5_config.is_model_gpt_5_model("gpt-5-codex")
        assert gpt5_config.is_model_gpt_5_codex_model("gpt-5-codex")

    def test_non_gpt5_models(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that non-GPT-5 models are not detected as GPT-5."""
        assert not gpt5_config.is_model_gpt_5_model("gpt-4")
        assert not gpt5_config.is_model_gpt_5_model("gpt-4o")
        assert not gpt5_config.is_model_gpt_5_model("o1")
        assert not gpt5_config.is_model_gpt_5_model("o1-mini")

    def test_codex_detection(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that only GPT-5-Codex is detected as codex."""
        assert not gpt5_config.is_model_gpt_5_codex_model("gpt-5")
        assert not gpt5_config.is_model_gpt_5_codex_model("gpt-5-mini")
        assert gpt5_config.is_model_gpt_5_codex_model("gpt-5-codex")


class TestGPT5TemperatureHandling:
    """Test temperature parameter handling for GPT-5 models."""

    def test_temperature_one_allowed(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that temperature=1 is allowed."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={"temperature": 1.0},
            model="gpt-5",
            drop_params=False,
        )
        assert params["temperature"] == 1.0

    def test_temperature_one_exact(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that exactly temperature=1 (int) is allowed."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={"temperature": 1},
            model="gpt-5",
            drop_params=False,
        )
        assert params["temperature"] == 1

    def test_temperature_zero_error(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that temperature=0 raises error when drop_params=False."""
        with pytest.raises(
            litellm.utils.UnsupportedParamsError,
            match="gpt-5 models.*don't support temperature=0",
        ):
            gpt5_config.map_openai_params(
                response_api_optional_params={"temperature": 0.0},
                model="gpt-5",
                drop_params=False,
            )

    def test_temperature_zero_dropped(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that temperature=0 is dropped when drop_params=True."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={"temperature": 0.0},
            model="gpt-5",
            drop_params=True,
        )
        assert "temperature" not in params

    def test_temperature_half_error(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that temperature=0.5 raises error when drop_params=False."""
        with pytest.raises(
            litellm.utils.UnsupportedParamsError,
            match="gpt-5 models.*don't support temperature=0.5",
        ):
            gpt5_config.map_openai_params(
                response_api_optional_params={"temperature": 0.5},
                model="gpt-5",
                drop_params=False,
            )

    def test_temperature_half_dropped(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that temperature=0.5 is dropped when drop_params=True."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={"temperature": 0.5},
            model="gpt-5",
            drop_params=True,
        )
        assert "temperature" not in params

    def test_temperature_seven_error(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that temperature=0.7 raises error when drop_params=False."""
        with pytest.raises(
            litellm.utils.UnsupportedParamsError,
            match="gpt-5 models.*don't support temperature=0.7",
        ):
            gpt5_config.map_openai_params(
                response_api_optional_params={"temperature": 0.7},
                model="gpt-5",
                drop_params=False,
            )

    def test_temperature_seven_dropped(
        self, gpt5_config: OpenAIGPT5ResponsesAPIConfig
    ):
        """Test that temperature=0.7 is dropped when drop_params=True."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={"temperature": 0.7},
            model="gpt-5",
            drop_params=True,
        )
        assert "temperature" not in params

    def test_temperature_none_ignored(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that temperature=None is ignored."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={"temperature": None},
            model="gpt-5",
            drop_params=False,
        )
        assert "temperature" not in params

    def test_no_temperature_param(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that missing temperature parameter is fine."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={},
            model="gpt-5",
            drop_params=False,
        )
        assert "temperature" not in params

    def test_litellm_drop_params_global(
        self, gpt5_config: OpenAIGPT5ResponsesAPIConfig
    ):
        """Test that litellm.drop_params global setting works."""
        original = litellm.drop_params
        try:
            litellm.drop_params = True
            params = gpt5_config.map_openai_params(
                response_api_optional_params={"temperature": 0.5},
                model="gpt-5",
                drop_params=False,  # Should be overridden by global
            )
            assert "temperature" not in params
        finally:
            litellm.drop_params = original


class TestGPT5CodexTemperature:
    """Test temperature handling for GPT-5-Codex specifically."""

    def test_gpt5_codex_temperature_one_allowed(
        self, gpt5_config: OpenAIGPT5ResponsesAPIConfig
    ):
        """Test that GPT-5-Codex allows temperature=1."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={"temperature": 1.0},
            model="gpt-5-codex",
            drop_params=False,
        )
        assert params["temperature"] == 1.0

    def test_gpt5_codex_temperature_error(
        self, gpt5_config: OpenAIGPT5ResponsesAPIConfig
    ):
        """Test that GPT-5-Codex raises error for unsupported temperature."""
        with pytest.raises(
            litellm.utils.UnsupportedParamsError,
            match="gpt-5 models \\(including gpt-5-codex\\)",
        ):
            gpt5_config.map_openai_params(
                response_api_optional_params={"temperature": 0.7},
                model="gpt-5-codex",
                drop_params=False,
            )

    def test_gpt5_codex_temperature_drop(
        self, gpt5_config: OpenAIGPT5ResponsesAPIConfig
    ):
        """Test that GPT-5-Codex drops unsupported temperature values."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={"temperature": 0.7},
            model="gpt-5-codex",
            drop_params=True,
        )
        assert "temperature" not in params


class TestGPT5OtherParams:
    """Test that other parameters are passed through correctly."""

    def test_other_params_preserved(self, gpt5_config: OpenAIGPT5ResponsesAPIConfig):
        """Test that non-temperature parameters are preserved."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={
                "temperature": 1.0,
                "max_output_tokens": 100,
                "instructions": "Be helpful",
            },
            model="gpt-5",
            drop_params=False,
        )
        assert params["temperature"] == 1.0
        assert params["max_output_tokens"] == 100
        assert params["instructions"] == "Be helpful"

    def test_params_without_temperature(
        self, gpt5_config: OpenAIGPT5ResponsesAPIConfig
    ):
        """Test that parameters work without temperature."""
        params = gpt5_config.map_openai_params(
            response_api_optional_params={
                "max_output_tokens": 200,
                "instructions": "Test",
            },
            model="gpt-5",
            drop_params=False,
        )
        assert params["max_output_tokens"] == 200
        assert params["instructions"] == "Test"
        assert "temperature" not in params
