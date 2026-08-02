import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.orcarouter.chat.transformation import OrcaRouterConfig
from litellm.llms.orcarouter.common_utils import OrcaRouterException


class TestOrcaRouterProviderRouting:
    """get_llm_provider must resolve orcarouter/* to the OrcaRouter endpoint."""

    def test_router_model_routes_to_orcarouter(self):
        model, provider, _, api_base = litellm.get_llm_provider(
            model="orcarouter/auto", api_key="sk-orca-test"
        )
        assert provider == "orcarouter"
        assert api_base == "https://api.orcarouter.ai/v1"

    def test_namespaced_model_routes_to_orcarouter(self):
        model, provider, _, api_base = litellm.get_llm_provider(
            model="orcarouter/openai/gpt-5", api_key="sk-orca-test"
        )
        assert provider == "orcarouter"
        assert model == "openai/gpt-5"
        assert api_base == "https://api.orcarouter.ai/v1"

    def test_api_key_from_environment(self):
        with patch.dict("os.environ", {"ORCAROUTER_API_KEY": "sk-orca-env"}):
            _, provider, api_key, _ = litellm.get_llm_provider(model="orcarouter/auto")
        assert provider == "orcarouter"
        assert api_key == "sk-orca-env"

    def test_validate_environment(self):
        with patch.dict("os.environ", {"ORCAROUTER_API_KEY": "sk-orca-env"}):
            result = litellm.validate_environment(model="orcarouter/auto")
        assert result["keys_in_environment"] is True
        assert "ORCAROUTER_API_KEY" not in result["missing_keys"]


class TestOrcaRouterModelNormalization:
    """OrcaRouter routes by namespaced model id.

    A bare router/model name (no "/") must be re-qualified as
    ``orcarouter/<name>`` (otherwise the backend returns 503), while names
    that already carry an upstream namespace are sent unchanged.
    """

    @pytest.mark.parametrize(
        "model_in,model_out",
        [
            ("auto", "orcarouter/auto"),
            ("openai/gpt-5", "openai/gpt-5"),
            ("anthropic/claude-opus-4.8", "anthropic/claude-opus-4.8"),
            ("orcarouter/auto", "orcarouter/auto"),
        ],
    )
    def test_normalize_orcarouter_model(self, model_in, model_out):
        assert OrcaRouterConfig._normalize_orcarouter_model(model_in) == model_out

    def test_transform_request_qualifies_bare_router_name(self):
        data = OrcaRouterConfig().transform_request(
            model="auto",
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert data["model"] == "orcarouter/auto"

    def test_transform_request_preserves_namespaced_model(self):
        data = OrcaRouterConfig().transform_request(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert data["model"] == "openai/gpt-5"

    def test_extra_body_routing_preferences_pass_through(self):
        data = OrcaRouterConfig().transform_request(
            model="auto",
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={
                "extra_body": {"models": ["openai/gpt-5"], "route": "fallback"}
            },
            litellm_params={},
            headers={},
        )
        assert data["extra_body"]["models"] == ["openai/gpt-5"]
        assert data["extra_body"]["route"] == "fallback"


class TestOrcaRouterConfig:
    def test_get_openai_compatible_provider_info(self):
        config = OrcaRouterConfig()
        with patch.dict(
            "os.environ",
            {
                "ORCAROUTER_API_BASE": "https://env.orcarouter.ai/v1",
                "ORCAROUTER_API_KEY": "sk-orca-env",
            },
        ):
            api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://env.orcarouter.ai/v1"
        assert api_key == "sk-orca-env"

    def test_get_supported_openai_params_includes_extra_body(self):
        supported_params = OrcaRouterConfig().get_supported_openai_params("auto")
        assert "extra_body" in supported_params
        assert "temperature" in supported_params
        assert "stream" in supported_params

    def test_custom_llm_provider(self):
        assert OrcaRouterConfig().custom_llm_provider == "orcarouter"

    def test_error_class(self):
        config = OrcaRouterConfig()
        err = config.get_error_class("boom", 400, {"Content-Type": "application/json"})
        assert isinstance(err, OrcaRouterException)
        assert err.status_code == 400

    def test_exception_inheritance(self):
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        exception = OrcaRouterException(message="test", status_code=500, headers={})
        assert isinstance(exception, BaseLLMException)
