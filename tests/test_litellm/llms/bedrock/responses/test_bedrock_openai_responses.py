"""Native OpenAI Responses API on the bedrock-runtime endpoint.

Without this config the bedrock provider has no Responses config, so /v1/responses
falls back to the Chat Completions bridge and rides Converse.
"""

import json
from importlib.resources import files
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.bedrock.common_utils import bedrock_supports_openai_responses
from litellm.llms.bedrock.responses.transformation import BedrockOpenAIResponsesConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODEL = "global.openai.gpt-5.6-sol"


def _cfg():
    return BedrockOpenAIResponsesConfig()


class TestCompleteURL:
    def test_default_host_and_path(self):
        url = _cfg().get_complete_url(None, {"aws_region_name": "us-east-1"})
        assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"

    def test_region_is_honoured(self):
        url = _cfg().get_complete_url(None, {"aws_region_name": "eu-west-1"})
        assert url == "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1/responses"

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://proxy.example.com",
            "https://proxy.example.com/",
            "https://proxy.example.com/openai/v1",
            "https://proxy.example.com/openai/v1/responses",
        ],
    )
    def test_custom_host_is_preserved_and_path_never_doubles(self, api_base):
        url = _cfg().get_complete_url(api_base, {"aws_region_name": "us-east-1"})
        assert url == "https://proxy.example.com/openai/v1/responses"

    def test_runtime_endpoint_param_is_honoured(self):
        url = _cfg().get_complete_url(
            None, {"aws_region_name": "us-east-1", "aws_bedrock_runtime_endpoint": "https://vpce.example.com"}
        )
        assert url == "https://vpce.example.com/openai/v1/responses"


class TestAuth:
    def test_bearer_token_is_used_when_present(self):
        headers = _cfg().validate_environment({}, MODEL, GenericLiteLLMParams(api_key="sk-bedrock"))
        assert headers["Authorization"] == "Bearer sk-bedrock"

    def test_no_authorization_header_without_a_token(self, monkeypatch):
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        headers = _cfg().validate_environment({}, MODEL, GenericLiteLLMParams())
        assert "Authorization" not in headers

    def test_sigv4_is_skipped_when_a_bearer_token_is_present(self):
        """Bedrock API keys are Bearer; signing on top would be wrong."""
        headers, body = _cfg().sign_request(
            headers={"Authorization": "Bearer sk-bedrock"},
            optional_params={},
            request_data={},
            api_base="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses",
            api_key="sk-bedrock",
        )
        assert headers["Authorization"] == "Bearer sk-bedrock"
        assert body is None


class TestProviderIdentity:
    def test_reports_the_bedrock_provider(self):
        """Cost tracking and callbacks key off this, so it must stay `bedrock` rather
        than becoming a separate provider."""
        assert _cfg().custom_llm_provider == LlmProviders.BEDROCK


class TestSigV4Fallback:
    def test_signs_with_sigv4_when_no_bearer_token_is_present(self, monkeypatch):
        """No Bedrock API key means SigV4 over the standard credential chain. Static
        credentials are set in the environment so signing stays a local computation."""
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        headers, body = _cfg().sign_request(
            headers={"content-type": "application/json"},
            optional_params={"aws_region_name": "us-east-1"},
            request_data={"model": MODEL, "input": "hi"},
            api_base="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses",
            api_key=None,
        )
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "Credential=AKIAIOSFODNN7EXAMPLE" in headers["Authorization"]


class TestPriceMapGate:
    def test_absent_model_has_no_signal(self):
        assert bedrock_supports_openai_responses(MODEL, {}) is False

    def test_none_model_is_false(self):
        assert bedrock_supports_openai_responses(None, {}) is False

    def test_signal_on_the_bare_key(self):
        cost = {MODEL: {"supported_endpoints": ["/v1/responses"]}}
        assert bedrock_supports_openai_responses(MODEL, cost) is True

    def test_signal_on_the_bedrock_prefixed_key(self):
        cost = {f"bedrock/{MODEL}": {"supported_endpoints": ["/v1/responses"]}}
        assert bedrock_supports_openai_responses(MODEL, cost) is True

    def test_other_endpoints_do_not_count(self):
        cost = {MODEL: {"supported_endpoints": ["/v1/messages"]}}
        assert bedrock_supports_openai_responses(MODEL, cost) is False


class TestForModelGate:
    """The capability decision lives on the adapter, not in the shared dispatch."""

    def test_returns_a_config_for_a_signalled_model(self):
        with patch.object(  # test-quality-ok: the gate reads the global cost map by design; no injection point exists
            litellm, "model_cost", {MODEL: {"supported_endpoints": ["/v1/responses"]}}
        ):
            assert isinstance(BedrockOpenAIResponsesConfig.for_model(MODEL), BedrockOpenAIResponsesConfig)

    def test_returns_none_for_an_unsignalled_model(self):
        with patch.object(  # test-quality-ok: the gate reads the global cost map by design; no injection point exists
            litellm, "model_cost", {}
        ):
            assert BedrockOpenAIResponsesConfig.for_model(MODEL) is None

    def test_returns_none_for_no_model(self):
        with patch.object(  # test-quality-ok: the gate reads the global cost map by design; no injection point exists
            litellm, "model_cost", {}
        ):
            assert BedrockOpenAIResponsesConfig.for_model(None) is None


class TestProviderResolution:
    """model_cost is patched explicitly: it is populated at import time from a GitHub
    fetch unless LITELLM_LOCAL_MODEL_COST_MAP is set, and conftest's monkeypatch of
    that variable lands after import — so these must not read the global."""

    def test_signalled_model_resolves_to_the_bedrock_responses_config(self):
        with patch.object(  # test-quality-ok: resolution reads the global cost map by design; no HTTP boundary or injection point exists
            litellm, "model_cost", {MODEL: {"supported_endpoints": ["/v1/responses"]}}
        ):
            cfg = ProviderConfigManager.get_provider_responses_api_config(model=MODEL, provider=LlmProviders.BEDROCK)
        assert isinstance(cfg, BedrockOpenAIResponsesConfig)

    def test_unsignalled_model_keeps_the_existing_bridge(self):
        """Claude on Bedrock has no OpenAI surface; it must keep falling through to
        the chat-completions bridge exactly as before."""
        with patch.object(litellm, "model_cost", {}):  # test-quality-ok: resolution reads the global cost map by design
            cfg = ProviderConfigManager.get_provider_responses_api_config(
                model="anthropic.claude-3-haiku-20240307-v1:0", provider=LlmProviders.BEDROCK
            )
        assert cfg is None

    def test_the_shipped_price_map_signals_the_gpt_56_family(self):
        """Reads the bundled backup directly rather than the network-fetched global."""
        shipped = json.loads(
            files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text(encoding="utf-8")
        )
        for prefix in ("us", "global"):
            for variant in ("sol", "terra", "luna"):
                model = f"{prefix}.openai.gpt-5.6-{variant}"
                assert bedrock_supports_openai_responses(model, shipped) is True, model


class TestCodexHistoryNormalization:
    def test_history_items_the_endpoint_rejects_are_rewritten(self):
        body = _cfg().transform_responses_api_request(
            model=MODEL,
            input=[
                {"type": "agent_message", "content": [{"type": "output_text", "text": "prior"}]},
                {"type": "context_compaction", "encrypted_content": "abc"},
                {"type": "local_shell_call", "call_id": "c1", "action": {"command": ["ls"]}},
                {"role": "user", "content": "carry on"},
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert [i.get("type") or i.get("role") for i in body["input"]] == [
            "message",
            "compaction",
            "function_call",
            "user",
        ]

    def test_a_first_turn_request_is_untouched(self):
        """The rejected types are history items, so turn one exercises none of this."""
        original = [{"role": "user", "content": "first turn"}]
        body = _cfg().transform_responses_api_request(
            model=MODEL,
            input=list(original),
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["input"] == original
