"""Native OpenAI Chat Completions API on the bedrock-runtime endpoint.

Sibling of the Responses config: when a bedrock model advertises /v1/chat/completions,
chat requests hit bedrock-runtime's native /openai/v1/chat/completions surface instead
of the Converse translation.
"""

import pytest

import litellm
from litellm.llms.bedrock.chat.openai_native.transformation import BedrockOpenAIChatConfig
from litellm.llms.bedrock.common_utils import (
    bedrock_chat_rejects_function_tools_while_reasoning,
    bedrock_supports_openai_chat,
    bedrock_uses_native_openai_chat,
    get_bedrock_chat_config,
)
from litellm.main import responses_api_bridge_check

MODEL = "global.openai.gpt-6-luna"


@pytest.fixture
def native_model():
    """Register a runtime model that advertises the native chat surface."""
    litellm.register_model(
        {
            MODEL: {
                "litellm_provider": "bedrock_converse",
                "mode": "chat",
                "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
                "supports_reasoning": True,
            }
        }
    )
    return MODEL


def _cfg():
    return BedrockOpenAIChatConfig()


class TestCompleteURL:
    def test_default_host_and_path(self):
        url = _cfg().get_complete_url(None, None, MODEL, {"aws_region_name": "us-east-1"}, {})
        assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"

    def test_region_is_honoured(self):
        url = _cfg().get_complete_url(None, None, MODEL, {"aws_region_name": "eu-west-1"}, {})
        assert url == "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1/chat/completions"

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://proxy.example.com",
            "https://proxy.example.com/",
            "https://proxy.example.com/openai/v1",
            "https://proxy.example.com/openai/v1/chat/completions",
            "https://proxy.example.com/v1",
        ],
    )
    def test_api_base_override_collapses_to_chat_path(self, api_base):
        url = _cfg().get_complete_url(api_base, None, MODEL, {"aws_region_name": "us-east-1"}, {})
        assert url == "https://proxy.example.com/openai/v1/chat/completions"


class TestAuth:
    def test_bearer_token_sets_header_and_skips_sigv4(self):
        cfg = _cfg()
        headers = cfg.validate_environment({}, MODEL, [], {}, {}, api_key="bedrock-key")
        assert headers["Authorization"] == "Bearer bedrock-key"
        signed, body = cfg.sign_request({}, {}, {"model": MODEL}, "https://x", api_key="bedrock-key")
        assert body is None  # SigV4 not applied on top of a Bearer credential

    def test_no_bearer_leaves_auth_to_sigv4(self):
        # No Authorization header without a token; signing is exercised elsewhere.
        headers = _cfg().validate_environment({}, MODEL, [], {}, {})
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"


class TestRequestBody:
    def test_aws_params_stripped_from_body(self):
        body = _cfg().transform_request(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"aws_region_name": "us-east-1", "max_completion_tokens": 10},
            litellm_params={},
            headers={},
        )
        assert "aws_region_name" not in body
        assert body["max_completion_tokens"] == 10
        assert body["model"] == MODEL

    def test_reasoning_temperature_dropped(self):
        # gpt-6 is a reasoning model; a non-default temperature must be dropped.
        mapped = _cfg().map_openai_params(
            non_default_params={"temperature": 0.5, "reasoning_effort": "low"},
            optional_params={},
            model=MODEL,
            drop_params=True,
        )
        assert "temperature" not in mapped
        assert mapped.get("reasoning_effort") == "low"

    def test_stream_flag_is_in_the_signed_body(self):
        """Under SigV4 the body is signed before the handler's stream-add step, so ``stream``
        must already be in the transformed body (it rides ``optional_params``), otherwise the
        signed request Bedrock receives would be non-streaming."""
        body = _cfg().transform_request(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"stream": True, "max_completion_tokens": 10, "aws_region_name": "us-east-1"},
            litellm_params={},
            headers={},
        )
        assert body.get("stream") is True


class TestRouting:
    def test_predicate_true_when_endpoint_advertised(self, native_model):
        assert bedrock_supports_openai_chat(native_model, litellm.model_cost) is True
        assert bedrock_uses_native_openai_chat(native_model) is True

    def test_config_selected_by_default(self, native_model):
        assert isinstance(get_bedrock_chat_config(native_model), BedrockOpenAIChatConfig)

    def test_explicit_converse_is_escape_hatch(self, native_model):
        assert bedrock_uses_native_openai_chat(f"bedrock/converse/{native_model}") is False
        cfg = get_bedrock_chat_config(f"bedrock/converse/{native_model}")
        assert isinstance(cfg, litellm.AmazonConverseConfig)

    def test_model_without_chat_endpoint_stays_converse(self):
        litellm.register_model(
            {
                "us.openai.gpt-6-astra": {
                    "litellm_provider": "bedrock_converse",
                    "mode": "chat",
                    "supported_endpoints": ["/v1/responses"],
                }
            }
        )
        assert bedrock_uses_native_openai_chat("us.openai.gpt-6-astra") is False
        assert isinstance(get_bedrock_chat_config("us.openai.gpt-6-astra"), litellm.AmazonConverseConfig)

    def test_claude_unaffected(self):
        assert bedrock_uses_native_openai_chat("anthropic.claude-3-5-sonnet-20241022-v2:0") is False


class TestFunctionToolsReasoningBridge:
    """gpt-5.6/6 reject function tools while reasoning on chat completions; those requests
    bridge to the native /v1/responses surface instead of regressing to Converse."""

    FUNCTION_TOOL = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]

    @staticmethod
    def _register(model, rejects):
        entry = {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
            "supports_reasoning": True,
        }
        if rejects:
            entry["bedrock_chat_rejects_function_tools_while_reasoning"] = True
        litellm.register_model({model: entry})

    @pytest.mark.parametrize(
        "model,rejects",
        [
            ("global.openai.gpt-6-luna", True),
            ("us.openai.gpt-5.6-sol", True),
            ("global.openai.gpt-5.5", False),
            ("us.openai.gpt-5.4", False),
        ],
    )
    def test_flag_read_from_model_cost(self, model, rejects):
        self._register(model, rejects)
        assert bedrock_chat_rejects_function_tools_while_reasoning(model, litellm.model_cost) is rejects

    def test_flag_absent_defaults_false(self):
        assert bedrock_chat_rejects_function_tools_while_reasoning("unknown.model", litellm.model_cost) is False

    def _bridge(self, model, rejects=True, **kw):
        self._register(model, rejects)
        info, _ = responses_api_bridge_check(model=model, custom_llm_provider="bedrock", **kw)
        return info.get("mode")

    def test_tools_with_reasoning_bridges_to_responses(self):
        assert self._bridge("global.openai.gpt-6-luna", tools=self.FUNCTION_TOOL, reasoning_effort="low") == "responses"

    def test_tools_with_default_reasoning_bridges(self):
        # unset reasoning_effort still means reasoning is active for these models
        assert self._bridge("global.openai.gpt-6-luna", tools=self.FUNCTION_TOOL) == "responses"

    def test_tools_with_reasoning_none_stays_chat(self):
        assert self._bridge("global.openai.gpt-6-luna", tools=self.FUNCTION_TOOL, reasoning_effort="none") != "responses"

    def test_gpt55_tools_with_reasoning_stays_chat(self):
        # gpt-5.5 isn't flagged (serves tools with reasoning natively), so no bridge.
        assert (
            self._bridge("global.openai.gpt-5.5", rejects=False, tools=self.FUNCTION_TOOL, reasoning_effort="low")
            != "responses"
        )

    def test_no_tools_stays_chat(self):
        assert self._bridge("global.openai.gpt-6-luna", reasoning_effort="low") != "responses"
