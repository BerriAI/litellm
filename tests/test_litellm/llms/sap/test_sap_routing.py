import litellm
import litellm.main as litellm_main
from litellm.llms.sap.chat.direct_transformation import SapDeploymentAnthropicChatConfig
from litellm.llms.sap.chat.gemini_direct_transformation import SapDeploymentGeminiChatConfig
from litellm.llms.sap.chat.openai_direct_transformation import SapDeploymentOpenAIChatConfig
from litellm.llms.sap.messages.transformation import SapDeploymentAnthropicMessagesConfig
from litellm.utils import ProviderConfigManager

_MODEL = "anthropic--claude-4.8-opus"
_SAP = litellm.LlmProviders.SAP_GENERATIVE_AI_HUB


def test_chat_plain_model_routes_to_orchestration():
    cfg = ProviderConfigManager._get_sap_chat_config(_MODEL)
    assert isinstance(cfg, litellm.GenAIHubOrchestrationConfig)


def test_chat_deployment_prefix_routes_to_direct_connect():
    cfg = ProviderConfigManager._get_sap_chat_config(f"deployment/{_MODEL}")
    assert isinstance(cfg, SapDeploymentAnthropicChatConfig)


def test_chat_gpt_deployment_prefix_routes_to_openai_direct_connect():
    cfg = ProviderConfigManager._get_sap_chat_config("deployment/gpt-4o")
    assert isinstance(cfg, SapDeploymentOpenAIChatConfig)


def test_chat_gemini_deployment_prefix_routes_to_gemini_direct_connect():
    cfg = ProviderConfigManager._get_sap_chat_config("deployment/gemini-2.5-flash-lite")
    assert isinstance(cfg, SapDeploymentGeminiChatConfig)


def test_chat_provider_prefixed_deployment_routes_to_direct_connect():
    cfg = ProviderConfigManager._get_sap_chat_config(f"sap/deployment/{_MODEL}")
    assert isinstance(cfg, SapDeploymentAnthropicChatConfig)


def test_provider_prefixed_deployment_maps_reasoning_effort_for_capability_probe():
    supported = litellm.get_supported_openai_params(model=f"sap/deployment/{_MODEL}", custom_llm_provider="sap") or []
    assert "reasoning_effort" in supported
    optional = litellm.get_optional_params(
        model=f"sap/deployment/{_MODEL}",
        custom_llm_provider="sap",
        reasoning_effort="high",
        max_tokens=1200,
        drop_params=True,
    )
    assert "reasoning_effort" not in optional
    assert optional["output_config"] == {"effort": "high"}
    assert optional["thinking"]["type"] == "adaptive"


def test_messages_plain_model_returns_none_to_trigger_bridge():
    cfg = ProviderConfigManager.get_provider_anthropic_messages_config(model=_MODEL, provider=_SAP)
    assert cfg is None


def test_messages_deployment_prefix_routes_to_direct_connect():
    cfg = ProviderConfigManager.get_provider_anthropic_messages_config(model=f"deployment/{_MODEL}", provider=_SAP)
    assert isinstance(cfg, SapDeploymentAnthropicMessagesConfig)


def test_messages_gpt_deployment_returns_none_to_trigger_bridge():
    cfg = ProviderConfigManager.get_provider_anthropic_messages_config(model="deployment/gpt-5.6-sol", provider=_SAP)
    assert cfg is None


def test_messages_gemini_deployment_returns_none_to_trigger_bridge():
    cfg = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deployment/gemini-2.5-flash-lite", provider=_SAP
    )
    assert cfg is None


def test_completion_forwards_resource_group_into_litellm_params(monkeypatch):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs.get("litellm_params") or {})
        return litellm.ModelResponse()

    monkeypatch.setattr(litellm_main.sap_gen_ai_hub_chat_completions, "completion", _capture)
    litellm.completion(
        model="sap/deployment/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "hi"}],
        resource_group="feedback-ai",
        api_key="dummy",
    )
    assert captured.get("resource_group") == "feedback-ai"
