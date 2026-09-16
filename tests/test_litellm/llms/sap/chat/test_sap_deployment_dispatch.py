import pytest

from litellm.llms.sap.chat.deployment_dispatch import (
    SapModelFamily,
    detect_sap_model_family,
    get_sap_deployment_chat_config,
)
from litellm.llms.sap.chat.direct_transformation import SapDeploymentAnthropicChatConfig
from litellm.llms.sap.chat.gemini_direct_transformation import SapDeploymentGeminiChatConfig
from litellm.llms.sap.chat.handler import GenAIHubOrchestrationError
from litellm.llms.sap.chat.openai_direct_transformation import SapDeploymentOpenAIChatConfig


@pytest.mark.parametrize(
    "model,family",
    [
        ("anthropic--claude-4.8-opus", SapModelFamily.ANTHROPIC),
        ("claude-3-5-sonnet", SapModelFamily.ANTHROPIC),
        ("gpt-4o", SapModelFamily.OPENAI),
        ("gpt-5", SapModelFamily.OPENAI),
        ("gemini-2.5-pro", SapModelFamily.GEMINI),
        ("gemini-2.5-flash-lite", SapModelFamily.GEMINI),
        ("mistral-large", None),
    ],
)
def test_detect_sap_model_family(model, family):
    assert detect_sap_model_family(model) is family


def test_dispatch_anthropic_returns_bedrock_invoke_config():
    assert isinstance(get_sap_deployment_chat_config("anthropic--claude-4.8-opus"), SapDeploymentAnthropicChatConfig)


def test_dispatch_gpt_returns_openai_chat_config():
    assert isinstance(get_sap_deployment_chat_config("gpt-4o"), SapDeploymentOpenAIChatConfig)


def test_dispatch_gemini_returns_vertex_generate_content_config():
    assert isinstance(get_sap_deployment_chat_config("gemini-2.5-flash-lite"), SapDeploymentGeminiChatConfig)


def test_dispatch_unknown_family_raises_400_listing_supported():
    with pytest.raises(GenAIHubOrchestrationError) as exc:
        get_sap_deployment_chat_config("mistral-large")
    assert exc.value.status_code == 400
    assert "mistral-large" in exc.value.message
