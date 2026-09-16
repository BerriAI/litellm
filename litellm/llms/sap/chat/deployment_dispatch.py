"""Pick the SAP direct-connect chat config for a foundation model by its family.

SAP AI Core serves each foundation model behind a provider-specific executable, so the native contract
at ``{deployment_url}`` differs by family: Claude runs on an ``aws-bedrock`` executable (Bedrock invoke
body at ``/invoke``), GPT runs on an Azure OpenAI executable (OpenAI chat body at ``/chat/completions``),
Gemini runs on a Vertex executable (Google ``generateContent`` body at ``/models/{model}:generateContent``).
This mirrors Bedrock's ``get_bedrock_chat_config`` dispatch: map the bare model name to the config that
wraps the right base transforms, so each family keeps its own broad native-param passthrough instead of
the single narrowed schema orchestration would force on all of them.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Final

from .direct_transformation import SapDeploymentAnthropicChatConfig
from .gemini_direct_transformation import SapDeploymentGeminiChatConfig
from .handler import GenAIHubOrchestrationError
from .openai_direct_transformation import SapDeploymentOpenAIChatConfig

if TYPE_CHECKING:
    from litellm.llms.base_llm.chat.transformation import BaseConfig


class SapModelFamily(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


def detect_sap_model_family(model: str) -> SapModelFamily | None:
    """Classify a bare SAP model name by native contract, or `None` when no family owns it."""
    lowered: Final = model.lower()
    if lowered.startswith("anthropic") or "claude" in lowered:
        return SapModelFamily.ANTHROPIC
    if "gpt" in lowered:
        return SapModelFamily.OPENAI
    if "gemini" in lowered:
        return SapModelFamily.GEMINI
    return None


def get_sap_deployment_chat_config(model: str) -> BaseConfig:
    """Return the direct-connect chat config for `model`, or raise a 400 naming supported families."""
    match detect_sap_model_family(model):
        case SapModelFamily.ANTHROPIC:
            return SapDeploymentAnthropicChatConfig()
        case SapModelFamily.OPENAI:
            return SapDeploymentOpenAIChatConfig()
        case SapModelFamily.GEMINI:
            return SapDeploymentGeminiChatConfig()
        case None:
            raise GenAIHubOrchestrationError(
                status_code=400,
                message=(
                    f"SAP direct-connect chat has no native transform for model '{model}'. "
                    "Supported families: Anthropic/Claude, OpenAI/GPT and Google/Gemini. "
                    "Drop the 'deployment/' prefix to route this model through orchestration instead."
                ),
            )
