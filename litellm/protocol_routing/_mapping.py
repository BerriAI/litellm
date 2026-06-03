from typing import Dict, List, Optional

from litellm.protocol_routing._types import SupportedProtocol

PROVIDER_DEFAULT_PROTOCOLS: Dict[str, List[SupportedProtocol]] = {
    "openai": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES],
    "azure": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES],
    "azure_ai": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES],
    "anthropic": [SupportedProtocol.ANTHROPIC_MESSAGES],
    "bedrock": [SupportedProtocol.ANTHROPIC_MESSAGES, SupportedProtocol.OPENAI_CHAT],
    "vertex_ai": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.GOOGLE_GENERATE_CONTENT],
    "gemini": [SupportedProtocol.GOOGLE_GENERATE_CONTENT, SupportedProtocol.OPENAI_CHAT],
    "deepseek": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.ANTHROPIC_MESSAGES],
    "minimax": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.ANTHROPIC_MESSAGES],
    "cohere": [SupportedProtocol.OPENAI_CHAT],
    "mistral": [SupportedProtocol.OPENAI_CHAT],
    "ollama": [SupportedProtocol.OPENAI_CHAT],
    "vllm": [SupportedProtocol.OPENAI_CHAT],
    "together_ai": [SupportedProtocol.OPENAI_CHAT],
    "groq": [SupportedProtocol.OPENAI_CHAT],
    "xai": [SupportedProtocol.OPENAI_CHAT],
}


def infer_protocols(
    provider: Optional[str],
    model_info: Optional[dict] = None,
) -> List[SupportedProtocol]:
    """Infer supported protocols for a deployment.

    Priority:
    1. Explicit model_info.supported_protocols (if provided)
    2. Provider default from PROVIDER_DEFAULT_PROTOCOLS
    3. Fallback to OPENAI_CHAT

    Args:
        provider: The LLM provider name (e.g., "openai", "anthropic")
        model_info: Optional model metadata dict, may contain "supported_protocols"

    Returns:
        List of SupportedProtocol values (always a new list to prevent mutation)
    """
    if model_info is not None:
        explicit = model_info.get("supported_protocols")
        if explicit is not None:
            # Convert string values to enum, pass through existing enums
            return [
                p if isinstance(p, SupportedProtocol) else SupportedProtocol(p)
                for p in explicit
            ]

    if provider is None:
        return [SupportedProtocol.OPENAI_CHAT]

    defaults = PROVIDER_DEFAULT_PROTOCOLS.get(provider)
    if defaults is not None:
        return list(defaults)  # Return copy to prevent mutation

    return [SupportedProtocol.OPENAI_CHAT]
