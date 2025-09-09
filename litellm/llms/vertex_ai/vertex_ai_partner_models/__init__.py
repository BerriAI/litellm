from litellm.llms.base_llm.chat.transformation import BaseConfig


def get_vertex_ai_partner_model_config(
    model: str, vertex_publisher_or_api_spec: str
) -> BaseConfig:
    """Return config for handling response transformation for vertex ai partner models"""
    if vertex_publisher_or_api_spec == "anthropic":
        from .anthropic.transformation import VertexAIAnthropicConfig

        return VertexAIAnthropicConfig()
    elif vertex_publisher_or_api_spec == "ai21":
        from .ai21.transformation import VertexAIAi21Config

        return VertexAIAi21Config()
    elif (
        vertex_publisher_or_api_spec == "openapi"
        or vertex_publisher_or_api_spec == "mistralai"
    ):
        from .llama3.transformation import VertexAILlama3Config

        return VertexAILlama3Config()
    else:
        raise ValueError(f"Unsupported model: {model}")
