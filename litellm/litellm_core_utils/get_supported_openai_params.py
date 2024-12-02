from typing import Literal, Optional

import litellm
from litellm.exceptions import BadRequestError


def get_supported_openai_params(  # noqa: PLR0915
    model: str,
    custom_llm_provider: Optional[str] = None,
    request_type: Literal["chat_completion", "embeddings"] = "chat_completion",
) -> Optional[list]:
    """
    Returns the supported openai params for a given model + provider

    Example:
    ```
    get_supported_openai_params(model="anthropic.claude-3", custom_llm_provider="bedrock")
    ```

    Returns:
    - List if custom_llm_provider is mapped
    - None if unmapped
    """
    if not custom_llm_provider:
        try:
            custom_llm_provider = litellm.get_llm_provider(model=model)[1]
        except BadRequestError:
            return None
    if custom_llm_provider == "bedrock":
        return litellm.AmazonConverseConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "ollama":
        return litellm.OllamaConfig().get_supported_openai_params()
    elif custom_llm_provider == "ollama_chat":
        return litellm.OllamaChatConfig().get_supported_openai_params()
    elif custom_llm_provider == "anthropic":
        return litellm.AnthropicConfig().get_supported_openai_params()
    elif custom_llm_provider == "fireworks_ai":
        if request_type == "embeddings":
            return litellm.FireworksAIEmbeddingConfig().get_supported_openai_params(
                model=model
            )
        else:
            return litellm.FireworksAIConfig().get_supported_openai_params()
    elif custom_llm_provider == "nvidia_nim":
        if request_type == "chat_completion":
            return litellm.nvidiaNimConfig.get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return litellm.nvidiaNimEmbeddingConfig.get_supported_openai_params()
    elif custom_llm_provider == "cerebras":
        return litellm.CerebrasConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "xai":
        return litellm.XAIChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "ai21_chat":
        return litellm.AI21ChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "volcengine":
        return litellm.VolcEngineConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "groq":
        return litellm.GroqChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "hosted_vllm":
        return litellm.HostedVLLMChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "deepseek":
        return [
            # https://platform.deepseek.com/api-docs/api/create-chat-completion
            "frequency_penalty",
            "max_tokens",
            "presence_penalty",
            "response_format",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "logprobs",
            "top_logprobs",
            "tools",
            "tool_choice",
        ]
    elif custom_llm_provider == "cohere":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "logit_bias",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "extra_headers",
        ]
    elif custom_llm_provider == "cohere_chat":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "tools",
            "tool_choice",
            "seed",
            "extra_headers",
        ]
    elif custom_llm_provider == "maritalk":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "presence_penalty",
            "stop",
        ]
    elif custom_llm_provider == "openai":
        return litellm.OpenAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "azure":
        if litellm.AzureOpenAIO1Config().is_o1_model(model=model):
            return litellm.AzureOpenAIO1Config().get_supported_openai_params(
                model=model
            )
        else:
            return litellm.AzureOpenAIConfig().get_supported_openai_params()
    elif custom_llm_provider == "openrouter":
        return [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "seed",
            "max_tokens",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "response_format",
            "stop",
            "tools",
            "tool_choice",
        ]
    elif custom_llm_provider == "mistral" or custom_llm_provider == "codestral":
        # mistal and codestral api have the exact same params
        if request_type == "chat_completion":
            return litellm.MistralConfig().get_supported_openai_params()
        elif request_type == "embeddings":
            return litellm.MistralEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "text-completion-codestral":
        return litellm.MistralTextCompletionConfig().get_supported_openai_params()
    elif custom_llm_provider == "replicate":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "seed",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
        ]
    elif custom_llm_provider == "huggingface":
        return litellm.HuggingfaceConfig().get_supported_openai_params()
    elif custom_llm_provider == "together_ai":
        return litellm.TogetherAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "ai21":
        return [
            "stream",
            "n",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        ]
    elif custom_llm_provider == "databricks":
        if request_type == "chat_completion":
            return litellm.DatabricksConfig().get_supported_openai_params()
        elif request_type == "embeddings":
            return litellm.DatabricksEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "palm" or custom_llm_provider == "gemini":
        return litellm.GoogleAIStudioGeminiConfig().get_supported_openai_params()
    elif custom_llm_provider == "vertex_ai":
        if request_type == "chat_completion":
            if model.startswith("meta/"):
                return litellm.VertexAILlama3Config().get_supported_openai_params()
            if model.startswith("mistral"):
                return litellm.MistralConfig().get_supported_openai_params()
            if model.startswith("codestral"):
                return (
                    litellm.MistralTextCompletionConfig().get_supported_openai_params()
                )
            if model.startswith("claude"):
                return litellm.VertexAIAnthropicConfig().get_supported_openai_params()
            return litellm.VertexAIConfig().get_supported_openai_params()
        elif request_type == "embeddings":
            return litellm.VertexAITextEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "vertex_ai_beta":
        if request_type == "chat_completion":
            return litellm.VertexGeminiConfig().get_supported_openai_params()
        elif request_type == "embeddings":
            return litellm.VertexAITextEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "sagemaker":
        return ["stream", "temperature", "max_tokens", "top_p", "stop", "n"]
    elif custom_llm_provider == "aleph_alpha":
        return [
            "max_tokens",
            "stream",
            "top_p",
            "temperature",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stop",
        ]
    elif custom_llm_provider == "cloudflare":
        return ["max_tokens", "stream"]
    elif custom_llm_provider == "nlp_cloud":
        return [
            "max_tokens",
            "stream",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stop",
        ]
    elif custom_llm_provider == "petals":
        return ["max_tokens", "temperature", "top_p", "stream"]
    elif custom_llm_provider == "deepinfra":
        return litellm.DeepInfraConfig().get_supported_openai_params()
    elif custom_llm_provider == "perplexity":
        return [
            "temperature",
            "top_p",
            "stream",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
        ]
    elif custom_llm_provider == "anyscale":
        return [
            "temperature",
            "top_p",
            "stream",
            "max_tokens",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        ]
    elif custom_llm_provider == "watsonx":
        return litellm.IBMWatsonXChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "custom_openai" or "text-completion-openai":
        return [
            "functions",
            "function_call",
            "temperature",
            "top_p",
            "n",
            "stream",
            "stream_options",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "max_retries",
            "logprobs",
            "top_logprobs",
            "extra_headers",
        ]
    return None
