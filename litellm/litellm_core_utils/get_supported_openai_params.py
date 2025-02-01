from typing import Literal, Optional

import litellm
from litellm import LlmProviders
from litellm.exceptions import BadRequestError


def get_supported_openai_params(  # noqa: PLR0915
    model: str,
    custom_llm_provider: Optional[str] = None,
    request_type: Literal[
        "chat_completion", "embeddings", "transcription"
    ] = "chat_completion",
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
        return litellm.OllamaConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "ollama_chat":
        return litellm.OllamaChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "anthropic":
        return litellm.AnthropicConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "anthropic_text":
        return litellm.AnthropicTextConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "fireworks_ai":
        if request_type == "embeddings":
            return litellm.FireworksAIEmbeddingConfig().get_supported_openai_params(
                model=model
            )
        elif request_type == "transcription":
            return litellm.FireworksAIAudioTranscriptionConfig().get_supported_openai_params(
                model=model
            )
        else:
            return litellm.FireworksAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "nvidia_nim":
        if request_type == "chat_completion":
            return litellm.nvidiaNimConfig.get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return litellm.nvidiaNimEmbeddingConfig.get_supported_openai_params()
    elif custom_llm_provider == "cerebras":
        return litellm.CerebrasConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "xai":
        return litellm.XAIChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "ai21_chat" or custom_llm_provider == "ai21":
        return litellm.AI21ChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "volcengine":
        return litellm.VolcEngineConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "groq":
        return litellm.GroqChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "hosted_vllm":
        return litellm.HostedVLLMChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "vllm":
        return litellm.VLLMConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "deepseek":
        return litellm.DeepSeekChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "cohere":
        return litellm.CohereConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "cohere_chat":
        return litellm.CohereChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "maritalk":
        return litellm.MaritalkConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "openai":
        return litellm.OpenAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "azure":
        if litellm.AzureOpenAIO1Config().is_o_series_model(model=model):
            return litellm.AzureOpenAIO1Config().get_supported_openai_params(
                model=model
            )
        else:
            return litellm.AzureOpenAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "openrouter":
        return litellm.OpenrouterConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "mistral" or custom_llm_provider == "codestral":
        # mistal and codestral api have the exact same params
        if request_type == "chat_completion":
            return litellm.MistralConfig().get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return litellm.MistralEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "text-completion-codestral":
        return litellm.CodestralTextCompletionConfig().get_supported_openai_params(
            model=model
        )
    elif custom_llm_provider == "sambanova":
        return litellm.SambanovaConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "replicate":
        return litellm.ReplicateConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "huggingface":
        return litellm.HuggingfaceConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "jina_ai":
        if request_type == "embeddings":
            return litellm.JinaAIEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "together_ai":
        return litellm.TogetherAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "databricks":
        if request_type == "chat_completion":
            return litellm.DatabricksConfig().get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return litellm.DatabricksEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "palm" or custom_llm_provider == "gemini":
        return litellm.GoogleAIStudioGeminiConfig().get_supported_openai_params(
            model=model
        )
    elif custom_llm_provider == "vertex_ai" or custom_llm_provider == "vertex_ai_beta":
        if request_type == "chat_completion":
            if model.startswith("meta/"):
                return litellm.VertexAILlama3Config().get_supported_openai_params()
            if model.startswith("mistral"):
                return litellm.MistralConfig().get_supported_openai_params(model=model)
            if model.startswith("codestral"):
                return (
                    litellm.CodestralTextCompletionConfig().get_supported_openai_params(
                        model=model
                    )
                )
            if model.startswith("claude"):
                return litellm.VertexAIAnthropicConfig().get_supported_openai_params(
                    model=model
                )
            return litellm.VertexGeminiConfig().get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return litellm.VertexAITextEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "sagemaker":
        return litellm.SagemakerConfig().get_supported_openai_params(model=model)
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
        return litellm.CloudflareChatConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "nlp_cloud":
        return litellm.NLPCloudConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "petals":
        return litellm.PetalsConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "deepinfra":
        return litellm.DeepInfraConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "perplexity":
        return litellm.PerplexityChatConfig().get_supported_openai_params(model=model)
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
    elif custom_llm_provider == "watsonx_text":
        return litellm.IBMWatsonXAIConfig().get_supported_openai_params(model=model)
    elif (
        custom_llm_provider == "custom_openai"
        or custom_llm_provider == "text-completion-openai"
    ):
        return litellm.OpenAITextCompletionConfig().get_supported_openai_params(
            model=model
        )
    elif custom_llm_provider == "predibase":
        return litellm.PredibaseConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "voyage":
        return litellm.VoyageEmbeddingConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "triton":
        if request_type == "embeddings":
            return litellm.TritonEmbeddingConfig().get_supported_openai_params(
                model=model
            )
        else:
            return litellm.TritonConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "deepgram":
        if request_type == "transcription":
            return (
                litellm.DeepgramAudioTranscriptionConfig().get_supported_openai_params(
                    model=model
                )
            )
    elif custom_llm_provider in litellm._custom_providers:
        if request_type == "chat_completion":
            provider_config = litellm.ProviderConfigManager.get_provider_chat_config(
                model=model, provider=LlmProviders.CUSTOM
            )
            return provider_config.get_supported_openai_params(model=model)
        elif request_type == "embeddings":
            return None
        elif request_type == "transcription":
            return None

    return None
