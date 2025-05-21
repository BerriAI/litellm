import json
import os
import sys
from unittest.mock import patch

import pytest
from jsonschema import validate

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.utils import LlmProviders
from litellm.utils import (
    ProviderConfigManager,
    get_llm_provider,
    get_optional_params_image_gen,
)

# Adds the parent directory to the system path


def test_get_optional_params_image_gen():
    from litellm.llms.azure.image_generation import AzureGPTImageGenerationConfig

    provider_config = AzureGPTImageGenerationConfig()
    optional_params = get_optional_params_image_gen(
        model="gpt-image-1",
        response_format="b64_json",
        n=3,
        custom_llm_provider="azure",
        drop_params=True,
        provider_config=provider_config,
    )
    assert optional_params is not None
    assert "response_format" not in optional_params
    assert optional_params["n"] == 3


def test_all_model_configs():
    from litellm.llms.vertex_ai.vertex_ai_partner_models.ai21.transformation import (
        VertexAIAi21Config,
    )
    from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
        VertexAILlama3Config,
    )

    assert (
        "max_completion_tokens"
        in VertexAILlama3Config().get_supported_openai_params(model="llama3")
    )
    assert VertexAILlama3Config().map_openai_params(
        {"max_completion_tokens": 10}, {}, "llama3", drop_params=False
    ) == {"max_tokens": 10}

    assert "max_completion_tokens" in VertexAIAi21Config().get_supported_openai_params(
        model="jamba-1.5-mini@001"
    )
    assert VertexAIAi21Config().map_openai_params(
        {"max_completion_tokens": 10}, {}, "jamba-1.5-mini@001", drop_params=False
    ) == {"max_tokens": 10}

    from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig

    assert "max_completion_tokens" in FireworksAIConfig().get_supported_openai_params(
        model="llama3"
    )
    assert FireworksAIConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig

    assert "max_completion_tokens" in NvidiaNimConfig().get_supported_openai_params(
        model="llama3"
    )
    assert NvidiaNimConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.ollama_chat import OllamaChatConfig

    assert "max_completion_tokens" in OllamaChatConfig().get_supported_openai_params(
        model="llama3"
    )
    assert OllamaChatConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"num_predict": 10}

    from litellm.llms.predibase.chat.transformation import PredibaseConfig

    assert "max_completion_tokens" in PredibaseConfig().get_supported_openai_params(
        model="llama3"
    )
    assert PredibaseConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_new_tokens": 10}

    from litellm.llms.codestral.completion.transformation import (
        CodestralTextCompletionConfig,
    )

    assert (
        "max_completion_tokens"
        in CodestralTextCompletionConfig().get_supported_openai_params(model="llama3")
    )
    assert CodestralTextCompletionConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.volcengine import VolcEngineConfig

    assert "max_completion_tokens" in VolcEngineConfig().get_supported_openai_params(
        model="llama3"
    )
    assert VolcEngineConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.ai21.chat.transformation import AI21ChatConfig

    assert "max_completion_tokens" in AI21ChatConfig().get_supported_openai_params(
        "jamba-1.5-mini@001"
    )
    assert AI21ChatConfig().map_openai_params(
        model="jamba-1.5-mini@001",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig

    assert "max_completion_tokens" in AzureOpenAIConfig().get_supported_openai_params(
        model="gpt-3.5-turbo"
    )
    assert AzureOpenAIConfig().map_openai_params(
        model="gpt-3.5-turbo",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        api_version="2022-12-01",
        drop_params=False,
    ) == {"max_completion_tokens": 10}

    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    assert (
        "max_completion_tokens"
        in AmazonConverseConfig().get_supported_openai_params(
            model="anthropic.claude-3-sonnet-20240229-v1:0"
        )
    )
    assert AmazonConverseConfig().map_openai_params(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"maxTokens": 10}

    from litellm.llms.codestral.completion.transformation import (
        CodestralTextCompletionConfig,
    )

    assert (
        "max_completion_tokens"
        in CodestralTextCompletionConfig().get_supported_openai_params(model="llama3")
    )
    assert CodestralTextCompletionConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm import AmazonAnthropicClaude3Config, AmazonAnthropicConfig

    assert (
        "max_completion_tokens"
        in AmazonAnthropicClaude3Config().get_supported_openai_params(
            model="anthropic.claude-3-sonnet-20240229-v1:0"
        )
    )

    assert AmazonAnthropicClaude3Config().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        drop_params=False,
    ) == {"max_tokens": 10}

    assert (
        "max_completion_tokens"
        in AmazonAnthropicConfig().get_supported_openai_params(model="")
    )

    assert AmazonAnthropicConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        model="",
        drop_params=False,
    ) == {"max_tokens_to_sample": 10}

    from litellm.llms.databricks.chat.transformation import DatabricksConfig

    assert "max_completion_tokens" in DatabricksConfig().get_supported_openai_params()

    assert DatabricksConfig().map_openai_params(
        model="databricks/llama-3-70b-instruct",
        drop_params=False,
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
    ) == {"max_tokens": 10}

    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    assert (
        "max_completion_tokens"
        in VertexAIAnthropicConfig().get_supported_openai_params(
            model="claude-3-5-sonnet-20240620"
        )
    )

    assert VertexAIAnthropicConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        model="claude-3-5-sonnet-20240620",
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    assert "max_completion_tokens" in VertexGeminiConfig().get_supported_openai_params(
        model="gemini-1.0-pro"
    )

    assert VertexGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}

    assert (
        "max_completion_tokens"
        in GoogleAIStudioGeminiConfig().get_supported_openai_params(
            model="gemini-1.0-pro"
        )
    )

    assert GoogleAIStudioGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}

    assert "max_completion_tokens" in VertexGeminiConfig().get_supported_openai_params(
        model="gemini-1.0-pro"
    )

    assert VertexGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}


def test_anthropic_web_search_in_model_info():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    supported_models = [
        "anthropic/claude-3-7-sonnet-20250219",
        "anthropic/claude-3-5-sonnet-latest",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-3-5-haiku-latest",
    ]
    for model in supported_models:
        from litellm.utils import get_model_info

        model_info = get_model_info(model)
        assert model_info is not None
        assert (
            model_info["supports_web_search"] is True
        ), f"Model {model} should support web search"
        assert (
            model_info["search_context_cost_per_query"] is not None
        ), f"Model {model} should have a search context cost per query"


def test_cohere_embedding_optional_params():
    from litellm import get_optional_params_embeddings

    optional_params = get_optional_params_embeddings(
        model="embed-v4.0",
        custom_llm_provider="cohere",
        input="Hello, world!",
        input_type="search_query",
        dimensions=512,
    )
    assert optional_params is not None


def test_aaamodel_prices_and_context_window_json_is_valid():
    """
    Validates the `model_prices_and_context_window.json` file.

    If this test fails after you update the json, you need to update the schema or correct the change you made.
    """

    INTENDED_SCHEMA = {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "supports_computer_use": {"type": "boolean"},
                "cache_creation_input_audio_token_cost": {"type": "number"},
                "cache_creation_input_token_cost": {"type": "number"},
                "cache_read_input_token_cost": {"type": "number"},
                "cache_read_input_audio_token_cost": {"type": "number"},
                "deprecation_date": {"type": "string"},
                "input_cost_per_audio_per_second": {"type": "number"},
                "input_cost_per_audio_per_second_above_128k_tokens": {"type": "number"},
                "input_cost_per_audio_token": {"type": "number"},
                "input_cost_per_character": {"type": "number"},
                "input_cost_per_character_above_128k_tokens": {"type": "number"},
                "input_cost_per_image": {"type": "number"},
                "input_cost_per_image_above_128k_tokens": {"type": "number"},
                "input_cost_per_token_above_200k_tokens": {"type": "number"},
                "input_cost_per_pixel": {"type": "number"},
                "input_cost_per_query": {"type": "number"},
                "input_cost_per_request": {"type": "number"},
                "input_cost_per_second": {"type": "number"},
                "input_cost_per_token": {"type": "number"},
                "input_cost_per_token_above_128k_tokens": {"type": "number"},
                "input_cost_per_token_batch_requests": {"type": "number"},
                "input_cost_per_token_batches": {"type": "number"},
                "input_cost_per_token_cache_hit": {"type": "number"},
                "input_cost_per_video_per_second": {"type": "number"},
                "input_cost_per_video_per_second_above_8s_interval": {"type": "number"},
                "input_cost_per_video_per_second_above_15s_interval": {
                    "type": "number"
                },
                "input_cost_per_video_per_second_above_128k_tokens": {"type": "number"},
                "input_dbu_cost_per_token": {"type": "number"},
                "litellm_provider": {"type": "string"},
                "max_audio_length_hours": {"type": "number"},
                "max_audio_per_prompt": {"type": "number"},
                "max_document_chunks_per_query": {"type": "number"},
                "max_images_per_prompt": {"type": "number"},
                "max_input_tokens": {"type": "number"},
                "max_output_tokens": {"type": "number"},
                "max_pdf_size_mb": {"type": "number"},
                "max_query_tokens": {"type": "number"},
                "max_tokens": {"type": "number"},
                "max_tokens_per_document_chunk": {"type": "number"},
                "max_video_length": {"type": "number"},
                "max_videos_per_prompt": {"type": "number"},
                "metadata": {"type": "object"},
                "mode": {
                    "type": "string",
                    "enum": [
                        "audio_speech",
                        "audio_transcription",
                        "chat",
                        "completion",
                        "embedding",
                        "image_generation",
                        "moderation",
                        "rerank",
                        "responses",
                    ],
                },
                "output_cost_per_audio_token": {"type": "number"},
                "output_cost_per_character": {"type": "number"},
                "output_cost_per_character_above_128k_tokens": {"type": "number"},
                "output_cost_per_image": {"type": "number"},
                "output_cost_per_pixel": {"type": "number"},
                "output_cost_per_second": {"type": "number"},
                "output_cost_per_token": {"type": "number"},
                "output_cost_per_token_above_128k_tokens": {"type": "number"},
                "output_cost_per_token_above_200k_tokens": {"type": "number"},
                "output_cost_per_token_batches": {"type": "number"},
                "output_cost_per_reasoning_token": {"type": "number"},
                "output_db_cost_per_token": {"type": "number"},
                "output_dbu_cost_per_token": {"type": "number"},
                "output_vector_size": {"type": "number"},
                "rpd": {"type": "number"},
                "rpm": {"type": "number"},
                "source": {"type": "string"},
                "supports_assistant_prefill": {"type": "boolean"},
                "supports_audio_input": {"type": "boolean"},
                "supports_audio_output": {"type": "boolean"},
                "supports_embedding_image_input": {"type": "boolean"},
                "supports_function_calling": {"type": "boolean"},
                "supports_image_input": {"type": "boolean"},
                "supports_parallel_function_calling": {"type": "boolean"},
                "supports_pdf_input": {"type": "boolean"},
                "supports_prompt_caching": {"type": "boolean"},
                "supports_response_schema": {"type": "boolean"},
                "supports_system_messages": {"type": "boolean"},
                "supports_tool_choice": {"type": "boolean"},
                "supports_video_input": {"type": "boolean"},
                "supports_vision": {"type": "boolean"},
                "supports_web_search": {"type": "boolean"},
                "supports_reasoning": {"type": "boolean"},
                "tool_use_system_prompt_tokens": {"type": "number"},
                "tpm": {"type": "number"},
                "supported_endpoints": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "/v1/responses",
                            "/v1/embeddings",
                            "/v1/chat/completions",
                            "/v1/completions",
                            "/v1/images/generations",
                            "/v1/images/variations",
                            "/v1/images/edits",
                            "/v1/batch",
                            "/v1/audio/transcriptions",
                            "/v1/audio/speech",
                        ],
                    },
                },
                "search_context_cost_per_query": {
                    "type": "object",
                    "properties": {
                        "search_context_size_low": {"type": "number"},
                        "search_context_size_medium": {"type": "number"},
                        "search_context_size_high": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "supported_modalities": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["text", "audio", "image", "video"],
                    },
                },
                "supported_output_modalities": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["text", "image", "audio", "code"],
                    },
                },
                "supports_native_streaming": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    }

    prod_json = "./model_prices_and_context_window.json"
    # prod_json = "../../model_prices_and_context_window.json"
    with open(prod_json, "r") as model_prices_file:
        actual_json = json.load(model_prices_file)
    assert isinstance(actual_json, dict)
    actual_json.pop(
        "sample_spec", None
    )  # remove the sample, whose schema is inconsistent with the real data

    validate(actual_json, INTENDED_SCHEMA)


def test_get_model_info_gemini():
    """
    Tests if ALL gemini models have 'tpm' and 'rpm' in the model info
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_map = litellm.model_cost
    for model, info in model_map.items():
        if (
            model.startswith("gemini/")
            and not "gemma" in model
            and not "learnlm" in model
        ):
            assert info.get("tpm") is not None, f"{model} does not have tpm"
            assert info.get("rpm") is not None, f"{model} does not have rpm"


def test_openai_models_in_model_info():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_map = litellm.model_cost
    violated_models = []
    for model, info in model_map.items():
        if (
            info.get("litellm_provider") == "openai"
            and info.get("supports_vision") is True
        ):
            if info.get("supports_pdf_input") is not True:
                violated_models.append(model)
    assert (
        len(violated_models) == 0
    ), f"The following models should support pdf input: {violated_models}"


def test_supports_tool_choice_simple_tests():
    """
    simple sanity checks
    """
    assert litellm.utils.supports_tool_choice(model="gpt-4o") == True
    assert (
        litellm.utils.supports_tool_choice(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        )
        == True
    )
    assert (
        litellm.utils.supports_tool_choice(
            model="anthropic.claude-3-sonnet-20240229-v1:0"
        )
        is True
    )

    assert (
        litellm.utils.supports_tool_choice(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            custom_llm_provider="bedrock_converse",
        )
        is True
    )

    assert (
        litellm.utils.supports_tool_choice(model="us.amazon.nova-micro-v1:0") is False
    )
    assert (
        litellm.utils.supports_tool_choice(model="bedrock/us.amazon.nova-micro-v1:0")
        is False
    )
    assert (
        litellm.utils.supports_tool_choice(
            model="us.amazon.nova-micro-v1:0", custom_llm_provider="bedrock_converse"
        )
        is False
    )

    assert litellm.utils.supports_tool_choice(model="perplexity/sonar") is False


def test_check_provider_match():
    """
    Test the _check_provider_match function for various provider scenarios
    """
    # Test bedrock and bedrock_converse cases
    model_info = {"litellm_provider": "bedrock"}
    assert litellm.utils._check_provider_match(model_info, "bedrock") is True
    assert litellm.utils._check_provider_match(model_info, "bedrock_converse") is True

    # Test bedrock_converse provider
    model_info = {"litellm_provider": "bedrock_converse"}
    assert litellm.utils._check_provider_match(model_info, "bedrock") is True
    assert litellm.utils._check_provider_match(model_info, "bedrock_converse") is True

    # Test non-matching provider
    model_info = {"litellm_provider": "bedrock"}
    assert litellm.utils._check_provider_match(model_info, "openai") is False


# Models that should be skipped during testing
OLD_PROVIDERS = ["aleph_alpha", "palm"]
SKIP_MODELS = [
    "azure/mistral",
    "azure/command-r",
    "jamba",
    "deepinfra",
    "mistral.",
    "groq/llama-guard-3-8b",
    "groq/gemma2-9b-it",
]

# Bedrock models to block - organized by type
BEDROCK_REGIONS = ["ap-northeast-1", "eu-central-1", "us-east-1", "us-west-2"]
BEDROCK_COMMITMENTS = ["1-month-commitment", "6-month-commitment"]
BEDROCK_MODELS = {
    "anthropic.claude-v1",
    "anthropic.claude-v2",
    "anthropic.claude-v2:1",
    "anthropic.claude-instant-v1",
}

# Generate block_list dynamically
block_list = set()
for region in BEDROCK_REGIONS:
    for commitment in BEDROCK_COMMITMENTS:
        for model in BEDROCK_MODELS:
            block_list.add(f"bedrock/{region}/{commitment}/{model}")
            block_list.add(f"bedrock/{region}/{model}")

# Add Cohere models
for commitment in BEDROCK_COMMITMENTS:
    block_list.add(f"bedrock/*/{commitment}/cohere.command-text-v14")
    block_list.add(f"bedrock/*/{commitment}/cohere.command-light-text-v14")

print("block_list", block_list)


@pytest.mark.asyncio
async def test_supports_tool_choice():
    """
    Test that litellm.utils.supports_tool_choice() returns the correct value
    for all models in model_prices_and_context_window.json.

    The test:
    1. Loads model pricing data
    2. Iterates through each model
    3. Checks if tool_choice support matches the model's supported parameters
    """
    # Load model prices
    litellm._turn_on_debug()
    # path = "../../model_prices_and_context_window.json"
    path = "./model_prices_and_context_window.json"
    with open(path, "r") as f:
        model_prices = json.load(f)
    litellm.model_cost = model_prices
    config_manager = ProviderConfigManager()

    for model_name, model_info in model_prices.items():
        print(f"testing model: {model_name}")

        # Skip certain models
        if (
            model_name == "sample_spec"
            or model_info.get("mode") != "chat"
            or any(skip in model_name for skip in SKIP_MODELS)
            or any(provider in model_name for provider in OLD_PROVIDERS)
            or model_info["litellm_provider"] in OLD_PROVIDERS
            or model_name in block_list
            or "azure/eu" in model_name
            or "azure/us" in model_name
            or "codestral" in model_name
            or "o1" in model_name
            or "o3" in model_name
            or "mistral" in model_name
        ):
            continue

        try:
            model, provider, _, _ = get_llm_provider(model=model_name)
        except Exception as e:
            print(f"\033[91mERROR for {model_name}: {e}\033[0m")
            continue

        # Get provider config and supported params
        print("LLM provider", provider)
        provider_enum = LlmProviders(provider)
        config = config_manager.get_provider_chat_config(model, provider_enum)
        print("config", config)

        if config:
            supported_params = config.get_supported_openai_params(model)
            print("supported_params", supported_params)
        else:
            raise Exception(f"No config found for {model_name}, provider: {provider}")

        # Check tool_choice support
        supports_tool_choice_result = litellm.utils.supports_tool_choice(
            model=model_name, custom_llm_provider=provider
        )
        tool_choice_in_params = "tool_choice" in supported_params

        assert (
            supports_tool_choice_result == tool_choice_in_params
        ), f"Tool choice support mismatch for {model_name}. supports_tool_choice() returned: {supports_tool_choice_result}, tool_choice in supported params: {tool_choice_in_params}\nConfig: {config}"


def test_supports_computer_use_utility():
    """
    Tests the litellm.utils.supports_computer_use utility function.
    """
    from litellm.utils import supports_computer_use
    # Ensure LITELLM_LOCAL_MODEL_COST_MAP is set for consistent test behavior,
    # as supports_computer_use relies on get_model_info.
    # This also requires litellm.model_cost to be populated.
    original_env_var = os.getenv("LITELLM_LOCAL_MODEL_COST_MAP")
    original_model_cost = getattr(litellm, "model_cost", None)

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="") # Load with local/backup
    
    try:
        # Test a model known to support computer_use from backup JSON
        supports_cu_anthropic = supports_computer_use(model="anthropic/claude-3-7-sonnet-20250219")
        assert supports_cu_anthropic is True

        # Test a model known not to have the flag or set to false (defaults to False via get_model_info)
        supports_cu_gpt = supports_computer_use(model="gpt-3.5-turbo")
        assert supports_cu_gpt is False
    finally:
        # Restore original environment and model_cost to avoid side effects
        if original_env_var is None:
            del os.environ["LITELLM_LOCAL_MODEL_COST_MAP"]
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = original_env_var
        
        if original_model_cost is not None:
            litellm.model_cost = original_model_cost
        elif hasattr(litellm, "model_cost"):
            delattr(litellm, "model_cost")

def test_get_model_info_shows_supports_computer_use():
    """
    Tests if 'supports_computer_use' is correctly retrieved by get_model_info.
    We'll use 'claude-3-7-sonnet-20250219' as it's configured
    in the backup JSON to have supports_computer_use: True.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    # Ensure litellm.model_cost is loaded, relying on the backup mechanism if primary fails
    # as per previous debugging.
    litellm.model_cost = litellm.get_model_cost_map(url="") 
    
    # This model should have 'supports_computer_use': True in the backup JSON
    model_known_to_support_computer_use = "claude-3-7-sonnet-20250219"
    info = litellm.get_model_info(model_known_to_support_computer_use)
    print(f"Info for {model_known_to_support_computer_use}: {info}")
    
    # After the fix in utils.py, this should now be present and True
    assert info.get("supports_computer_use") is True

    # Optionally, test a model known NOT to support it, or where it's undefined (should default to False)
    # For example, if "gpt-3.5-turbo" doesn't have it defined, it should be False.
    model_known_not_to_support_computer_use = "gpt-3.5-turbo"
    info_gpt = litellm.get_model_info(model_known_not_to_support_computer_use)
    print(f"Info for {model_known_not_to_support_computer_use}: {info_gpt}")
    assert info_gpt.get("supports_computer_use") is False # Expecting False due to the default in ModelInfoBase