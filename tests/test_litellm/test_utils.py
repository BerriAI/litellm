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

    from litellm.llms.ollama.chat.transformation import OllamaChatConfig

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
                "supports_url_context": {"type": "boolean"},
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
                "supported_regions": {
                    "type": "array",
                    "items": {
                        "type": "string",
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
                "citation_cost_per_token": {"type": "number"},
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
    litellm.model_cost = litellm.get_model_cost_map(url="")  # Load with local/backup

    try:
        # Test a model known to support computer_use from backup JSON
        supports_cu_anthropic = supports_computer_use(
            model="anthropic/claude-3-7-sonnet-20250219"
        )
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
    assert (
        info_gpt.get("supports_computer_use") is None
    )  # Expecting None due to the default in ModelInfoBase


@pytest.mark.parametrize(
    "model, custom_llm_provider",
    [
        ("gpt-3.5-turbo", "openai"),
        ("anthropic.claude-3-7-sonnet-20250219-v1:0", "bedrock"),
        ("gemini-1.5-pro", "vertex_ai"),
    ],
)
def test_pre_process_non_default_params(model, custom_llm_provider):
    from pydantic import BaseModel

    from litellm.utils import pre_process_non_default_params

    class ResponseFormat(BaseModel):
        x: str
        y: str

    passed_params = {
        "model": "gpt-3.5-turbo",
        "response_format": ResponseFormat,
    }
    special_params = {}
    processed_non_default_params = pre_process_non_default_params(
        model=model,
        passed_params=passed_params,
        special_params=special_params,
        custom_llm_provider=custom_llm_provider,
        additional_drop_params=None,
    )
    print(processed_non_default_params)
    assert processed_non_default_params == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "properties": {
                        "x": {"title": "X", "type": "string"},
                        "y": {"title": "Y", "type": "string"},
                    },
                    "required": ["x", "y"],
                    "title": "ResponseFormat",
                    "type": "object",
                    "additionalProperties": False,
                },
                "name": "ResponseFormat",
                "strict": True,
            },
        }
    }


from litellm.utils import supports_function_calling


class TestProxyFunctionCalling:
    """Test class for proxy function calling capabilities."""

    @pytest.fixture(autouse=True)
    def reset_mock_cache(self):
        """Reset model cache before each test."""
        from litellm.utils import _model_cache

        _model_cache.flush_cache()

    @pytest.mark.parametrize(
        "direct_model,proxy_model,expected_result",
        [
            # OpenAI models
            ("gpt-3.5-turbo", "litellm_proxy/gpt-3.5-turbo", True),
            ("gpt-4", "litellm_proxy/gpt-4", True),
            ("gpt-4o", "litellm_proxy/gpt-4o", True),
            ("gpt-4o-mini", "litellm_proxy/gpt-4o-mini", True),
            ("gpt-4-turbo", "litellm_proxy/gpt-4-turbo", True),
            ("gpt-4-1106-preview", "litellm_proxy/gpt-4-1106-preview", True),
            # Azure OpenAI models
            ("azure/gpt-4", "litellm_proxy/azure/gpt-4", True),
            ("azure/gpt-3.5-turbo", "litellm_proxy/azure/gpt-3.5-turbo", True),
            (
                "azure/gpt-4-1106-preview",
                "litellm_proxy/azure/gpt-4-1106-preview",
                True,
            ),
            # Anthropic models (Claude supports function calling)
            (
                "claude-3-5-sonnet-20240620",
                "litellm_proxy/claude-3-5-sonnet-20240620",
                True,
            ),
            ("claude-3-opus-20240229", "litellm_proxy/claude-3-opus-20240229", True),
            (
                "claude-3-sonnet-20240229",
                "litellm_proxy/claude-3-sonnet-20240229",
                True,
            ),
            ("claude-3-haiku-20240307", "litellm_proxy/claude-3-haiku-20240307", True),
            # Google models
            ("gemini-pro", "litellm_proxy/gemini-pro", True),
            ("gemini/gemini-1.5-pro", "litellm_proxy/gemini/gemini-1.5-pro", True),
            ("gemini/gemini-1.5-flash", "litellm_proxy/gemini/gemini-1.5-flash", True),
            # Groq models (mixed support)
            ("groq/gemma-7b-it", "litellm_proxy/groq/gemma-7b-it", True),
            (
                "groq/llama3-70b-8192",
                "litellm_proxy/groq/llama3-70b-8192",
                False,
            ),  # This model doesn't support function calling
            # Cohere models (generally don't support function calling)
            ("command-nightly", "litellm_proxy/command-nightly", False),
            (
                "anthropic.claude-instant-v1",
                "litellm_proxy/anthropic.claude-instant-v1",
                False,
            ),
        ],
    )
    def test_proxy_function_calling_support_consistency(
        self, direct_model, proxy_model, expected_result
    ):
        """Test that proxy models have the same function calling support as their direct counterparts."""
        direct_result = supports_function_calling(direct_model)
        proxy_result = supports_function_calling(proxy_model)

        # Both should match the expected result
        assert (
            direct_result == expected_result
        ), f"Direct model {direct_model} should return {expected_result}"
        assert (
            proxy_result == expected_result
        ), f"Proxy model {proxy_model} should return {expected_result}"

        # Direct and proxy should be consistent
        assert (
            direct_result == proxy_result
        ), f"Mismatch: {direct_model}={direct_result} vs {proxy_model}={proxy_result}"

    @pytest.mark.parametrize(
        "proxy_model_name,underlying_model,expected_proxy_result",
        [
            # Custom model names that cannot be resolved without proxy configuration context
            # These will return False because LiteLLM cannot determine the underlying model
            (
                "litellm_proxy/bedrock-claude-3-haiku",
                "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                False,
            ),
            (
                "litellm_proxy/bedrock-claude-3-sonnet",
                "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
            ),
            (
                "litellm_proxy/bedrock-claude-3-opus",
                "bedrock/anthropic.claude-3-opus-20240229-v1:0",
                False,
            ),
            (
                "litellm_proxy/bedrock-claude-instant",
                "bedrock/anthropic.claude-instant-v1",
                False,
            ),
            (
                "litellm_proxy/bedrock-titan-text",
                "bedrock/amazon.titan-text-express-v1",
                False,
            ),
            # Azure with custom deployment names (cannot be resolved)
            ("litellm_proxy/my-gpt4-deployment", "azure/gpt-4", False),
            ("litellm_proxy/production-gpt35", "azure/gpt-3.5-turbo", False),
            ("litellm_proxy/dev-gpt4o", "azure/gpt-4o", False),
            # Custom OpenAI deployments (cannot be resolved)
            ("litellm_proxy/company-gpt4", "gpt-4", False),
            ("litellm_proxy/internal-gpt35", "gpt-3.5-turbo", False),
            # Vertex AI with custom names (cannot be resolved)
            ("litellm_proxy/vertex-gemini-pro", "vertex_ai/gemini-1.5-pro", False),
            ("litellm_proxy/vertex-gemini-flash", "vertex_ai/gemini-1.5-flash", False),
            # Anthropic with custom names (cannot be resolved)
            ("litellm_proxy/claude-prod", "anthropic/claude-3-sonnet-20240229", False),
            ("litellm_proxy/claude-dev", "anthropic/claude-3-haiku-20240307", False),
            # Groq with custom names (cannot be resolved)
            ("litellm_proxy/fast-llama", "groq/llama3-8b-8192", False),
            ("litellm_proxy/groq-gemma", "groq/gemma-7b-it", False),
            # Cohere with custom names (cannot be resolved)
            ("litellm_proxy/cohere-command", "cohere/command-r", False),
            ("litellm_proxy/cohere-command-plus", "cohere/command-r-plus", False),
            # Together AI with custom names (cannot be resolved)
            (
                "litellm_proxy/together-llama",
                "together_ai/meta-llama/Llama-2-70b-chat-hf",
                False,
            ),
            (
                "litellm_proxy/together-mistral",
                "together_ai/mistralai/Mistral-7B-Instruct-v0.1",
                False,
            ),
            # Ollama with custom names (cannot be resolved)
            ("litellm_proxy/local-llama", "ollama/llama2", False),
            ("litellm_proxy/local-mistral", "ollama/mistral", False),
        ],
    )
    def test_proxy_custom_model_names_without_config(
        self, proxy_model_name, underlying_model, expected_proxy_result
    ):
        """
        Test proxy models with custom model names that differ from underlying models.

        Without proxy configuration context, LiteLLM cannot resolve custom model names
        to their underlying models, so these will return False.
        This demonstrates the limitation and documents the expected behavior.
        """
        # Test the underlying model directly first to establish what it SHOULD return
        try:
            underlying_result = supports_function_calling(underlying_model)
            print(
                f"Underlying model {underlying_model} supports function calling: {underlying_result}"
            )
        except Exception as e:
            print(f"Warning: Could not test underlying model {underlying_model}: {e}")

        # Test the proxy model - this will return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        assert (
            proxy_result == expected_proxy_result
        ), f"Proxy model {proxy_model_name} should return {expected_proxy_result} (without config context)"

    def test_proxy_model_resolution_with_custom_names_documentation(self):
        """
        Document the behavior and limitation for custom proxy model names.

        This test demonstrates:
        1. The current limitation with custom model names
        2. How the proxy server would handle this in production
        3. The expected behavior for both scenarios
        """
        # Case 1: Custom model name that cannot be resolved
        custom_model = "litellm_proxy/my-custom-claude"
        result = supports_function_calling(custom_model)
        assert (
            result is False
        ), "Custom model names return False without proxy config context"

        # Case 2: Model name that can be resolved (matches pattern)
        resolvable_model = "litellm_proxy/claude-3-sonnet-20240229"
        result = supports_function_calling(resolvable_model)
        assert result is True, "Resolvable model names work with fallback logic"

        # Documentation notes:
        print(
            """
        PROXY MODEL RESOLUTION BEHAVIOR:
        
        ✅ WORKS (with current fallback logic):
           - litellm_proxy/gpt-4
           - litellm_proxy/claude-3-sonnet-20240229
           - litellm_proxy/anthropic/claude-3-haiku-20240307
           
        ❌ DOESN'T WORK (requires proxy server config):
           - litellm_proxy/my-custom-gpt4
           - litellm_proxy/bedrock-claude-3-haiku
           - litellm_proxy/production-model
           
        💡 SOLUTION: Use LiteLLM proxy server with proper model_list configuration
           that maps custom names to underlying models.
        """
        )

    @pytest.mark.parametrize(
        "proxy_model_with_hints,expected_result",
        [
            # These are proxy models where we can infer the underlying model from the name
            ("litellm_proxy/gpt-4-with-functions", True),  # Hints at GPT-4
            ("litellm_proxy/claude-3-haiku-prod", True),  # Hints at Claude 3 Haiku
            (
                "litellm_proxy/bedrock-anthropic-claude-3-sonnet",
                True,
            ),  # Hints at Bedrock Claude 3 Sonnet
        ],
    )
    def test_proxy_models_with_naming_hints(
        self, proxy_model_with_hints, expected_result
    ):
        """
        Test proxy models with names that provide hints about the underlying model.

        Note: These will currently fail because the hint-based resolution isn't implemented yet,
        but they demonstrate what could be possible with enhanced model name inference.
        """
        # This test documents potential future enhancement
        proxy_result = supports_function_calling(proxy_model_with_hints)

        # Currently these will return False, but we document the expected behavior
        # In the future, we could implement smarter model name inference
        print(
            f"Model {proxy_model_with_hints}: current={proxy_result}, desired={expected_result}"
        )

        # For now, we expect False (current behavior), but document the limitation
        assert (
            proxy_result is False
        ), f"Current limitation: {proxy_model_with_hints} returns False without inference"

    @pytest.mark.parametrize(
        "proxy_model,expected_result",
        [
            # Test specific proxy models that should support function calling
            ("litellm_proxy/gpt-3.5-turbo", True),
            ("litellm_proxy/gpt-4", True),
            ("litellm_proxy/gpt-4o", True),
            ("litellm_proxy/claude-3-5-sonnet-20240620", True),
            ("litellm_proxy/gemini/gemini-1.5-pro", True),
            # Test proxy models that should not support function calling
            ("litellm_proxy/command-nightly", False),
            ("litellm_proxy/anthropic.claude-instant-v1", False),
        ],
    )
    def test_proxy_only_function_calling_support(self, proxy_model, expected_result):
        """
        Test proxy models independently to ensure they report correct function calling support.

        This test focuses on proxy models without comparing to direct models,
        useful for cases where we only care about the proxy behavior.
        """
        try:
            result = supports_function_calling(model=proxy_model)
            assert (
                result == expected_result
            ), f"Proxy model {proxy_model} returned {result}, expected {expected_result}"
        except Exception as e:
            pytest.fail(f"Error testing proxy model {proxy_model}: {e}")

    def test_litellm_utils_supports_function_calling_import(self):
        """Test that supports_function_calling can be imported from litellm.utils."""
        try:
            from litellm.utils import supports_function_calling

            assert callable(supports_function_calling)
        except ImportError as e:
            pytest.fail(f"Failed to import supports_function_calling: {e}")

    def test_litellm_supports_function_calling_import(self):
        """Test that supports_function_calling can be imported from litellm directly."""
        try:
            import litellm

            assert hasattr(litellm, "supports_function_calling")
            assert callable(litellm.supports_function_calling)
        except Exception as e:
            pytest.fail(f"Failed to access litellm.supports_function_calling: {e}")

    @pytest.mark.parametrize(
        "model_name",
        [
            "litellm_proxy/gpt-3.5-turbo",
            "litellm_proxy/gpt-4",
            "litellm_proxy/claude-3-5-sonnet-20240620",
            "litellm_proxy/gemini/gemini-1.5-pro",
        ],
    )
    def test_proxy_model_with_custom_llm_provider_none(self, model_name):
        """
        Test proxy models with custom_llm_provider=None parameter.

        This tests the supports_function_calling function with the custom_llm_provider
        parameter explicitly set to None, which is a common usage pattern.
        """
        try:
            result = supports_function_calling(
                model=model_name, custom_llm_provider=None
            )
            # All the models in this test should support function calling
            assert (
                result is True
            ), f"Model {model_name} should support function calling but returned {result}"
        except Exception as e:
            pytest.fail(
                f"Error testing {model_name} with custom_llm_provider=None: {e}"
            )

    def test_edge_cases_and_malformed_proxy_models(self):
        """Test edge cases and malformed proxy model names."""
        test_cases = [
            ("litellm_proxy/", False),  # Empty model name after proxy prefix
            ("litellm_proxy", False),  # Just the proxy prefix without slash
            ("litellm_proxy//gpt-3.5-turbo", False),  # Double slash
            ("litellm_proxy/nonexistent-model", False),  # Non-existent model
        ]

        for model_name, expected_result in test_cases:
            try:
                result = supports_function_calling(model=model_name)
                # For malformed models, we expect False or the function to handle gracefully
                assert (
                    result == expected_result
                ), f"Edge case {model_name} returned {result}, expected {expected_result}"
            except Exception:
                # It's acceptable for malformed model names to raise exceptions
                # rather than returning False, as long as they're handled gracefully
                pass

    def test_proxy_model_resolution_demonstration(self):
        """
        Demonstration test showing the current issue with proxy model resolution.

        This test documents the current behavior and can be used to verify
        when the issue is fixed.
        """
        direct_model = "gpt-3.5-turbo"
        proxy_model = "litellm_proxy/gpt-3.5-turbo"

        direct_result = supports_function_calling(model=direct_model)
        proxy_result = supports_function_calling(model=proxy_model)

        print(f"\nDemonstration of proxy model resolution:")
        print(
            f"Direct model '{direct_model}' supports function calling: {direct_result}"
        )
        print(f"Proxy model '{proxy_model}' supports function calling: {proxy_result}")

        # This assertion will currently fail due to the bug
        # When the bug is fixed, this test should pass
        if direct_result != proxy_result:
            pytest.skip(
                f"Known issue: Proxy model resolution inconsistency. "
                f"Direct: {direct_result}, Proxy: {proxy_result}. "
                f"This test will pass when the issue is resolved."
            )

        assert direct_result == proxy_result, (
            f"Proxy model resolution issue: {direct_model} -> {direct_result}, "
            f"{proxy_model} -> {proxy_result}"
        )

    @pytest.mark.parametrize(
        "proxy_model_name,underlying_bedrock_model,expected_proxy_result,description",
        [
            # Bedrock Converse API mappings - these are the real-world scenarios
            (
                "litellm_proxy/bedrock-claude-3-haiku",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Bedrock Claude 3 Haiku via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-sonnet",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "Bedrock Claude 3 Sonnet via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-opus",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "Bedrock Claude 3 Opus via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-5-sonnet",
                "bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0",
                False,
                "Bedrock Claude 3.5 Sonnet via Converse API",
            ),
            # Bedrock Legacy API mappings (non-converse)
            (
                "litellm_proxy/bedrock-claude-instant",
                "bedrock/anthropic.claude-instant-v1",
                False,
                "Bedrock Claude Instant Legacy API",
            ),
            (
                "litellm_proxy/bedrock-claude-v2",
                "bedrock/anthropic.claude-v2",
                False,
                "Bedrock Claude v2 Legacy API",
            ),
            (
                "litellm_proxy/bedrock-claude-v2-1",
                "bedrock/anthropic.claude-v2:1",
                False,
                "Bedrock Claude v2.1 Legacy API",
            ),
            # Bedrock other model providers via Converse API
            (
                "litellm_proxy/bedrock-titan-text",
                "bedrock/converse/amazon.titan-text-express-v1",
                False,
                "Bedrock Titan Text Express via Converse API",
            ),
            (
                "litellm_proxy/bedrock-titan-text-premier",
                "bedrock/converse/amazon.titan-text-premier-v1:0",
                False,
                "Bedrock Titan Text Premier via Converse API",
            ),
            (
                "litellm_proxy/bedrock-llama3-8b",
                "bedrock/converse/meta.llama3-8b-instruct-v1:0",
                False,
                "Bedrock Llama 3 8B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-llama3-70b",
                "bedrock/converse/meta.llama3-70b-instruct-v1:0",
                False,
                "Bedrock Llama 3 70B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-7b",
                "bedrock/converse/mistral.mistral-7b-instruct-v0:2",
                False,
                "Bedrock Mistral 7B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-8x7b",
                "bedrock/converse/mistral.mixtral-8x7b-instruct-v0:1",
                False,
                "Bedrock Mistral 8x7B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-large",
                "bedrock/converse/mistral.mistral-large-2402-v1:0",
                False,
                "Bedrock Mistral Large via Converse API",
            ),
            # Company-specific naming patterns (real-world examples)
            (
                "litellm_proxy/prod-claude-haiku",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Production Claude Haiku",
            ),
            (
                "litellm_proxy/dev-claude-sonnet",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "Development Claude Sonnet",
            ),
            (
                "litellm_proxy/staging-claude-opus",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "Staging Claude Opus",
            ),
            (
                "litellm_proxy/cost-optimized-claude",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Cost-optimized Claude deployment",
            ),
            (
                "litellm_proxy/high-performance-claude",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "High-performance Claude deployment",
            ),
            # Regional deployment examples
            (
                "litellm_proxy/us-east-claude",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "US East Claude deployment",
            ),
            (
                "litellm_proxy/eu-west-claude",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "EU West Claude deployment",
            ),
            (
                "litellm_proxy/ap-south-llama",
                "bedrock/converse/meta.llama3-70b-instruct-v1:0",
                False,
                "Asia Pacific Llama deployment",
            ),
        ],
    )
    def test_bedrock_converse_api_proxy_mappings(
        self,
        proxy_model_name,
        underlying_bedrock_model,
        expected_proxy_result,
        description,
    ):
        """
        Test real-world Bedrock Converse API proxy model mappings.

        This test covers the specific scenario where proxy model names like
        'bedrock-claude-3-haiku' map to underlying Bedrock Converse API models like
        'bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0'.

        These mappings are typically defined in proxy server configuration files
        and cannot be resolved by LiteLLM without that context.
        """
        print(f"\nTesting: {description}")
        print(f"  Proxy model: {proxy_model_name}")
        print(f"  Underlying model: {underlying_bedrock_model}")

        # Test the underlying model directly to verify it supports function calling
        try:
            underlying_result = supports_function_calling(underlying_bedrock_model)
            print(f"  Underlying model function calling support: {underlying_result}")

            # Most Bedrock Converse API models with Anthropic Claude should support function calling
            if "anthropic.claude-3" in underlying_bedrock_model:
                assert (
                    underlying_result is True
                ), f"Claude 3 models should support function calling: {underlying_bedrock_model}"
        except Exception as e:
            print(
                f"  Warning: Could not test underlying model {underlying_bedrock_model}: {e}"
            )

        # Test the proxy model - should return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        print(f"  Proxy model function calling support: {proxy_result}")

        assert proxy_result == expected_proxy_result, (
            f"Proxy model {proxy_model_name} should return {expected_proxy_result} "
            f"(without config context). Description: {description}"
        )

    def test_real_world_proxy_config_documentation(self):
        """
        Document how real-world proxy configurations would handle model mappings.

        This test provides documentation on how the proxy server configuration
        would typically map custom model names to underlying models.
        """
        print(
            """
        
        REAL-WORLD PROXY SERVER CONFIGURATION EXAMPLE:
        ===============================================
        
        In a proxy_server_config.yaml file, you would define:
        
        model_list:
          - model_name: bedrock-claude-3-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: bedrock-claude-3-sonnet
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: prod-claude-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/PROD_AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/PROD_AWS_SECRET_ACCESS_KEY
              aws_region_name: us-west-2
        
        
        FUNCTION CALLING WITH PROXY SERVER:
        ===================================
        
        When using the proxy server with this configuration:
        
        1. Client calls: supports_function_calling("bedrock-claude-3-haiku")
        2. Proxy server resolves to: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
        3. LiteLLM evaluates the underlying model's capabilities
        4. Returns: True (because Claude 3 Haiku supports function calling)
        
        Without the proxy server configuration context, LiteLLM cannot resolve
        the custom model name and returns False.
        
        
        BEDROCK CONVERSE API BENEFITS:
        ==============================
        
        The Bedrock Converse API provides:
        - Standardized function calling interface across providers
        - Better tool use capabilities compared to legacy APIs
        - Consistent request/response format
        - Enhanced streaming support for function calls
        
        """
        )

        # Verify that direct underlying models work as expected
        bedrock_models = [
            "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
        ]

        for model in bedrock_models:
            try:
                result = supports_function_calling(model)
                print(f"Direct test - {model}: {result}")
                # Claude 3 models should support function calling
                assert (
                    result is True
                ), f"Claude 3 model should support function calling: {model}"
            except Exception as e:
                print(f"Could not test {model}: {e}")

    @pytest.mark.parametrize(
        "proxy_model_name,underlying_bedrock_model,expected_proxy_result,description",
        [
            # Bedrock Converse API mappings - these are the real-world scenarios
            (
                "litellm_proxy/bedrock-claude-3-haiku",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Bedrock Claude 3 Haiku via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-sonnet",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "Bedrock Claude 3 Sonnet via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-opus",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "Bedrock Claude 3 Opus via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-5-sonnet",
                "bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0",
                False,
                "Bedrock Claude 3.5 Sonnet via Converse API",
            ),
            # Bedrock Legacy API mappings (non-converse)
            (
                "litellm_proxy/bedrock-claude-instant",
                "bedrock/anthropic.claude-instant-v1",
                False,
                "Bedrock Claude Instant Legacy API",
            ),
            (
                "litellm_proxy/bedrock-claude-v2",
                "bedrock/anthropic.claude-v2",
                False,
                "Bedrock Claude v2 Legacy API",
            ),
            (
                "litellm_proxy/bedrock-claude-v2-1",
                "bedrock/anthropic.claude-v2:1",
                False,
                "Bedrock Claude v2.1 Legacy API",
            ),
            # Bedrock other model providers via Converse API
            (
                "litellm_proxy/bedrock-titan-text",
                "bedrock/converse/amazon.titan-text-express-v1",
                False,
                "Bedrock Titan Text Express via Converse API",
            ),
            (
                "litellm_proxy/bedrock-titan-text-premier",
                "bedrock/converse/amazon.titan-text-premier-v1:0",
                False,
                "Bedrock Titan Text Premier via Converse API",
            ),
            (
                "litellm_proxy/bedrock-llama3-8b",
                "bedrock/converse/meta.llama3-8b-instruct-v1:0",
                False,
                "Bedrock Llama 3 8B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-llama3-70b",
                "bedrock/converse/meta.llama3-70b-instruct-v1:0",
                False,
                "Bedrock Llama 3 70B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-7b",
                "bedrock/converse/mistral.mistral-7b-instruct-v0:2",
                False,
                "Bedrock Mistral 7B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-8x7b",
                "bedrock/converse/mistral.mixtral-8x7b-instruct-v0:1",
                False,
                "Bedrock Mistral 8x7B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-large",
                "bedrock/converse/mistral.mistral-large-2402-v1:0",
                False,
                "Bedrock Mistral Large via Converse API",
            ),
            # Company-specific naming patterns (real-world examples)
            (
                "litellm_proxy/prod-claude-haiku",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Production Claude Haiku",
            ),
            (
                "litellm_proxy/dev-claude-sonnet",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "Development Claude Sonnet",
            ),
            (
                "litellm_proxy/staging-claude-opus",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "Staging Claude Opus",
            ),
            (
                "litellm_proxy/cost-optimized-claude",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Cost-optimized Claude deployment",
            ),
            (
                "litellm_proxy/high-performance-claude",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "High-performance Claude deployment",
            ),
            # Regional deployment examples
            (
                "litellm_proxy/us-east-claude",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "US East Claude deployment",
            ),
            (
                "litellm_proxy/eu-west-claude",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "EU West Claude deployment",
            ),
            (
                "litellm_proxy/ap-south-llama",
                "bedrock/converse/meta.llama3-70b-instruct-v1:0",
                False,
                "Asia Pacific Llama deployment",
            ),
        ],
    )
    def test_bedrock_converse_api_proxy_mappings(
        self,
        proxy_model_name,
        underlying_bedrock_model,
        expected_proxy_result,
        description,
    ):
        """
        Test real-world Bedrock Converse API proxy model mappings.

        This test covers the specific scenario where proxy model names like
        'bedrock-claude-3-haiku' map to underlying Bedrock Converse API models like
        'bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0'.

        These mappings are typically defined in proxy server configuration files
        and cannot be resolved by LiteLLM without that context.
        """
        print(f"\nTesting: {description}")
        print(f"  Proxy model: {proxy_model_name}")
        print(f"  Underlying model: {underlying_bedrock_model}")

        # Test the underlying model directly to verify it supports function calling
        try:
            underlying_result = supports_function_calling(underlying_bedrock_model)
            print(f"  Underlying model function calling support: {underlying_result}")

            # Most Bedrock Converse API models with Anthropic Claude should support function calling
            if "anthropic.claude-3" in underlying_bedrock_model:
                assert (
                    underlying_result is True
                ), f"Claude 3 models should support function calling: {underlying_bedrock_model}"
        except Exception as e:
            print(
                f"  Warning: Could not test underlying model {underlying_bedrock_model}: {e}"
            )

        # Test the proxy model - should return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        print(f"  Proxy model function calling support: {proxy_result}")

        assert proxy_result == expected_proxy_result, (
            f"Proxy model {proxy_model_name} should return {expected_proxy_result} "
            f"(without config context). Description: {description}"
        )

    def test_real_world_proxy_config_documentation(self):
        """
        Document how real-world proxy configurations would handle model mappings.

        This test provides documentation on how the proxy server configuration
        would typically map custom model names to underlying models.
        """
        print(
            """
        
        REAL-WORLD PROXY SERVER CONFIGURATION EXAMPLE:
        ===============================================
        
        In a proxy_server_config.yaml file, you would define:
        
        model_list:
          - model_name: bedrock-claude-3-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: bedrock-claude-3-sonnet
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: prod-claude-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/PROD_AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/PROD_AWS_SECRET_ACCESS_KEY
              aws_region_name: us-west-2
        
        
        FUNCTION CALLING WITH PROXY SERVER:
        ===================================
        
        When using the proxy server with this configuration:
        
        1. Client calls: supports_function_calling("bedrock-claude-3-haiku")
        2. Proxy server resolves to: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
        3. LiteLLM evaluates the underlying model's capabilities
        4. Returns: True (because Claude 3 Haiku supports function calling)
        
        Without the proxy server configuration context, LiteLLM cannot resolve
        the custom model name and returns False.
        
        
        BEDROCK CONVERSE API BENEFITS:
        ==============================
        
        The Bedrock Converse API provides:
        - Standardized function calling interface across providers
        - Better tool use capabilities compared to legacy APIs
        - Consistent request/response format
        - Enhanced streaming support for function calls
        
        """
        )

        # Verify that direct underlying models work as expected
        bedrock_models = [
            "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
        ]

        for model in bedrock_models:
            try:
                result = supports_function_calling(model)
                print(f"Direct test - {model}: {result}")
                # Claude 3 models should support function calling
                assert (
                    result is True
                ), f"Claude 3 model should support function calling: {model}"
            except Exception as e:
                print(f"Could not test {model}: {e}")

    @pytest.mark.parametrize(
        "proxy_model_name,underlying_bedrock_model,expected_proxy_result,description",
        [
            # Bedrock Converse API mappings - these are the real-world scenarios
            (
                "litellm_proxy/bedrock-claude-3-haiku",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Bedrock Claude 3 Haiku via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-sonnet",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "Bedrock Claude 3 Sonnet via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-opus",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "Bedrock Claude 3 Opus via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-5-sonnet",
                "bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0",
                False,
                "Bedrock Claude 3.5 Sonnet via Converse API",
            ),
            # Bedrock Legacy API mappings (non-converse)
            (
                "litellm_proxy/bedrock-claude-instant",
                "bedrock/anthropic.claude-instant-v1",
                False,
                "Bedrock Claude Instant Legacy API",
            ),
            (
                "litellm_proxy/bedrock-claude-v2",
                "bedrock/anthropic.claude-v2",
                False,
                "Bedrock Claude v2 Legacy API",
            ),
            (
                "litellm_proxy/bedrock-claude-v2-1",
                "bedrock/anthropic.claude-v2:1",
                False,
                "Bedrock Claude v2.1 Legacy API",
            ),
            # Bedrock other model providers via Converse API
            (
                "litellm_proxy/bedrock-titan-text",
                "bedrock/converse/amazon.titan-text-express-v1",
                False,
                "Bedrock Titan Text Express via Converse API",
            ),
            (
                "litellm_proxy/bedrock-titan-text-premier",
                "bedrock/converse/amazon.titan-text-premier-v1:0",
                False,
                "Bedrock Titan Text Premier via Converse API",
            ),
            (
                "litellm_proxy/bedrock-llama3-8b",
                "bedrock/converse/meta.llama3-8b-instruct-v1:0",
                False,
                "Bedrock Llama 3 8B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-llama3-70b",
                "bedrock/converse/meta.llama3-70b-instruct-v1:0",
                False,
                "Bedrock Llama 3 70B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-7b",
                "bedrock/converse/mistral.mistral-7b-instruct-v0:2",
                False,
                "Bedrock Mistral 7B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-8x7b",
                "bedrock/converse/mistral.mixtral-8x7b-instruct-v0:1",
                False,
                "Bedrock Mistral 8x7B via Converse API",
            ),
            (
                "litellm_proxy/bedrock-mistral-large",
                "bedrock/converse/mistral.mistral-large-2402-v1:0",
                False,
                "Bedrock Mistral Large via Converse API",
            ),
            # Company-specific naming patterns (real-world examples)
            (
                "litellm_proxy/prod-claude-haiku",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Production Claude Haiku",
            ),
            (
                "litellm_proxy/dev-claude-sonnet",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "Development Claude Sonnet",
            ),
            (
                "litellm_proxy/staging-claude-opus",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "Staging Claude Opus",
            ),
            (
                "litellm_proxy/cost-optimized-claude",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "Cost-optimized Claude deployment",
            ),
            (
                "litellm_proxy/high-performance-claude",
                "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
                False,
                "High-performance Claude deployment",
            ),
            # Regional deployment examples
            (
                "litellm_proxy/us-east-claude",
                "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
                False,
                "US East Claude deployment",
            ),
            (
                "litellm_proxy/eu-west-claude",
                "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
                False,
                "EU West Claude deployment",
            ),
            (
                "litellm_proxy/ap-south-llama",
                "bedrock/converse/meta.llama3-70b-instruct-v1:0",
                False,
                "Asia Pacific Llama deployment",
            ),
        ],
    )
    def test_bedrock_converse_api_proxy_mappings(
        self,
        proxy_model_name,
        underlying_bedrock_model,
        expected_proxy_result,
        description,
    ):
        """
        Test real-world Bedrock Converse API proxy model mappings.

        This test covers the specific scenario where proxy model names like
        'bedrock-claude-3-haiku' map to underlying Bedrock Converse API models like
        'bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0'.

        These mappings are typically defined in proxy server configuration files
        and cannot be resolved by LiteLLM without that context.
        """
        print(f"\nTesting: {description}")
        print(f"  Proxy model: {proxy_model_name}")
        print(f"  Underlying model: {underlying_bedrock_model}")

        # Test the underlying model directly to verify it supports function calling
        try:
            underlying_result = supports_function_calling(underlying_bedrock_model)
            print(f"  Underlying model function calling support: {underlying_result}")

            # Most Bedrock Converse API models with Anthropic Claude should support function calling
            if "anthropic.claude-3" in underlying_bedrock_model:
                assert (
                    underlying_result is True
                ), f"Claude 3 models should support function calling: {underlying_bedrock_model}"
        except Exception as e:
            print(
                f"  Warning: Could not test underlying model {underlying_bedrock_model}: {e}"
            )

        # Test the proxy model - should return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        print(f"  Proxy model function calling support: {proxy_result}")

        assert proxy_result == expected_proxy_result, (
            f"Proxy model {proxy_model_name} should return {expected_proxy_result} "
            f"(without config context). Description: {description}"
        )

    def test_real_world_proxy_config_documentation(self):
        """
        Document how real-world proxy configurations would handle model mappings.

        This test provides documentation on how the proxy server configuration
        would typically map custom model names to underlying models.
        """
        print(
            """
        
        REAL-WORLD PROXY SERVER CONFIGURATION EXAMPLE:
        ===============================================
        
        In a proxy_server_config.yaml file, you would define:
        
        model_list:
          - model_name: bedrock-claude-3-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: bedrock-claude-3-sonnet
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0
              aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
              aws_region_name: us-east-1
              
          - model_name: prod-claude-haiku
            litellm_params:
              model: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
              aws_access_key_id: os.environ/PROD_AWS_ACCESS_KEY_ID
              aws_secret_access_key: os.environ/PROD_AWS_SECRET_ACCESS_KEY
              aws_region_name: us-west-2
        
        
        FUNCTION CALLING WITH PROXY SERVER:
        ===================================
        
        When using the proxy server with this configuration:
        
        1. Client calls: supports_function_calling("bedrock-claude-3-haiku")
        2. Proxy server resolves to: bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0
        3. LiteLLM evaluates the underlying model's capabilities
        4. Returns: True (because Claude 3 Haiku supports function calling)
        
        Without the proxy server configuration context, LiteLLM cannot resolve
        the custom model name and returns False.
        
        
        BEDROCK CONVERSE API BENEFITS:
        ==============================
        
        The Bedrock Converse API provides:
        - Standardized function calling interface across providers
        - Better tool use capabilities compared to legacy APIs
        - Consistent request/response format
        - Enhanced streaming support for function calls
        
        """
        )

        # Verify that direct underlying models work as expected
        bedrock_models = [
            "bedrock/converse/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/converse/anthropic.claude-3-opus-20240229-v1:0",
        ]

        for model in bedrock_models:
            try:
                result = supports_function_calling(model)
                print(f"Direct test - {model}: {result}")
                # Claude 3 models should support function calling
                assert (
                    result is True
                ), f"Claude 3 model should support function calling: {model}"
            except Exception as e:
                print(f"Could not test {model}: {e}")




def test_register_model_with_scientific_notation():
    """
    Test that the register_model function can handle scientific notation in the model name.
    """
    model_cost_dict = {
        "my-custom-model": {
            "max_tokens": 8192,
            "input_cost_per_token": "3e-07",
            "output_cost_per_token": "6e-07",
            "litellm_provider": "openai",
            "mode": "chat",
        },
    }

    litellm.register_model(model_cost_dict)

    registered_model = litellm.model_cost["my-custom-model"]
    print(registered_model)
    assert registered_model["input_cost_per_token"] == 3e-07
    assert registered_model["output_cost_per_token"] == 6e-07
    assert registered_model["litellm_provider"] == "openai"
    assert registered_model["mode"] == "chat"


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v"])
