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
from litellm.utils import get_optional_params_image_gen
from litellm import UnsupportedParamsError
from litellm.utils import (
    _get_optional_params_defaults,
    _get_optional_params_non_default_params,
    get_optional_params,
)

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

class TestGetOptionalParamsDefaults:
    def test_returns_dictionary(self):
        """Test that the function returns a dictionary."""
        result = _get_optional_params_defaults()
        assert isinstance(result, dict)

    def test_return_value_is_not_mutated(self):
        """Test that subsequent calls return independent copies of the default dictionary."""
        first_call = _get_optional_params_defaults()
        second_call = _get_optional_params_defaults()

        # Verify they're equal but not the same object
        assert first_call == second_call
        assert first_call is not second_call

        # Modify the first result and verify the second isn't affected
        first_call["temperature"] = 0.7
        assert second_call["temperature"] is None

    @pytest.mark.parametrize(
        "param_name, expected_value",
        [
            ("additional_drop_params", None),
            ("allowed_openai_params", None),
            ("api_version", None),
            ("audio", None),
            ("custom_llm_provider", ""),
            ("drop_params", None),
            ("extra_headers", None),
            ("frequency_penalty", None),
            ("function_call", None),
            ("functions", None),
            ("logit_bias", None),
            ("logprobs", None),
            ("max_completion_tokens", None),
            ("max_retries", None),
            ("max_tokens", None),
            ("messages", None),
            ("modalities", None),
            ("model", None),
            ("n", None),
            ("parallel_tool_calls", None),
            ("prediction", None),
            ("presence_penalty", None),
            ("reasoning_effort", None),
            ("response_format", None),
            ("seed", None),
            ("stop", None),
            ("stream", False),
            ("stream_options", None),
            ("temperature", None),
            ("thinking", None),
            ("tool_choice", None),
            ("tools", None),
            ("top_logprobs", None),
            ("top_p", None),
            ("user", None),
        ],
    )
    def test_individual_defaults(self, param_name, expected_value):
        """Test that each parameter has the expected default value."""
        defaults = _get_optional_params_defaults()
        assert param_name in defaults
        assert defaults[param_name] == expected_value

    def test_completeness(self):
        """Test that the function returns all expected parameters with no extras or missing items."""
        expected_params = {
            "additional_drop_params",
            "allowed_openai_params",
            "api_version",
            "audio",
            "custom_llm_provider",
            "drop_params",
            "extra_headers",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_retries",
            "max_tokens",
            "messages",
            "modalities",
            "model",
            "n",
            "parallel_tool_calls",
            "prediction",
            "presence_penalty",
            "reasoning_effort",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "thinking",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
        }

        actual_params = set(_get_optional_params_defaults().keys())

        # Check for extra parameters
        extra_params = actual_params - expected_params
        assert not extra_params, f"Unexpected parameters found: {extra_params}"

        # Check for missing parameters
        missing_params = expected_params - actual_params
        assert not missing_params, f"Expected parameters missing: {missing_params}"

    def test_custom_llm_provider_is_empty_string(self):
        """Specifically test that custom_llm_provider has empty string as default (not None)."""
        defaults = _get_optional_params_defaults()
        assert defaults["custom_llm_provider"] == ""
        assert defaults["custom_llm_provider"] is not None

    def test_stream_is_false(self):
        """Specifically test that stream has False as default (not None)."""
        defaults = _get_optional_params_defaults()
        assert not defaults["stream"]

    def test_all_others_are_none(self):
        """Test that all parameters except custom_llm_provider have None as default.

        This test may change in the future or no longer be required, but is included for now.
        """
        defaults = _get_optional_params_defaults()
        for key, value in defaults.items():
            if key in ["custom_llm_provider", "stream"]:
                continue
            assert value is None, f"Expected {key} to be None, but got {value}"


class TestGetOptionalParamsNonDefaultParams:
    @pytest.mark.parametrize(
        "passed_params, default_params, additional_drop_params, expected",
        [
            # no non-defaults, should return empty
            (
                {"model": "gpt-4", "api_version": "v1"},
                _get_optional_params_defaults(),
                None,
                {},
            ),
            # one non-default parameter not excluded
            (
                {
                    "temperature": 0.9,
                    "additional_drop_params": None,
                    "allowed_openai_params": "test",
                    "api_version": "v1",
                    "custom_llm_provider": "llamafile",
                    "drop_params": ["foo"],
                    "messages": ["bar"],
                    "model": "gpt-4",
                },
                _get_optional_params_defaults(),
                None,
                {"temperature": 0.9},
            ),
            # specifically exclude (drop) a parameter that is not default
            (
                {
                    "temperature": 0.9,
                    "additional_drop_params": None,
                    "allowed_openai_params": "test",
                    "api_version": "v1",
                    "custom_llm_provider": "llamafile",
                    "drop_params": ["foo"],
                    "messages": ["bar"],
                    "model": "gpt-4",
                },
                _get_optional_params_defaults(),
                ["temperature"],
                {},
            ),
            # non-default param dropped, not default param left alone
            (
                {"temperature": 0.9, "top_p": 0.95},
                _get_optional_params_defaults(),
                ["top_p"],
                {"temperature": 0.9},
            ),
        ],
    )
    def test_get_optional_params_non_default_params(
        self, passed_params, default_params, additional_drop_params, expected
    ):
        result = _get_optional_params_non_default_params(
            passed_params,
            default_params,
            additional_drop_params=additional_drop_params,
        )
        assert result == expected


class TestGetOptionalParms:
    def test_raises_on_unsupported_function_calling(self):
        original_flag = litellm.add_function_to_prompt

        try:
            litellm.add_function_to_prompt = False

            with pytest.raises(
                UnsupportedParamsError,
                match=r"^litellm.UnsupportedParamsError: Function calling is not supported by bad_provider.",
            ):
                get_optional_params(
                    model="qwerty",
                    custom_llm_provider="bad_provider",
                    functions="not_supported",
                )
        finally:
            litellm.add_function_to_prompt = original_flag

    def test_ollama_sets_json_and_removes_tool_choice(self):
        original_flag = litellm.add_function_to_prompt

        try:
            optional_params = get_optional_params(
                model="qwerty",
                custom_llm_provider="ollama",
                functions="my_function",
                tool_choice="auto",
            )

            assert optional_params["format"] == "json"
            assert litellm.add_function_to_prompt
            assert optional_params["functions_unsupported_model"] == "my_function"
        finally:
            litellm.add_function_to_prompt = original_flag

    @pytest.mark.parametrize(
        "tools, functions, function_call, expected_value",
        [
            ("foo", None, None, "foo"),
            (None, None, "baz", "baz"),
            ("foo", "bar", None, "foo"),
            ("foo", None, "baz", "foo"),
            (None, "bar", "baz", "bar"),
            ("foo", "bar", "baz", "foo"),
        ],
    )
    def test_supplying_tools_funcs_calls(
        self, tools, functions, function_call, expected_value
    ):
        original_flag = litellm.add_function_to_prompt
        try:
            optional_params = get_optional_params(
                model="qwerty",
                custom_llm_provider="ollama",
                tools=tools,
                functions=functions,
                function_call=function_call,
            )

            assert optional_params["format"] == "json"
            assert litellm.add_function_to_prompt
            assert optional_params["functions_unsupported_model"] == expected_value
        finally:
            litellm.add_function_to_prompt = original_flag
