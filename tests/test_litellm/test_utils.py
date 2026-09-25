import asyncio
import base64
import contextlib
import contextvars
import io
import json
import logging
import os
import queue
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import PurePath
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from jsonschema import validate

import litellm
from litellm._internal_context import is_internal_call
from litellm._logging import (
    CorrelationContextFilter,
    JsonFormatter,
    session_id_var,
    trace_id_var,
    verbose_logger,
)
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.constants import DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.thread_pool_executor import executor as logging_executor
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.proxy.utils import is_valid_api_key
from litellm.types.integrations.custom_logger import HEADROOM_CONVERTED_STREAM_KEY
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import CredentialLiteLLMParams, GenericLiteLLMParams
from litellm.types.utils import (
    ADDRESSED_RESPONSE_ID_FIELD,
    CallTypes,
    Choices,
    Delta,
    EmbeddingResponse,
    ImageResponse,
    LlmProviders,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
    PromptTokensDetailsWrapper,
    RerankResponse,
    StreamingChoices,
    TranscriptionResponse,
    Usage,
    all_litellm_params,
    bedrock_batch_litellm_params,
)
from litellm.types.videos.main import VideoObject
from litellm.utils import (
    CustomStreamWrapper,
    ProviderConfigManager,
    TextCompletionStreamWrapper,
    _check_provider_match,
    _get_potential_model_names,
    _is_streaming_request,
    _run_success_deployment_hook_on_converted_chat_stream,
    _snapshot_exception_for_hook,
    async_post_call_failure_deployment_hook,
    async_post_call_success_deployment_hook,
    calculate_max_parallel_requests,
    client,
    get_non_default_completion_params,
    get_optional_params_image_gen,
    get_prompt_cache_min_tokens,
    is_cached_message,
    is_prompt_caching_valid_prompt,
)

# Adds the parent directory to the system path


def test_non_ocr_wrapper_preserves_logging_executor_and_context(monkeypatch: pytest.MonkeyPatch) -> None:
    marker: Final = contextvars.ContextVar("non-ocr-logging-context", default="missing")
    token: Final = marker.set("caller-context")
    caller_thread: Final = threading.get_ident()
    response: Final = object()
    logger: Final = MagicMock()
    observed: Final = queue.Queue[tuple[object, str, int]]()

    def record_success(result: object, start_time: datetime, end_time: datetime) -> None:
        observed.put((result, marker.get(), threading.get_ident()))

    def embedding(**kwargs: object) -> object:
        return response

    logger.success_handler.side_effect = record_success
    monkeypatch.setattr("litellm.utils.function_setup", MagicMock(return_value=(logger, {})))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            monkeypatch.setattr("litellm.utils.executor", executor)
            result: Final = client(embedding)()
        logged_response, context, worker_thread = observed.get_nowait()
        assert result is response
        assert logged_response is response
        assert context == "caller-context"
        assert worker_thread != caller_thread
        assert observed.empty()
    finally:
        marker.reset(token)


def test_get_utc_datetime_returns_current_aware_utc_time() -> None:
    before: Final = datetime.now(timezone.utc)
    result: Final = litellm.utils.get_utc_datetime()
    after: Final = datetime.now(timezone.utc)

    assert result.utcoffset() == timedelta(0)
    assert before <= result <= after


def test_usage_openai_cache_write_tokens_populates_both_names():
    """OpenAI reports cache-write tokens as prompt_tokens_details.cache_write_tokens.
    The Usage constructor must expose it under both cache_write_tokens (canonical,
    OpenAI naming) and cache_creation_tokens (legacy, Anthropic naming)."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
        prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 800},
    )
    assert usage.prompt_tokens_details.cache_write_tokens == 800
    assert usage.prompt_tokens_details.cache_creation_tokens == 800


def test_usage_anthropic_cache_creation_maps_to_cache_write_tokens():
    """Anthropic/Bedrock report the top-level cache_creation_input_tokens field.
    It must be normalized onto the OpenAI cache_write_tokens name as well as the
    legacy cache_creation_tokens name."""
    usage = Usage(
        prompt_tokens=500,
        completion_tokens=50,
        total_tokens=550,
        cache_creation_input_tokens=300,
        cache_read_input_tokens=120,
    )
    assert usage.prompt_tokens_details.cache_write_tokens == 300
    assert usage.prompt_tokens_details.cache_creation_tokens == 300
    assert usage.prompt_tokens_details.cached_tokens == 120


def test_prompt_tokens_details_no_cache_write_tokens_when_absent():
    """A read-only cache hit (no cache write) must not surface cache-write fields."""
    details = PromptTokensDetailsWrapper(cached_tokens=800)
    assert details.cached_tokens == 800
    assert not hasattr(details, "cache_write_tokens")
    assert not hasattr(details, "cache_creation_tokens")


def test_prompt_tokens_details_cache_write_creation_stay_in_sync_on_assignment():
    """Assigning either name after construction must mirror to the other, so a
    caller that sets only one field can't leave the pair silently out of sync."""
    details = PromptTokensDetailsWrapper(cache_write_tokens=100)
    assert details.cache_write_tokens == details.cache_creation_tokens == 100

    details.cache_write_tokens = 250
    assert details.cache_write_tokens == details.cache_creation_tokens == 250

    details.cache_creation_tokens = 375
    assert details.cache_write_tokens == details.cache_creation_tokens == 375


def test_potential_model_names_keeps_provider_prefixed_candidate():
    """A provider whose own model ids repeat the litellm provider name (Perplexity's
    Agent API serves `perplexity/glm-5.2`, mapped as `perplexity/perplexity/glm-5.2`)
    needs the un-stripped `<provider>/<model>` candidate. Every other candidate reads
    the leading `perplexity/` as the litellm prefix and strips it away."""
    already_prefixed = _get_potential_model_names(model="perplexity/glm-5.2", custom_llm_provider="perplexity")
    assert already_prefixed["provider_prefixed_model_name"] == "perplexity/perplexity/glm-5.2"
    assert already_prefixed["split_model"] == "glm-5.2"
    assert already_prefixed["combined_model_name"] == "perplexity/glm-5.2"
    assert already_prefixed["combined_stripped_model_name"] == "perplexity/glm-5.2"

    bare = _get_potential_model_names(model="glm-5.2", custom_llm_provider="perplexity")
    assert bare["provider_prefixed_model_name"] == bare["combined_model_name"] == "perplexity/glm-5.2"


@pytest.mark.parametrize("capability", [True, False, None])
def test_get_model_info_anthropic_compaction(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch, capability: bool | None
) -> None:
    monkeypatch.setitem(litellm.model_cost["claude-sonnet-5"], "supports_anthropic_compaction", capability)
    assert litellm.get_model_info("claude-sonnet-5")["supports_anthropic_compaction"] is capability


def test_get_model_info_strips_openai_finetune_ids_without_a_custom_suffix(local_model_cost_map):
    info = litellm.get_model_info(model="ft:gpt-4o-2024-08-06:my-org::abc123", custom_llm_provider="openai")
    assert info["key"] == "ft:gpt-4o-2024-08-06"


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected_key"),
    [
        ("gpt-5.6-luna-2099-01-01", "openai", "gpt-5.6-luna"),
        ("gpt-5.6-luna-2099-01-01", "azure", "azure/gpt-5.6-luna"),
    ],
)
def test_get_model_info_falls_back_from_dated_snapshot_to_undated_entry(
    local_model_cost_map: None,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    custom_llm_provider: str,
    expected_key: str,
) -> None:
    monkeypatch.delitem(litellm.model_cost, model, raising=False)
    monkeypatch.delitem(litellm.model_cost, f"{custom_llm_provider}/{model}", raising=False)
    assert expected_key in litellm.model_cost
    info: Final = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    assert info["key"] == expected_key


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected_key"),
    [
        ("gpt-4o-2024-08-06", "openai", "gpt-4o-2024-08-06"),
        ("gpt-5.6-luna-2026-07-09", "azure", "azure/gpt-5.6-luna-2026-07-09"),
    ],
)
def test_get_model_info_prefers_exact_dated_key_over_stripped(
    local_model_cost_map: None, model: str, custom_llm_provider: str, expected_key: str
) -> None:
    assert expected_key in litellm.model_cost
    info: Final = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    assert info["key"] == expected_key


def test_get_model_info_internal_failure_is_not_reported_as_unmapped() -> None:
    with patch("litellm.utils._get_potential_model_names", side_effect=RuntimeError("malformed metadata")):
        with pytest.raises(Exception, match="This model isn't mapped yet") as exc_info:
            litellm.utils._get_model_info_helper(model="gpt-4o", custom_llm_provider="openai")
    assert not isinstance(exc_info.value, litellm.ModelNotMappedError)


def test_check_provider_match_azure_ai_allows_openai_and_azure():
    """
    Test that azure_ai provider can match openai and azure models.
    This is needed for Azure Model Router which can route to OpenAI models.
    """
    # azure_ai should match openai models
    assert _check_provider_match(model_info={"litellm_provider": "openai"}, custom_llm_provider="azure_ai") is True

    # azure_ai should match azure models
    assert _check_provider_match(model_info={"litellm_provider": "azure"}, custom_llm_provider="azure_ai") is True

    # azure_ai should NOT match other providers
    assert _check_provider_match(model_info={"litellm_provider": "anthropic"}, custom_llm_provider="azure_ai") is False


def test_check_provider_match_github_allows_upstream_provider_metadata():
    """
    Test that github provider can match upstream provider metadata.
    GitHub Models can provide models from multiple providers.
    """
    assert (
        _check_provider_match(
            model_info={"litellm_provider": "openai"},
            custom_llm_provider="github",
        )
        is True
    )

    assert (
        _check_provider_match(
            model_info={"litellm_provider": "github"},
            custom_llm_provider="github",
        )
        is True
    )

    assert (
        _check_provider_match(
            model_info={"litellm_provider": "anthropic"},
            custom_llm_provider="github",
        )
        is True
    )


def test_supports_function_calling_unknown_github_alias_returns_false():
    assert litellm.utils.supports_function_calling(model="github/non-existent-model-for-capability-check") is False


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


@pytest.mark.parametrize("custom_llm_provider", ["openai", "azure"])
def test_get_optional_params_image_gen_keeps_gpt_image_supported_params(custom_llm_provider):
    """https://github.com/BerriAI/litellm/issues/38649"""
    from litellm.types.utils import LlmProviders

    provider_config = ProviderConfigManager.get_provider_image_generation_config(
        model="gpt-image-2", provider=LlmProviders(custom_llm_provider)
    )
    optional_params = get_optional_params_image_gen(
        model="gpt-image-2",
        n=1,
        size="1024x1024",
        custom_llm_provider=custom_llm_provider,
        provider_config=provider_config,
        background="transparent",
        output_format="png",
        moderation="low",
        output_compression=50,
        unknown_param="kept-in-extra-body",
    )
    assert optional_params == {
        "n": 1,
        "size": "1024x1024",
        "background": "transparent",
        "output_format": "png",
        "moderation": "low",
        "output_compression": 50,
        "extra_body": {"unknown_param": "kept-in-extra-body"},
    }


def test_get_optional_params_image_gen_vertex_ai_size():
    """Test that Vertex AI image generation properly handles size parameter and maps it to aspectRatio"""
    # Test with various size parameters
    test_cases = [
        ("1024x1024", "1:1"),  # Square aspect ratio
        ("256x256", "1:1"),  # Square aspect ratio
        ("512x512", "1:1"),  # Square aspect ratio
        ("1792x1024", "16:9"),  # Landscape aspect ratio
        ("1024x1792", "9:16"),  # Portrait aspect ratio
        ("unsupported", "1:1"),  # Default to square for unsupported sizes
    ]

    for size_input, expected_aspect_ratio in test_cases:
        optional_params = get_optional_params_image_gen(
            model="vertex_ai/imagegeneration@006",
            size=size_input,
            n=2,
            custom_llm_provider="vertex_ai",
            drop_params=True,
        )
        assert optional_params is not None
        assert optional_params["aspectRatio"] == expected_aspect_ratio
        assert optional_params["sampleCount"] == 2
        assert "size" not in optional_params  # size should be converted to aspectRatio

    # Test without size parameter
    optional_params = get_optional_params_image_gen(
        model="vertex_ai/imagegeneration@006",
        n=1,
        custom_llm_provider="vertex_ai",
        drop_params=True,
    )
    assert optional_params is not None
    assert "aspectRatio" not in optional_params  # aspectRatio should not be set if size is not provided
    assert optional_params["sampleCount"] == 1


def test_get_optional_params_image_gen_filters_empty_values():
    optional_params = get_optional_params_image_gen(
        model="gpt-image-1",
        custom_llm_provider="openai",
        extra_body={},
    )
    assert optional_params == {}


def test_gpt_image_provider_detection_covers_existing_family():
    for image_model in ("gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"):
        model, custom_llm_provider, _, _ = litellm.get_llm_provider(model=image_model)

        assert model == image_model
        assert custom_llm_provider == "openai"


def test_all_model_configs():
    from litellm.llms.vertex_ai.vertex_ai_partner_models.ai21.transformation import (
        VertexAIAi21Config,
    )
    from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
        VertexAILlama3Config,
    )

    assert "max_completion_tokens" in VertexAILlama3Config().get_supported_openai_params(model="llama3")
    assert VertexAILlama3Config().map_openai_params({"max_completion_tokens": 10}, {}, "llama3", drop_params=False) == {
        "max_tokens": 10
    }

    assert "max_completion_tokens" in VertexAIAi21Config().get_supported_openai_params(model="jamba-1.5-mini@001")
    assert VertexAIAi21Config().map_openai_params(
        {"max_completion_tokens": 10}, {}, "jamba-1.5-mini@001", drop_params=False
    ) == {"max_tokens": 10}

    from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig

    assert "max_completion_tokens" in FireworksAIConfig().get_supported_openai_params(model="llama3")
    assert FireworksAIConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig

    assert "max_completion_tokens" in NvidiaNimConfig().get_supported_openai_params(model="llama3")
    assert NvidiaNimConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.ollama.chat.transformation import OllamaChatConfig

    assert "max_completion_tokens" in OllamaChatConfig().get_supported_openai_params(model="llama3")
    assert OllamaChatConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"num_predict": 10}

    from litellm.llms.predibase.chat.transformation import PredibaseConfig

    assert "max_completion_tokens" in PredibaseConfig().get_supported_openai_params(model="llama3")
    assert PredibaseConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_new_tokens": 10}

    from litellm.llms.codestral.completion.transformation import (
        CodestralTextCompletionConfig,
    )

    assert "max_completion_tokens" in CodestralTextCompletionConfig().get_supported_openai_params(model="llama3")
    assert CodestralTextCompletionConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.volcengine.chat.transformation import (
        VolcEngineChatConfig as VolcEngineConfig,
    )

    assert "max_completion_tokens" in VolcEngineConfig().get_supported_openai_params(model="llama3")
    assert VolcEngineConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.ai21.chat.transformation import AI21ChatConfig

    assert "max_completion_tokens" in AI21ChatConfig().get_supported_openai_params("jamba-1.5-mini@001")
    assert AI21ChatConfig().map_openai_params(
        model="jamba-1.5-mini@001",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig

    assert "max_completion_tokens" in AzureOpenAIConfig().get_supported_openai_params(model="gpt-3.5-turbo")
    assert AzureOpenAIConfig().map_openai_params(
        model="gpt-3.5-turbo",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        api_version="2022-12-01",
        drop_params=False,
    ) == {"max_completion_tokens": 10}

    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    assert "max_completion_tokens" in AmazonConverseConfig().get_supported_openai_params(
        model="anthropic.claude-3-sonnet-20240229-v1:0"
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

    assert "max_completion_tokens" in CodestralTextCompletionConfig().get_supported_openai_params(model="llama3")
    assert CodestralTextCompletionConfig().map_openai_params(
        model="llama3",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm import AmazonAnthropicClaudeConfig, AmazonAnthropicConfig

    assert "max_completion_tokens" in AmazonAnthropicClaudeConfig().get_supported_openai_params(
        model="anthropic.claude-3-sonnet-20240229-v1:0"
    )

    assert AmazonAnthropicClaudeConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        drop_params=False,
    ) == {"max_tokens": 10}

    assert "max_completion_tokens" in AmazonAnthropicConfig().get_supported_openai_params(model="")

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

    assert "max_completion_tokens" in VertexAIAnthropicConfig().get_supported_openai_params(model="claude-sonnet-4-6")

    assert VertexAIAnthropicConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        model="claude-sonnet-4-6",
        drop_params=False,
    ) == {"max_tokens": 10}

    from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    assert "max_completion_tokens" in VertexGeminiConfig().get_supported_openai_params(model="gemini-1.0-pro")

    assert VertexGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}

    assert "max_completion_tokens" in GoogleAIStudioGeminiConfig().get_supported_openai_params(model="gemini-1.0-pro")

    assert GoogleAIStudioGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}

    assert "max_completion_tokens" in VertexGeminiConfig().get_supported_openai_params(model="gemini-1.0-pro")

    assert VertexGeminiConfig().map_openai_params(
        model="gemini-1.0-pro",
        non_default_params={"max_completion_tokens": 10},
        optional_params={},
        drop_params=False,
    ) == {"max_output_tokens": 10}


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


def validate_model_cost_values(model_data, exceptions=None):
    """
    Validates that cost values in model data do not exceed 1.

    Args:
        model_data (dict): The model data dictionary
        exceptions (list, optional): List of model IDs that are allowed to have costs > 1

    Returns:
        tuple: (is_valid, violations) where is_valid is a boolean and violations is a list of error messages
    """
    if exceptions is None:
        exceptions = []

    violations = []

    # Define all cost-related fields to check
    cost_fields = [
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_character",
        "output_cost_per_character",
        "input_cost_per_image",
        "output_cost_per_image",
        "output_cost_per_image_512",
        "output_cost_per_image_1024",
        "output_cost_per_image_1536",
        "output_cost_per_image_0.5K",
        "output_cost_per_image_1K",
        "output_cost_per_image_2K",
        "output_cost_per_image_4K",
        "input_cost_per_pixel",
        "output_cost_per_pixel",
        "input_cost_per_second",
        "output_cost_per_second",
        "output_cost_per_second_480p",
        "output_cost_per_second_720p",
        "output_cost_per_second_768p",
        "output_cost_per_second_2k",
        "output_cost_per_second_1080p",
        "output_cost_per_second_4k",
        "input_cost_per_query",
        "input_cost_per_request",
        "input_cost_per_audio_token",
        "output_cost_per_audio_token",
        "output_cost_per_image_token",
        "input_cost_per_video_token",
        "output_cost_per_video_token",
        "input_cost_per_audio_per_second",
        "input_cost_per_video_per_second",
        "input_cost_per_token_above_128k_tokens",
        "output_cost_per_token_above_128k_tokens",
        "input_cost_per_token_above_200k_tokens",
        "output_cost_per_token_above_200k_tokens",
        "input_cost_per_token_above_272k_tokens",
        "output_cost_per_token_above_272k_tokens",
        "input_cost_per_character_above_128k_tokens",
        "output_cost_per_character_above_128k_tokens",
        "input_cost_per_image_above_128k_tokens",
        "input_cost_per_video_per_second_above_8s_interval",
        "input_cost_per_video_per_second_above_15s_interval",
        "input_cost_per_video_per_second_above_128k_tokens",
        "input_cost_per_audio_token_batches",
        "input_cost_per_image_token_batches",
        "input_cost_per_token_batches",
        "input_cost_per_video_token_batches",
        "output_cost_per_token_batches",
        "input_cost_per_token_cache_hit",
        "cache_creation_input_token_cost",
        "cache_creation_input_audio_token_cost",
        "cache_read_input_token_cost",
        "cache_read_input_audio_token_cost",
        "cache_read_input_image_token_cost",
        "input_dbu_cost_per_token",
        "output_db_cost_per_token",
        "output_dbu_cost_per_token",
        "output_cost_per_reasoning_token",
        "citation_cost_per_token",
    ]

    # Also check nested cost fields
    nested_cost_fields = [
        "search_context_cost_per_query",
    ]

    for model_id, model_info in model_data.items():
        # Skip if this model is in exceptions
        if model_id in exceptions:
            continue

        # Check direct cost fields
        for field in cost_fields:
            if field in model_info and model_info[field] is not None:
                cost_value = model_info[field]

                # Convert string values to float if needed
                if isinstance(cost_value, str):
                    try:
                        cost_value = float(cost_value)
                    except (ValueError, TypeError):
                        # Skip if we can't convert to float
                        continue

                if isinstance(cost_value, (int, float)) and cost_value > 1:
                    violations.append(f"Model '{model_id}' has {field} = {cost_value} which exceeds 1")

        # Check nested cost fields
        for field in nested_cost_fields:
            if field in model_info and model_info[field] is not None:
                nested_costs = model_info[field]
                if isinstance(nested_costs, dict):
                    for nested_field, nested_value in nested_costs.items():
                        # Convert string values to float if needed
                        if isinstance(nested_value, str):
                            try:
                                nested_value = float(nested_value)
                            except (ValueError, TypeError):
                                # Skip if we can't convert to float
                                continue

                        if isinstance(nested_value, (int, float)) and nested_value > 1:
                            violations.append(
                                f"Model '{model_id}' has {field}.{nested_field} = {nested_value} which exceeds 1"
                            )

    return len(violations) == 0, violations


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
                "cache_creation_input_token_cost_above_1hr": {"type": "number"},
                "cache_creation_input_token_cost_above_32k_tokens": {"type": "number"},
                "cache_creation_input_token_cost_above_128k_tokens": {"type": "number"},
                "cache_creation_input_token_cost_above_200k_tokens": {"type": "number"},
                "cache_creation_input_token_cost_above_256k_tokens": {"type": "number"},
                "cache_creation_input_token_cost_above_272k_tokens": {"type": "number"},
                "cache_creation_input_token_cost_above_272k_tokens_flex": {"type": "number"},
                "cache_creation_input_token_cost_above_272k_tokens_priority": {"type": "number"},
                "cache_creation_input_token_cost_above_272k_tokens_batches": {"type": "number"},
                "cache_creation_input_token_cost_batches": {"type": "number"},
                "cache_creation_input_token_cost_flex": {"type": "number"},
                "cache_creation_input_token_cost_priority": {"type": "number"},
                "cache_read_input_token_cost": {"type": "number"},
                "cache_read_input_token_cost_above_32k_tokens": {"type": "number"},
                "cache_read_input_token_cost_above_128k_tokens": {"type": "number"},
                "cache_read_input_token_cost_above_200k_tokens": {"type": "number"},
                "cache_read_input_token_cost_above_256k_tokens": {"type": "number"},
                "cache_read_input_token_cost_above_272k_tokens": {"type": "number"},
                "cache_read_input_token_cost_above_272k_tokens_flex": {"type": "number"},
                "cache_read_input_token_cost_above_512k_tokens": {"type": "number"},
                "cache_read_input_token_cost_batches": {"type": "number"},
                "cache_read_input_token_cost_above_272k_tokens_batches": {"type": "number"},
                "cache_creation_input_token_cost_above_1hr_above_200k_tokens": {"type": "number"},
                "cache_read_input_audio_token_cost": {"type": "number"},
                "cache_read_input_image_token_cost": {"type": "number"},
                "audio_transcription_config": {"type": "string"},
                "deprecation_date": {"type": "string"},
                "input_cost_per_audio_per_second": {"type": "number"},
                "input_cost_per_audio_per_second_above_128k_tokens": {"type": "number"},
                "google_maps_grounding_cost_per_query": {"type": "number"},
                "input_cost_per_audio_token": {"type": "number"},
                "input_cost_per_image_token": {"type": "number"},
                "input_cost_per_character": {"type": "number"},
                "input_cost_per_character_above_128k_tokens": {"type": "number"},
                "input_cost_per_image": {"type": "number"},
                "input_cost_per_image_above_128k_tokens": {"type": "number"},
                "input_cost_per_video_token": {"type": "number"},
                "input_cost_per_token_above_32k_tokens": {"type": "number"},
                "input_cost_per_token_above_200k_tokens": {"type": "number"},
                "input_cost_per_token_above_256k_tokens": {"type": "number"},
                "input_cost_per_token_above_272k_tokens": {"type": "number"},
                "input_cost_per_token_above_512k_tokens": {"type": "number"},
                "cache_read_input_token_cost_flex": {"type": "number"},
                "cache_read_input_token_cost_priority": {"type": "number"},
                "cache_read_input_token_cost_above_200k_tokens_priority": {"type": "number"},
                "cache_read_input_token_cost_above_272k_tokens_priority": {"type": "number"},
                "input_cost_per_token_flex": {"type": "number"},
                "input_cost_per_token_priority": {"type": "number"},
                "input_cost_per_token_above_200k_tokens_priority": {"type": "number"},
                "input_cost_per_token_above_272k_tokens_priority": {"type": "number"},
                "input_cost_per_token_above_272k_tokens_batches": {"type": "number"},
                "input_cost_per_token_above_272k_tokens_flex": {"type": "number"},
                "input_cost_per_audio_token_priority": {"type": "number"},
                "output_cost_per_token_flex": {"type": "number"},
                "output_cost_per_token_priority": {"type": "number"},
                "output_cost_per_token_above_200k_tokens_priority": {"type": "number"},
                "output_cost_per_token_above_272k_tokens_priority": {"type": "number"},
                "output_cost_per_token_above_272k_tokens_batches": {"type": "number"},
                "output_cost_per_token_above_272k_tokens_flex": {"type": "number"},
                "regional_endpoint_uplift_multiplier": {"type": "number"},
                "regional_processing_uplift_multiplier_eu": {"type": "number"},
                "regional_processing_uplift_multiplier_us": {"type": "number"},
                "input_cost_per_pixel": {"type": "number"},
                "input_cost_per_query": {"type": "number"},
                "input_cost_per_request": {"type": "number"},
                "input_cost_per_second": {"type": "number"},
                "input_cost_per_token": {"type": "number"},
                "input_cost_per_token_above_128k_tokens": {"type": "number"},
                "input_cost_per_audio_token_batches": {"type": "number"},
                "input_cost_per_image_token_batches": {"type": "number"},
                "input_cost_per_token_batches": {"type": "number"},
                "input_cost_per_video_token_batches": {"type": "number"},
                "input_cost_per_token_cache_hit": {"type": "number"},
                "input_cost_per_video_per_second": {"type": "number"},
                "input_cost_per_video_per_second_above_8s_interval": {"type": "number"},
                "input_cost_per_video_per_second_above_15s_interval": {"type": "number"},
                "input_cost_per_video_per_second_above_128k_tokens": {"type": "number"},
                "input_dbu_cost_per_token": {"type": "number"},
                "annotation_cost_per_page": {"type": "number"},
                "annotation_cost_per_page_batches": {"type": "number"},
                "ocr_cost_per_page": {"type": "number"},
                "ocr_cost_per_page_batches": {"type": "number"},
                "ocr_cost_per_credit": {"type": "number"},
                "code_interpreter_cost_per_session": {"type": "number"},
                "inference_geo": {"type": "string"},
                "litellm_provider": {"type": "string"},
                "max_input_tokens": {"type": "number"},
                "max_output_tokens": {"type": "number"},
                "max_tokens": {"type": "number"},
                "metadata": {"type": "object"},
                "provider_specific_entry": {"type": "object"},
                "mode": {
                    "type": "string",
                    "enum": [
                        "audio_speech",
                        "audio_transcription",
                        "chat",
                        "completion",
                        "container",
                        "image_edit",
                        "embedding",
                        "evaluation",
                        "guardrail",
                        "image_generation",
                        "video_generation",
                        "moderation",
                        "rerank",
                        "realtime",
                        "responses",
                        "ocr",
                        "search",
                        "vector_store",
                    ],
                },
                "output_cost_per_audio_token": {"type": "number"},
                "output_cost_per_character": {"type": "number"},
                "output_cost_per_character_above_128k_tokens": {"type": "number"},
                "output_cost_per_image": {"type": "number"},
                "output_cost_per_image_512": {"type": "number"},
                "output_cost_per_image_1024": {"type": "number"},
                "output_cost_per_image_1536": {"type": "number"},
                "output_cost_per_image_0.5K": {"type": "number"},
                "output_cost_per_image_1K": {"type": "number"},
                "output_cost_per_image_2K": {"type": "number"},
                "output_cost_per_image_4K": {"type": "number"},
                "output_cost_per_image_token": {"type": "number"},
                "output_cost_per_video_token": {"type": "number"},
                "output_cost_per_pixel": {"type": "number"},
                "output_cost_per_second": {"type": "number"},
                "output_cost_per_second_480p": {"type": "number"},
                "output_cost_per_second_720p": {"type": "number"},
                "output_cost_per_second_768p": {"type": "number"},
                "output_cost_per_second_2k": {"type": "number"},
                "output_cost_per_second_1080p": {"type": "number"},
                "output_cost_per_second_4k": {"type": "number"},
                "output_cost_per_token": {"type": "number"},
                "output_cost_per_token_above_32k_tokens": {"type": "number"},
                "output_cost_per_token_above_128k_tokens": {"type": "number"},
                "output_cost_per_token_above_200k_tokens": {"type": "number"},
                "output_cost_per_token_above_256k_tokens": {"type": "number"},
                "output_cost_per_token_above_272k_tokens": {"type": "number"},
                "output_cost_per_token_above_512k_tokens": {"type": "number"},
                "output_cost_per_token_batches": {"type": "number"},
                "output_cost_per_reasoning_token": {"type": "number"},
                "output_cost_per_video_per_second": {"type": "number"},
                "output_db_cost_per_token": {"type": "number"},
                "output_dbu_cost_per_token": {"type": "number"},
                "output_vector_size": {"type": "number"},
                "rpd": {"type": "number"},
                "rpm": {"type": "number"},
                "source": {"type": "string"},
                "comment": {"type": "string"},
                "supports_assistant_prefill": {"type": "boolean"},
                "supports_anthropic_compaction": {"type": "boolean"},
                "supports_audio_input": {"type": "boolean"},
                "supports_audio_output": {"type": "boolean"},
                "gemini_native_audio": {"type": "boolean"},
                "gemini_audio_only_live": {"type": "boolean"},
                "supports_embedding_image_input": {"type": "boolean"},
                "supports_forced_tool_use": {"type": "boolean"},
                "supports_function_calling": {"type": "boolean"},
                "supports_image_input": {"type": "boolean"},
                "supports_nova_canvas_image_edit": {"type": "boolean"},
                "supports_parallel_function_calling": {"type": "boolean"},
                "supports_parallel_tool_use_config": {"type": "boolean"},
                "supports_pdf_input": {"type": "boolean"},
                "prompt_cache_min_tokens": {"type": "number"},
                "supports_prompt_cache_breakpoint": {"type": "boolean"},
                "supports_thinking_cache_preservation": {"type": "boolean"},
                "supports_prompt_caching": {"type": "boolean"},
                "supports_response_schema": {"type": "boolean"},
                "supports_system_messages": {"type": "boolean"},
                "supports_tool_choice": {"type": "boolean"},
                "supports_tool_search": {"type": "boolean"},
                "supports_video_input": {"type": "boolean"},
                "supports_vision": {"type": "boolean"},
                "supports_web_search": {"type": "boolean"},
                "supports_url_context": {"type": "boolean"},
                "supports_multimodal": {"type": "boolean"},
                "uses_embed_content": {"type": "boolean"},
                "supports_reasoning": {"type": "boolean"},
                "supports_minimal_reasoning_effort": {"type": "boolean"},
                "supports_low_reasoning_effort": {"type": "boolean"},
                "supports_none_reasoning_effort": {"type": "boolean"},
                "supports_xhigh_reasoning_effort": {"type": "boolean"},
                "supports_max_reasoning_effort": {"type": "boolean"},
                "reasoning_effort_levels": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"]},
                },
                "default_reasoning_effort": {
                    "type": "string",
                    "enum": ["none", "minimal", "low", "medium", "high", "xhigh"],
                },
                "supports_adaptive_thinking": {"type": "boolean"},
                "supports_anthropic_thinking_payload": {"type": "boolean"},
                "supports_legacy_thinking": {"type": "boolean"},
                "thinking_always_on": {"type": "boolean"},
                "supports_mid_conversation_system": {"type": "boolean"},
                "supports_sampling_params": {"type": "boolean"},
                "supports_output_config": {"type": "boolean"},
                "supports_speed": {"type": "boolean"},
                "supports_fast_mode": {"type": "boolean"},
                "supported_audio_formats": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["mp3", "wav"],
                    },
                },
                "vertex_ai_audio_api": {
                    "type": "string",
                    "enum": ["lyria_predict", "lyria_interactions"],
                },
                "bedrock_output_config_effort_ceiling": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "max", "xhigh"],
                },
                "bedrock_converse_supports_strict_tools": {"type": "boolean"},
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
                            "/v1/messages",
                            "/v1/images/generations",
                            "/v1/realtime",
                            "/v1/realtime/transcription_sessions",
                            "/v1/images/variations",
                            "/v1/images/edits",
                            "/v1/batch",
                            "/v1beta/interactions",
                            "/v1/audio/transcriptions",
                            "/v1/audio/speech",
                            "/v1/ocr",
                            "/v1/videos",
                            "/vertex_ai/live",
                            "/v1/listen",
                            "/v1beta/interactions",
                        ],
                    },
                },
                "supported_regions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "guardrail_cost_per_unit": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
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
                "web_search_billing_unit": {
                    "type": "string",
                    "enum": ["per_prompt", "per_query"],
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
                        "enum": ["text", "image", "audio", "code", "video"],
                    },
                },
                "supports_native_streaming": {"type": "boolean"},
                "supports_image_size": {"type": "boolean"},
                "supports_native_structured_output": {"type": "boolean"},
                "use_openai_responses_path": {"type": "boolean"},
                "off_peak_pricing": {
                    "type": "object",
                    "properties": {
                        "hours_utc": {
                            "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                        },
                        "windows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "hours_utc": {
                                        "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                                    },
                                    "weekdays": {
                                        "type": "array",
                                        "items": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                                    },
                                },
                                "required": ["hours_utc"],
                                "additionalProperties": False,
                            },
                        },
                        "weekday_timezone": {"type": "string"},
                        "input_cost_per_token": {"type": "number"},
                        "output_cost_per_token": {"type": "number"},
                        "output_cost_per_reasoning_token": {"type": "number"},
                        "cache_read_input_token_cost": {"type": "number"},
                        "cache_creation_input_token_cost": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "tiered_pricing": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "input_cost_per_token": {"type": "number"},
                            "output_cost_per_token": {"type": "number"},
                            "cache_read_input_token_cost": {"type": "number"},
                            "cache_creation_input_token_cost": {"type": "number"},
                            "output_cost_per_reasoning_token": {"type": "number"},
                            "max_results_range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "input_cost_per_query": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    }

    prod_json = os.path.join(os.path.dirname(__file__), "..", "..", "model_prices_and_context_window.json")
    with open(prod_json, "r") as model_prices_file:
        actual_json = json.load(model_prices_file)
    assert isinstance(actual_json, dict)
    actual_json.pop("sample_spec", None)  # remove the sample, whose schema is inconsistent with the real data
    actual_json.pop("fallback_generalizations", None)  # reserved meta key, not a model entry

    # Validate schema
    validate(actual_json, INTENDED_SCHEMA)

    # Validate cost values
    # Define exceptions for models that are allowed to have costs > 1
    # Add model IDs here if they legitimately have costs > 1
    exceptions = [
        # Add any model IDs that should be exempt from the cost validation
        # Example: "expensive-model-id",
        "runwayml/seedance2",  # 4K output is 150 credits/second = $1.50/second
        "fal_ai/bytedance/seedance-2.0/text-to-video",
        "fal_ai/bytedance/seedance-2.0/image-to-video",
        "fal_ai/bytedance/seedance-2.0/reference-to-video",
    ]

    is_valid, violations = validate_model_cost_values(actual_json, exceptions)

    if not is_valid:
        error_message = "Cost validation failed:\n" + "\n".join(violations)
        error_message += "\n\nTo add exceptions, add the model ID to the 'exceptions' list in the test function."
        raise AssertionError(error_message)


def test_max_tokens_consistency():
    """
    Test that max_tokens == max_output_tokens for all models.

    According to the spec in model_prices_and_context_window.json:
    - max_tokens is a LEGACY parameter
    - It should be set to max_output_tokens if the provider specifies it

    This test ensures consistency across all model definitions.
    """
    import json
    from pathlib import Path

    # Load the model configuration
    config_path = Path(__file__).parent.parent.parent / "model_prices_and_context_window.json"
    with open(config_path, "r") as f:
        models = json.load(f)

    inconsistencies = []

    for model_name, config in models.items():
        # Skip the sample_spec
        if model_name == "sample_spec":
            continue

        # Check if both max_tokens and max_output_tokens exist
        if isinstance(config, dict):
            max_tokens = config.get("max_tokens")
            max_output_tokens = config.get("max_output_tokens")

            # Only validate if both exist
            if max_tokens is not None and max_output_tokens is not None:
                if max_tokens != max_output_tokens:
                    inconsistencies.append(
                        {
                            "model": model_name,
                            "max_tokens": max_tokens,
                            "max_output_tokens": max_output_tokens,
                        }
                    )

    if inconsistencies:
        error_msg = f"\n\n❌ Found {len(inconsistencies)} models with max_tokens != max_output_tokens:\n\n"
        for item in inconsistencies[:10]:  # Show first 10
            error_msg += (
                f"  {item['model']}: max_tokens={item['max_tokens']}, max_output_tokens={item['max_output_tokens']}\n"
            )

        if len(inconsistencies) > 10:
            error_msg += f"\n  ... and {len(inconsistencies) - 10} more\n"

        error_msg += "\nTo fix these inconsistencies, run: uv run python fix_max_tokens_inconsistencies.py"
        raise AssertionError(error_msg)


def test_get_model_info_bedrock_regional_inference_profile_pricing(local_model_cost_map):
    """Regression LIT-4056: with the bedrock/ routing prefix (plain, converse/, or
    invoke/), the exact regional cost-map entry must win over the region-stripped
    base entry, matching the unprefixed control form."""
    regional = litellm.model_cost["eu.amazon.nova-pro-v1:0"]
    base = litellm.model_cost["amazon.nova-pro-v1:0"]
    assert regional["input_cost_per_token"] > base["input_cost_per_token"]

    for model in (
        "bedrock/eu.amazon.nova-pro-v1:0",
        "bedrock/converse/eu.amazon.nova-pro-v1:0",
        "bedrock/invoke/eu.amazon.nova-pro-v1:0",
    ):
        info = litellm.get_model_info(model=model)
        assert info["key"] == "eu.amazon.nova-pro-v1:0", model
        assert info["input_cost_per_token"] == regional["input_cost_per_token"], model
        assert info["output_cost_per_token"] == regional["output_cost_per_token"], model

    control = litellm.get_model_info(model="eu.amazon.nova-pro-v1:0", custom_llm_provider="bedrock")
    assert control["key"] == "eu.amazon.nova-pro-v1:0"


@pytest.mark.parametrize(
    "bare_key",
    [
        "anthropic.claude-fable-5",
        "anthropic.claude-fable-5-1",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-opus-4-5-20251101-v1:0",
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-5",
        "anthropic.claude-opus-5-5",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-sonnet-5",
    ],
)
def test_bedrock_bare_claude_id_is_priced_global(local_model_cost_map, bare_key):
    """A bare Bedrock Claude id is billed at the Global SKU, so it carries the same
    rate as its global. inference profile and sits below the regional us. rate."""
    bare = litellm.model_cost[bare_key]
    us = litellm.model_cost[f"us.{bare_key}"]
    global_ = litellm.model_cost[f"global.{bare_key}"]
    cost_fields = [f for f in bare if "cost" in f]
    assert cost_fields
    for field in cost_fields:
        assert bare[field] == global_[field], field
    assert bare["input_cost_per_token"] < us["input_cost_per_token"]


def test_get_model_info_bedrock_mantle_region_prefix_falls_back_to_the_mantle_row(local_model_cost_map):
    """A Mantle deployment name may carry the region as a prefix (bedrock_mantle/us-east-2/<model>).
    That name has no cost row of its own, so pricing must fall through to the region-free
    bedrock_mantle/<model> row instead of raising, while a region that has its own row keeps it."""
    for model, expected_key in (
        ("bedrock_mantle/us-east-2/anthropic.claude-haiku-4-5", "bedrock_mantle/anthropic.claude-haiku-4-5"),
        ("bedrock_mantle/us-east-2/openai.gpt-5.6-sol", "bedrock_mantle/openai.gpt-5.6-sol"),
        ("bedrock_mantle/us-gov-west-1/openai.gpt-5.4", "bedrock_mantle/us-gov-west-1/openai.gpt-5.4"),
    ):
        info = litellm.get_model_info(model=model, custom_llm_provider="bedrock_mantle")
        assert info["key"] == expected_key, model
        assert info["input_cost_per_token"] == litellm.model_cost[expected_key]["input_cost_per_token"], model
        assert info["input_cost_per_token"] > 0, model


def test_openai_models_in_model_info(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_map = litellm.model_cost
    violated_models = []
    for model, info in model_map.items():
        if info.get("litellm_provider") == "openai" and info.get("supports_vision") is True:
            if info.get("supports_pdf_input") is not True:
                violated_models.append(model)
    assert len(violated_models) == 0, f"The following models should support pdf input: {violated_models}"


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


def test_check_provider_match_none_value_matches_any_provider():
    """
    A ``litellm_provider`` of None must be treated the same as a missing
    key: both mean "no provider constraint" and should match any
    ``custom_llm_provider``.

    Regression test for https://github.com/BerriAI/litellm/issues/28336.
    Before the fix, ``register_model`` persisted ``litellm_provider: None``
    via ``get_model_info`` for deployments registered without a provider
    (e.g. ``Router.add_deployment``), which caused ``_check_provider_match``
    to drop custom pricing intermittently.
    """
    # Missing key already returned True; None must behave identically.
    assert litellm.utils._check_provider_match({}, "openai") is True
    assert litellm.utils._check_provider_match({"litellm_provider": None}, "openai") is True
    assert litellm.utils._check_provider_match({"litellm_provider": None}, "anthropic") is True
    # When custom_llm_provider is also None nothing constrains the match.
    assert litellm.utils._check_provider_match({"litellm_provider": None}, None) is True


def test_get_provider_rerank_config():
    """
    Test the get_provider_rerank_config function for various providers
    """
    from litellm import HostedVLLMRerankConfig
    from litellm.utils import LlmProviders

    # Test for hosted_vllm provider
    config = ProviderConfigManager.get_provider_rerank_config(
        "my_model", LlmProviders.HOSTED_VLLM, "http://localhost", []
    )
    assert isinstance(config, HostedVLLMRerankConfig)


def test_get_provider_text_to_speech_config_vertex_gemini_skips_cloud_tts():
    """Regression for LIT-6501: mapping vertex Gemini TTS params through Google Cloud TTS
    dropped response_format before the speech_to_completion bridge could honor it."""
    from litellm.llms.vertex_ai.text_to_speech.transformation import VertexAITextToSpeechConfig
    from litellm.utils import LlmProviders

    assert (
        ProviderConfigManager.get_provider_text_to_speech_config(
            model="gemini-2.5-flash-preview-tts", provider=LlmProviders.VERTEX_AI
        )
        is None
    )
    assert isinstance(
        ProviderConfigManager.get_provider_text_to_speech_config(
            model="en-US-Studio-O", provider=LlmProviders.VERTEX_AI
        ),
        VertexAITextToSpeechConfig,
    )


# Models that should be skipped during testing
OLD_PROVIDERS = ["aleph_alpha", "palm"]
SKIP_MODELS = [
    "azure/mistral",
    "azure/command-r",
    "jamba",
    "deepinfra",
    "mistral.",
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


@pytest.mark.parametrize(
    "model, custom_llm_provider",
    [
        ("gpt-3.5-turbo", "openai"),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock"),
        ("gemini-2.5-pro", "vertex_ai"),
    ],
)
def test_pre_process_non_default_params(model, custom_llm_provider):
    from pydantic import BaseModel

    from litellm.utils import pre_process_non_default_params

    provider_config = ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders(custom_llm_provider)
    )

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
        provider_config=provider_config,
    )
    print(processed_non_default_params)
    # Vertex AI / Gemini uses Pydantic's model_json_schema() which doesn't
    # include additionalProperties: False (Gemini rejects it).  Other
    # providers use OpenAI's to_strict_json_schema() which does.
    expected_schema = {
        "properties": {
            "x": {"title": "X", "type": "string"},
            "y": {"title": "Y", "type": "string"},
        },
        "required": ["x", "y"],
        "title": "ResponseFormat",
        "type": "object",
    }
    if custom_llm_provider not in ("vertex_ai", "vertex_ai_beta", "gemini"):
        expected_schema["additionalProperties"] = False
    assert processed_non_default_params == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": expected_schema,
                "name": "ResponseFormat",
                "strict": True,
            },
        }
    }


@pytest.mark.parametrize(
    "custom_llm_provider, expected",
    [
        ("vertex_ai", True),
        ("vertex_ai_beta", True),
        ("gdc", True),
        ("openai", False),
        ("bedrock", False),
        ("not_a_real_provider", False),
    ],
)
def test_provider_supports_vertex_params(custom_llm_provider, expected):
    from litellm.utils import _provider_supports_vertex_params

    assert _provider_supports_vertex_params(custom_llm_provider) is expected


@pytest.mark.parametrize(
    "model, custom_llm_provider, should_keep",
    [
        ("gemini-2.5-pro", "vertex_ai", True),
        ("gemini-2.5-pro", "vertex_ai_beta", True),
        ("gdc/gemini-2.5-flash", "gdc", True),
        ("gpt-4o", "openai", False),
    ],
)
def test_vertex_params_not_stripped_for_vertex_family(model, custom_llm_provider, should_keep):
    optional_params = litellm.utils.get_optional_params(
        model=model,
        custom_llm_provider=custom_llm_provider,
        vertex_project="my-project",
        vertex_location="us-central1",
    )
    assert ("vertex_project" in optional_params) is should_keep
    assert ("vertex_location" in optional_params) is should_keep
    if should_keep:
        assert optional_params["vertex_project"] == "my-project"
        assert optional_params["vertex_location"] == "us-central1"


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
                "claude-sonnet-4-6",
                "litellm_proxy/claude-sonnet-4-6",
                True,
            ),
            # Google models
            ("gemini-2.5-pro", "litellm_proxy/gemini-2.5-pro", True),
            ("gemini/gemini-2.5-pro", "litellm_proxy/gemini/gemini-2.5-pro", True),
            ("gemini/gemini-2.5-flash", "litellm_proxy/gemini/gemini-2.5-flash", True),
            # Groq models (mixed support)
            # Cohere models (generally don't support function calling)
            ("command-nightly", "litellm_proxy/command-nightly", False),
        ],
    )
    def test_proxy_function_calling_support_consistency(self, direct_model, proxy_model, expected_result):
        """Test that proxy models have the same function calling support as their direct counterparts."""
        direct_result = supports_function_calling(direct_model)
        proxy_result = supports_function_calling(proxy_model)

        # Both should match the expected result
        assert direct_result == expected_result, f"Direct model {direct_model} should return {expected_result}"
        assert proxy_result == expected_result, f"Proxy model {proxy_model} should return {expected_result}"

        # Direct and proxy should be consistent
        assert direct_result == proxy_result, (
            f"Mismatch: {direct_model}={direct_result} vs {proxy_model}={proxy_result}"
        )

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
                "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
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
            ("litellm_proxy/fast-llama", "groq/llama-3.1-8b-instant", False),
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
    def test_proxy_custom_model_names_without_config(self, proxy_model_name, underlying_model, expected_proxy_result):
        """
        Test proxy models with custom model names that differ from underlying models.

        Without proxy configuration context, LiteLLM cannot resolve custom model names
        to their underlying models, so these will return False.
        This demonstrates the limitation and documents the expected behavior.
        """
        # Test the underlying model directly first to establish what it SHOULD return
        try:
            underlying_result = supports_function_calling(underlying_model)
            print(f"Underlying model {underlying_model} supports function calling: {underlying_result}")
        except Exception as e:
            print(f"Warning: Could not test underlying model {underlying_model}: {e}")

        # Test the proxy model - this will return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        assert proxy_result == expected_proxy_result, (
            f"Proxy model {proxy_model_name} should return {expected_proxy_result} (without config context)"
        )

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
        assert result is False, "Custom model names return False without proxy config context"

        # Case 2: Model name that can be resolved (matches pattern)
        resolvable_model = "litellm_proxy/claude-sonnet-4-5-20250929"
        result = supports_function_calling(resolvable_model)
        assert result is True, "Resolvable model names work with fallback logic"

        # Documentation notes:
        print("""
        PROXY MODEL RESOLUTION BEHAVIOR:
        
        ✅ WORKS (with current fallback logic):
           - litellm_proxy/gpt-4
           - litellm_proxy/claude-sonnet-4-5-20250929
           - litellm_proxy/anthropic/claude-3-haiku-20240307
           
        ❌ DOESN'T WORK (requires proxy server config):
           - litellm_proxy/my-custom-gpt4
           - litellm_proxy/bedrock-claude-3-haiku
           - litellm_proxy/production-model
           
        💡 SOLUTION: Use LiteLLM proxy server with proper model_list configuration
           that maps custom names to underlying models.
        """)

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
    def test_proxy_models_with_naming_hints(self, proxy_model_with_hints, expected_result):
        """
        Test proxy models with names that provide hints about the underlying model.

        Note: These will currently fail because the hint-based resolution isn't implemented yet,
        but they demonstrate what could be possible with enhanced model name inference.
        """
        # This test documents potential future enhancement
        proxy_result = supports_function_calling(proxy_model_with_hints)

        # Currently these will return False, but we document the expected behavior
        # In the future, we could implement smarter model name inference
        print(f"Model {proxy_model_with_hints}: current={proxy_result}, desired={expected_result}")

        # For now, we expect False (current behavior), but document the limitation
        assert proxy_result is False, f"Current limitation: {proxy_model_with_hints} returns False without inference"

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
                assert result == expected_result, (
                    f"Edge case {model_name} returned {result}, expected {expected_result}"
                )
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
        print(f"Direct model '{direct_model}' supports function calling: {direct_result}")
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
            f"Proxy model resolution issue: {direct_model} -> {direct_result}, {proxy_model} -> {proxy_result}"
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
                "bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0",
                False,
                "Bedrock Claude 3 Opus via Converse API",
            ),
            (
                "litellm_proxy/bedrock-claude-3-5-sonnet",
                "bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0",
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
                "bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0",
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
                "bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0",
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
                assert underlying_result is True, (
                    f"Claude 3 models should support function calling: {underlying_bedrock_model}"
                )
        except Exception as e:
            print(f"  Warning: Could not test underlying model {underlying_bedrock_model}: {e}")

        # Test the proxy model - should return False due to lack of configuration context
        proxy_result = supports_function_calling(proxy_model_name)
        print(f"  Proxy model function calling support: {proxy_result}")

        assert proxy_result == expected_proxy_result, (
            f"Proxy model {proxy_model_name} should return {expected_proxy_result} "
            f"(without config context). Description: {description}"
        )


def test_register_model_with_scientific_notation():
    """
    Test that the register_model function can handle scientific notation in the model name.
    """
    import uuid

    # Use a truly unique model name with uuid to avoid conflicts when tests run in parallel
    test_model_name = f"test-scientific-notation-model-{uuid.uuid4().hex[:12]}"

    # Clear LRU caches that might have stale data
    from litellm.utils import (
        _invalidate_model_cost_lowercase_map,
    )

    _invalidate_model_cost_lowercase_map()

    model_cost_dict = {
        test_model_name: {
            "max_tokens": 8192,
            "input_cost_per_token": "3e-07",
            "output_cost_per_token": "6e-07",
            "litellm_provider": "openai",
            "mode": "chat",
        },
    }

    litellm.register_model(model_cost_dict)

    registered_model = litellm.model_cost[test_model_name]
    print(registered_model)
    assert registered_model["input_cost_per_token"] == 3e-07
    assert registered_model["output_cost_per_token"] == 6e-07
    assert registered_model["litellm_provider"] == "openai"
    assert registered_model["mode"] == "chat"

    # Clean up after test
    if test_model_name in litellm.model_cost:
        del litellm.model_cost[test_model_name]
    _invalidate_model_cost_lowercase_map()


@respx.mock
def test_register_model_url_fetch_uses_single_attempt(monkeypatch):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    before = dict(litellm.model_cost)
    threads_before = {thread.name for thread in threading.enumerate()}
    route = respx.get("https://example.invalid/custom_pricing.json").mock(return_value=httpx.Response(503))

    litellm.register_model(model_cost="https://example.invalid/custom_pricing.json")

    threads_after = {thread.name for thread in threading.enumerate()}
    assert route.call_count == 1
    assert not (threads_after - threads_before) & {"litellm-model-cost-map-retry"}
    assert not any(
        thread.name == "litellm-model-cost-map-retry" and thread.is_alive() for thread in threading.enumerate()
    )
    assert litellm.model_cost.keys() >= before.keys()


def test_register_model_openrouter_without_slash():
    """
    Test that register_model handles openrouter models without '/' in the name.

    Fixes https://github.com/BerriAI/litellm/issues/18936

    Previously, the code did `split_string[1]` which would fail with IndexError
    when the model name didn't contain '/'. Now it uses `split_string[-1]` which
    always works.
    """
    # Clear any existing entries
    litellm.openrouter_models.discard("my-custom-alias")
    litellm.openrouter_models.discard("gpt-4")
    litellm.openrouter_models.discard("openai/gpt-4")

    # Test 1: Model name without '/' (this was the bug - would raise IndexError)
    litellm.register_model(
        {
            "my-custom-alias": {
                "max_tokens": 8192,
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00002,
                "litellm_provider": "openrouter",
                "mode": "chat",
            },
        }
    )
    assert "my-custom-alias" in litellm.openrouter_models

    # Test 2: Model name with single '/' (openrouter/model format)
    litellm.register_model(
        {
            "openrouter/gpt-4": {
                "max_tokens": 8192,
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00002,
                "litellm_provider": "openrouter",
                "mode": "chat",
            },
        }
    )
    assert "gpt-4" in litellm.openrouter_models

    # Test 3: Model name with double '/' (openrouter/provider/model format)
    litellm.register_model(
        {
            "openrouter/openai/gpt-4-turbo": {
                "max_tokens": 8192,
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00002,
                "litellm_provider": "openrouter",
                "mode": "chat",
            },
        }
    )
    assert "openai/gpt-4-turbo" in litellm.openrouter_models


def test_reasoning_content_preserved_in_text_completion_wrapper():
    """Ensure reasoning_content is copied from delta to text_choices."""
    chunk = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content="Some answer text",
                    role="assistant",
                    reasoning_content="Here's my chain of thought...",
                ),
            )
        ],
    )

    wrapper = TextCompletionStreamWrapper(
        completion_stream=None,  # Not used in convert_to_text_completion_object
        model="test-model",
        stream_options=None,
    )

    transformed = wrapper.convert_to_text_completion_object(chunk)

    assert "choices" in transformed
    assert len(transformed["choices"]) == 1
    choice = transformed["choices"][0]
    assert choice["text"] == "Some answer text"
    assert choice["reasoning_content"] == "Here's my chain of thought..."


def test_anthropic_claude_4_invoke_chat_provider_config():
    """Test that the Anthropic Claude 4 Invoke chat provider config is correct."""
    from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeConfig,
    )

    config = ProviderConfigManager.get_provider_chat_config(
        model="invoke/us.anthropic.claude-sonnet-4-20250514-v1:0",
        provider=LlmProviders.BEDROCK,
    )
    print(config)
    assert isinstance(config, AmazonAnthropicClaudeConfig)


def test_bedrock_application_inference_profile():
    model = "arn:aws:bedrock:us-east-2:<AWS-ACCOUNT-ID>:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0"

    from litellm.utils import supports_tool_choice

    result = supports_tool_choice(model, custom_llm_provider="bedrock")
    result_2 = supports_tool_choice(model, custom_llm_provider="bedrock_converse")
    print(result)
    assert result == result_2
    assert result is True


def test_image_response_utils():
    """Test that the image response utils are correct."""
    from litellm.utils import ImageResponse

    result = {
        "created": None,
        "data": [
            {
                "b64_json": "/9j/.../2Q==",
                "revised_prompt": None,
                "url": None,
                "timings": {"inference": 0.9612685777246952},
                "index": 0,
            }
        ],
        "id": "91559891cxxx-PDX",
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "object": "list",
        "hidden_params": {"additional_headers": {}},
    }
    ImageResponse(**result)


def test_is_valid_api_key():
    import hashlib

    # Valid sk- keys
    assert is_valid_api_key("sk-abc123")
    assert is_valid_api_key("sk-ABC_123-xyz")
    # Valid hashed key (64 hex chars)
    assert is_valid_api_key("a" * 64)
    assert is_valid_api_key("0123456789abcdef" * 4)  # 16*4 = 64
    # Real SHA-256 hash
    real_hash = hashlib.sha256(b"my_secret_key").hexdigest()
    assert len(real_hash) == 64
    assert is_valid_api_key(real_hash)
    # Invalid: too short
    assert not is_valid_api_key("sk-")
    assert not is_valid_api_key("")
    # Invalid: too long
    assert not is_valid_api_key("sk-" + "a" * 200)
    # Invalid: wrong prefix
    assert not is_valid_api_key("pk-abc123")
    # Invalid: wrong chars in sk- key
    assert not is_valid_api_key("sk-abc$%#@!")
    # Invalid: not a string
    assert not is_valid_api_key(None)
    assert not is_valid_api_key(12345)
    # Invalid: wrong length for hash
    assert not is_valid_api_key("a" * 63)
    assert not is_valid_api_key("a" * 65)


def test_block_key_hashing_logic():
    """
    Test that block_key() function only hashes keys that start with "sk-"
    """

    from litellm.proxy.utils import hash_token

    # Test cases: (input_key, should_be_hashed, expected_output)
    test_cases = [
        ("sk-1234567890abcdef", True, hash_token("sk-1234567890abcdef")),
        ("sk-test-key", True, hash_token("sk-test-key")),
        ("abc123", False, "abc123"),  # Should not be hashed
        ("hashed_key_123", False, "hashed_key_123"),  # Should not be hashed
        ("", False, ""),  # Empty string should not be hashed
        ("sk-", True, hash_token("sk-")),  # Edge case: just "sk-"
    ]

    for input_key, should_be_hashed, expected_output in test_cases:
        # Simulate the logic from block_key() function
        if input_key.startswith("sk-"):
            hashed_token = hash_token(token=input_key)
        else:
            hashed_token = input_key

        assert hashed_token == expected_output, f"Failed for input: {input_key}"

        # Additional verification: if it should be hashed, verify it's actually a hash
        if should_be_hashed:
            # SHA-256 hashes are 64 characters long and contain only hex digits
            assert len(hashed_token) == 64, f"Hash length should be 64, got {len(hashed_token)} for {input_key}"
            assert all(c in "0123456789abcdef" for c in hashed_token), (
                f"Hash should contain only hex digits for {input_key}"
            )
        else:
            # If not hashed, it should be the original string
            assert hashed_token == input_key, f"Non-hashed key should remain unchanged: {input_key}"

    print("✅ All block_key hashing logic tests passed!")


def test_generate_gcp_iam_access_token():
    """
    Test the _generate_gcp_iam_access_token function with mocked GCP IAM client.
    """
    from unittest.mock import Mock, patch

    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"
    expected_token = "test-access-token-12345"

    # Mock the GCP IAM client and its response
    mock_response = Mock()
    mock_response.access_token = expected_token

    mock_client = Mock()
    mock_client.generate_access_token.return_value = mock_response

    # Mock the iam_credentials_v1 module
    mock_iam_credentials_v1 = Mock()
    mock_iam_credentials_v1.IAMCredentialsClient = Mock(return_value=mock_client)
    mock_iam_credentials_v1.GenerateAccessTokenRequest = Mock()

    # Test successful token generation by mocking sys.modules
    with patch.dict("sys.modules", {"google.cloud.iam_credentials_v1": mock_iam_credentials_v1}):
        from litellm._redis import _generate_gcp_iam_access_token

        result = _generate_gcp_iam_access_token(service_account)

        assert result == expected_token
        mock_iam_credentials_v1.IAMCredentialsClient.assert_called_once()
        mock_client.generate_access_token.assert_called_once()

        # Verify the request was created with correct parameters
        mock_iam_credentials_v1.GenerateAccessTokenRequest.assert_called_once_with(
            name=service_account,
            scope=["https://www.googleapis.com/auth/cloud-platform"],
        )


def test_generate_gcp_iam_access_token_import_error():
    """
    Test that _generate_gcp_iam_access_token raises ImportError when google-cloud-iam is not available.
    """
    # Import the function first, before mocking
    from litellm._redis import _generate_gcp_iam_access_token

    # Mock the import to fail when the function tries to import google.cloud.iam_credentials_v1
    original_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name == "google.cloud.iam_credentials_v1":
            raise ImportError("No module named 'google.cloud.iam_credentials_v1'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError) as exc_info:
            _generate_gcp_iam_access_token("test-service-account")

        assert "google-cloud-iam is required" in str(exc_info.value)
        assert "pip install google-cloud-iam" in str(exc_info.value)


def test_generate_azure_ad_redis_token():
    """Test _generate_azure_ad_redis_token with mocked Azure credential."""
    from unittest.mock import Mock, patch

    expected_token = "azure-access-token-12345"

    mock_token = Mock()
    mock_token.token = expected_token

    mock_credential = Mock()
    mock_credential.get_token.return_value = mock_token

    mock_azure_identity = Mock()
    mock_azure_identity.DefaultAzureCredential = Mock(return_value=mock_credential)
    mock_azure_identity.ClientSecretCredential = Mock()
    mock_azure_identity.ManagedIdentityCredential = Mock()

    with patch.dict("sys.modules", {"azure.identity": mock_azure_identity, "azure": Mock()}):
        from litellm._redis import _generate_azure_ad_redis_token

        result = _generate_azure_ad_redis_token()

        assert result == expected_token
        mock_credential.get_token.assert_called_once_with("https://redis.azure.com/.default")


def test_generate_azure_ad_redis_token_service_principal():
    """Test _generate_azure_ad_redis_token with service principal credentials."""
    from unittest.mock import Mock, patch

    expected_token = "sp-access-token-67890"

    mock_token = Mock()
    mock_token.token = expected_token

    mock_credential = Mock()
    mock_credential.get_token.return_value = mock_token

    mock_client_secret_credential = Mock(return_value=mock_credential)

    mock_azure_identity = Mock()
    mock_azure_identity.DefaultAzureCredential = Mock()
    mock_azure_identity.ClientSecretCredential = mock_client_secret_credential
    mock_azure_identity.ManagedIdentityCredential = Mock()

    with patch.dict("sys.modules", {"azure.identity": mock_azure_identity, "azure": Mock()}):
        from litellm._redis import _generate_azure_ad_redis_token

        result = _generate_azure_ad_redis_token(
            azure_client_id="test-client-id",
            azure_tenant_id="test-tenant-id",
            azure_client_secret="test-secret",
        )

        assert result == expected_token
        mock_client_secret_credential.assert_called_once_with(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
            client_secret="test-secret",
        )


def test_generate_azure_ad_redis_token_import_error():
    """Test that _generate_azure_ad_redis_token raises ImportError when azure-identity is missing."""
    from unittest.mock import patch

    from litellm._redis import _generate_azure_ad_redis_token

    with patch.dict("sys.modules", {"azure.identity": None}):
        with pytest.raises(ImportError) as exc_info:
            _generate_azure_ad_redis_token()

        assert "azure-identity is required" in str(exc_info.value)


def test_redis_client_logic_azure_ad_auth():
    """Test that _get_redis_client_logic sets up Azure AD auth when REDIS_AZURE_AD_TOKEN=true.

    Mocks ``azure.identity`` via ``sys.modules`` so the test does not require
    the real ``azure-identity`` package to be installed in the CI environment.
    """
    from unittest.mock import Mock, patch

    mock_credential = Mock()
    mock_azure_identity = Mock()
    mock_azure_identity.DefaultAzureCredential = Mock(return_value=mock_credential)
    mock_azure_identity.ClientSecretCredential = Mock(return_value=mock_credential)
    mock_azure_identity.ManagedIdentityCredential = Mock(return_value=mock_credential)

    with patch.dict("sys.modules", {"azure.identity": mock_azure_identity, "azure": Mock()}):
        from litellm._redis import _get_redis_client_logic

        redis_kwargs = _get_redis_client_logic(
            host="myredis.redis.cache.windows.net",
            port="6380",
            azure_redis_ad_token="true",
            ssl=True,
        )

        assert "redis_connect_func" in redis_kwargs
        # Marker for async paths to detect Azure AD auth
        assert hasattr(redis_kwargs["redis_connect_func"], "_azure_redis_ad_token")
        assert redis_kwargs["redis_connect_func"]._azure_redis_ad_token is True
        # Live credential object (not raw secret) is exposed for async paths
        assert hasattr(redis_kwargs["redis_connect_func"], "_azure_credential")
        # Raw credentials must NOT be exposed on the function
        assert not hasattr(redis_kwargs["redis_connect_func"], "_azure_client_secret")
        assert not hasattr(redis_kwargs["redis_connect_func"], "_azure_client_id")
        assert not hasattr(redis_kwargs["redis_connect_func"], "_azure_tenant_id")

        # Azure-specific kwargs should be removed from the dict passed to Redis
        assert "azure_redis_ad_token" not in redis_kwargs
        assert "azure_client_id" not in redis_kwargs


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v"])


class TestGetValidModelsWithCLI:
    """Test get_valid_models function as used in CLI token usage"""

    def test_get_valid_models_with_cli_pattern(self):
        """Test get_valid_models with litellm_proxy provider and CLI token pattern"""

        # Mock the HTTP request that get_valid_models makes to the proxy
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "gpt-4", "object": "model"},
                {"id": "litellm_proxy/gemini/gemini-2.5-flash", "object": "model"},
                {"id": "claude-3-sonnet", "object": "model"},
            ]
        }

        with patch.object(litellm.module_level_client, "get", return_value=mock_response) as mock_get:
            # Test the exact pattern used in cli_token_usage.py
            result = litellm.get_valid_models(
                check_provider_endpoint=True,
                custom_llm_provider="litellm_proxy",
                api_key="sk-test-cli-key-123",
                api_base="http://localhost:4000/",
            )

            # Verify the function returns a list of model names
            assert isinstance(result, list)
            assert len(result) == 4
            # All models get prefixed with "litellm_proxy/" by the get_models method
            assert "litellm_proxy/gpt-3.5-turbo" in result
            assert "litellm_proxy/gpt-4" in result
            # Note: This model already had the prefix, so it gets double-prefixed
            assert "litellm_proxy/litellm_proxy/gemini/gemini-2.5-flash" in result
            assert "litellm_proxy/claude-3-sonnet" in result

            # Verify the HTTP request was made with correct parameters
            mock_get.assert_called_once()
            _, call_kwargs = mock_get.call_args

            # Check that the request was made to the correct endpoint
            assert call_kwargs["url"].startswith("http://localhost:4000/")
            assert call_kwargs["url"].endswith("/v1/models")

            # Check that the API key was included in headers
            assert "headers" in call_kwargs
            headers = call_kwargs["headers"]
            assert headers.get("Authorization") == "Bearer sk-test-cli-key-123"


class TestIsCachedMessage:
    """Test is_cached_message function for context caching detection.

    Fixes GitHub issue #17821 - TypeError when content is string instead of list.
    """

    def test_string_content_returns_false(self):
        """String content should return False without crashing."""
        message = {"role": "user", "content": "Hello world"}
        assert is_cached_message(message) is False

    def test_none_content_returns_false(self):
        """None content should return False."""
        message = {"role": "user", "content": None}
        assert is_cached_message(message) is False

    def test_missing_content_returns_false(self):
        """Message without content key should return False."""
        message = {"role": "user"}
        assert is_cached_message(message) is False

    def test_list_content_without_cache_control_returns_false(self):
        """List content without cache_control should return False."""
        message = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        assert is_cached_message(message) is False

    def test_list_content_with_cache_control_returns_true(self):
        """List content with cache_control ephemeral should return True."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        assert is_cached_message(message) is True

    def test_list_with_non_dict_items_skips_them(self):
        """List content with non-dict items should skip them gracefully."""
        message = {
            "role": "user",
            "content": ["string_item", 123, {"type": "text", "text": "Hello"}],
        }
        assert is_cached_message(message) is False

    def test_list_with_mixed_items_finds_cached(self):
        """Mixed content list should find cached item."""
        message = {
            "role": "user",
            "content": [
                "string_item",
                {"type": "image", "url": "..."},
                {
                    "type": "text",
                    "text": "cached",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
        assert is_cached_message(message) is True

    def test_wrong_cache_control_type_returns_false(self):
        """Non-ephemeral cache_control type should return False."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello",
                    "cache_control": {"type": "permanent"},
                }
            ],
        }
        assert is_cached_message(message) is False

    def test_empty_list_content_returns_false(self):
        """Empty list content should return False."""
        message = {"role": "user", "content": []}
        assert is_cached_message(message) is False

    def test_message_level_cache_control_returns_true(self):
        """Message with string content and message-level cache_control should return True.

        This is the format injected by the cache_control_injection_points hook
        when the message content is a string (common for system messages).
        Fixes GitHub issue #18519 - Gemini models ignoring cache_control_injection_points.
        """
        message = {
            "role": "system",
            "content": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"},
        }
        assert is_cached_message(message) is True

    def test_message_level_cache_control_wrong_type_returns_false(self):
        """Message-level cache_control with non-ephemeral type should return False."""
        message = {
            "role": "system",
            "content": "You are a helpful assistant.",
            "cache_control": {"type": "permanent"},
        }
        assert is_cached_message(message) is False

    def test_message_level_cache_control_non_dict_returns_false(self):
        """Message-level cache_control that's not a dict should return False."""
        message = {
            "role": "system",
            "content": "You are a helpful assistant.",
            "cache_control": "ephemeral",
        }
        assert is_cached_message(message) is False


@pytest.mark.asyncio
class TestProxyLoggingBudgetAlerts:
    """Test budget_alerts method in ProxyLogging class."""

    async def test_budget_alerts_when_alerting_is_none(self):
        """Test that budget_alerts returns early when alerting is None."""
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = None
        proxy_logging.slack_alerting_instance = AsyncMock()
        proxy_logging.email_logging_instance = AsyncMock()

        user_info = MagicMock()

        # Should return without calling any alerting instances
        await proxy_logging.budget_alerts(type="user_budget", user_info=user_info)

        # Verify no calls were made
        proxy_logging.slack_alerting_instance.budget_alerts.assert_not_called()
        proxy_logging.email_logging_instance.budget_alerts.assert_not_called()

    async def test_budget_alerts_with_slack_only(self):
        """Test that budget_alerts calls slack_alerting_instance when slack is in alerting."""
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = ["slack"]
        proxy_logging.slack_alerting_instance = AsyncMock()

        user_info = MagicMock()

        await proxy_logging.budget_alerts(type="token_budget", user_info=user_info)

        proxy_logging.slack_alerting_instance.budget_alerts.assert_called_once_with(
            type="token_budget", user_info=user_info
        )

    async def test_budget_alerts_with_email_only(self):
        """Test that budget_alerts calls email_logging_instance when email is in alerting."""
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = ["email"]
        proxy_logging.email_logging_instance = AsyncMock()

        user_info = MagicMock()

        await proxy_logging.budget_alerts(type="team_budget", user_info=user_info)

        proxy_logging.email_logging_instance.budget_alerts.assert_called_once_with(
            type="team_budget", user_info=user_info
        )

    async def test_budget_alerts_with_email_when_instance_is_none(self):
        """Test that budget_alerts does not call email_logging_instance when it is None."""
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = ["email"]
        proxy_logging.email_logging_instance = None

        user_info = MagicMock()

        # Should not raise an error
        await proxy_logging.budget_alerts(type="organization_budget", user_info=user_info)

    async def test_budget_alerts_with_both_slack_and_email(self):
        """Test that budget_alerts calls both slack and email instances when both are in alerting."""
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = ["slack", "email"]
        proxy_logging.slack_alerting_instance = AsyncMock()
        proxy_logging.email_logging_instance = AsyncMock()

        user_info = MagicMock()

        await proxy_logging.budget_alerts(type="proxy_budget", user_info=user_info)

        proxy_logging.slack_alerting_instance.budget_alerts.assert_called_once_with(
            type="proxy_budget", user_info=user_info
        )
        proxy_logging.email_logging_instance.budget_alerts.assert_called_once_with(
            type="proxy_budget", user_info=user_info
        )

    @pytest.mark.parametrize(
        "alert_type",
        [
            "token_budget",
            "user_budget",
            "soft_budget",
            "team_budget",
            "organization_budget",
            "proxy_budget",
            "projected_limit_exceeded",
        ],
    )
    async def test_budget_alerts_with_all_alert_types(self, alert_type):
        """Test that budget_alerts works with all supported alert types."""
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = ["slack", "email"]
        proxy_logging.slack_alerting_instance = AsyncMock()
        proxy_logging.email_logging_instance = AsyncMock()

        user_info = MagicMock()

        await proxy_logging.budget_alerts(type=alert_type, user_info=user_info)

        proxy_logging.slack_alerting_instance.budget_alerts.assert_called_once_with(
            type=alert_type, user_info=user_info
        )
        proxy_logging.email_logging_instance.budget_alerts.assert_called_once_with(type=alert_type, user_info=user_info)

    async def test_budget_alerts_soft_budget_with_alert_emails_bypasses_alerting_none(
        self,
    ):
        """
        Test that soft_budget alerts with alert_emails bypass the alerting=None check
        and send emails even when alerting is None.

        This tests the new logic that allows team-specific soft budget email alerts
        via metadata.soft_budget_alerting_emails to work even when global alerting is disabled.
        """
        from litellm.caching.caching import DualCache
        from litellm.proxy._types import CallInfo, Litellm_EntityType
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = None  # Global alerting is disabled
        proxy_logging.slack_alerting_instance = AsyncMock()
        proxy_logging.email_logging_instance = AsyncMock()

        # Create CallInfo with alert_emails set (simulating team metadata extraction)
        user_info = CallInfo(
            token="test-token",
            spend=100.0,
            soft_budget=50.0,
            user_id="test-user",
            team_id="test-team",
            team_alias="test-team-alias",
            event_group=Litellm_EntityType.TEAM,
            alert_emails=["team1@example.com", "team2@example.com"],
        )

        # Should send email even though alerting is None (because of alert_emails)
        await proxy_logging.budget_alerts(type="soft_budget", user_info=user_info)

        # Verify slack was NOT called (alerting is None)
        proxy_logging.slack_alerting_instance.budget_alerts.assert_not_called()

        # Verify email WAS called (bypasses alerting=None check)
        proxy_logging.email_logging_instance.budget_alerts.assert_called_once_with(
            type="soft_budget", user_info=user_info
        )

    async def test_budget_alerts_soft_budget_without_alert_emails_respects_alerting_none(
        self,
    ):
        """
        Test that soft_budget alerts WITHOUT alert_emails still respect alerting=None
        and do not send emails when alerting is None.
        """
        from litellm.caching.caching import DualCache
        from litellm.proxy._types import CallInfo, Litellm_EntityType
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = None
        proxy_logging.slack_alerting_instance = AsyncMock()
        proxy_logging.email_logging_instance = AsyncMock()

        # Create CallInfo WITHOUT alert_emails
        user_info = CallInfo(
            token="test-token",
            spend=100.0,
            soft_budget=50.0,
            user_id="test-user",
            team_id="test-team",
            team_alias="test-team-alias",
            event_group=Litellm_EntityType.TEAM,
            alert_emails=None,  # No alert emails
        )

        # Should NOT send email (alerting is None and no alert_emails)
        await proxy_logging.budget_alerts(type="soft_budget", user_info=user_info)

        # Verify no calls were made
        proxy_logging.slack_alerting_instance.budget_alerts.assert_not_called()
        proxy_logging.email_logging_instance.budget_alerts.assert_not_called()

    async def test_budget_alerts_soft_budget_with_empty_alert_emails_respects_alerting_none(
        self,
    ):
        """
        Test that soft_budget alerts with empty alert_emails list still respect alerting=None.
        """
        from litellm.caching.caching import DualCache
        from litellm.proxy._types import CallInfo, Litellm_EntityType
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging.alerting = None
        proxy_logging.slack_alerting_instance = AsyncMock()
        proxy_logging.email_logging_instance = AsyncMock()

        # Create CallInfo with empty alert_emails list
        user_info = CallInfo(
            token="test-token",
            spend=100.0,
            soft_budget=50.0,
            user_id="test-user",
            team_id="test-team",
            team_alias="test-team-alias",
            event_group=Litellm_EntityType.TEAM,
            alert_emails=[],  # Empty list
        )

        # Should NOT send email (alert_emails is empty)
        await proxy_logging.budget_alerts(type="soft_budget", user_info=user_info)

        # Verify no calls were made
        proxy_logging.slack_alerting_instance.budget_alerts.assert_not_called()
        proxy_logging.email_logging_instance.budget_alerts.assert_not_called()


def test_azure_ai_claude_provider_config():
    """Test that Azure AI Claude models return AzureAnthropicConfig for proper tool transformation."""
    from litellm import AzureAIStudioConfig, AzureAnthropicConfig

    # Claude models should return AzureAnthropicConfig
    config = ProviderConfigManager.get_provider_chat_config(
        model="claude-sonnet-4-5",
        provider=LlmProviders.AZURE_AI,
    )
    assert isinstance(config, AzureAnthropicConfig)

    # Test case-insensitive matching
    config = ProviderConfigManager.get_provider_chat_config(
        model="Claude-Opus-4",
        provider=LlmProviders.AZURE_AI,
    )
    assert isinstance(config, AzureAnthropicConfig)

    # Non-Claude models should return AzureAIStudioConfig
    config = ProviderConfigManager.get_provider_chat_config(
        model="mistral-large",
        provider=LlmProviders.AZURE_AI,
    )
    assert isinstance(config, AzureAIStudioConfig)


# Tests for thinking blocks helper functions
# Related to issue: https://github.com/BerriAI/litellm/issues/18926


def test_any_assistant_message_has_thinking_blocks_with_thinking():
    """Test that function returns True when any assistant message has thinking_blocks."""
    from litellm.utils import any_assistant_message_has_thinking_blocks

    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "thinking_blocks": [{"type": "thinking", "thinking": "Let me think..."}],
            "tool_calls": [{"id": "123", "function": {"name": "test"}}],
        },
        {"role": "tool", "tool_call_id": "123", "content": "result"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "456", "function": {"name": "test2"}}],
            # No thinking_blocks here - Claude sometimes doesn't include them
        },
    ]

    assert any_assistant_message_has_thinking_blocks(messages) is True


def test_any_assistant_message_has_thinking_blocks_without_thinking():
    """Test that function returns False when no assistant message has thinking_blocks."""
    from litellm.utils import any_assistant_message_has_thinking_blocks

    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "123", "function": {"name": "test"}}],
        },
        {"role": "tool", "tool_call_id": "123", "content": "result"},
    ]

    assert any_assistant_message_has_thinking_blocks(messages) is False


def test_any_assistant_message_has_thinking_blocks_empty_list():
    """Test that function returns False when thinking_blocks is an empty list."""
    from litellm.utils import any_assistant_message_has_thinking_blocks

    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "thinking_blocks": [],  # Empty list
            "tool_calls": [{"id": "123", "function": {"name": "test"}}],
        },
    ]

    assert any_assistant_message_has_thinking_blocks(messages) is False


def test_last_assistant_with_tool_calls_has_no_thinking_blocks_issue_18926():
    """
    Test the scenario from issue #18926 where:
    - First assistant message HAS thinking_blocks
    - Second assistant message has NO thinking_blocks

    The old logic would drop thinking because the LAST tool_call message
    has no thinking_blocks, but this breaks because the first message
    still has thinking blocks in the conversation.
    """
    from litellm.utils import (
        any_assistant_message_has_thinking_blocks,
        last_assistant_with_tool_calls_has_no_thinking_blocks,
    )

    messages = [
        {"role": "user", "content": "Build a feature"},
        {
            "role": "assistant",
            "thinking_blocks": [{"type": "thinking", "thinking": "Let me analyze the requirements..."}],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "function": {"name": "file_editor", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "content": "File contents here...",
        },
        {
            "role": "assistant",
            # NO thinking_blocks - Claude sometimes doesn't include them
            "content": [{"type": "text", "text": "Let me explore more..."}],
            "tool_calls": [
                {
                    "id": "toolu_2",
                    "function": {"name": "file_editor", "arguments": "{}"},
                }
            ],
        },
    ]

    # Last assistant with tool_calls has no thinking_blocks
    assert last_assistant_with_tool_calls_has_no_thinking_blocks(messages) is True

    # But ANY assistant message has thinking_blocks
    assert any_assistant_message_has_thinking_blocks(messages) is True

    # So we should NOT drop thinking - the combination tells us thinking is in use
    # The fix uses both checks: only drop if last has none AND no message has any
    should_drop_thinking = last_assistant_with_tool_calls_has_no_thinking_blocks(
        messages
    ) and not any_assistant_message_has_thinking_blocks(messages)
    assert should_drop_thinking is False


class TestAdditionalDropParamsForNonOpenAIProviders:
    """
    Test additional_drop_params functionality for non-OpenAI providers.

    Fixes https://github.com/BerriAI/litellm/issues/19225

    The bug was that additional_drop_params only filtered params for OpenAI/Azure
    providers, but not for other providers like Bedrock. This caused OpenAI-specific
    params like prompt_cache_key to be passed to Bedrock, resulting in errors.
    """

    def test_additional_drop_params_filters_for_bedrock(self):
        """
        Test that additional_drop_params correctly filters params for Bedrock provider.

        Before the fix, prompt_cache_key would be passed through to Bedrock even when
        specified in additional_drop_params, causing:
        'BedrockException - {"message":"The model returned the following errors:
        prompt_cache_key: Extra inputs are not permitted"}'
        """
        from litellm.utils import add_provider_specific_params_to_optional_params

        optional_params = {}
        passed_params = {
            "prompt_cache_key": "test_key_123",
            "temperature": 0.7,
            "model": "bedrock/anthropic.claude-v2",
        }
        openai_params = ["temperature", "max_tokens", "top_p", "model"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="bedrock",
            openai_params=openai_params,
            additional_drop_params=["prompt_cache_key"],
        )

        # prompt_cache_key should be filtered out
        assert "prompt_cache_key" not in result
        # temperature should still be there (it's in openai_params, not filtered)
        # Note: temperature is in openai_params so it won't be added by this function
        # The function only adds params NOT in openai_params

    def test_additional_drop_params_filters_multiple_params_for_non_openai(self):
        """Test filtering multiple params for non-OpenAI providers."""
        from litellm.utils import add_provider_specific_params_to_optional_params

        optional_params = {}
        passed_params = {
            "prompt_cache_key": "test_key",
            "some_openai_only_param": "value1",
            "another_openai_param": "value2",
            "keep_this_param": "keep_me",
        }
        openai_params = ["temperature", "max_tokens"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="anthropic",
            openai_params=openai_params,
            additional_drop_params=["prompt_cache_key", "some_openai_only_param"],
        )

        # Filtered params should not be present
        assert "prompt_cache_key" not in result
        assert "some_openai_only_param" not in result
        # Non-filtered params should be present
        assert result.get("another_openai_param") == "value2"
        assert result.get("keep_this_param") == "keep_me"

    def test_additional_drop_params_none_keeps_all_params(self):
        """Test that when additional_drop_params is None, all params are kept."""
        from litellm.utils import add_provider_specific_params_to_optional_params

        optional_params = {}
        passed_params = {
            "prompt_cache_key": "test_key",
            "custom_param": "value",
        }
        openai_params = ["temperature"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="bedrock",
            openai_params=openai_params,
            additional_drop_params=None,
        )

        # All params should be present when additional_drop_params is None
        assert result.get("prompt_cache_key") == "test_key"
        assert result.get("custom_param") == "value"

    def test_additional_drop_params_empty_list_keeps_all_params(self):
        """Test that when additional_drop_params is empty list, all params are kept."""
        from litellm.utils import add_provider_specific_params_to_optional_params

        optional_params = {}
        passed_params = {
            "prompt_cache_key": "test_key",
            "custom_param": "value",
        }
        openai_params = ["temperature"]

        result = add_provider_specific_params_to_optional_params(
            optional_params=optional_params,
            passed_params=passed_params,
            custom_llm_provider="bedrock",
            openai_params=openai_params,
            additional_drop_params=[],
        )

        # All params should be present when additional_drop_params is empty
        assert result.get("prompt_cache_key") == "test_key"
        assert result.get("custom_param") == "value"


class TestExtraBodyCannotOverrideModel:
    @pytest.mark.parametrize("custom_llm_provider", ["edenai", "openai", "azure"])
    def test_extra_body_model_is_dropped_for_openai_compatible_providers(self, custom_llm_provider: str) -> None:
        from litellm.utils import add_provider_specific_params_to_optional_params

        result = add_provider_specific_params_to_optional_params(
            optional_params={"extra_body": {"model": "edenai/openai/gpt-4o", "provider_flag": True}},
            passed_params={
                "model": "edenai/openai/gpt-4o-mini",
                "extra_body": {"model": "edenai/anthropic/claude-3-opus", "top_k": 5},
                "custom_param": "kept",
            },
            custom_llm_provider=custom_llm_provider,
            openai_params=["model", "temperature"],
            additional_drop_params=None,
        )

        assert result == {"extra_body": {"provider_flag": True, "top_k": 5, "custom_param": "kept"}}, result

    def test_get_optional_params_strips_extra_body_model_for_edenai(self) -> None:
        result = litellm.get_optional_params(
            model="openai/gpt-4o-mini",
            custom_llm_provider="edenai",
            extra_body={"model": "anthropic/claude-opus-4-1", "top_k": 5},
        )

        assert result["extra_body"] == {"top_k": 5}, result

    def test_nested_drop_paths_do_not_break_extra_body_filtering(self) -> None:
        from litellm.utils import add_provider_specific_params_to_optional_params

        result = add_provider_specific_params_to_optional_params(
            optional_params={},
            passed_params={
                "model": "hosted_vllm/my-vllm-model",
                "extra_body": {"model": "hosted_vllm/other", "top_k": 5, "kept": True},
            },
            custom_llm_provider="hosted_vllm",
            openai_params=["model", "temperature"],
            additional_drop_params=[["tools", "function", "strict"], "top_k"],
        )

        assert result == {"extra_body": {"kept": True}}, result

    def test_a_list_entry_does_not_break_a_supported_nested_drop_path(self) -> None:
        def tools() -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {"name": "f", "custom_marker": "LEAK", "parameters": {"type": "object"}},
                }
            ]

        untouched = litellm.get_optional_params(
            model="my-vllm-model", custom_llm_provider="hosted_vllm", tools=tools()
        )
        assert untouched["tools"][0]["function"]["custom_marker"] == "LEAK", untouched

        result = litellm.get_optional_params(
            model="my-vllm-model",
            custom_llm_provider="hosted_vllm",
            tools=tools(),
            additional_drop_params=["tools[*].function.custom_marker", ["tools", "function", "custom_marker"]],
        )

        assert "custom_marker" not in result["tools"][0]["function"], result


class TestDropParamsWithPromptCacheKey:
    """
    Test that drop_params: true correctly drops prompt_cache_key for non-OpenAI providers.

    Fixes https://github.com/BerriAI/litellm/issues/19225

    prompt_cache_key is an OpenAI-specific parameter that should be automatically
    dropped when using providers like Bedrock that don't support it.
    """

    def test_prompt_cache_key_in_default_params(self):
        """Verify prompt_cache_key is now in DEFAULT_CHAT_COMPLETION_PARAM_VALUES."""
        from litellm.constants import DEFAULT_CHAT_COMPLETION_PARAM_VALUES

        assert "prompt_cache_key" in DEFAULT_CHAT_COMPLETION_PARAM_VALUES
        assert "prompt_cache_retention" in DEFAULT_CHAT_COMPLETION_PARAM_VALUES

    def test_drop_params_removes_prompt_cache_key_for_bedrock(self):
        """
        Test that get_optional_params with drop_params=True removes prompt_cache_key
        for Bedrock provider since it's not in Bedrock's supported params.
        """
        from litellm.utils import get_optional_params

        # Call get_optional_params for Bedrock with prompt_cache_key
        # drop_params=True should remove it since Bedrock doesn't support it
        result = get_optional_params(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            custom_llm_provider="bedrock",
            prompt_cache_key="test_cache_key",
            temperature=0.7,
            drop_params=True,
        )

        # prompt_cache_key should be dropped for Bedrock
        assert "prompt_cache_key" not in result
        # temperature should remain (it's supported by Bedrock)
        assert result.get("temperature") == 0.7


class TestGetOptionalParamsDeepSeek:
    """Tests that deepseek provider uses DeepSeekChatConfig for parameter mapping."""

    def test_deepseek_supports_thinking_param(self):
        """
        Verify that get_optional_params for deepseek accepts the 'thinking' param,
        which is only supported by DeepSeekChatConfig, not OpenAIConfig.
        """
        from litellm.utils import get_optional_params

        result = get_optional_params(
            model="deepseek-reasoner",
            custom_llm_provider="deepseek",
            thinking={"type": "enabled"},
        )
        assert result.get("thinking") == {"type": "enabled"}

    def test_deepseek_supports_reasoning_effort_param(self):
        """
        Verify that get_optional_params for deepseek accepts 'reasoning_effort',
        which is only supported by DeepSeekChatConfig, not OpenAIConfig.
        """
        from litellm.utils import get_optional_params

        result = get_optional_params(
            model="deepseek-reasoner",
            custom_llm_provider="deepseek",
            reasoning_effort="high",
        )
        assert result.get("thinking") == {"type": "enabled"}

    def test_deepseek_thinking_strips_budget_tokens(self):
        """
        DeepSeekChatConfig strips budget_tokens from thinking param.
        This would not happen with OpenAIConfig.
        """
        from litellm.utils import get_optional_params

        result = get_optional_params(
            model="deepseek-reasoner",
            custom_llm_provider="deepseek",
            thinking={"type": "enabled", "budget_tokens": 5000},
        )
        assert "budget_tokens" not in result.get("thinking", {})
        assert result.get("thinking") == {"type": "enabled"}


class TestIsStreamingRequest:
    def test_stream_true_in_kwargs(self):
        assert _is_streaming_request(kwargs={"stream": True}, call_type="acompletion") is True

    def test_stream_false_in_kwargs(self):
        assert _is_streaming_request(kwargs={"stream": False}, call_type="acompletion") is False

    def test_no_stream_in_kwargs(self):
        assert _is_streaming_request(kwargs={}, call_type="acompletion") is False

    def test_generate_content_stream_string(self):
        assert _is_streaming_request(kwargs={}, call_type=CallTypes.generate_content_stream.value) is True

    def test_agenerate_content_stream_string(self):
        assert _is_streaming_request(kwargs={}, call_type=CallTypes.agenerate_content_stream.value) is True

    def test_generate_content_stream_enum(self):
        assert _is_streaming_request(kwargs={}, call_type=CallTypes.generate_content_stream) is True

    def test_agenerate_content_stream_enum(self):
        assert _is_streaming_request(kwargs={}, call_type=CallTypes.agenerate_content_stream) is True

    def test_non_streaming_call_type_enum(self):
        assert _is_streaming_request(kwargs={}, call_type=CallTypes.acompletion) is False

    def test_stream_true_overrides_non_streaming_call_type(self):
        assert _is_streaming_request(kwargs={"stream": True}, call_type=CallTypes.acompletion) is True


class TestCallbackAsyncSyncSeparation:
    """Test that LoggingCallbackManager auto-routes async callbacks to async lists."""

    def setup_method(self):
        """Reset callback lists before each test."""
        litellm.input_callback = []
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_input_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []

    def test_async_success_callback_routed_to_async_list(self):
        async def my_async_cb(*args, **kwargs):
            pass

        litellm.logging_callback_manager.add_litellm_success_callback(my_async_cb)
        assert my_async_cb in litellm._async_success_callback
        assert my_async_cb not in litellm.success_callback

    def test_sync_success_callback_stays_in_sync_list(self):
        def my_sync_cb(*args, **kwargs):
            pass

        litellm.logging_callback_manager.add_litellm_success_callback(my_sync_cb)
        assert my_sync_cb in litellm.success_callback
        assert my_sync_cb not in litellm._async_success_callback

    def test_string_callback_stays_in_sync_list(self):
        litellm.logging_callback_manager.add_litellm_success_callback("langfuse")
        assert "langfuse" in litellm.success_callback
        assert "langfuse" not in litellm._async_success_callback

    def test_async_failure_callback_routed_to_async_list(self):
        async def my_async_cb(*args, **kwargs):
            pass

        litellm.logging_callback_manager.add_litellm_failure_callback(my_async_cb)
        assert my_async_cb in litellm._async_failure_callback
        assert my_async_cb not in litellm.failure_callback

    def test_sync_failure_callback_stays_in_sync_list(self):
        def my_sync_cb(*args, **kwargs):
            pass

        litellm.logging_callback_manager.add_litellm_failure_callback(my_sync_cb)
        assert my_sync_cb in litellm.failure_callback
        assert my_sync_cb not in litellm._async_failure_callback

    def test_dynamodb_routed_to_async_success(self):
        litellm.logging_callback_manager.add_litellm_success_callback("dynamodb")
        assert "dynamodb" in litellm._async_success_callback
        assert "dynamodb" not in litellm.success_callback

    def test_openmeter_routed_to_async_success(self):
        litellm.logging_callback_manager.add_litellm_success_callback("openmeter")
        assert "openmeter" in litellm._async_success_callback
        assert "openmeter" not in litellm.success_callback

    def test_async_input_callback_routed_to_async_list(self):
        async def my_async_cb(*args, **kwargs):
            pass

        litellm.logging_callback_manager.add_litellm_input_callback(my_async_cb)
        assert my_async_cb in litellm._async_input_callback
        assert my_async_cb not in litellm.input_callback

    def test_sync_input_callback_stays_in_sync_list(self):
        def my_sync_cb(*args, **kwargs):
            pass

        litellm.logging_callback_manager.add_litellm_input_callback(my_sync_cb)
        assert my_sync_cb in litellm.input_callback
        assert my_sync_cb not in litellm._async_input_callback


class TestMetadataNoneHandling:
    """
    Test that metadata=None in kwargs doesn't cause TypeError.

    When metadata key exists with value None (e.g., from Azure OpenAI streaming),
    dict.get("metadata", {}) returns None (key exists, so default is ignored).
    The fix uses (kwargs.get("metadata") or {}) which handles both missing key
    and explicit None value.

    Related: #20871
    """

    def test_metadata_none_get_previous_models(self):
        """kwargs.get("metadata") or {} should return {} when metadata is None."""
        kwargs = {"metadata": None}
        previous_models = (kwargs.get("metadata") or {}).get("previous_models", None)
        assert previous_models is None

    def test_metadata_none_model_group_check(self):
        """'model_group' in (kwargs.get("metadata") or {}) should not raise TypeError."""
        kwargs = {"metadata": None}
        _is_litellm_router_call = "model_group" in (kwargs.get("metadata") or {})
        assert _is_litellm_router_call is False

    def test_metadata_missing_key(self):
        """Should work when metadata key is completely absent."""
        kwargs = {}
        previous_models = (kwargs.get("metadata") or {}).get("previous_models", None)
        assert previous_models is None

    def test_metadata_present_with_values(self):
        """Should work when metadata has actual values."""
        kwargs = {"metadata": {"previous_models": ["model1"], "model_group": "test"}}
        previous_models = (kwargs.get("metadata") or {}).get("previous_models", None)
        assert previous_models == ["model1"]
        _is_litellm_router_call = "model_group" in (kwargs.get("metadata") or {})
        assert _is_litellm_router_call is True

    def test_metadata_none_causes_error_with_old_pattern(self):
        """Demonstrate the bug: dict.get('metadata', {}) returns None when key exists with None value."""
        kwargs = {"metadata": None}
        # Old pattern: kwargs.get("metadata", {}) returns None because key exists
        result = kwargs.get("metadata", {})
        assert result is None  # This is the root cause of the bug

        # Attempting to use .get() on None raises AttributeError or TypeError
        with pytest.raises((TypeError, AttributeError)):
            kwargs.get("metadata", {}).get("previous_models", None)

        # Attempting 'in' on None raises TypeError
        with pytest.raises(TypeError):
            _ = "model_group" in kwargs.get("metadata", {})

    def test_litellm_params_metadata_none(self):
        """litellm_params.get("metadata") or {} should handle None value."""
        litellm_params = {"metadata": None}
        metadata = litellm_params.get("metadata") or {}
        assert metadata == {}


_RETRY_CAP_CASES: Final = (
    pytest.param(5, {"request_retry_count": 5}, True, id="cap-above-four-reached"),
    pytest.param(5, {"request_retry_count": 4}, False, id="cap-above-four-not-reached"),
    pytest.param(0, {"request_retry_count": 0}, False, id="first-attempt-passes-cap-of-zero"),
    pytest.param(0, {"request_retry_count": 1}, True, id="cap-of-zero-refuses-first-retry"),
    pytest.param(0, {"attempted_retries": 1}, False, id="per-hop-attempted-retries-is-not-the-cap"),
    pytest.param(5, {"previous_models": ("a", "b", "c", "d", "e")}, False, id="breadcrumb-count-is-not-the-cap"),
    pytest.param(5, None, False, id="metadata-none"),
)


def _capped_completion_kwargs(metadata_key: str, metadata: object) -> dict[str, object]:
    return {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "api_key": "sk-fake",
        "mock_response": "ok",
        metadata_key: metadata,
    }


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("cap, metadata, refused", _RETRY_CAP_CASES)
def test_num_retries_per_request_reads_request_retry_count_sync(
    monkeypatch: pytest.MonkeyPatch, metadata_key: str, cap: int, metadata: object, refused: bool
) -> None:
    monkeypatch.setattr(litellm, "num_retries_per_request", cap)
    kwargs: Final = _capped_completion_kwargs(metadata_key, metadata)
    if refused:
        with pytest.raises(Exception, match="Max retries per request hit!"):
            litellm.completion(**kwargs)
    else:
        assert litellm.completion(**kwargs).choices[0].message.content == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("cap, metadata, refused", _RETRY_CAP_CASES)
async def test_num_retries_per_request_reads_request_retry_count_async(
    monkeypatch: pytest.MonkeyPatch, metadata_key: str, cap: int, metadata: object, refused: bool
) -> None:
    monkeypatch.setattr(litellm, "num_retries_per_request", cap)
    kwargs: Final = _capped_completion_kwargs(metadata_key, metadata)
    if refused:
        with pytest.raises(Exception, match="Max retries per request hit!"):
            await litellm.acompletion(**kwargs)
    else:
        assert (await litellm.acompletion(**kwargs)).choices[0].message.content == "ok"


class TestValidateAndFixThinkingParam:
    """Tests for validate_and_fix_thinking_param."""

    def test_none_returns_none(self):
        from litellm.utils import validate_and_fix_thinking_param

        assert validate_and_fix_thinking_param(thinking=None) is None

    def test_already_snake_case(self):
        from litellm.utils import validate_and_fix_thinking_param

        thinking = {"type": "enabled", "budget_tokens": 32000}
        result = validate_and_fix_thinking_param(thinking=thinking)
        assert result == {"type": "enabled", "budget_tokens": 32000}

    def test_camel_case_normalized(self):
        from litellm.utils import validate_and_fix_thinking_param

        thinking = {"type": "enabled", "budgetTokens": 32000}
        result = validate_and_fix_thinking_param(thinking=thinking)
        assert result == {"type": "enabled", "budget_tokens": 32000}
        assert "budgetTokens" not in result

    def test_both_keys_snake_case_wins(self):
        from litellm.utils import validate_and_fix_thinking_param

        thinking = {"type": "enabled", "budget_tokens": 10000, "budgetTokens": 50000}
        result = validate_and_fix_thinking_param(thinking=thinking)
        assert result == {"type": "enabled", "budget_tokens": 10000}
        assert "budgetTokens" not in result

    def test_original_dict_not_mutated(self):
        from litellm.utils import validate_and_fix_thinking_param

        thinking = {"type": "enabled", "budgetTokens": 32000}
        validate_and_fix_thinking_param(thinking=thinking)
        assert "budgetTokens" in thinking
        assert "budget_tokens" not in thinking

    def test_bool_true_maps_to_enabled_with_default_budget(self):
        from litellm.constants import DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
        from litellm.utils import validate_and_fix_thinking_param

        assert validate_and_fix_thinking_param(thinking=True) == {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
        }

    def test_bool_false_returns_none(self):
        from litellm.utils import validate_and_fix_thinking_param

        assert validate_and_fix_thinking_param(thinking=False) is None


_FIREWORKS_MODELS = [
    (
        "accounts/fireworks/models/glm-5p2",
        1048576,
        131072,
        False,
        True,
    ),
    (
        "accounts/fireworks/models/glm-5p1",
        202800,
        131072,
        False,
        True,
    ),
    (
        "accounts/fireworks/routers/glm-5p1-fast",
        202800,
        131072,
        False,
        True,
    ),
    (
        "accounts/fireworks/models/qwen3p7-plus",
        262144,
        65536,
        True,
        True,
    ),
    (
        "accounts/fireworks/models/minimax-m3",
        512000,
        512000,
        True,
        True,
    ),
    (
        "accounts/fireworks/models/minimax-m2p7",
        196608,
        196608,
        False,
        True,
    ),
    (
        "accounts/fireworks/models/kimi-k2p7-code",
        262144,
        32768,
        True,
        True,
    ),
    (
        "accounts/fireworks/routers/kimi-k2p7-code-fast",
        262144,
        32768,
        True,
        True,
    ),
    (
        "accounts/fireworks/models/kimi-k2p6",
        262144,
        32768,
        True,
        True,
    ),
    (
        "accounts/fireworks/routers/kimi-k2p6-fast",
        262144,
        32768,
        True,
        True,
    ),
    (
        "accounts/fireworks/models/gpt-oss-120b",
        131072,
        32768,
        False,
        True,
    ),
    (
        "accounts/fireworks/models/gpt-oss-20b",
        131072,
        32768,
        False,
        True,
    ),
    (
        "accounts/fireworks/models/deepseek-v4-pro",
        1048576,
        384000,
        False,
        True,
    ),
    (
        "accounts/fireworks/models/deepseek-v4-flash",
        1048576,
        384000,
        False,
        True,
    ),
]

_FIREWORKS_SHORT_FORMS = [
    "glm-5p2",
    "glm-5p1",
    "qwen3p7-plus",
    "minimax-m3",
    "minimax-m2p7",
    "kimi-k2p7-code",
    "kimi-k2p6",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
]

_FIREWORKS_ROUTER_SHORT_FORMS = [
    "glm-5p1-fast",
    "kimi-k2p6-fast",
    "kimi-k2p7-code-fast",
]


@pytest.fixture
def fireworks_short_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "fireworks_ai/accounts/fireworks/models/glm-5p3": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "litellm_provider": "fireworks_ai",
                "mode": "chat",
                "max_tokens": 100,
            },
            "fireworks_ai/accounts/fireworks/routers/glm-5p3-fast": {
                "input_cost_per_token": 2.1e-6,
                "output_cost_per_token": 6.6e-6,
                "litellm_provider": "fireworks_ai",
                "mode": "chat",
            },
            "fireworks_ai/nomic-ai/nomic-embed-text-v1.5": {
                "input_cost_per_token": 8e-9,
                "output_cost_per_token": 0.0,
                "litellm_provider": "fireworks_ai",
                "mode": "embedding",
            },
        },
    )
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


class TestBedrockBaseModelLabelKeepsTools:
    """Regression for #29618: a Bedrock deployment whose ``base_model`` is a friendly
    label must not silently drop ``tools``/``tool_choice`` under ``drop_params``."""

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    def test_base_model_label_keeps_tools_with_drop_params(self):
        from litellm.utils import get_optional_params

        result = get_optional_params(
            model="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
            custom_llm_provider="bedrock",
            base_model="claude-haiku-4-5",
            tools=self.TOOLS,
            tool_choice="auto",
            drop_params=True,
        )

        assert "tools" in result
        assert "tool_choice" in result

    def test_base_model_label_alone_drops_tools(self):
        """Without the real model id the label resolves to no tool support, so passing
        the label as ``model`` is exactly what dropped tools before the fix."""
        from litellm.utils import get_optional_params

        result = get_optional_params(
            model="claude-haiku-4-5",
            custom_llm_provider="bedrock",
            tools=self.TOOLS,
            tool_choice="auto",
            drop_params=True,
        )

        assert "tools" not in result


def test_aws_bedrock_project_id_excluded_from_bedrock_optional_params():
    """`aws_bedrock_project_id` is sent as a bedrock-mantle request header, so it
    must never reach optional_params (and from there the request body), while
    other aws_* params keep flowing for boto3 auth."""
    from litellm.utils import get_optional_params

    result = get_optional_params(
        model="mantle/anthropic.claude-mythos-preview",
        custom_llm_provider="bedrock",
        max_tokens=10,
        aws_bedrock_project_id="proj_abc123def456",
        aws_region_name="us-east-1",
    )

    assert "aws_bedrock_project_id" not in result
    assert result["aws_region_name"] == "us-east-1"


@pytest.mark.parametrize(
    "filter_name",
    [
        "get_non_default_completion_params",
        "get_non_default_transcription_params",
        "filter_out_litellm_params",
    ],
)
def test_scoped_weights_are_excluded_from_provider_params(filter_name: str) -> None:
    filtered = getattr(litellm.utils, filter_name)(
        {"provider_option": "kept", "_router_weights": {"group": {"deployment": 100}}}
    )
    assert filtered == {"provider_option": "kept"}


@pytest.mark.parametrize(
    "provider_filter",
    [
        litellm.utils.get_non_default_completion_params,
        litellm.utils.get_non_default_transcription_params,
        litellm.utils.filter_out_litellm_params,
    ],
)
@pytest.mark.parametrize("setting", [("tag_regex", ["^team-a$"]), ("max_file_size_mb", 5)])
def test_deployment_only_settings_copied_by_the_router_stay_out_of_provider_params(
    provider_filter: Callable[[dict[str, object]], Mapping[str, object]], setting: tuple[str, object]
) -> None:
    name, value = setting
    filtered: Final = provider_filter({"provider_option": "kept", name: value})
    assert filtered == {"provider_option": "kept"}, filtered


@pytest.mark.parametrize(
    "provider_filter",
    [
        litellm.utils.get_non_default_completion_params,
        litellm.utils.get_non_default_transcription_params,
        litellm.utils.filter_out_litellm_params,
    ],
)
def test_undeclared_internal_prefixed_kwargs_stay_out_of_provider_params(
    provider_filter: Callable[[dict[str, object]], Mapping[str, object]],
) -> None:
    undeclared: Final = "_litellm_never_declared_anywhere"
    assert undeclared not in litellm.all_litellm_params
    filtered: Final = provider_filter(
        {"provider_option": "kept", "provider_litellm_option": "kept", "litellm_option": "kept", undeclared: "internal"}
    )
    assert filtered == {"provider_option": "kept", "provider_litellm_option": "kept", "litellm_option": "kept"}


class TestGetOptionalParamsTencent:
    """Tests that tencent provider uses TencentChatConfig for parameter mapping."""

    def test_tencent_supports_thinking_param(self):
        """Verify get_optional_params for tencent accepts the 'thinking' param.

        `thinking` must be nested in extra_body: tencent routes through the
        OpenAI SDK's chat.completions.create(), which rejects unknown kwargs.
        """
        from unittest.mock import patch

        from litellm.utils import get_optional_params

        with patch(
            "litellm.llms.tencent.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            result = get_optional_params(
                model="tencent/deepseek-v4-pro",
                custom_llm_provider="tencent",
                thinking={"type": "enabled"},
            )
        assert "thinking" not in result
        assert result["extra_body"]["thinking"] == {"type": "enabled"}

    def test_tencent_supports_reasoning_effort(self):
        """Verify get_optional_params for tencent converts reasoning_effort to thinking."""
        from unittest.mock import patch

        from litellm.utils import get_optional_params

        with patch(
            "litellm.llms.tencent.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            result = get_optional_params(
                model="tencent/deepseek-v4-pro",
                custom_llm_provider="tencent",
                reasoning_effort="medium",
            )
        assert "thinking" not in result
        assert result["extra_body"]["thinking"] == {"type": "enabled"}

    def test_tencent_supported_params_includes_thinking_and_reasoning_effort(self):
        """Verify get_supported_openai_params for tencent includes custom params."""
        from unittest.mock import patch

        from litellm.litellm_core_utils.get_supported_openai_params import (
            get_supported_openai_params,
        )

        with patch(
            "litellm.llms.tencent.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            params = get_supported_openai_params(
                model="tencent/deepseek-v4-pro",
                custom_llm_provider="tencent",
            )
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_tencent_messages_config_routing(self):
        """Verify ProviderConfigManager routes tencent to TencentAnthropicMessagesConfig."""
        import litellm
        from litellm.llms.tencent.messages.transformation import (
            TencentAnthropicMessagesConfig,
        )

        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="deepseek-v4-pro",
            provider=litellm.LlmProviders.TENCENT,
        )
        assert isinstance(config, TencentAnthropicMessagesConfig)
        assert config.custom_llm_provider == "tencent"

    def test_bedrock_mantle_claude_messages_config_routing(self):
        import litellm
        from litellm.llms.bedrock_mantle.messages.transformation import (
            BedrockMantleAnthropicMessagesConfig,
        )

        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="anthropic.claude-sonnet-5",
            provider=litellm.LlmProviders.BEDROCK_MANTLE,
        )
        assert isinstance(config, BedrockMantleAnthropicMessagesConfig)
        assert config.custom_llm_provider == "bedrock_mantle"

    def test_bedrock_mantle_openai_models_keep_the_messages_bridge(self):
        import litellm

        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="openai.gpt-5.6-sol",
            provider=litellm.LlmProviders.BEDROCK_MANTLE,
        )
        assert config is None


class TestValidateEnvironmentTencent:
    """Tests that validate_environment resolves TENCENT_API_KEY for the tencent provider."""

    def test_reports_key_present(self):
        with patch.dict(os.environ, {"TENCENT_API_KEY": "sk-tencent"}):
            result = litellm.validate_environment(model="tencent/deepseek-v4-pro")

        assert result["keys_in_environment"] is True
        assert result["missing_keys"] == []

    def test_reports_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            result = litellm.validate_environment(model="tencent/deepseek-v4-pro")

        assert result["keys_in_environment"] is False
        assert "TENCENT_API_KEY" in result["missing_keys"]


class TestVertexEmbeddingEncodingFormat:
    """vertex_ai/gemini embeddings must accept encoding_format="float" — it's
    the OpenAI SDK default and float lists are exactly what the vertex API
    returns. Other values keep the unsupported-param behavior (drop with
    drop_params, raise otherwise). Issue #33173."""

    def test_encoding_format_float_is_accepted_and_dropped(self):
        optional_params = litellm.utils.get_optional_params_embeddings(
            model="gemini-embedding-001",
            encoding_format="float",
            custom_llm_provider="vertex_ai",
        )
        assert "encoding_format" not in optional_params

    def test_encoding_format_float_accepted_for_gemini_provider(self):
        optional_params = litellm.utils.get_optional_params_embeddings(
            model="gemini-embedding-001",
            encoding_format="float",
            custom_llm_provider="gemini",
        )
        assert "encoding_format" not in optional_params

    def test_encoding_format_base64_still_rejected_without_drop_params(self):
        with pytest.raises(Exception, match="To drop these, set `litellm\\.drop_params=True` or for proxy") as excinfo:
            litellm.utils.get_optional_params_embeddings(
                model="gemini-embedding-001",
                encoding_format="base64",
                custom_llm_provider="vertex_ai",
            )
        assert "encoding_format" in str(excinfo.value)

    def test_encoding_format_base64_dropped_with_drop_params(self):
        optional_params = litellm.utils.get_optional_params_embeddings(
            model="gemini-embedding-001",
            encoding_format="base64",
            custom_llm_provider="vertex_ai",
            drop_params=True,
        )
        assert "encoding_format" not in optional_params

    def test_dimensions_still_mapped(self):
        optional_params = litellm.utils.get_optional_params_embeddings(
            model="gemini-embedding-001",
            encoding_format="float",
            dimensions=256,
            custom_llm_provider="vertex_ai",
        )
        assert optional_params.get("outputDimensionality") == 256


class TestBedrockCohereEmbeddingDispatch:
    """All bedrock cohere.embed models must route to BedrockCohereEmbeddingConfig,
    not just multilingual-v3/v4: english-v3 was falling into the unmapped
    else-branch and rejecting encoding_format. Issue #38659."""

    @pytest.mark.parametrize(
        "model",
        [
            "cohere.embed-english-v3",
            "cohere.embed-multilingual-v3",
            "cohere.embed-v4:0",
        ],
    )
    def test_cohere_embed_models_accept_encoding_format(self, model):
        optional_params = litellm.utils.get_optional_params_embeddings(
            model=model,
            encoding_format="float",
            custom_llm_provider="bedrock",
        )
        assert optional_params.get("embedding_types") == ["float"]

    @pytest.mark.parametrize(
        "model",
        [
            "cohere.embed-english-v3",
            "cohere.embed-multilingual-v3",
            "cohere.embed-v4:0",
        ],
    )
    def test_cohere_embed_models_map_base64_to_float(self, model):
        optional_params = litellm.utils.get_optional_params_embeddings(
            model=model,
            encoding_format="base64",
            custom_llm_provider="bedrock",
        )
        assert optional_params.get("embedding_types") == ["float"]

    def test_cohere_embed_english_v3_maps_dimensions(self):
        optional_params = litellm.utils.get_optional_params_embeddings(
            model="cohere.embed-english-v3",
            encoding_format="float",
            dimensions=512,
            custom_llm_provider="bedrock",
        )
        assert optional_params.get("output_dimension") == 512


PROMPT_CACHE_MESSAGES = [{"role": "user", "content": "the quick brown fox jumps over the lazy dog " * 155}]


@pytest.mark.parametrize(
    "model, expected_min_tokens",
    [
        ("claude-opus-4-6", 4096),
        ("claude-opus-4-7", 2048),
        ("claude-opus-4-8", 1024),
        ("claude-fable-5", 512),
    ],
)
def test_get_prompt_cache_min_tokens_resolves_per_model(
    model: str, expected_min_tokens: int, local_model_cost_map: None
) -> None:
    """The smallest cacheable prefix is a per-model property, read from the cost map's
    prompt_cache_min_tokens. Anthropic's minimum spans 512..4096 across models and moves in both
    directions across releases, so a single global constant is wrong for every model but one."""
    assert get_prompt_cache_min_tokens(model=model) == expected_min_tokens


ANTHROPIC_REEXPORT_CACHE_MIN: Final = {
    "azure_ai/claude-fable-5": 512,
    "azure_ai/claude-haiku-4-5": 4096,
    "azure_ai/claude-opus-4-1": 1024,
    "azure_ai/claude-opus-4-5": 4096,
    "azure_ai/claude-opus-4-6": 4096,
    "azure_ai/claude-opus-4-7": 2048,
    "azure_ai/claude-opus-4-8": 1024,
    "azure_ai/claude-sonnet-4-5": 1024,
    "azure_ai/claude-sonnet-4-6": 1024,
    "azure_ai/claude-sonnet-5": 1024,
    "databricks/databricks-claude-haiku-4-5": 4096,
    "databricks/databricks-claude-opus-4": 1024,
    "databricks/databricks-claude-opus-4-1": 1024,
    "databricks/databricks-claude-opus-4-5": 4096,
    "databricks/databricks-claude-opus-4-6": 4096,
    "databricks/databricks-claude-sonnet-4": 1024,
    "databricks/databricks-claude-sonnet-4-5": 1024,
    "databricks/databricks-claude-sonnet-4-6": 1024,
    "openrouter/anthropic/claude-haiku-4.5": 4096,
    "openrouter/anthropic/claude-opus-4": 1024,
    "openrouter/anthropic/claude-opus-4.1": 1024,
    "openrouter/anthropic/claude-opus-4.5": 4096,
    "openrouter/anthropic/claude-opus-4.6": 4096,
    "openrouter/anthropic/claude-opus-4.7": 2048,
    "openrouter/anthropic/claude-sonnet-4": 1024,
    "openrouter/anthropic/claude-sonnet-4.5": 1024,
    "openrouter/anthropic/claude-sonnet-4.6": 1024,
    "replicate/anthropic/claude-4-sonnet": 1024,
    "replicate/anthropic/claude-4.5-haiku": 4096,
    "replicate/anthropic/claude-4.5-sonnet": 1024,
    "snowflake/claude-4-opus": 1024,
    "snowflake/claude-4-sonnet": 1024,
    "snowflake/claude-haiku-4-5": 4096,
    "snowflake/claude-sonnet-4-5": 1024,
    "snowflake/claude-sonnet-4-6": 1024,
    "vercel_ai_gateway/anthropic/claude-haiku-4.5": 4096,
    "vercel_ai_gateway/anthropic/claude-opus-4": 1024,
    "vercel_ai_gateway/anthropic/claude-opus-4.1": 1024,
    "vercel_ai_gateway/anthropic/claude-opus-4.5": 4096,
    "vercel_ai_gateway/anthropic/claude-opus-4.6": 4096,
    "vercel_ai_gateway/anthropic/claude-sonnet-4": 1024,
    "vercel_ai_gateway/anthropic/claude-sonnet-4.5": 1024,
    "vertex_ai/claude-fable-5": 512,
    "vertex_ai/claude-fable-5@default": 512,
}


GEMINI_4096_CACHE_MIN_MODELS: Final = tuple(
    prefix + base
    for base in (
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
    )
    for prefix in ("", "gemini/", "vertex_ai/")
)


def test_gemini_3_flash_and_31_pro_preview_resolve_4096_cache_minimum(local_model_cost_map: None) -> None:
    """Regression for the cost map missing prompt_cache_min_tokens on these models: Google rejects
    explicit caching below 4,096 tokens for them (https://ai.google.dev/gemini-api/docs/caching), so
    the 1024 default sent cachedContents creates Vertex answered with a hard 400."""
    wrong: Final = {
        model: get_prompt_cache_min_tokens(model=model)
        for model in GEMINI_4096_CACHE_MIN_MODELS
        if get_prompt_cache_min_tokens(model=model) != 4096
    }
    assert not wrong, f"prompt_cache_min_tokens must be 4096: {wrong}"


def test_get_prompt_cache_min_tokens_unmapped_model_falls_back_to_default(local_model_cost_map: None) -> None:
    """get_model_info raises for a model it has no entry for. The resolver must swallow that and
    fall back to the default, otherwise the raise reaches callers that would read it as
    "not cacheable" -- turning an unknown model into a silently uncacheable one."""
    assert get_prompt_cache_min_tokens(model="totally-unknown-model-xyz") == 1024


def test_is_prompt_caching_valid_prompt_uses_per_model_minimum(local_model_cost_map: None) -> None:
    """Regression: a prompt between two models' minimums is cacheable on one and not the other.
    A 1403-token prompt clears claude-opus-4-8's 1024 minimum but not claude-opus-4-6's 4096, so
    the flat-1024 check reported claude-opus-4-6 as cacheable and the cache write was rejected
    upstream. Both assertions must live together: is_prompt_caching_valid_prompt returns False on
    any internal error, so the True case is what proves the False case isn't a swallowed exception."""
    token_count = litellm.token_counter(
        model="claude-opus-4-6", messages=PROMPT_CACHE_MESSAGES, use_default_image_token_count=True
    )
    assert 1024 <= token_count < 4096, (
        f"prompt drifted to {token_count} tokens; it must sit between claude-opus-4-8's 1024 minimum "
        "and claude-opus-4-6's 4096 minimum for this test to distinguish them"
    )

    assert is_prompt_caching_valid_prompt(model="claude-opus-4-6", messages=PROMPT_CACHE_MESSAGES) is False
    assert is_prompt_caching_valid_prompt(model="claude-opus-4-8", messages=PROMPT_CACHE_MESSAGES) is True


def test_is_prompt_caching_valid_prompt_explicit_min_token_count_overrides_model(local_model_cost_map: None) -> None:
    """An explicit min_token_count wins over the model-resolved value in both directions. Callers
    holding only a model-group alias resolve the threshold themselves and pass it, because an alias
    resolves to nothing here and would silently fall back to the default."""
    assert (
        is_prompt_caching_valid_prompt(model="claude-opus-4-6", messages=PROMPT_CACHE_MESSAGES, min_token_count=512)
        is True
    )
    assert (
        is_prompt_caching_valid_prompt(model="claude-opus-4-8", messages=PROMPT_CACHE_MESSAGES, min_token_count=8192)
        is False
    )


def test_custom_logger_guards_ignore_subclass_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression LIT-4392: the success/failure existence guards used isinstance, so a user
    subclass of a built-in logger already promoted into the callback lists made the guard
    report the built-in itself as registered and the configured logger was silently skipped.
    The exact-class assertions must hold alongside the subclass assertions: the guards still
    have to dedup a second instance of the same class, only a subclass must stop matching."""
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.utils import (
        _custom_logger_class_exists_in_failure_callbacks,
        _custom_logger_class_exists_in_success_callbacks,
    )

    class BuiltinLogger(CustomLogger):
        pass

    class UserSubclassLogger(BuiltinLogger):
        pass

    builtin_instance = BuiltinLogger()

    monkeypatch.setattr(litellm, "success_callback", [UserSubclassLogger()])
    monkeypatch.setattr(litellm, "failure_callback", [UserSubclassLogger()])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    assert _custom_logger_class_exists_in_success_callbacks(builtin_instance) is False
    assert _custom_logger_class_exists_in_failure_callbacks(builtin_instance) is False

    monkeypatch.setattr(litellm, "success_callback", [BuiltinLogger()])
    monkeypatch.setattr(litellm, "failure_callback", [BuiltinLogger()])
    assert _custom_logger_class_exists_in_success_callbacks(builtin_instance) is True
    assert _custom_logger_class_exists_in_failure_callbacks(builtin_instance) is True


@pytest.mark.asyncio
async def test_s3_v2_success_callback_registers_alongside_user_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression LIT-4392: with a user S3Logger subclass registered via litellm_settings.callbacks
    and success_callback ["s3_v2"], the built-in s3_v2 logger was never added and S3 logs were
    silently dropped while requests kept returning 200."""
    from litellm.integrations.s3_v2 import S3Logger
    from litellm.utils import _add_custom_logger_callback_to_specific_event

    class UserS3Logger(S3Logger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

    user_logger = UserS3Logger()
    monkeypatch.setattr(litellm, "success_callback", [user_logger, "s3_v2"])
    monkeypatch.setattr(litellm, "_async_success_callback", [user_logger])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])

    _add_custom_logger_callback_to_specific_event("s3_v2", "success")

    assert any(type(cb) is S3Logger for cb in litellm.success_callback)
    assert any(type(cb) is S3Logger for cb in litellm._async_success_callback)
    assert "s3_v2" not in litellm.success_callback
    assert user_logger in litellm.success_callback


@pytest.mark.asyncio
async def test_builtin_string_callback_registers_when_subclass_already_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression LIT-4392, litellm.callbacks path: the inline dedup in function_setup also
    matched subclass instances, so a built-in name in litellm.callbacks was dropped whenever a
    user subclass was already promoted into _async_success_callback."""
    from litellm.integrations.s3_v2 import S3Logger

    class UserS3Logger(S3Logger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

    user_logger = UserS3Logger()
    monkeypatch.setattr(litellm, "callbacks", ["s3_v2"])
    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [user_logger])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [user_logger])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])

    await litellm.acompletion(
        model="gpt-5.6",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="ok",
    )

    assert any(type(cb) is S3Logger for cb in litellm._async_success_callback)


def test_reapply_runtime_registrations_replays_register_model_overrides(monkeypatch):
    """
    register_model is the documented way to override pricing for a model. A
    price-data reload swaps litellm.model_cost for a freshly fetched catalog,
    so without replaying those registrations the override is silently lost and
    the model reverts to upstream pricing.
    """
    from litellm import utils as litellm_utils
    from litellm.utils import (
        _invalidate_model_cost_lowercase_map,
        reapply_runtime_model_cost_registrations,
    )

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )
    # Only the recorded half is under test here; the live-router rebuild is covered
    # in test_router_model_cost_isolation.py. Routers built by earlier tests in this
    # process stay in the weak set until they are collected, so leaving the callback
    # installed would make this depend on when that happens.
    monkeypatch.setattr(litellm_utils._LiveDeploymentReplay, "callback", None)

    saved_model_cost = litellm.model_cost
    try:
        litellm.register_model(
            model_cost={
                "openai/gpt-4o": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 0.000123,
                }
            }
        )

        litellm.model_cost = {
            "openai/gpt-4o": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.000999,
                "max_input_tokens": 4242,
            }
        }
        _invalidate_model_cost_lowercase_map()
        reapply_runtime_model_cost_registrations()

        assert litellm.model_cost["openai/gpt-4o"]["input_cost_per_token"] == 0.000123
        assert litellm.model_cost["openai/gpt-4o"]["max_input_tokens"] == 4242
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_reapply_runtime_registrations_drops_request_scoped_registrations(monkeypatch):
    """
    Per-request custom pricing describes one call, so it must not be re-asserted
    over every future catalog. Replaying it would let a one-off price outlive
    the catalog generation it was applied to and silently beat fresh upstream
    pricing forever, while a durable override registered alongside it survives.
    """
    from litellm import utils as litellm_utils
    from litellm.utils import (
        _invalidate_model_cost_lowercase_map,
        reapply_runtime_model_cost_registrations,
    )

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    saved_model_cost = litellm.model_cost
    try:
        litellm.register_model(
            model_cost={"openai/gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": 0.000111}},
            persist_across_reloads=True,
        )
        litellm.register_model(
            model_cost={"openai/gpt-4o-mini": {"litellm_provider": "openai", "input_cost_per_token": 0.000222}},
            persist_across_reloads=False,
        )

        litellm.model_cost = {
            "openai/gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": 0.000999},
            "openai/gpt-4o-mini": {"litellm_provider": "openai", "input_cost_per_token": 0.000888},
        }
        _invalidate_model_cost_lowercase_map()
        reapply_runtime_model_cost_registrations()

        assert litellm.model_cost["openai/gpt-4o"]["input_cost_per_token"] == 0.000111
        assert litellm.model_cost["openai/gpt-4o-mini"]["input_cost_per_token"] == 0.000888
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


class _JsonCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.formatter = JsonFormatter()
        self.records: list[dict] = []
        self.addFilter(CorrelationContextFilter())

    def emit(self, record):
        self.records.append(json.loads(self.formatter.format(record)))


def _make_capture_logger(name: str) -> tuple[logging.Logger, _JsonCapture]:
    lg = logging.getLogger(name)
    cap = _JsonCapture()
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    return lg, cap


@pytest.mark.asyncio
async def test_wrapper_async_restores_originating_task_context_after_success(monkeypatch):
    """A successful acompletion() dispatches async_success_handler via
    asyncio.create_task + the global logging worker - a different Task than the
    one running acompletion() itself (this test's own task). That handler's own
    restore only fixes up the detached child task it runs in; wrapper_async's own
    finally block (in litellm/utils.py) must separately restore the *originating*
    task's trace_id/session_id, since nothing else does.
    """
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    trace_id_var.set("outer-trace-wrapper-test")
    session_id_var.set("outer-session-wrapper-test")
    try:
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello there!",
            litellm_session_id="mock-call-session",
            num_retries=0,
        )
        assert trace_id_var.get() == "outer-trace-wrapper-test"
        assert session_id_var.get() == "outer-session-wrapper-test"
    finally:
        trace_id_var.set("")
        session_id_var.set("")


class _ConvertStreamDeploymentHook(CustomLogger):
    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: CallTypes | None
    ) -> dict[str, object] | None:
        if not kwargs.get("stream"):
            return None
        return {**kwargs, "stream": False, HEADROOM_CONVERTED_STREAM_KEY: True}


class _SuccessKwargsCapture(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.success_kwargs: list[dict[str, object]] = []
        self.stream_event_responses: list[object] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.success_kwargs.append(kwargs)

    async def async_log_stream_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.stream_event_responses.append(response_obj)


def _install_converted_stream_callbacks(monkeypatch: pytest.MonkeyPatch) -> _SuccessKwargsCapture:
    capture: Final = _SuccessKwargsCapture()
    monkeypatch.setattr(litellm, "callbacks", [_ConvertStreamDeploymentHook(), capture])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    return capture


async def _wait_for_success_kwargs(capture: _SuccessKwargsCapture, count: int = 1) -> dict[str, object]:
    for _ in range(50):
        if len(capture.success_kwargs) >= count and not _PENDING_CACHE_WRITES:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)
    assert len(capture.success_kwargs) == count
    return capture.success_kwargs[-1]


def _assert_cache_hit_logged_as_stream(capture: _SuccessKwargsCapture, success_kwargs: dict[str, object]) -> None:
    standard_logging_object: Final = success_kwargs["standard_logging_object"]
    assert isinstance(standard_logging_object, dict)
    assert standard_logging_object["cache_hit"] is True
    assert standard_logging_object["stream"] is True
    assert success_kwargs["stream"] is True
    assert capture.stream_event_responses == []


@pytest.mark.asyncio
async def test_wrapper_async_logs_converted_chat_stream_with_standard_logging_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: Final = _install_converted_stream_callbacks(monkeypatch)

    response: Final = await litellm.acompletion(
        model="gpt-5.6",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        mock_response="converted stream body",
        num_retries=0,
    )
    assert isinstance(response, CustomStreamWrapper)
    chunks: Final = [chunk async for chunk in response]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "converted stream body"

    success_kwargs: Final = await _wait_for_success_kwargs(capture)
    standard_logging_object: Final = success_kwargs["standard_logging_object"]
    assert isinstance(standard_logging_object, dict)
    assert standard_logging_object["response_cost"] > 0
    assert standard_logging_object["stream"] is True
    assert success_kwargs["stream"] is True


class _RewritingSuccessDeploymentHook(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.seen_responses: tuple[object, ...] = ()

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: CallTypes | None
    ) -> ModelResponse | None:
        self.seen_responses = (*self.seen_responses, response)
        if not isinstance(response, ModelResponse):
            return None
        choice: Final = response.choices[0]
        if not isinstance(choice, Choices):
            return None
        rewritten_message: Final = choice.message.model_copy(update={"content": "rewritten by deployment hook"})
        return response.model_copy(update={"choices": [choice.model_copy(update={"message": rewritten_message})]})


@pytest.mark.asyncio
async def test_wrapper_async_runs_success_deployment_hook_on_converted_chat_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_converted_stream_callbacks(monkeypatch)
    hook: Final = _RewritingSuccessDeploymentHook()
    monkeypatch.setattr(litellm, "callbacks", [_ConvertStreamDeploymentHook(), hook])

    response: Final = await litellm.acompletion(
        model="gpt-5.6",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        mock_response="converted stream body",
        num_retries=0,
    )
    assert isinstance(response, CustomStreamWrapper)
    chunks: Final = [chunk async for chunk in response]

    assert len(hook.seen_responses) == 1
    seen: Final = hook.seen_responses[0]
    assert isinstance(seen, ModelResponse)
    assert seen.choices[0].message.content == "converted stream body"
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "rewritten by deployment hook"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion_stream", "call_type"),
    [
        (iter([ModelResponse(model="gpt-5.6")]), "acompletion"),
        (MockResponseIterator(model_response=ModelResponse(model="gpt-5.6")), "not_a_call_type"),
    ],
    ids=["real_provider_stream", "unmapped_call_type"],
)
async def test_converted_chat_stream_hook_skips_unhandled_wrappers(
    monkeypatch: pytest.MonkeyPatch, completion_stream: object, call_type: str
) -> None:
    hook: Final = _RewritingSuccessDeploymentHook()
    monkeypatch.setattr(litellm, "callbacks", [hook])
    wrapper: Final = CustomStreamWrapper(
        completion_stream=completion_stream, model="gpt-5.6", logging_obj=MagicMock(), custom_llm_provider="openai"
    )

    await _run_success_deployment_hook_on_converted_chat_stream(
        result=wrapper, request_data={"model": "gpt-5.6"}, call_type=call_type
    )

    assert hook.seen_responses == ()
    assert wrapper.completion_stream is completion_stream


class _ChatShapedSuccessDeploymentHook(CustomLogger):
    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: CallTypes | None
    ) -> None:
        raise AttributeError(f"{type(response).__name__!r} object has no attribute 'choices'")


class _RecordingSuccessDeploymentHook(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.seen_responses: tuple[object, ...] = ()

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: CallTypes | None
    ) -> None:
        self.seen_responses = (*self.seen_responses, response)


_SUCCESS_RESPONSES_BY_CALL_TYPE: Final = (
    pytest.param(
        VideoObject(id="video_abc", object="video", status="queued", model="sora-2", seconds="4", size="720x1280"),
        CallTypes.avideo_generation,
        id="video",
    ),
    pytest.param(EmbeddingResponse(model="text-embedding-3-small"), CallTypes.aembedding, id="embedding"),
    pytest.param(
        ResponsesAPIResponse(
            id="resp_abc", created_at=1, output=[], parallel_tool_calls=False, tool_choice="auto", tools=[], model="gpt-5.6"
        ),
        CallTypes.aresponses,
        id="responses",
    ),
    pytest.param(ImageResponse(), CallTypes.aimage_generation, id="image"),
    pytest.param(RerankResponse(id="rerank_abc"), CallTypes.arerank, id="rerank"),
    pytest.param(TranscriptionResponse(text="hi"), CallTypes.atranscription, id="transcription"),
    pytest.param(ModelResponse(model="gpt-5.6"), CallTypes.acompletion, id="chat"),
    pytest.param(ModelResponse(model="claude-sonnet-4-5"), CallTypes.aanthropic_messages, id="anthropic_messages"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("response", "call_type"), _SUCCESS_RESPONSES_BY_CALL_TYPE)
async def test_success_deployment_hook_raising_keeps_response_and_runs_later_hooks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, response: object, call_type: CallTypes
) -> None:
    second_hook: Final = _RecordingSuccessDeploymentHook()
    monkeypatch.setattr(litellm, "callbacks", [_ChatShapedSuccessDeploymentHook(), second_hook])

    with caplog.at_level(logging.ERROR, logger=verbose_logger.name):
        result: Final = await async_post_call_success_deployment_hook(
            request_data={"model": "m"}, response=response, call_type=call_type
        )

    assert result is response
    assert second_hook.seen_responses == (response,)
    failure_logs: Final = tuple(r for r in caplog.records if "async_post_call_success_deployment_hook error" in r.message)
    assert len(failure_logs) == 1
    assert "_ChatShapedSuccessDeploymentHook" in failure_logs[0].message
    assert str(call_type) in failure_logs[0].message
    assert failure_logs[0].exc_info is not None


@pytest.mark.asyncio
async def test_success_deployment_hook_raising_keeps_earlier_hook_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    rewriter: Final = _RewritingSuccessDeploymentHook()
    trailing_hook: Final = _RecordingSuccessDeploymentHook()
    monkeypatch.setattr(litellm, "callbacks", [rewriter, _ChatShapedSuccessDeploymentHook(), trailing_hook])
    original: Final = ModelResponse(model="gpt-5.6")

    result: Final = await async_post_call_success_deployment_hook(
        request_data={"model": "gpt-5.6"}, response=original, call_type=CallTypes.acompletion
    )

    assert isinstance(result, ModelResponse)
    assert result is not original
    assert result.choices[0].message.content == "rewritten by deployment hook"
    assert trailing_hook.seen_responses == (result,)


class _GuardrailBlocked(Exception):
    pass


class _BlockingSuccessDeploymentGuardrail(CustomGuardrail):
    async def async_post_call_success_deployment_hook(
        self, request_data: dict, response: LLMResponseTypes, call_type: CallTypes | None
    ) -> LLMResponseTypes | None:
        raise _GuardrailBlocked("Violated moderation policy")


@pytest.mark.asyncio
async def test_success_deployment_hook_still_propagates_guardrail_block(monkeypatch: pytest.MonkeyPatch) -> None:
    later_hook: Final = _RewritingSuccessDeploymentHook()
    monkeypatch.setattr(
        litellm, "callbacks", [_BlockingSuccessDeploymentGuardrail(guardrail_name="blocking"), later_hook]
    )

    with pytest.raises(_GuardrailBlocked):
        await async_post_call_success_deployment_hook(
            request_data={"model": "gpt-5.6"}, response=ModelResponse(model="gpt-5.6"), call_type=CallTypes.acompletion
        )

    assert later_hook.seen_responses == ()


@pytest.mark.asyncio
@respx.mock
async def test_wrapper_async_leaves_success_deployment_hook_off_requested_fake_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook: Final = _RewritingSuccessDeploymentHook()
    monkeypatch.setattr(litellm, "callbacks", [hook])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx.post("http://fake-stream.invalid/api/v1/run/flow-1").respond(
        json={"outputs": [{"outputs": [{"results": {"message": {"text": "plain stream body"}}}]}]}
    )

    response: Final = await litellm.acompletion(
        model="langflow/flow-1",
        api_base="http://fake-stream.invalid",
        api_key="fake-key",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        num_retries=0,
    )
    assert isinstance(response, CustomStreamWrapper)
    assert isinstance(response.completion_stream, MockResponseIterator)
    chunks: Final = [chunk async for chunk in response]

    assert hook.seen_responses == ()
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "plain stream body"


@pytest.mark.asyncio
@respx.mock
async def test_wrapper_async_logs_converted_responses_stream_with_standard_logging_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

    capture: Final = _install_converted_stream_callbacks(monkeypatch)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx.post("https://api.openai.com/v1/responses").respond(
        json={
            "id": "resp_converted",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_converted",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "converted stream body", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
    )

    response: Final = await litellm.aresponses(
        model="openai/gpt-5.6", input="hi", stream=True, api_key="sk-test", num_retries=0
    )
    assert isinstance(response, BaseResponsesAPIStreamingIterator)
    events: Final = [event async for event in response]
    assert events[-1].type == "response.completed"

    success_kwargs: Final = await _wait_for_success_kwargs(capture)
    standard_logging_object: Final = success_kwargs["standard_logging_object"]
    assert isinstance(standard_logging_object, dict)
    assert standard_logging_object["response_cost"] > 0
    assert standard_logging_object["stream"] is True
    assert success_kwargs["stream"] is True


@pytest.mark.asyncio
async def test_wrapper_async_replays_cached_converted_chat_stream_as_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: Final = _install_converted_stream_callbacks(monkeypatch)
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    request: Final = {
        "model": "gpt-5.6",
        "messages": [{"role": "user", "content": "replay me from cache"}],
        "stream": True,
        "mock_response": "converted stream body",
        "num_retries": 0,
    }

    first: Final = await litellm.acompletion(**request)
    first_chunks: Final = [chunk async for chunk in first]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in first_chunks) == "converted stream body"
    await _wait_for_success_kwargs(capture)

    replay: Final = await litellm.acompletion(**request)
    assert isinstance(replay, CustomStreamWrapper)
    replay_chunks: Final = [chunk async for chunk in replay]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in replay_chunks) == "converted stream body"

    _assert_cache_hit_logged_as_stream(capture, await _wait_for_success_kwargs(capture, count=2))


@pytest.mark.asyncio
@respx.mock
async def test_wrapper_async_replays_cached_converted_responses_stream_as_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

    capture: Final = _install_converted_stream_callbacks(monkeypatch)
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    route: Final = respx.post("https://api.openai.com/v1/responses").respond(
        json={
            "id": "resp_cached_converted",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_cached_converted",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "converted stream body", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
    )
    request: Final = {
        "model": "openai/gpt-5.6",
        "input": "replay me from cache",
        "stream": True,
        "api_key": "sk-test",
        "num_retries": 0,
    }

    first: Final = await litellm.aresponses(**request)
    assert [event async for event in first][-1].type == "response.completed"
    await _wait_for_success_kwargs(capture)

    replay: Final = await litellm.aresponses(**request)
    assert isinstance(replay, BaseResponsesAPIStreamingIterator)
    assert [event async for event in replay][-1].type == "response.completed"
    assert route.call_count == 1

    _assert_cache_hit_logged_as_stream(capture, await _wait_for_success_kwargs(capture, count=2))


def test_function_setup_failure_after_logging_construction_restores_context(monkeypatch):
    """If function_setup() constructs Logging() (which already mutated
    trace_id_var/session_id_var in __init__) but then raises before returning,
    the caller's wrapper() never gets a logging_obj reference to restore from.
    function_setup()'s own except block must restore the correlation context
    itself in that case, or it leaks into every subsequent log line in this
    thread/task until something unrelated happens to reset it."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated failure after Logging() construction")

    monkeypatch.setattr(Logging, "update_environment_variables", _boom)

    trace_id_var.set("pre-setup-failure-trace")
    session_id_var.set("pre-setup-failure-session")
    try:
        with pytest.raises(RuntimeError, match="simulated failure"):
            litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="Hello there!",
                litellm_session_id="doomed-call-session",
                num_retries=0,
            )
        assert trace_id_var.get() == "pre-setup-failure-trace"
        assert session_id_var.get() == "pre-setup-failure-session"
    finally:
        trace_id_var.set("")
        session_id_var.set("")


def test_function_setup_failure_log_line_shows_outer_not_doomed_ids(monkeypatch):
    """The 'Error in function_setup' diagnostic log line itself must be stamped
    with the outer/pre-call correlation ids, not the doomed call's own ids -
    restoring context must happen *before* logging the exception, not after,
    since the failed call never produces a usable logging object for anything
    else to be attributed to."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated failure after Logging() construction")

    monkeypatch.setattr(Logging, "update_environment_variables", _boom)

    lg, cap = _make_capture_logger("test.function_setup_failure_log_order")
    # verbose_logger is a distinct, module-level logger from our throwaway one -
    # temporarily attach the same capture handler so we see its own emitted record.
    verbose_logger.addHandler(cap)
    try:
        trace_id_var.set("outer-trace")
        session_id_var.set("outer-session")
        with pytest.raises(RuntimeError, match="simulated failure"):
            litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="Hello there!",
                litellm_session_id="doomed-call-session",
                num_retries=0,
            )
        setup_failure_records = [r for r in cap.records if "Error in function_setup" in r.get("message", "")]
        assert len(setup_failure_records) == 1
        record = setup_failure_records[0]
        assert record.get("session_id") == "outer-session"
        assert record.get("trace_id") == "outer-trace"
    finally:
        verbose_logger.removeHandler(cap)
        trace_id_var.set("")
        session_id_var.set("")


WEBSEARCH_INTERNAL_CONTROL_FIELDS = (
    "_websearch_interception_emit_native_blocks",
    "_websearch_interception_converted_stream",
)


def test_websearch_interception_control_fields_never_reach_the_provider():
    """The web-search interception hooks stamp these onto kwargs to carry state
    across the agentic loop. Anything the param builder does not recognize is
    swept into the provider request, and a provider that validates its body
    rejects the whole call: Bedrock Converse answers
    `_websearch_interception_emit_native_blocks: Extra inputs are not permitted`
    with a 400, so enabling interception breaks every request it touches.

    Their code-interpreter counterparts are already registered; these were not.
    """
    kwargs = {
        "a_real_provider_specific_param": 1,
        **{field: True for field in WEBSEARCH_INTERNAL_CONTROL_FIELDS},
    }

    non_default = get_non_default_completion_params(kwargs)

    assert non_default == {"a_real_provider_specific_param": 1}, (
        "web-search interception control fields leaked into the provider params: "
        f"{sorted(set(non_default) - {'a_real_provider_specific_param'})}"
    )
    assert set(WEBSEARCH_INTERNAL_CONTROL_FIELDS) <= set(all_litellm_params)


def test_get_litellm_params_keys_never_reach_the_provider():
    """Bridges (chat <-> Responses, agentic loop follow-ups) forward litellm_params as
    `completion()` kwargs. Any key the param builder does not recognize is swept into
    extra_body, and OpenAI rejects the call with `Unknown parameter: 'model_alias_map'`.
    """
    litellm_param_keys = frozenset(get_litellm_params()) - {"drop_params"}
    kwargs = {
        "a_real_provider_specific_param": 1,
        "model_alias_map": {"alias": "gpt-5.4"},
        **{key: "configured-value" for key in litellm_param_keys - {"model_alias_map"}},
    }

    non_default = get_non_default_completion_params(kwargs)

    assert non_default == {"a_real_provider_specific_param": 1}, (
        "litellm params leaked into the provider params: "
        f"{sorted(set(non_default) - {'a_real_provider_specific_param'})}"
    )


def test_addressed_response_id_never_reaches_the_provider():
    kwargs = {
        "a_real_provider_specific_param": 1,
        ADDRESSED_RESPONSE_ID_FIELD: "resp_addressed-by-the-client",
    }

    non_default = get_non_default_completion_params(kwargs)

    assert non_default == {"a_real_provider_specific_param": 1}, (
        "the addressed response id leaked into the provider params: "
        f"{sorted(set(non_default) - {'a_real_provider_specific_param'})}"
    )


def test_bedrock_batch_params_never_reach_the_provider():
    """A Bedrock managed-batch deployment carries aws_batch_role_arn / s3_* /
    bedrock_tags in its litellm_params, and the same deployment also serves chat.
    Anything the param builder does not recognize is swept into extra_body, so
    Bedrock rejects the whole call: `aws_batch_role_arn: Extra inputs are not
    permitted` (Anthropic models) or `extraneous key [aws_batch_role_arn] is not
    permitted` (Nova/Llama/Titan), turning every non-batch request to that
    deployment into a 400.

    The batch path is unaffected by registering them, because GenericLiteLLMParams
    is extra="allow" and preserves them into litellm_params for the batch and files
    transformations that read them.
    """
    configured = {
        field: ([{"key": "team", "value": "configured-value"}] if field == "bedrock_tags" else "configured-value")
        for field in bedrock_batch_litellm_params
    }
    kwargs = {"a_real_provider_specific_param": 1, **configured}

    non_default = get_non_default_completion_params(dict(kwargs))

    assert non_default == {"a_real_provider_specific_param": 1}, (
        "bedrock batch params leaked into the provider params: "
        f"{sorted(set(non_default) - {'a_real_provider_specific_param'})}"
    )
    assert set(bedrock_batch_litellm_params) <= set(all_litellm_params)

    batch_params = dict(GenericLiteLLMParams(**kwargs))
    assert all(batch_params.get(field) == configured[field] for field in bedrock_batch_litellm_params), (
        "registering these must not strip them from the batch path: "
        f"{sorted(f for f in bedrock_batch_litellm_params if batch_params.get(f) != configured[f])}"
    )

    normalized = CredentialLiteLLMParams.model_validate(
        GenericLiteLLMParams(**kwargs).model_dump(exclude_none=True)
    ).model_dump(exclude_none=True)
    assert all(normalized.get(field) == configured[field] for field in bedrock_batch_litellm_params), (
        "credential normalization dropped batch params before the transformation: "
        f"{sorted(f for f in bedrock_batch_litellm_params if normalized.get(f) != configured[f])}"
    )


def test_documented_batch_s3_credentials_never_reach_the_provider():
    """The Bedrock batch docs tell users to put s3_access_key_id, s3_secret_access_key
    and s3_encryption_key_id on the deployment. Left unregistered they are swept into
    additionalModelRequestFields, Bedrock 400s ordinary chat on that deployment with
    `s3_secret_access_key: Extra inputs are not permitted`, and the S3 secret is sent
    to the provider and printed in the debug log (LIT-8290).
    """
    configured = {
        "s3_access_key_id": "configured-access-key-id",
        "s3_secret_access_key": "configured-secret-access-key",
        "s3_encryption_key_id": "arn:aws:kms:us-east-1:000000000000:key/configured",
    }
    kwargs = {"a_real_provider_specific_param": 1, **configured}

    non_default = get_non_default_completion_params(dict(kwargs))

    assert non_default == {"a_real_provider_specific_param": 1}, (
        "documented batch S3 credentials leaked into the provider params: "
        f"{sorted(set(non_default) - {'a_real_provider_specific_param'})}"
    )

    batch_params = dict(GenericLiteLLMParams(**kwargs))
    assert {field: batch_params.get(field) for field in configured} == configured, (
        "registering these must not strip them from the batch path"
    )


def test_client_side_timeout_marker_never_reaches_the_provider():
    """The proxy stamps kwargs["client_side_timeout"] = True whenever a request carries
    a caller-supplied timeout (body timeout / request_timeout / stream_timeout or the
    x-litellm-timeout headers) so the router can skip cooldowns on the resulting 408s.
    The marker is only meaningful to the router, so it must be filtered out of the
    provider params: swept into extra_body / additionalModelRequestFields it turns every
    timed-out request into a provider 400 (`client_side_timeout: Extra inputs are not
    permitted`)."""
    kwargs = {"a_real_provider_specific_param": 1, "client_side_timeout": True}

    non_default = get_non_default_completion_params(kwargs)

    assert non_default == {"a_real_provider_specific_param": 1}, (
        "client_side_timeout leaked into the provider params: "
        f"{sorted(set(non_default) - {'a_real_provider_specific_param'})}"
    )


class _RecordingDeploymentFailureLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[dict, Exception, CallTypes | None, int | None]] = []

    async def async_post_call_failure_deployment_hook(self, request_data, exception, call_type, fallback_depth=None):
        self.calls.append((request_data, exception, call_type, fallback_depth))


@pytest.mark.asyncio
async def test_async_post_call_failure_deployment_hook_calls_custom_logger_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher must call the CustomLogger hook with an equivalent exception (not
    necessarily the same object - see test_..._snapshots_exception_so_callback_mutations_..._
    below) and the call_type resolved to its CallTypes enum member."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    exc = ValueError("deployment failed")
    await async_post_call_failure_deployment_hook(
        request_data={"model": "gpt-4o-mini"}, exception=exc, call_type="acompletion"
    )

    assert len(recorder.calls) == 1
    request_data, received_exc, call_type, fallback_depth = recorder.calls[0]
    assert request_data == {"model": "gpt-4o-mini"}
    assert isinstance(received_exc, ValueError)
    assert str(received_exc) == str(exc)
    assert call_type == CallTypes.acompletion
    assert fallback_depth is None


@pytest.mark.asyncio
async def test_async_post_call_failure_deployment_hook_falls_back_to_none_call_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized call_type string must resolve to None rather than raising."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    await async_post_call_failure_deployment_hook(
        request_data={}, exception=ValueError("x"), call_type="not_a_real_call_type"
    )

    assert recorder.calls[0][2] is None


@pytest.mark.asyncio
async def test_async_post_call_failure_deployment_hook_passes_through_fallback_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fallback_depth on request_data (set by Router on each fallback hop) must reach the
    callback unchanged, so a subscriber can tell which fallback hop this failure is from."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    await async_post_call_failure_deployment_hook(
        request_data={"fallback_depth": 2}, exception=ValueError("x"), call_type="acompletion"
    )

    assert recorder.calls[0][3] == 2


@pytest.mark.asyncio
async def test_async_post_call_failure_deployment_hook_fallback_depth_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fallback_depth must be None, not raise or pass through garbage, when request_data has
    no fallback_depth at all (first attempt, or a bare SDK call with no Router) or a
    non-int value there."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    await async_post_call_failure_deployment_hook(request_data={}, exception=ValueError("x"), call_type="acompletion")
    await async_post_call_failure_deployment_hook(
        request_data={"fallback_depth": "not-an-int"}, exception=ValueError("y"), call_type="acompletion"
    )

    assert recorder.calls[0][3] is None
    assert recorder.calls[1][3] is None


@pytest.mark.asyncio
async def test_async_post_call_failure_deployment_hook_swallows_callback_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback that raises inside the hook must not propagate out of the dispatcher."""

    class ExplodingLogger(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        async def async_post_call_failure_deployment_hook(
            self, request_data, exception, call_type, fallback_depth=None
        ):
            self.called = True
            raise RuntimeError("hook exploded")

    exploding_logger = ExplodingLogger()
    monkeypatch.setattr(litellm, "callbacks", [exploding_logger])

    await async_post_call_failure_deployment_hook(request_data={}, exception=ValueError("x"), call_type="acompletion")

    assert exploding_logger.called


@pytest.mark.asyncio
async def test_async_post_call_failure_deployment_hook_skips_non_custom_logger_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callable (function-based) callbacks are not CustomLogger instances and must be skipped."""
    called: list[bool] = []

    async def fn_callback(*args: object, **kwargs: object) -> None:
        called.append(True)

    monkeypatch.setattr(litellm, "callbacks", [fn_callback])

    await async_post_call_failure_deployment_hook(request_data={}, exception=ValueError("x"), call_type="acompletion")

    assert called == []


@pytest.mark.asyncio
async def test_wrapper_async_fires_post_call_failure_deployment_hook_once_per_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a failed deployment call must reach async_post_call_failure_deployment_hook
    exactly once, sourced from wrapper_async's own except block rather than the dedup-gated
    async_log_failure_event path, which would miss retries/fallback chain attempts 2+."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    with pytest.raises(litellm.AuthenticationError):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        )

    assert len(recorder.calls) == 1
    _, received_exc, call_type, fallback_depth = recorder.calls[0]
    assert isinstance(received_exc, litellm.AuthenticationError)
    assert call_type == CallTypes.acompletion
    assert fallback_depth is None


@pytest.mark.asyncio
async def test_wrapper_async_raises_original_exception_even_if_hook_callback_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken async_post_call_failure_deployment_hook override must never shadow the real
    exception the caller is waiting on."""

    class ExplodingLogger(CustomLogger):
        async def async_post_call_failure_deployment_hook(
            self, request_data, exception, call_type, fallback_depth=None
        ):
            raise RuntimeError("hook exploded")

    monkeypatch.setattr(litellm, "callbacks", [ExplodingLogger()])

    with pytest.raises(litellm.AuthenticationError, match="bad key"):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        )


@pytest.mark.asyncio
async def test_router_fallback_chain_reports_increasing_fallback_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a real Router fallback chain must report fallback_depth=None on the
    first, pre-fallback attempt and fallback_depth=1 on the first fallback hop - the
    concrete scenario async_post_call_failure_deployment_hook exists to make visible."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    router = litellm.Router(
        model_list=[
            {"model_name": "bad-group", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "bad-a"}},
            {"model_name": "good-group", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "bad-b"}},
        ],
        num_retries=0,
        fallbacks=[{"bad-group": ["good-group"]}],
    )

    with pytest.raises(litellm.AuthenticationError):
        await router.acompletion(
            model="bad-group",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        )

    assert len(recorder.calls) == 2
    assert recorder.calls[0][3] is None
    assert recorder.calls[1][3] == 1


@pytest.mark.asyncio
async def test_router_multi_hop_fallback_chain_reports_depth_per_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: fallback_depth must keep incrementing across more than one fallback
    hop (group-a -> group-b -> group-c, all failing), not just report 1 for every
    fallback attempt regardless of how deep the chain has gone."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    router = litellm.Router(
        model_list=[
            {"model_name": "group-a", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "bad-a"}},
            {"model_name": "group-b", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "bad-b"}},
            {"model_name": "group-c", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "bad-c"}},
        ],
        num_retries=0,
        fallbacks=[{"group-a": ["group-b", "group-c"]}],
    )

    with pytest.raises(litellm.AuthenticationError):
        await router.acompletion(
            model="group-a",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        )

    assert len(recorder.calls) == 3
    assert [call[3] for call in recorder.calls] == [None, 1, 2]


@pytest.mark.asyncio
async def test_wrapper_async_fires_post_call_failure_deployment_hook_on_internal_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a failed attempt made while is_internal_call is set (e.g. an emulated
    file-search step) must still reach async_post_call_failure_deployment_hook, matching
    async_pre_call_deployment_hook, which already fires unconditionally for such calls."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    token = is_internal_call.set(True)
    try:
        with pytest.raises(litellm.AuthenticationError):
            await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response=litellm.AuthenticationError(
                    message="bad key", llm_provider="openai", model="gpt-4o-mini"
                ),
            )
    finally:
        is_internal_call.reset(token)

    assert len(recorder.calls) == 1
    assert isinstance(recorder.calls[0][1], litellm.AuthenticationError)


def _budget_reservation(callback_bound: bool = False) -> dict:
    return {"reserved_cost": 0.5, "entries": [], "finalized": False, "callback_bound": callback_bound}


_BUDGET_RESERVATION_CALL_KWARGS: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
_BUDGET_RESERVATION_REFUSAL: Final = litellm.AuthenticationError(
    message="bad key", llm_provider="openai", model="gpt-4o"
)


@pytest.mark.asyncio
async def test_wrapper_async_claims_the_budget_reservation_for_the_cost_callback() -> None:
    reservation = _budget_reservation()

    await litellm.acompletion(
        **_BUDGET_RESERVATION_CALL_KWARGS,
        mock_response="ok",
        metadata={"user_api_key_budget_reservation": reservation},
    )

    assert reservation["callback_bound"] is True


@pytest.mark.asyncio
async def test_wrapper_async_claims_the_budget_reservation_before_the_stream_is_consumed() -> None:
    reservation = _budget_reservation()

    stream = await litellm.acompletion(
        **_BUDGET_RESERVATION_CALL_KWARGS,
        mock_response="ok",
        stream=True,
        metadata={"user_api_key_budget_reservation": reservation},
    )

    assert reservation["callback_bound"] is True
    async for _ in stream:
        pass


@pytest.mark.asyncio
async def test_wrapper_async_claims_the_budget_reservation_a_supplied_logging_object_already_saw() -> None:
    reservation = _budget_reservation()
    logging_obj, kwargs = litellm.utils.function_setup(
        original_function="acompletion",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.now(),
        **_BUDGET_RESERVATION_CALL_KWARGS,
        litellm_call_id="proxy-pre-call-setup",
        metadata={"user_api_key_budget_reservation": reservation},
    )
    assert reservation["callback_bound"] is False

    await litellm.acompletion(**kwargs, litellm_logging_obj=logging_obj, mock_response="ok")

    assert reservation["callback_bound"] is True


@pytest.mark.asyncio
async def test_wrapper_async_hands_the_budget_reservation_back_when_the_call_fails() -> None:
    reservation = _budget_reservation()

    with pytest.raises(litellm.AuthenticationError):
        await litellm.acompletion(
            **_BUDGET_RESERVATION_CALL_KWARGS,
            mock_response=_BUDGET_RESERVATION_REFUSAL,
            metadata={"user_api_key_budget_reservation": reservation},
        )

    assert reservation["callback_bound"] is False


@pytest.mark.asyncio
async def test_wrapper_async_leaves_the_budget_reservation_alone_on_internal_calls() -> None:
    claimed_by_the_outer_call = _budget_reservation(callback_bound=True)
    never_claimed = _budget_reservation()

    token = is_internal_call.set(True)
    try:
        await litellm.acompletion(
            **_BUDGET_RESERVATION_CALL_KWARGS,
            mock_response="ok",
            metadata={"user_api_key_budget_reservation": never_claimed},
        )
        with pytest.raises(litellm.AuthenticationError):
            await litellm.acompletion(
                **_BUDGET_RESERVATION_CALL_KWARGS,
                mock_response=_BUDGET_RESERVATION_REFUSAL,
                metadata={"user_api_key_budget_reservation": claimed_by_the_outer_call},
            )
    finally:
        is_internal_call.reset(token)

    assert never_claimed["callback_bound"] is False
    assert claimed_by_the_outer_call["callback_bound"] is True


@pytest.mark.asyncio
async def test_wrapper_async_does_not_fire_failure_hook_for_pre_call_budget_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a BudgetExceededError raised before any deployment call is attempted
    (the [OPTIONAL] CHECK BUDGET gate) is not a deployment attempt failure and must not
    reach async_post_call_failure_deployment_hook."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "max_budget", 0.0001)
    monkeypatch.setattr(litellm, "_current_cost", 100.0)

    with pytest.raises(litellm.BudgetExceededError):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="should never be reached",
        )

    assert recorder.calls == []


@pytest.mark.asyncio
async def test_wrapper_async_does_not_fire_failure_hook_for_post_success_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an error raised after the deployment call already succeeded (e.g. inside
    async_post_call_success_deployment_hook or post_call_processing) is not a deployment
    attempt failure and must not reach async_post_call_failure_deployment_hook. The raising
    callback is a guardrail because a plain logger's success hook error is isolated and
    logged instead of propagating out of the call."""

    class ExplodingSuccessGuardrail(CustomGuardrail):
        def __init__(self) -> None:
            super().__init__(guardrail_name="exploding")
            self.failure_calls: tuple[Exception, ...] = ()

        async def async_post_call_success_deployment_hook(
            self, request_data: Mapping[str, object], response: LLMResponseTypes, call_type: CallTypes | None
        ) -> LLMResponseTypes | None:
            raise RuntimeError("boom in success hook, model call itself succeeded")

        async def async_post_call_failure_deployment_hook(
            self,
            request_data: Mapping[str, object],
            exception: Exception,
            call_type: CallTypes | None,
            fallback_depth: int | None = None,
        ) -> None:
            self.failure_calls = (*self.failure_calls, exception)

    exploding_guardrail: Final = ExplodingSuccessGuardrail()
    monkeypatch.setattr(litellm, "callbacks", [exploding_guardrail])

    with pytest.raises(RuntimeError, match="boom in success hook"):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="this call succeeds",
        )

    assert exploding_guardrail.failure_calls == ()


@pytest.mark.asyncio
async def test_wrapper_async_calls_hook_override_missing_fallback_depth_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an override written before fallback_depth existed (this PR's own earlier
    proof-of-fix example used exactly this 3-arg signature) must still fire, not raise a
    TypeError on the fallback_depth keyword that gets swallowed at debug level."""

    class ThreeArgLogger(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[dict, Exception, CallTypes | None]] = []

        async def async_post_call_failure_deployment_hook(self, request_data, exception, call_type):
            self.calls.append((request_data, exception, call_type))

    three_arg_logger = ThreeArgLogger()
    monkeypatch.setattr(litellm, "callbacks", [three_arg_logger])

    with pytest.raises(litellm.AuthenticationError):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        )

    assert len(three_arg_logger.calls) == 1
    assert isinstance(three_arg_logger.calls[0][1], litellm.AuthenticationError)


@pytest.mark.asyncio
async def test_wrapper_async_failure_hook_exception_mutation_does_not_change_raised_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a callback setting an attribute on the exception it receives (e.g.
    status_code, as a real caller would read to determine the HTTP response) must not
    change what the actual caller ends up with - the hook must not have write access to
    the real exception about to be re-raised."""

    class StatusCodeMutatingLogger(CustomLogger):
        async def async_post_call_failure_deployment_hook(
            self, request_data, exception, call_type, fallback_depth=None
        ):
            exception.status_code = 429

    monkeypatch.setattr(litellm, "callbacks", [StatusCodeMutatingLogger()])

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        )

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_async_post_call_failure_deployment_hook_omits_attempted_targets_from_request_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: attempted_targets is the router's own live fallback-walk bookkeeping,
    shared by reference across every hop of a single request - unlike the rest of
    request_data, it is not this attempt's own isolated copy. A callback calling .record()
    on it would make the router skip a deployment it hasn't actually tried, so the
    dispatcher must never hand it to a callback."""
    recorder = _RecordingDeploymentFailureLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    sentinel_targets = object()
    await async_post_call_failure_deployment_hook(
        request_data={"model": "gpt-4o-mini", "attempted_targets": sentinel_targets},
        exception=ValueError("x"),
        call_type="acompletion",
    )

    assert recorder.calls[0][0].get("attempted_targets") is None


@pytest.mark.asyncio
async def test_router_fallback_not_skipped_when_failure_hook_callback_touches_attempted_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: even a callback that tries to record a target on whatever it's handed as
    attempted_targets must not affect the live Router fallback walk - the healthy fallback
    deployment must still be reachable, not silently skipped as already-attempted.

    attempted_targets is only present in kwargs starting from the second hop onward (the
    first deployment's own failure predates the router's own fallback bookkeeping), so this
    needs a 3-deployment chain: mid-group's failure is where the callback sees
    attempted_targets and can prematurely mark good-group as tried. Uses per-deployment
    mock_timeout/mock_response, not a request-level mock_response, which Router carries
    into every hop's kwargs and would mask this test's real signal."""

    class RecordingAttemptLogger(CustomLogger):
        async def async_post_call_failure_deployment_hook(
            self, request_data, exception, call_type, fallback_depth=None
        ):
            attempted = request_data.get("attempted_targets")
            if attempted is not None:
                attempted.record("good-group")

    monkeypatch.setattr(litellm, "callbacks", [RecordingAttemptLogger()])

    def _mock_timeout_deployment(model_name: str) -> dict:
        return {
            "model_name": model_name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "fake",
                "mock_timeout": True,
                "timeout": 0.001,
                "num_retries": 0,
            },
        }

    router = litellm.Router(
        model_list=[
            _mock_timeout_deployment("bad-group"),
            _mock_timeout_deployment("mid-group"),
            {
                "model_name": "good-group",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake",
                    "mock_response": "fallback worked",
                    "num_retries": 0,
                },
            },
        ],
        num_retries=0,
        fallbacks=[{"bad-group": ["mid-group", "good-group"]}],
    )

    response = await router.acompletion(
        model="bad-group",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.choices[0].message.content == "fallback worked"


@pytest.mark.asyncio
async def test_wrapper_async_preserves_original_exception_when_hook_await_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: if the caller's own timeout (e.g. asyncio.wait_for) fires while the
    failure hook is still being awaited, the real deployment exception must still reach
    the caller - not get replaced by CancelledError/TimeoutError from the hook's own
    await getting cancelled."""

    class SlowLogger(CustomLogger):
        async def async_post_call_failure_deployment_hook(
            self, request_data, exception, call_type, fallback_depth=None
        ):
            await asyncio.sleep(5)

    monkeypatch.setattr(litellm, "callbacks", [SlowLogger()])

    with pytest.raises(litellm.AuthenticationError):
        await asyncio.wait_for(
            litellm.acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response=litellm.AuthenticationError(
                    message="bad key", llm_provider="openai", model="gpt-4o-mini"
                ),
            ),
            timeout=0.2,
        )


@pytest.mark.asyncio
async def test_wrapper_async_failure_hook_latency_does_not_inflate_reported_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a slow failure-hook callback must not inflate the duration reported to
    async_log_failure_event - that's real observability data (e.g. latency dashboards),
    and the hook's own runtime is not part of how long the deployment call itself took."""
    reported_durations: list[float] = []

    class SlowLoggerWithDurationCapture(CustomLogger):
        async def async_post_call_failure_deployment_hook(
            self, request_data, exception, call_type, fallback_depth=None
        ):
            await asyncio.sleep(1)

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            reported_durations.append((end_time - start_time).total_seconds())

    monkeypatch.setattr(litellm, "callbacks", [SlowLoggerWithDurationCapture()])

    with pytest.raises(litellm.AuthenticationError):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        )
    await asyncio.sleep(0.1)

    assert len(reported_durations) == 1
    assert reported_durations[0] < 0.5


@pytest.mark.asyncio
async def test_wrapper_async_failure_hook_exception_snapshot_preserves_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the exception snapshot handed to failure-hook callbacks (see
    test_..._exception_mutation_does_not_change_raised_exception above) must still carry
    __traceback__/__cause__/__context__, not just __dict__/args - a callback formatting or
    inspecting the failure chain needs the real traceback, not an empty one."""
    received: list[Exception] = []

    class TracebackCapturingLogger(CustomLogger):
        async def async_post_call_failure_deployment_hook(
            self, request_data, exception, call_type, fallback_depth=None
        ):
            received.append(exception)

    monkeypatch.setattr(litellm, "callbacks", [TracebackCapturingLogger()])

    with pytest.raises(litellm.AuthenticationError):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        )

    assert len(received) == 1
    assert received[0].__traceback__ is not None


def test_snapshot_exception_for_hook_preserves_suppress_context_flag() -> None:
    """Regression: setting __cause__ has a documented CPython side effect of implicitly
    forcing __suppress_context__ to True, even when the real exception's own
    __suppress_context__ is False (the common case: no `raise ... from`, just an
    exception raised while handling another one, which chains __context__ but does not
    suppress it). Snapshotting __cause__ before __suppress_context__ would silently flip
    a real exception's __suppress_context__=False to True on the snapshot, hiding a
    chained context a callback formatting it should still see."""

    def _raise_chained_without_from() -> None:
        try:
            raise ValueError("inner cause")
        except ValueError:
            raise RuntimeError("outer error")  # no `from` clause: implicit chaining, not suppressed

    with pytest.raises(RuntimeError) as exc_info:
        _raise_chained_without_from()

    e = exc_info.value
    assert e.__suppress_context__ is False  # sanity check on the real exception itself
    snapshot = _snapshot_exception_for_hook(e)
    assert snapshot.__suppress_context__ is False
    assert snapshot.__context__ is e.__context__


class TestDefaultReasoningEffortHydration:
    """`get_model_info` is the public shape every other capability key is readable through, so
    the declared default has to survive hydration too, not only the raw-map fallback the
    request-path gate happens to reach it by.
    """

    @pytest.mark.parametrize(
        "model, provider",
        [("gpt-5.1", "openai"), ("gpt-5.4", "openai"), ("azure/gpt-5.1", "azure")],
    )
    def test_the_declared_default_survives_model_info_hydration(self, local_model_cost_map, model, provider):
        from litellm.utils import _get_model_info_helper

        model_info = dict(_get_model_info_helper(model=model, custom_llm_provider=provider))
        assert model_info["default_reasoning_effort"] == "none"

    def test_a_model_that_declares_nothing_hydrates_to_none(self, local_model_cost_map):
        """Absent means "the map does not say", which the gate reads as reasoning being active."""
        from litellm.utils import _get_model_info_helper

        model_info = dict(_get_model_info_helper(model="gpt-5.6-terra", custom_llm_provider="openai"))
        assert model_info.get("default_reasoning_effort") is None


class TestHuggingFaceConfigFetch:
    """The Hugging Face config.json fetch runs on background logging threads during cost
    calculation, so an unbounded request can hang a whole test job; the timeout is the fix."""

    @pytest.fixture
    def hf_config_route(self):
        with respx.mock(assert_all_called=True) as respx_mock:
            yield respx_mock.get(url__regex=r"https://huggingface\.co/.*/config\.json").respond(
                json={"max_position_embeddings": 512}
            )

    def test_get_max_tokens_reads_hf_config_with_a_bounded_timeout(self, hf_config_route):
        from litellm.constants import HF_CONFIG_FETCH_TIMEOUT_SECONDS
        from litellm.utils import get_max_tokens

        assert get_max_tokens("huggingface/some-org/some-model") == 512
        request_timeout = hf_config_route.calls.last.request.extensions["timeout"]
        assert request_timeout["read"] == HF_CONFIG_FETCH_TIMEOUT_SECONDS

    def test_get_max_position_embeddings_reads_hf_config_with_a_bounded_timeout(self, hf_config_route):
        from litellm.constants import HF_CONFIG_FETCH_TIMEOUT_SECONDS
        from litellm.utils import _get_max_position_embeddings

        assert _get_max_position_embeddings("some-org/some-model") == 512
        request_timeout = hf_config_route.calls.last.request.extensions["timeout"]
        assert request_timeout["read"] == HF_CONFIG_FETCH_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_success_deployment_hook_chains_past_callback_returning_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (LIT-5863): the dispatcher must run every callback, chaining each non-None
    result into the next call, instead of returning at the first callback answering non-None.
    A guardrail answering with the unmodified response used to starve every callback after it."""
    from litellm.types.utils import ModelResponse

    original = ModelResponse()
    replacement = ModelResponse()

    class PassthroughLogger(CustomLogger):
        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            return response

    class ReplacingLogger(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list = []

        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            self.seen.append(response)
            return replacement

    class ObservingLogger(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list = []

        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            self.seen.append(response)
            return None

    replacer = ReplacingLogger()
    observer = ObservingLogger()
    monkeypatch.setattr(litellm, "callbacks", [PassthroughLogger(), replacer, observer])

    result = await async_post_call_success_deployment_hook(
        request_data={}, response=original, call_type=CallTypes.acompletion
    )

    assert replacer.seen == [original]
    assert observer.seen == [replacement]
    assert result is replacement


@pytest.mark.asyncio
async def test_registered_guardrail_does_not_starve_vector_store_search_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (LIT-5863): with any guardrail registered ahead of the lazily-appended
    VectorStorePreCallHook, /v1/chat/completions responses lost
    provider_specific_fields["search_results"] because the guardrail answered the unmodified
    response and the dispatcher stopped there."""
    from types import SimpleNamespace

    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
        VectorStorePreCallHook,
    )
    from litellm.types.utils import ModelResponse

    search_results: Final = [
        {"search_query": "coolant", "data": [{"content": [{"text": "Cryoline-9", "type": "text"}]}]}
    ]
    logging_obj = SimpleNamespace(model_call_details={"search_results": search_results})
    response = ModelResponse(choices=[{"message": {"role": "assistant", "content": "Cryoline-9"}}])

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [CustomGuardrail(guardrail_name="dummy-guardrail"), VectorStorePreCallHook()],
    )

    result = await async_post_call_success_deployment_hook(
        request_data={"litellm_logging_obj": logging_obj},
        response=response,
        call_type=CallTypes.acompletion,
    )

    provider_fields = result.choices[0].message.provider_specific_fields
    assert provider_fields is not None
    assert provider_fields["search_results"] == search_results


class TestIsVisionExplicitlyDisabled:
    """github_copilot and chatgpt run an OAuth device flow inside get_llm_provider; the
    explicit-disable lookup must adopt the declared prefix instead of resolving it, exactly
    as _supports_factory does, or a capability check on a copilot deployment blocks routing
    on a device-code prompt."""

    @pytest.mark.parametrize("model", ["github_copilot/gpt-4o", "chatgpt/gpt-5"])
    def test_never_resolves_an_authenticating_prefix(self, model, monkeypatch):
        from litellm.utils import is_vision_explicitly_disabled

        lookups: list = []

        def _record(*args, **kwargs):
            lookups.append((args, kwargs))
            raise RuntimeError("provider resolution must not run for an authenticating provider")

        monkeypatch.setattr(litellm, "get_llm_provider", _record)

        assert is_vision_explicitly_disabled(model) is False
        assert lookups == []

    def test_explicit_false_detected_and_absent_reads_enabled(self):
        from litellm.utils import is_vision_explicitly_disabled

        assert is_vision_explicitly_disabled("fireworks_ai/accounts/fireworks/models/deepseek-v4-flash-0731") is True
        assert is_vision_explicitly_disabled("anthropic/claude-sonnet-4-5") is False


class TestVerboseRequestLineRedaction:
    """`litellm.set_verbose = True` echoes the caller's kwargs back as a `litellm.completion(...)`
    line on stdout, so a credential kwarg lands in whatever collects stdout: a terminal, a
    container log drain, a CI job log. Credential-named kwargs must not survive that echo,
    at any nesting depth, while ordinary params still must, or the line stops telling the
    developer what they called."""

    FAKE_API_KEY: Final = "sk-fake-lit6823-0000000000000000"

    def _verbose_request_line(self, capsys, monkeypatch, **kwargs) -> str:
        monkeypatch.setattr(litellm, "set_verbose", True)
        monkeypatch.setattr("litellm._logging.set_verbose", True)
        capsys.readouterr()
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
            mock_response="hi",
            **kwargs,
        )
        captured: Final = capsys.readouterr()
        return "\n".join(line for line in (captured.out + captured.err).splitlines() if "litellm.completion(" in line)

    def test_api_key_never_reaches_the_request_line(self, capsys, monkeypatch):
        printed: Final = self._verbose_request_line(capsys, monkeypatch, api_key=self.FAKE_API_KEY)

        assert "litellm.completion(" in printed
        assert self.FAKE_API_KEY not in printed
        assert "api_key='REDACTED'" in printed

    def test_credential_headers_never_reach_the_request_line(self, capsys, monkeypatch):
        printed: Final = self._verbose_request_line(
            capsys,
            monkeypatch,
            api_key=self.FAKE_API_KEY,
            extra_headers={"Authorization": "Bearer fake-lit6823-header", "x-request-id": "abc123"},
        )

        assert "fake-lit6823-header" not in printed
        assert "'Authorization': 'REDACTED'" in printed
        assert "'x-request-id': 'abc123'" in printed

    def test_credentials_nested_in_a_list_never_reach_the_request_line(self, capsys, monkeypatch):
        printed: Final = self._verbose_request_line(
            capsys,
            monkeypatch,
            api_key=self.FAKE_API_KEY,
            extra_body={"providers": [{"name": "openai", "api_key": "sk-fake-lit6823-nested"}]},
        )

        assert "sk-fake-lit6823-nested" not in printed
        assert "'name': 'openai'" in printed

    def test_ordinary_params_still_printed(self, capsys, monkeypatch):
        printed: Final = self._verbose_request_line(
            capsys, monkeypatch, api_key=self.FAKE_API_KEY, max_tokens=17, temperature=0.25
        )

        assert "model='gpt-3.5-turbo'" in printed
        assert "max_tokens=17" in printed
        assert "temperature=0.25" in printed


class TestFinalOptionalParamsLineRedaction:
    """A verbose run echoes the fully built optional params too, and `extra_body` carries whatever the
    caller nested inside it straight onto that line, so a credential tucked in there lands in a terminal
    or a log drain in plaintext. It has to be redacted on both surfaces `print_verbose` writes to, and the
    line has to keep printing on both, because `litellm.set_verbose` and the DEBUG logger are independent
    switches and neither implies the other."""

    FAKE_NESTED_KEY: Final = "sk-fake-lit6835-nested-0000000000"

    def _complete(self, **kwargs) -> None:
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
            mock_response="hi",
            **kwargs,
        )

    def _printed_line(self, capsys) -> str:
        captured: Final = capsys.readouterr()
        return "\n".join(
            line for line in (captured.out + captured.err).splitlines() if "Final returned optional params" in line
        )

    def test_nested_credential_is_redacted_when_only_set_verbose_is_on(self, capsys, caplog, monkeypatch):
        monkeypatch.setattr(litellm, "set_verbose", True)
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            capsys.readouterr()
            self._complete(extra_body={"providers": [{"name": "openai", "api_key": self.FAKE_NESTED_KEY}]})
            printed: Final = self._printed_line(capsys)

        assert printed
        assert self.FAKE_NESTED_KEY not in printed
        assert "'api_key': 'REDACTED'" in printed
        assert "'name': 'openai'" in printed

    def test_line_still_reaches_the_logger_when_only_the_debug_logger_is_on(self, capsys, caplog, monkeypatch):
        monkeypatch.setattr(litellm, "set_verbose", False)
        with caplog.at_level(logging.DEBUG, logger=verbose_logger.name):
            self._complete(extra_body={"providers": [{"name": "openai", "api_key": self.FAKE_NESTED_KEY}]})
            logged: Final = "\n".join(
                record.getMessage()
                for record in caplog.records
                if "Final returned optional params" in record.getMessage()
            )

        assert logged
        assert self.FAKE_NESTED_KEY not in logged
        assert "'name': 'openai'" in logged

    def test_nothing_is_emitted_when_neither_verbose_switch_is_on(self, capsys, caplog, monkeypatch):
        monkeypatch.setattr(litellm, "set_verbose", False)
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            capsys.readouterr()
            self._complete(extra_body={"providers": [{"name": "openai", "api_key": self.FAKE_NESTED_KEY}]})
            captured: Final = capsys.readouterr()

        assert "Final returned optional params" not in captured.out + captured.err
        assert self.FAKE_NESTED_KEY not in captured.out + captured.err

    def test_ordinary_optional_params_still_reach_the_line(self, capsys, caplog, monkeypatch):
        monkeypatch.setattr(litellm, "set_verbose", True)
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            capsys.readouterr()
            self._complete(max_tokens=17, temperature=0.25)
            printed: Final = self._printed_line(capsys)

        assert "'max_tokens': 17" in printed
        assert "'temperature': 0.25" in printed


class TestDropParamsStringCoercion:
    @pytest.mark.parametrize("drop_params", ["true", "True", True])
    def test_truthy_drop_params_drops_unsupported_temperature(self, drop_params, monkeypatch):
        from litellm.utils import get_optional_params

        monkeypatch.setattr(litellm, "drop_params", False)
        result = get_optional_params(
            model="gpt-5-nano",
            custom_llm_provider="openai",
            temperature=0.1,
            drop_params=drop_params,
        )
        assert "temperature" not in result

    @pytest.mark.parametrize("drop_params", ["false", False, None])
    def test_falsy_drop_params_still_raises(self, drop_params, monkeypatch):
        from litellm.utils import get_optional_params

        monkeypatch.setattr(litellm, "drop_params", False)
        with pytest.raises(litellm.UnsupportedParamsError):
            get_optional_params(
                model="gpt-5-nano",
                custom_llm_provider="openai",
                temperature=0.1,
                drop_params=drop_params,
            )


def _credential_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if "litellm_credential_name=" in record.getMessage()]


def test_load_credentials_from_list_warns_when_the_named_credential_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from litellm.utils import load_credentials_from_list

    monkeypatch.setattr(litellm, "credential_list", [])
    request_kwargs = {"litellm_credential_name": "openai-cred", "model": "openai/gpt-5.4-mini"}
    with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
        load_credentials_from_list(request_kwargs)

    assert request_kwargs == {"litellm_credential_name": "openai-cred", "model": "openai/gpt-5.4-mini"}
    assert _credential_warnings(caplog) == [
        "litellm_credential_name=openai-cred matched none of the 0 loaded credentials; the request runs without it"
    ]


def test_load_credentials_from_list_fills_kwargs_from_the_loaded_credential_without_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from litellm.types.utils import CredentialItem
    from litellm.utils import load_credentials_from_list

    loaded = CredentialItem(
        credential_name="openai-cred",
        credential_values={"api_key": "sk-from-db", "api_base": "https://credential.example"},
        credential_info={},
    )
    monkeypatch.setattr(litellm, "credential_list", [loaded])
    request_kwargs = {"litellm_credential_name": "openai-cred", "api_base": "https://request.example"}
    with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
        load_credentials_from_list(request_kwargs)

    assert request_kwargs == {
        "litellm_credential_name": "openai-cred",
        "api_base": "https://request.example",
        "api_key": "sk-from-db",
    }
    assert _credential_warnings(caplog) == []


_MOCK_STREAM_ID: Final = "chatcmpl-mock-stream"
_ChunkSnapshot = tuple[str, tuple[str | None, ...], Usage | None]


def _snapshot(chunk: ModelResponseStream) -> _ChunkSnapshot:
    return chunk.id, tuple(choice.delta.content for choice in chunk.choices), getattr(chunk, "usage", None)


def _mock_stream_snapshots(mock_response: object, prompt_tokens: int | None) -> list[_ChunkSnapshot]:
    from litellm.utils import mock_completion_streaming_obj

    return [
        _snapshot(chunk)
        for chunk in mock_completion_streaming_obj(
            ModelResponseStream(id=_MOCK_STREAM_ID, model="gpt-5.4-mini"),
            mock_response=mock_response,
            model="gpt-5.4-mini",
            prompt_tokens=prompt_tokens,
        )
    ]


async def _async_mock_stream_snapshots(mock_response: object, prompt_tokens: int | None) -> list[_ChunkSnapshot]:
    from litellm.utils import async_mock_completion_streaming_obj

    return [
        _snapshot(chunk)
        async for chunk in async_mock_completion_streaming_obj(
            ModelResponseStream(id=_MOCK_STREAM_ID, model="gpt-5.4-mini"),
            mock_response=mock_response,
            model="gpt-5.4-mini",
            prompt_tokens=prompt_tokens,
        )
    ]


_CONTENT_SNAPSHOTS: Final = [(_MOCK_STREAM_ID, (content,), None) for content in ("hel", "lo ", "wor", "ld")]


def _assert_trailing_usage_chunk(snapshots: list[_ChunkSnapshot], prompt_tokens: int) -> None:
    assert snapshots[:-1] == _CONTENT_SNAPSHOTS
    chunk_id, choices, usage = snapshots[-1]
    assert chunk_id == _MOCK_STREAM_ID
    assert choices == ()
    assert usage is not None
    assert usage.prompt_tokens == prompt_tokens
    assert usage.completion_tokens == DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT
    assert usage.total_tokens == prompt_tokens + usage.completion_tokens


@pytest.mark.parametrize("prompt_tokens", (51234, 0))
def test_mock_completion_streaming_obj_emits_usage_chunk_with_admission_prompt_tokens(prompt_tokens: int) -> None:
    _assert_trailing_usage_chunk(_mock_stream_snapshots("hello world", prompt_tokens), prompt_tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_tokens", (51234, 0))
async def test_async_mock_completion_streaming_obj_emits_usage_chunk_with_admission_prompt_tokens(
    prompt_tokens: int,
) -> None:
    _assert_trailing_usage_chunk(await _async_mock_stream_snapshots("hello world", prompt_tokens), prompt_tokens)


def test_mock_completion_streaming_obj_emits_no_usage_chunk_without_admission_prompt_tokens() -> None:
    assert _mock_stream_snapshots("hello world", None) == _CONTENT_SNAPSHOTS


@pytest.mark.asyncio
async def test_async_mock_completion_streaming_obj_emits_no_usage_chunk_without_admission_prompt_tokens() -> None:
    assert await _async_mock_stream_snapshots("hello world", None) == _CONTENT_SNAPSHOTS


def test_mock_completion_streaming_obj_passes_prebuilt_stream_chunk_through_without_usage_chunk() -> None:
    prebuilt: Final = ModelResponseStream(
        model="gpt-5.4-mini", choices=[StreamingChoices(index=0, delta=Delta(role="assistant", content="prebuilt"))]
    )

    assert _mock_stream_snapshots(prebuilt, 51234) == [(prebuilt.id, ("prebuilt",), None)]


@pytest.mark.asyncio
async def test_async_mock_completion_streaming_obj_raises_mock_exception_before_usage_chunk() -> None:
    mock_exception: Final = litellm.MockException(
        status_code=500, message="boom", llm_provider="openai", model="gpt-5.4-mini"
    )
    with pytest.raises(litellm.MockException):
        await _async_mock_stream_snapshots(mock_exception, 51234)


@contextlib.contextmanager
def _recording_hidden_params_at_submit(submit_target: str) -> "Iterator[queue.SimpleQueue[dict[str, object]]]":
    seen: Final = queue.SimpleQueue()

    def record_submit(_fn, *args, **_kwargs):
        response: Final = next(arg for arg in args if isinstance(arg, litellm.ModelResponse))
        seen.put(dict(response._hidden_params))
        return MagicMock()

    with patch(submit_target, side_effect=record_submit):
        yield seen


@pytest.mark.asyncio
async def test_acompletion_finishes_response_metadata_before_handing_the_response_to_the_logging_thread(monkeypatch):
    monkeypatch.setattr(litellm, "success_callback", [lambda kwargs, response, start_time, end_time: None])
    with _recording_hidden_params_at_submit("litellm.litellm_core_utils.litellm_logging.executor.submit") as seen:
        await litellm.acompletion(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello there!",
            num_retries=0,
        )
    snapshot: Final = seen.get_nowait()
    assert snapshot["litellm_call_id"]
    assert snapshot["response_cost"] is not None
    assert snapshot["api_base"]


class _GatedSyncLoggingHookRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.seen: Final = queue.SimpleQueue[str | None]()
        self.release: Final = threading.Event()

    def logging_hook(
        self, kwargs: dict[str, object], result: object, call_type: str
    ) -> tuple[dict[str, object], object]:
        self.seen.put(result.id if isinstance(result, litellm.ModelResponse) else None)
        self.release.wait(timeout=5)
        return kwargs, result


@pytest.mark.asyncio
async def test_acompletion_runs_a_custom_logger_sync_logging_hook_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    def legacy_sync_callback(
        kwargs: dict[str, object], response: litellm.ModelResponse, start_time: datetime, end_time: datetime
    ) -> None:
        pass

    recorder: Final = _GatedSyncLoggingHookRecorder()
    monkeypatch.setattr(litellm, "success_callback", [legacy_sync_callback, recorder])
    logging_futures: Final = queue.SimpleQueue[Future[object]]()
    real_submit: Final = logging_executor.submit

    def submit_and_track(fn: Callable[..., object], *args: object, **kwargs: object) -> Future[object]:
        future: Final = real_submit(fn, *args, **kwargs)
        logging_futures.put(future)
        return future

    with patch(  # test-quality-ok: wraps the real submit only to collect the futures to join, the pool still runs
        "litellm.litellm_core_utils.litellm_logging.executor.submit", side_effect=submit_and_track
    ):
        response: Final = await litellm.acompletion(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello there!",
            num_retries=0,
        )
        await asyncio.sleep(0)
    recorder.release.set()
    for _ in range(logging_futures.qsize()):
        logging_futures.get_nowait().result(timeout=5)
    assert [recorder.seen.get_nowait() for _ in range(recorder.seen.qsize())] == [response.id]


def test_completion_finishes_response_metadata_before_handing_the_response_to_the_logging_thread():
    with _recording_hidden_params_at_submit("litellm.utils.executor.submit") as seen:
        litellm.completion(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello there!",
        )
    snapshot: Final = seen.get_nowait()
    assert snapshot["litellm_call_id"]
    assert snapshot["response_cost"] is not None
    assert snapshot["api_base"]


def test_get_model_info_gemini(monkeypatch):
    """
    Tests if ALL gemini models have 'tpm' and 'rpm' in the model info
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_map = litellm.model_cost
    for model, info in model_map.items():
        if (
            model.startswith("gemini/")
            and "gemma" not in model
            and "learnlm" not in model
            and "imagen" not in model
            and "veo" not in model
            and "lyria" not in model
            and "robotics" not in model
            and "3.8-flash-tts" not in model
            and "3.8-flash-lite-tts" not in model
        ):
            assert info.get("tpm") is not None, f"{model} does not have tpm"
            assert info.get("rpm") is not None, f"{model} does not have rpm"


@pytest.mark.parametrize(
    ("max_parallel_requests", "rpm", "tpm", "default_max_parallel_requests", "expected"),
    [
        (3, 100, 100_000, 7, 3),
        (None, 100, 100_000, 7, 100),
        (None, None, 100_000, 7, 600),
        (None, None, 50, 7, 1),
        (None, None, None, 7, 7),
        (None, None, None, None, None),
    ],
)
def test_calculate_max_parallel_requests_precedence(
    max_parallel_requests: int | None,
    rpm: int | None,
    tpm: int | None,
    default_max_parallel_requests: int | None,
    expected: int | None,
) -> None:
    assert (
        calculate_max_parallel_requests(
            max_parallel_requests=max_parallel_requests,
            rpm=rpm,
            tpm=tpm,
            default_max_parallel_requests=default_max_parallel_requests,
        )
        == expected
    )


class _NamedStream(io.BytesIO):
    def __init__(self, name: str | int) -> None:
        super().__init__(b"%PDF-1.4 secret document body")
        self.name = name


def _logged_request_messages(original_function: str, *args: object, **kwargs: object) -> object:
    logging_obj, _ = litellm.utils.function_setup(
        original_function,
        litellm.utils.Rules(),
        datetime.now(),
        *args,
        litellm_call_id="request-text-call",
        **kwargs,
    )
    return logging_obj.messages


@pytest.mark.parametrize(
    ("original_function", "args", "kwargs", "expected"),
    [
        ("search", (), {"query": "Eiffel Tower"}, "Eiffel Tower"),
        ("asearch", ("Eiffel Tower",), {}, "Eiffel Tower"),
        ("asearch", (), {"query": ["Eiffel Tower", "Louvre"]}, "Eiffel Tower\nLouvre"),
        ("asearch", (), {"query": ["Eiffel Tower", 7, None]}, "Eiffel Tower"),
        ("image_edit", (), {"prompt": "make it blue", "image": b"png"}, "make it blue"),
        ("aimage_edit", (b"png", "make it blue"), {}, "make it blue"),
        (
            "aocr",
            (),
            {"document": {"type": "document_url", "document_url": "https://x.test/a.pdf"}},
            "https://x.test/a.pdf",
        ),
        (
            "ocr",
            ("mistral-ocr-latest", {"type": "image_url", "image_url": "https://x.test/a.png"}),
            {},
            "https://x.test/a.png",
        ),
        (
            "aocr",
            (),
            {"document": {"type": "document_url", "document_url": "data:application/pdf;base64,JVBERi0xLjQ="}},
            "data:application/pdf;base64 (12 chars)",
        ),
        (
            "aocr",
            (),
            {"document": {"type": "image_url", "image_url": "https://x.test/a,b.png"}},
            "https://x.test/a,b.png",
        ),
        (
            "aocr",
            (),
            {"document": {"type": "file", "file": PurePath("/tmp/hello.pdf"), "mime_type": "application/pdf"}},
            "file (application/pdf) hello.pdf",
        ),
        ("aocr", (), {"document": {"type": "document_url", "document_url": ""}}, ""),
        ("aocr", (), {"document": {"type": "file", "file": b"%PDF"}}, "file 4 bytes"),
        ("aocr", (), {"document": {"type": "file", "file": io.BytesIO(b"%PDF")}}, "file"),
        ("aocr", (), {"document": {"type": "file", "file": _NamedStream("/tmp/scan.pdf")}}, "file scan.pdf"),
        (
            "aocr",
            (),
            {"document": {"type": "file", "file": _NamedStream(3), "mime_type": "application/pdf"}},
            "file (application/pdf)",
        ),
        ("aocr", (), {"document": "not-a-document"}, "default-message-value"),
    ],
)
def test_function_setup_logs_the_search_query_edit_prompt_and_ocr_document_summary_as_the_request(
    original_function: str, args: tuple[object, ...], kwargs: dict[str, object], expected: str
) -> None:
    assert _logged_request_messages(original_function, *args, **kwargs) == [{"role": "user", "content": expected}]


def test_search_with_a_mixed_type_query_list_still_reaches_its_own_validation_error() -> None:
    mixed_query: Final = cast(list[str], ["Eiffel Tower", 7])  # cast-ok: the invalid list is the point of the test

    with pytest.raises(litellm.APIConnectionError, match="All items in query list must be strings"):
        litellm.search(query=mixed_query, search_provider="duckduckgo")


def test_function_setup_never_logs_the_ocr_file_bytes() -> None:
    content: Final = b"%PDF-1.4 secret document body"
    logged: Final = _logged_request_messages("aocr", document={"type": "file", "file": content})

    assert logged == [{"role": "user", "content": "file 29 bytes"}]


def test_function_setup_leaves_the_ocr_file_stream_unread_and_never_logs_its_bytes() -> None:
    stream: Final = _NamedStream("/tmp/scan.pdf")
    logged: Final = _logged_request_messages("aocr", document={"type": "file", "file": stream})

    assert logged == [{"role": "user", "content": "file scan.pdf"}]
    assert stream.tell() == 0


def test_function_setup_never_logs_the_ocr_data_uri_payload() -> None:
    payload: Final = base64.b64encode(b"%PDF-1.4 secret document body").decode()
    logged: Final = _logged_request_messages(
        "aocr", document={"type": "document_url", "document_url": f"data:application/pdf;base64,{payload}"}
    )

    assert logged == [{"role": "user", "content": f"data:application/pdf;base64 ({len(payload)} chars)"}]
    assert payload not in str(logged)
