# What is this?
## File for 'response_cost' calculation in Logging
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import accumulate, chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias, cast

from httpx import Response
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm import verbose_logger
from litellm.constants import (
    DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND,
    OPENAI_LIVE_TERMINAL_RESPONSE_EVENT_TYPES,
    REALTIME_METERED_USAGE_KEY,
)
from litellm.litellm_core_utils.audio_utils.utils import normalized_audio_duration_seconds
from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    attribute_or_none,
    billed_usage_cost,
    require_attributes,
)
from litellm.litellm_core_utils.decisions_pricing import decisions_fields, is_decisions_call_type
from litellm.litellm_core_utils.hidden_params import HIDDEN_PARAMS_ATTR
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
    AdjustedCost,
    AdjustFacts,
    BatchLineDocument,
    BranchFacts,
    BreakdownFrom,
    CallFacts,
    CandidateFacts,
    CandidateFee,
    Characters,
    ClipUsage,
    ClipUse,
    CostRow,
    CustomPricing,
    CustomRates,
    DurationClip,
    Elapsed,
    HiddenFacts,
    ImageDocument,
    MeteredUse,
    ModelFacts,
    OcrDocument,
    OtherClip,
    PricedSides,
    PricedUsage,
    ReceiptFacts,
    ReportedCost,
    RerankDocument,
    SessionCandidate,
    SessionDocument,
    SessionTokensUse,
    SessionTurnUse,
    SpeechDocument,
    TokenCandidateUse,
    TokenClip,
    ToolsOfCall,
    ToolUse,
    TranscriptionDocument,
    TranslationUse,
    VectorStoreSearchDocument,
    adjusted_cost,
    batch_usage,
    billed_token_rates,
    branch_receipt,
    cost_candidates,
    cost_rows,
    custom_pricing_usage,
    custom_receipt,
    part_costs,
    price,
    price_call,
    price_reported,
    price_session,
    price_tools,
    price_units,
    priced_names,
    priced_usage,
    prompt_and_completion,
    raw_counts,
    total_or_zero,
)
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    BilledTokenRates,
    CostCalculatorUtils,
    billed_usage,
)
from litellm.litellm_core_utils.llm_response_utils.response_metadata import response_ms_since_logging_start
from litellm.litellm_core_utils.model_cost_table import AliasedCostEntry
from litellm.litellm_core_utils.redact_messages import should_redact_message_logging
from litellm.litellm_core_utils.usage.token_usage_processors import (
    BaseTokenUsageProcessor as BaseTokenUsageProcessor,  # noqa: PLC0414  # re-export at the old path, tests import it
)
from litellm.litellm_core_utils.usage.token_usage_processors import (
    RealtimeAPITokenUsageProcessor as RealtimeAPITokenUsageProcessor,  # noqa: PLC0414  # re-export, tests import it
)
from litellm.litellm_core_utils.usage.token_usage_processors import (
    ResponsesWebSocketTokenUsageProcessor,
    translation_session_seconds,
)
from litellm.litellm_core_utils.usage.usage_object import (
    decisions_usage,
    extraction_usage,
    get_usage_object,
    is_known_usage_object,
    usage_from_model,
)
from litellm.litellm_core_utils.usage.usage_object_transformation import (
    InteractionsUsageObjectTransformation,
    TranscriptionUsageObjectTransformation,
)
from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo
from litellm.llms.azure_ai.cost_calculator import (
    is_azure_model_router as azure_ai_is_model_router_name,
)
from litellm.llms.azure_ai.cost_calculator import (
    receipt as azure_ai_receipt,
)
from litellm.llms.azure_ai.cost_calculator import (
    router_fee_row as azure_ai_router_fee_row,
)
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.databricks.cost_calculator import pricing_row as databricks_pricing_row
from litellm.llms.fireworks_ai.cost_calculator import pricing_row as fireworks_pricing_row
from litellm.llms.openai.cost_calculation import (
    cost_per_second as openai_cost_per_second,
)
from litellm.llms.openai.cost_calculation import (
    priced_video,
)
from litellm.llms.perplexity.cost_calculator import pricing_row as perplexity_pricing_row
from litellm.llms.perplexity.cost_calculator import stated_cost as perplexity_stated_cost
from litellm.llms.together_ai.cost_calculator import (
    get_model_params_and_category,
    has_together_registry_pricing,
)
from litellm.llms.vertex_ai.common_utils import get_vertex_ai_lyria_generation_cost
from litellm.llms.xai.cost_calculator import pricing_row as xai_pricing_row
from litellm.llms.xai.cost_calculator import stated_cost as xai_stated_cost
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.rust_bridge._native import NativeReceipt, NativeUnpriced, billable_service_tier
from litellm.types.agents import LiteLLMSendMessageResponse
from litellm.types.decisions import (
    DecisionsResponse,
    DecisionsUsage,
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
)
from litellm.types.extraction import ExtractionResponse, ExtractionUsage
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.llms.openai import (
    HttpxBinaryResponseContent,
    ImageGenerationRequestQuality,
    OpenAIModerationResponse,
    OpenAIRealtimeStreamList,
    ResponsesAPIResponse,
)
from litellm.types.llms.xai import RealtimeMeteredUsage
from litellm.types.rerank import RerankBilledUnits, RerankResponse
from litellm.types.utils import (
    CallTypesLiteral,
    LiteLLMRealtimeStreamLoggingObject,
    LlmProviders,
    ModelInfo,
    PromptTokensDetailsWrapper,
    ServiceTier,
    StandardBuiltInToolsParams,
    TranscriptionUsageDurationObject,
    TranscriptionUsageTokensObject,
    Usage,
    VectorStoreSearchResponse,
)
from litellm.utils import (
    CallTypes,
    CostPerToken,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
    ProviderConfigManager,
    TextCompletionResponse,
    TranscriptionResponse,
    cached_get_model_info_helper,
    mapped_model_fill_missing,
    model_info_with_pricing_overrides,
    token_counter,
    unmapped_model_generalization,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LitellmLoggingObject,
    )
    from litellm.llms.base_llm.ocr.transformation import OCRUsageInfo
else:
    LitellmLoggingObject = Any

# Pre-resolved CallTypes enum values for fast membership checks
_A2A_CALL_TYPES: Final = frozenset(
    {
        CallTypes.asend_message.value,
        CallTypes.send_message.value,
    }
)

_VIDEO_CALL_TYPES: Final = frozenset(
    {
        CallTypes.create_video.value,
        CallTypes.acreate_video.value,
        CallTypes.video_edit.value,
        CallTypes.avideo_edit.value,
        CallTypes.video_remix.value,
        CallTypes.avideo_remix.value,
    }
)


_SPEECH_CALL_TYPES: Final = frozenset(
    {
        CallTypes.speech.value,
        CallTypes.aspeech.value,
    }
)

_TRANSCRIPTION_CALL_TYPES: Final = frozenset(
    {
        CallTypes.atranscription.value,
        CallTypes.transcription.value,
    }
)

_RERANK_CALL_TYPES: Final = frozenset(
    {
        CallTypes.rerank.value,
        CallTypes.arerank.value,
    }
)

_SEARCH_CALL_TYPES: Final = frozenset(
    {
        CallTypes.search.value,
        CallTypes.asearch.value,
    }
)

_AREALTIME_CALL_TYPE: Final = CallTypes.arealtime.value
_RUST_PRICED_CALL_TYPES: Final = frozenset(
    {
        CallTypes.completion.value,
        CallTypes.acompletion.value,
        CallTypes.text_completion.value,
        CallTypes.atext_completion.value,
        CallTypes.embedding.value,
        CallTypes.aembedding.value,
        CallTypes.anthropic_messages.value,
        CallTypes.aanthropic_messages.value,
        CallTypes.responses.value,
        CallTypes.aresponses.value,
    }
)
_MCP_CALL_TYPE: Final = CallTypes.call_mcp_tool.value


def _router_fee_row(
    model: str,
    custom_llm_provider: str | None,
    request_model: str | None,
    completion_response: object,
) -> ModelInfo | None:
    """The Model Router row whose fee ``completion_cost`` bills beside a routed model's own tokens, ``None`` when no fee
    applies. The fee is named by the request model, or by a router name the response's hidden params carry."""
    if custom_llm_provider not in ("azure_ai", "azure") or azure_ai_is_model_router_name(model):
        return None
    hidden_params: Final = _hidden_params_of(completion_response) if completion_response is not None else None
    hidden: Final[object] = (
        None if hidden_params is None else hidden_params.get("model") or hidden_params.get("litellm_model_name")
    )
    hidden_model: Final = hidden if isinstance(hidden, str) and hidden else None
    hidden_router: Final = hidden_model is not None and (
        "model_router" in hidden_model.lower() or "model-router" in hidden_model.lower()
    )
    fee_model: Final = (hidden_model if hidden_router or request_model is None else request_model) or model
    if AzureFoundryModelInfo.get_azure_ai_route(fee_model) != "model_router" or not azure_ai_is_model_router_name(
        fee_model
    ):
        return None
    try:
        return azure_ai_router_fee_row(fee_model)
    except Exception as e:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model
        verbose_logger.debug("Error calculating additional costs: %s", e)
        return None


def _breakdown_row(
    model: str,
    custom_llm_provider: str | None,
    custom_cost_per_token: CostPerToken | None,
    bedrock_model_info: ModelInfo | None,
    usage: Usage | None,
    stored: bool,
) -> tuple[BreakdownFrom, ModelInfo | None]:
    """The row main's stored token-type breakdown read: the custom rates, the Bedrock row the total priced, else the
    candidate's own cost map row. A call without usage or a logging object to store it on reads none."""
    if usage is None or not stored:
        return "absent", None
    if custom_cost_per_token is not None:
        return "priced", None
    if bedrock_model_info is not None:
        return "row", bedrock_model_info
    try:
        return "row", litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model
        return "absent", None


def _unit_kind_cost(
    sides: tuple[float, float],
    model: str,
    custom_llm_provider: str | None,
    usage: Usage | None,
    service_tier: str | None,
    data_residency: str | None,
    vertex_location: str | None,
    stored_row: tuple[BreakdownFrom, ModelInfo | None],
    tool_fees: float,
    fee_row: ModelInfo | None,
    adjust: AdjustFacts,
    litellm_logging_obj: LitellmLoggingObject | None,
) -> float:
    """``completion_cost`` of a per-unit kind priced by its own calculator: its sides, the tool and router fees and the
    token-type breakdown its usage stores, adjusted in Rust."""
    beside: Final = (
        None
        if usage is None
        else _stored_breakdown_receipt(
            model, custom_llm_provider or "", usage, service_tier, data_residency, vertex_location, stored_row, fee_row
        )
    )
    additional: Final = part_costs(beside) if beside is not None else MappingProxyType({})
    original: Final = sides[0] + sides[1] + tool_fees + sum(additional.values())
    adjusted: Final = adjusted_cost(original, adjust["provider"], adjust["model"], adjust["region"])
    _store_cost_breakdown_in_logging_obj(
        litellm_logging_obj=litellm_logging_obj,
        prompt_tokens_cost_usd_dollar=sides[0],
        completion_tokens_cost_usd_dollar=sides[1],
        cost_for_built_in_tools_cost_usd_dollar=tool_fees,
        total_cost_usd_dollar=adjusted.cost,
        original_cost=original,
        additional_costs=dict(additional) or None,
        discount_percent=adjusted.discount_percent,
        discount_amount=adjusted.discount_amount,
        margin_percent=adjusted.margin_percent,
        margin_fixed_amount=adjusted.margin_fixed_amount,
        margin_total_amount=adjusted.margin_total_amount,
        cache_read_cost=None if beside is None else beside.cache_read_cost,
        cache_creation_cost=None if beside is None else beside.cache_creation_cost,
        reasoning_cost=None if beside is None else beside.reasoning_cost,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        billed_token_rates=None if beside is None else billed_token_rates(beside),
    )
    return adjusted.cost


def _stored_breakdown_receipt(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
    service_tier: str | None,
    data_residency: str | None,
    vertex_location: str | None,
    stored_row: tuple[BreakdownFrom, ModelInfo | None],
    fee_row: ModelInfo | None,
) -> NativeReceipt | None:
    """The token receipt of ``usage`` on the breakdown row alone, for the breakdown and router fee a per-unit kind
    stores beside its own sides."""
    facts: Final = ReceiptFacts(
        breakdown="priced" if stored_row[1] is not None else "absent",
        router_fee=None if fee_row is None else "additional",
        fee_row=fee_row,
    )
    if stored_row[1] is None:
        return (
            None
            if fee_row is None
            else custom_receipt(
                model,
                custom_llm_provider,
                priced_usage(usage, web_search=None),
                CustomPricing(per_token=None, per_second=0.0),
                None,
                receipt=facts,
            )
        )
    return price(
        ModelFacts(by="model", model=model, provider=custom_llm_provider, region=None),
        priced_usage(usage, web_search=None),
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        row=stored_row[1],
        branch=BranchFacts(rules="generic"),
        receipt=facts,
    )


_TranscriptionRateKey: TypeAlias = Literal[
    "input_cost_per_second",
    "output_cost_per_second",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
    "input_cost_per_token",
    "output_cost_per_token",
]
_TRANSCRIPTION_TOKEN_RATE_KEYS: Final[tuple[_TranscriptionRateKey, ...]] = (
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
)
_TRANSCRIPTION_SECOND_RATE_KEYS: Final[tuple[_TranscriptionRateKey, ...]] = (
    "input_cost_per_second",
    "output_cost_per_second",
)


def _usage_reports_tokens(usage: Usage | None) -> bool:
    if usage is None:
        return False
    prompt_details: Final[PromptTokensDetailsWrapper | None] = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None and (
        (getattr(prompt_details, "audio_tokens", 0) or 0) > 0 or (getattr(prompt_details, "text_tokens", 0) or 0) > 0
    ):
        return True
    return (getattr(usage, "prompt_tokens", 0) or 0) > 0 or (getattr(usage, "completion_tokens", 0) or 0) > 0


def _transcription_cost(
    model: str,
    model_without_prefix: str,
    custom_llm_provider: str,
    duration: float,
    usage: Usage,
    service_tier: str | None,
    data_residency: str | None,
) -> tuple[float, float]:
    if not _usage_reports_tokens(usage):
        return openai_cost_per_second(
            model=model_without_prefix, custom_llm_provider=custom_llm_provider, duration=duration
        )
    model_info: Final = cached_get_model_info_helper(
        model=model_without_prefix, custom_llm_provider=custom_llm_provider
    )
    declared: Final[Mapping[str, object] | None] = litellm.model_cost.get(model_info.get("key"))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # model cost map is dynamically loaded
    outcome: Final = price_units(
        TranscriptionDocument(
            kind="transcription",
            model=model_without_prefix,
            provider=custom_llm_provider,
            seconds=duration,
            tokens=billed_usage(usage),
            service_tier=service_tier,
            data_residency=data_residency,
        ),
        (litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider),),
        catalog=declared if declared is not None else model_info,
    )
    return prompt_and_completion(outcome)


def _speech_cost(
    model: str,
    custom_llm_provider: str,
    prompt_characters: int | None,
    usage: Usage,
    service_tier: str | None,
    data_residency: str | None,
) -> tuple[float, float]:
    lyria_generation_cost: Final = (
        get_vertex_ai_lyria_generation_cost(model=model)
        if custom_llm_provider in ("vertex_ai", "vertex_ai_beta")
        else None
    )
    if lyria_generation_cost is not None:
        return 0.0, lyria_generation_cost
    catalog: Final = cached_get_model_info_helper(
        model=model, custom_llm_provider=custom_llm_provider, preserve_unset_costs=True
    )
    outcome: Final = price_units(
        SpeechDocument(
            kind="speech",
            model=model,
            provider=custom_llm_provider,
            characters=prompt_characters,
            generations=None,
            tokens=billed_usage(usage),
            service_tier=service_tier,
            data_residency=data_residency,
        ),
        (litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider),),
        catalog=catalog,
    )
    if isinstance(outcome, NativeReceipt):
        return outcome.prompt_cost, outcome.completion_cost
    if outcome.reason == "missing_usage":
        raise ValueError(
            f"prompt_characters must be provided for tts calls. prompt_characters={prompt_characters}, "
            f"model={model}, custom_llm_provider={custom_llm_provider}"
        )
    raise ValueError(f"Model {model} does not have 'input_cost_per_character' or 'input_cost_per_token'")


def _transcription_duration_for_cost(completion_response: object) -> float:
    billed_duration: Final = _get_transcription_usage_duration(completion_response)
    if billed_duration is not None and billed_duration > 0:
        return billed_duration
    hidden_params: Final = as_str_mapping(attribute_or_none(completion_response, "_hidden_params")) or {}
    measured_duration: Final = normalized_audio_duration_seconds(
        hidden_params.get("audio_transcription_duration", attribute_or_none(completion_response, "duration"))
    )
    return measured_duration if measured_duration is not None and measured_duration > 0 else 0.0


class OCRPricing(TypedDict, total=False):
    ocr_cost_per_page: ReadOnly[float | None]
    ocr_cost_per_credit: ReadOnly[float | None]
    annotation_cost_per_page: ReadOnly[float | None]


def cost_per_token(
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    response_time_ms: float | None = 0.0,
    custom_llm_provider: str | None = None,
    region_name=None,
    ### CHARACTER PRICING ###
    prompt_characters: int | None = None,
    completion_characters: int | None = None,
    ### PROMPT CACHING PRICING ### - used for anthropic
    cache_creation_input_tokens: int | None = 0,
    cache_read_input_tokens: int | None = 0,
    ### CUSTOM PRICING ###
    custom_cost_per_token: CostPerToken | None = None,
    custom_cost_per_second: float | None = None,
    requested_model_info_override: ModelInfo | None = None,
    ### NUMBER OF QUERIES ###
    number_of_queries: int | None = None,
    ### USAGE OBJECT ###
    usage_object: Usage | None = None,  # just read the usage object if provided
    ### BILLED UNITS ###
    rerank_billed_units: RerankBilledUnits | None = None,
    ### CALL TYPE ###
    call_type: CallTypesLiteral = "completion",
    audio_transcription_file_duration: float = 0.0,  # for audio transcription calls - the file time in seconds
    ### SERVICE TIER ###
    service_tier: str | None = None,  # for OpenAI service tier pricing
    ### DATA RESIDENCY ###
    data_residency: str | None = None,  # for OpenAI regional-processing uplift (e.g. "eu", "us")
    ### VERTEX LOCATION ###
    vertex_location: str | None = None,  # for Vertex AI regional-endpoint uplift (e.g. "us-east5", "global")
    response: object | None = None,
    ### REQUEST MODEL ###
    request_model: str | None = None,  # original request model for router detection
    custom_model_info: OCRPricing | Sequence[Mapping[str, object]] | None = None,
    bedrock_model_info: ModelInfo | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Parameters:
        model (str): The name of the model to use. Default is ""
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        response_time (float): The amount of time, in milliseconds, it took the call to complete.
        prompt_characters (float): The number of characters in the prompt. Used for vertex ai cost calculation.
        completion_characters (float): The number of characters in the completion response. Used for vertex ai cost calculation.
        custom_llm_provider (str): The llm provider to whom the call was made (see init.py for full list)
        custom_cost_per_token: Optional[CostPerToken]: the cost per input + output token for the llm api call.
        custom_cost_per_second: Optional[float]: the cost per second for the llm api call.
        call_type: Optional[str]: the call type

    Returns:
        tuple: A tuple containing the cost in USD dollars for prompt tokens and completion tokens, respectively.
    """

    priced: Final = _priced_tokens(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_time_ms=response_time_ms,
        custom_llm_provider=custom_llm_provider,
        region_name=region_name,
        prompt_characters=prompt_characters,
        completion_characters=completion_characters,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        custom_cost_per_token=custom_cost_per_token,
        custom_cost_per_second=custom_cost_per_second,
        requested_model_info_override=requested_model_info_override,
        number_of_queries=number_of_queries,
        usage_object=usage_object,
        rerank_billed_units=rerank_billed_units,
        call_type=call_type,
        audio_transcription_file_duration=audio_transcription_file_duration,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        response=response,
        request_model=request_model,
        custom_model_info=custom_model_info,
        bedrock_model_info=bedrock_model_info,
        receipt=None,
        adjust=None,
    )
    return (priced.prompt_cost, priced.completion_cost) if isinstance(priced, NativeReceipt) else priced


def _priced_tokens(
    *,
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    response_time_ms: float | None = 0.0,
    custom_llm_provider: str | None = None,
    region_name: object = None,
    prompt_characters: int | None = None,
    completion_characters: int | None = None,
    cache_creation_input_tokens: int | None = 0,
    cache_read_input_tokens: int | None = 0,
    custom_cost_per_token: CostPerToken | None = None,
    custom_cost_per_second: float | None = None,
    requested_model_info_override: ModelInfo | None = None,
    number_of_queries: int | None = None,
    usage_object: Usage | None = None,
    rerank_billed_units: RerankBilledUnits | None = None,
    call_type: CallTypesLiteral = "completion",
    audio_transcription_file_duration: float = 0.0,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    response: object | None = None,
    request_model: str | None = None,
    custom_model_info: OCRPricing | Sequence[Mapping[str, object]] | None = None,
    bedrock_model_info: ModelInfo | None = None,
    receipt: ReceiptFacts | None,
    adjust: AdjustFacts | None,
) -> NativeReceipt | tuple[float, float]:
    """``cost_per_token``'s pricing: the receipt of a token call, the prompt and completion cost of any other kind.
    ``receipt`` and ``adjust`` are what ``completion_cost`` adds to a token call's receipt."""
    if model is None:
        raise Exception("Invalid arg. Model cannot be none.")

    ## RECONSTRUCT USAGE BLOCK ##
    if usage_object is not None:
        usage_block = usage_object
    else:
        usage_block = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

    ## CUSTOM PRICING ##
    # Normalize cache token counts across providers:
    #   - OpenAI-compatible: usage.prompt_tokens_details.cached_tokens
    #     (prompt_tokens already INCLUDES cached_tokens)
    #   - Anthropic: usage.cache_read_input_tokens / cache_creation_input_tokens
    #     (prompt_tokens does NOT include these — adjust before calling helper)
    _cache_read_tokens: float = 0
    _cache_creation_tokens: float = 0
    _is_anthropic_style = False

    if usage_object is not None:
        _pt_details: Final[PromptTokensDetailsWrapper | None] = getattr(usage_object, "prompt_tokens_details", None)
        if _pt_details is not None:
            _cache_read_tokens = float(getattr(_pt_details, "cached_tokens", 0) or 0)
            # OpenAI-compatible providers report cache-write tokens under
            # either `cache_write_tokens` (kimi-k2) or `cache_creation_tokens`.
            # Mirror db_spend_update_writer to stay symmetric.
            _cache_creation_tokens = float(
                getattr(_pt_details, "cache_write_tokens", 0) or getattr(_pt_details, "cache_creation_tokens", 0) or 0
            )

        _anthropic_read: Final[int | None] = getattr(usage_object, "cache_read_input_tokens", None)
        _anthropic_create: Final[int | None] = getattr(usage_object, "cache_creation_input_tokens", None)
        if _anthropic_read is not None or _anthropic_create is not None:
            _is_anthropic_style = True
            if _anthropic_read is not None:
                _cache_read_tokens = float(_anthropic_read)
            if _anthropic_create is not None:
                _cache_creation_tokens = float(_anthropic_create)

    if not _cache_read_tokens and cache_read_input_tokens:
        _cache_read_tokens = float(cache_read_input_tokens)
        _is_anthropic_style = True
    if not _cache_creation_tokens and cache_creation_input_tokens:
        _cache_creation_tokens = float(cache_creation_input_tokens)
        _is_anthropic_style = True

    # Anthropic reports prompt_tokens as input_tokens (excluding cache tokens).
    # Adjust so the helper's "prompt_tokens includes cache tokens" invariant holds.
    _normalized_prompt_tokens = float(prompt_tokens)
    if _is_anthropic_style:
        _normalized_prompt_tokens += _cache_read_tokens + _cache_creation_tokens

    elapsed: Final = Elapsed(response_ms=response_time_ms or 0.0, audio_seconds=audio_transcription_file_duration)
    custom: Final = _custom_pricing(custom_cost_per_token, custom_cost_per_second)
    if custom is not None:
        return custom_receipt(
            model,
            custom_llm_provider or "",
            custom_pricing_usage(
                prompt_tokens=int(_normalized_prompt_tokens),
                completion_tokens=completion_tokens,
                cached_tokens=int(_cache_read_tokens),
                cache_creation_tokens=int(_cache_creation_tokens),
                reasoning_tokens=_reasoning_tokens(usage_block),
            ),
            custom,
            response_time_ms,
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=vertex_location,
            receipt=receipt,
            adjust=adjust,
        )

    if requested_model_info_override is not None:
        return branch_receipt(
            model,
            custom_llm_provider or "",
            usage_block,
            requested_model_info_override,
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=vertex_location,
            branch=BranchFacts(rules="override", elapsed=elapsed),
            receipt=receipt,
            adjust=adjust,
        )

    caller_supplied_provider: Final = custom_llm_provider is not None
    cost_provider: Final[str] = (
        custom_llm_provider if custom_llm_provider is not None else litellm.get_llm_provider(model=model)[1]
    )
    cost_model, model_without_prefix = priced_names(
        model,
        custom_llm_provider if caller_supplied_provider else None,
        None if region_name is None else str(region_name),
    )

    if is_decisions_call_type(call_type):
        decisions_model_info: Final = cached_get_model_info_helper(model=cost_model, custom_llm_provider=cost_provider)
        return branch_receipt(
            cost_model,
            cost_provider,
            usage_block,
            {
                **decisions_model_info,
                **decisions_fields(_raw_cost_map_entry(decisions_model_info["key"]) or _NO_COST_MAP_ENTRY),
            },
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=vertex_location,
            branch=BranchFacts(elapsed=elapsed, decisions=True),
            receipt=receipt,
            adjust=adjust,
        )
    if call_type == "speech" or call_type == "aspeech":
        return _speech_cost(
            model=model_without_prefix,
            custom_llm_provider=cost_provider,
            prompt_characters=prompt_characters,
            usage=usage_block,
            service_tier=service_tier,
            data_residency=data_residency,
        )
    elif call_type == "arerank" or call_type == "rerank":
        return rerank_cost(
            model=cost_model,
            custom_llm_provider=cost_provider,
            billed_units=rerank_billed_units,
        )
    elif call_type == "avector_store_search" or call_type == "vector_store_search":
        return vector_store_search_cost(
            model=cost_model,
            custom_llm_provider=cost_provider,
            response=cast(VectorStoreSearchResponse, response),
        )
    elif call_type == "ocr" or call_type == "aocr":
        return ocr_cost(
            model=cost_model,
            custom_llm_provider=cost_provider,
            response=response,
            model_info=custom_model_info,
        )
    elif (
        call_type == "aretrieve_batch"
        or call_type == "retrieve_batch"
        or call_type == CallTypes.aretrieve_batch
        or call_type == CallTypes.retrieve_batch
    ):
        return batch_cost_calculator(
            usage=usage_block,
            model=cost_model,
            custom_llm_provider=cost_provider,
            data_residency=data_residency,
            vertex_location=vertex_location,
        )
    elif call_type == "atranscription" or call_type == "transcription":
        return _transcription_cost(
            model=cost_model,
            model_without_prefix=model_without_prefix,
            custom_llm_provider=cost_provider,
            duration=audio_transcription_file_duration,
            usage=usage_block,
            service_tier=service_tier,
            data_residency=data_residency,
        )
    elif call_type == "search" or call_type == "asearch":
        # Search providers use per-query pricing
        from litellm.search import search_provider_cost_per_query

        return search_provider_cost_per_query(
            model=cost_model,
            custom_llm_provider=cost_provider,
            number_of_queries=number_of_queries or 1,
            optional_params=(getattr(response, "_hidden_params", None) if response else None),
        )
    elif cost_provider == "azure_ai":
        return azure_ai_receipt(
            model=cost_model,
            usage=usage_block,
            request_model=request_model,
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=vertex_location,
            facts=receipt,
            adjust=adjust,
        )
    bedrock_row: Final = bedrock_model_info if cost_provider == "bedrock" else None
    stated: Final = _stated_cost(cost_provider, usage_block)
    priced_model, row, row_provider = (
        (cost_model, bedrock_row, None)
        if bedrock_row is not None
        else _branch_row(cost_model, model_without_prefix, cost_provider, usage_block, stated is not None)
    )
    return branch_receipt(
        priced_model,
        cost_provider,
        usage_block,
        row,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        branch=BranchFacts(
            rules="generic_tier" if bedrock_row is not None else "branch",
            elapsed=elapsed,
            characters=None
            if prompt_characters is None and completion_characters is None
            else Characters(prompt=prompt_characters, completion=completion_characters),
            embedding=call_type in _EMBEDDING_CALL_TYPES,
            reported=stated,
        ),
        row_provider=row_provider,
        receipt=receipt,
        adjust=adjust,
    )


_EMBEDDING_CALL_TYPES: Final = frozenset({CallTypes.embedding.value, CallTypes.aembedding.value})


def _reasoning_tokens(usage: Usage) -> int:
    """The reasoning tokens the stored breakdown bills: the completion details' count, else the usage's own."""
    details: Final[object] = getattr(usage, "completion_tokens_details", None)
    counted: Final[object] = getattr(details, "reasoning_tokens", 0)
    stated: Final[object] = getattr(usage, "reasoning_tokens", 0)
    return next((count for count in (counted, stated) if isinstance(count, int) and count > 0), 0)


def _custom_pricing(
    custom_cost_per_token: CostPerToken | None, custom_cost_per_second: float | None
) -> CustomPricing | None:
    if custom_cost_per_token is not None:
        return CustomPricing(
            per_token=CustomRates(
                input=custom_cost_per_token["input_cost_per_token"],
                output=custom_cost_per_token["output_cost_per_token"],
                cache_read=custom_cost_per_token.get("cache_read_input_token_cost"),
                cache_write=custom_cost_per_token.get("cache_creation_input_token_cost"),
            ),
            per_second=None,
        )
    if custom_cost_per_second is not None:
        return CustomPricing(per_token=None, per_second=custom_cost_per_second)
    return None


def _stated_cost(custom_llm_provider: str, usage: Usage) -> ReportedCost | None:
    """The cost a provider billed from its own usage fields reported, which its branch bills instead of tokens."""
    if custom_llm_provider == "perplexity":
        return perplexity_stated_cost(usage)
    if custom_llm_provider != "xai":
        return None
    reported: Final = xai_stated_cost(usage)
    return None if reported is None else ReportedCost(source="usage.cost", usd=reported)


def _branch_row(
    model: str, model_without_prefix: str, custom_llm_provider: str, usage: Usage, stated: bool
) -> tuple[str, Mapping[str, object] | None, str | None]:
    """The model name, the row and the provider the row is lowered under for the providers whose branch of
    ``cost_per_token`` finds its row through code of its own. Every other provider's row is Model Catalog's, so the
    row is ``None``. A lookup raises for an unmapped model, as main's did."""
    if custom_llm_provider == "vertex_ai":
        return model_without_prefix, None, None
    if custom_llm_provider == "databricks":
        return model, databricks_pricing_row(model), None
    if custom_llm_provider == "fireworks_ai":
        fireworks_row: Final = fireworks_pricing_row(model)
        return model, fireworks_row, fireworks_row["litellm_provider"]
    if custom_llm_provider == "perplexity":
        return model, perplexity_pricing_row(model, usage), None
    if custom_llm_provider == "xai":
        return model, xai_pricing_row(model, stated), None
    return model, None, None


def get_replicate_completion_pricing(completion_response: dict, total_time=0.0):
    """Replicate bills an unmapped model by the second it ran on an A100 80GB (https://replicate.com/pricing).
    ``total_time`` is in milliseconds; without it the span between the response's ``created`` and ``ended``."""
    elapsed_ms: Final = (
        total_time
        if total_time != 0.0
        else getattr(completion_response, "ended", time.time()) - completion_response.get("created", time.time())
    )
    return custom_receipt(
        "replicate",
        "replicate",
        custom_pricing_usage(prompt_tokens=0, completion_tokens=0, cached_tokens=0, cache_creation_tokens=0),
        CustomPricing(per_token=None, per_second=DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND),
        elapsed_ms,
    ).total


def has_hidden_params(obj: object) -> bool:
    return hasattr(obj, "_hidden_params")


def _get_hidden_str_for_cost_calc(hidden_params: object, key: str) -> str | None:
    if not isinstance(hidden_params, Mapping):
        return None
    value: Final[object] = hidden_params.get(key)
    return value if isinstance(value, str) and value else None


_NON_TOKEN_RATE_FIELDS: Final = frozenset(
    {
        "cost_per_second",
        "input_cost_per_second",
        "output_cost_per_second",
        "input_cost_per_query",
        "input_cost_per_text_message",
        "tiered_pricing",
    }
)


def _cost_map_entry_prices_anything(entry: Mapping[str, object]) -> bool:
    return any(
        value is not None and (field in _NON_TOKEN_RATE_FIELDS or ("cost_per" in field and "token" in field))
        for field, value in entry.items()
    )


def cost_map_key_prices_anything(key: str) -> bool:
    entry: Final = _raw_cost_map_entry(key)
    return entry is not None and _cost_map_entry_prices_anything(entry)


def _registered_fields(key: str | None) -> Mapping[str, object]:
    """The fields ``register_model`` wrote onto the row under ``key``."""
    entry: Final = None if key is None else litellm.model_cost.get(key)
    if not isinstance(entry, AliasedCostEntry):
        return MappingProxyType({})
    return MappingProxyType({name: value for name, value in entry.items() if name in entry.registered_fields})


def _bedrock_model_info(pricing: tuple[str, str | None, bool] | None) -> ModelInfo | None:
    """The Bedrock row Model Catalog found for a call (an inference profile's or the one a ``register_model`` price
    landed on), with the registration merged over the profile's row where main merged it."""
    if pricing is None:
        return None
    key, registration, merged = pricing
    row: Final = _cost_map_model_info(key, None if key == registration else "bedrock")
    if row is None or not merged:
        return row
    return model_info_with_pricing_overrides(row, _registered_fields(registration))


def _pricing_provider_for_cost_candidate(
    candidate_provider: str | None,
    request_provider: str | None,
    current_provider: object,
) -> str | None:
    if candidate_provider != request_provider:
        return candidate_provider
    return current_provider if isinstance(current_provider, str) else None


def _invoked_model_id_for_cost(
    optional_params: object,
    logging_obj: LitellmLoggingObject | None,
) -> str | None:
    recorded: Final[object] = getattr(logging_obj, "model_call_details", None)
    return _get_hidden_str_for_cost_calc(optional_params, "model_id") or _get_hidden_str_for_cost_calc(
        recorded, "model_id"
    )


def _loop_reads_hidden_params(completion_response: object) -> bool:
    """Whether main's candidate loop read the response's usage, and with it the provider and region its hidden params
    carry."""
    return completion_response is not None and (
        isinstance(completion_response, (BaseModel, dict))
        or isinstance(completion_response, HttpxBinaryResponseContent)
        and completion_response.usage is not None
    )


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _call_facts(
    completion_response: object,
    model: str | None,
    custom_llm_provider: object,
    base_model: str | None,
    custom_pricing: bool | None,
    router_model_id: str | None,
    region_name: object,
    invoked_model_id: str | None,
    pricing_provider: str | None = None,
) -> CallFacts:
    """What Model Catalog picks a call's candidates from: the request, the response's model and hidden params, and
    the provider and region main's candidate loop priced with once it read them."""
    hidden: Final[object] = getattr(completion_response, "_hidden_params", None)
    hidden_params: Final = as_str_mapping(hidden) if _loop_reads_hidden_params(completion_response) else None
    response_model: Final[object] = _get_response_model(completion_response)
    provider: Final = custom_llm_provider if isinstance(custom_llm_provider, str) else None
    loop_provider: Final[object] = (
        provider
        if hidden_params is None
        else pricing_provider or hidden_params.get("custom_llm_provider", provider or None)
    )
    loop_region: Final[object] = region_name if hidden_params is None else hidden_params.get("region_name", region_name)
    return CallFacts(
        by="call",
        provider=provider,
        request_model=_text(model) or None,
        response_model=_text(response_model),
        base_model=_text(base_model),
        custom_pricing=custom_pricing is True,
        deployment_id=_text(router_model_id),
        region=_text(region_name),
        invoked_model_id=_text(invoked_model_id),
        hidden=None
        if hidden is None
        else HiddenFacts(
            model=_get_hidden_str_for_cost_calc(hidden, "model"),
            litellm_model_name=_get_hidden_str_for_cost_calc(hidden, "litellm_model_name"),
            provider_response_model=_get_hidden_str_for_cost_calc(hidden, "provider_response_model"),
            region=_get_hidden_str_for_cost_calc(hidden, "region_name"),
        ),
        loop_provider=loop_provider if isinstance(loop_provider, str) else None,
        loop_region=loop_region if isinstance(loop_region, str) else None,
    )


def select_model_name_for_cost_calc(
    model: str | None,
    completion_response: object | None,
    base_model: str | None = None,
    custom_pricing: bool | None = None,
    custom_llm_provider: str | None = None,
    router_model_id: str | None = None,
    region_name: str | None = None,
    invoked_model_id: str | None = None,
) -> str | None:
    """The model name ``completion_cost`` prices first, as Model Catalog picks it: the deployment's custom-priced row,
    else the base model or the Bedrock row the call prices with, else the served model unless only the requested
    model's row has rates, else the requested model."""
    return cost_candidates(
        _call_facts(
            completion_response,
            model,
            custom_llm_provider,
            base_model,
            custom_pricing,
            router_model_id,
            region_name,
            invoked_model_id,
        )
    ).selected


_select_model_name_for_cost_calc = select_model_name_for_cost_calc


def declared_cost_map_pricing(key: str, model: str, custom_llm_provider: str | None) -> Mapping[str, object]:
    entry: Final = _raw_cost_map_entry(key)
    if entry is None:
        return unmapped_model_generalization(model, custom_llm_provider) or MappingProxyType({})
    fill_missing: Final = mapped_model_fill_missing(key, entry)
    return entry if fill_missing is None else MappingProxyType({**fill_missing, **entry})


class _SupportsGet(Protocol):
    def get(self, key: str, /) -> object: ...


def _get_response_model(completion_response: object) -> str | None:
    """
    Extract the model name from a completion response object.

    Used as a fallback for cost calculation when the input model name
    doesn't exist in model_cost (e.g., Azure Model Router).
    """
    if completion_response is None:
        return None

    if isinstance(completion_response, InteractionsAPIResponse):
        return completion_response.model or completion_response.agent
    if isinstance(completion_response, BaseModel):
        return getattr(completion_response, "model", None)
    elif isinstance(completion_response, dict):
        return completion_response.get("model", None)

    return None


_GEMINI_TRAFFIC_TYPE_TO_SERVICE_TIER: Final[dict] = {
    # ON_DEMAND_PRIORITY maps to "priority" — selects input_cost_per_token_priority, etc.
    "ON_DEMAND_PRIORITY": "priority",
    # FLEX / BATCH / ON_DEMAND_FLEX maps to "flex" — selects input_cost_per_token_flex, etc.
    # Vertex AI reports flex/shared-capacity traffic as ON_DEMAND_FLEX, not FLEX.
    "FLEX": "flex",
    "BATCH": "flex",
    "ON_DEMAND_FLEX": "flex",
    # ON_DEMAND is standard pricing — no service_tier suffix applied
    "ON_DEMAND": None,
}


def _map_traffic_type_to_service_tier(traffic_type: object) -> str | None:
    """
    Map a Gemini usageMetadata.trafficType value to a LiteLLM service_tier string.

    This allows the same `_priority` / `_flex` cost-key suffix logic used for
    OpenAI/Azure to work for Gemini and Vertex AI models.

    trafficType values seen in practice
    ------------------------------------
    ON_DEMAND               -> standard pricing  (service_tier = None)
    ON_DEMAND_PRIORITY      -> priority pricing  (service_tier = "priority")
    FLEX / BATCH / ON_DEMAND_FLEX -> batch/flex pricing (service_tier = "flex")
    """
    if traffic_type is None:
        return None
    service_tier: Final = _GEMINI_TRAFFIC_TYPE_TO_SERVICE_TIER.get(str(traffic_type).upper())
    return service_tier


def _normalize_service_tier(service_tier: object) -> str | None:
    """
    Reduce a service_tier value to a concrete billable tier string or None.

    "auto" is a routing preference and any non-string value is not a billable
    tier, so both defer to standard pricing (or to the tier the provider reports
    on the response usage) instead of crashing the downstream cost-key lookup,
    which calls service_tier.lower()
    """
    if not isinstance(service_tier, str) or service_tier.lower() == ServiceTier.AUTO.value:
        return None
    return service_tier


def _served_image_service_tier(completion_response: object, usage_object: Usage | None) -> str | None:
    """Image requests never send a tier upstream, so only the tier the provider reports can price them."""
    return _resolve_billable_service_tier(
        requested=None,
        served=_served_service_tier(completion_response, usage_object),
    )


def _resolve_billable_service_tier(requested: object, served: object) -> str | None:
    """Served tier wins when it names a priced tier or explicitly says base pricing; otherwise the request decides."""
    return billable_service_tier(
        requested if isinstance(requested, str) else None, served if isinstance(served, str) else None
    )


def _served_service_tier(completion_response: object, usage_object: Usage | None) -> str | None:
    """Find the tier the provider actually served: response, then usage, then Gemini trafficType."""
    response_tier: Final = _extract_service_tier(completion_response)
    if isinstance(response_tier, str):
        return response_tier
    usage_tier: Final = _extract_service_tier(usage_object)
    if isinstance(usage_tier, str):
        return usage_tier
    hidden_params: Final[object] = getattr(completion_response, "_hidden_params", None)
    if hidden_params is None:
        return None
    provider_specific: Final[object] = (
        require_attributes(hidden_params, _SupportsGet, "get").get("provider_specific_fields") or {}
    )
    raw_traffic_type: Final = require_attributes(provider_specific, _SupportsGet, "get").get("traffic_type")
    if not raw_traffic_type:
        return None
    return _map_traffic_type_to_service_tier(raw_traffic_type) or "default"


def _extract_service_tier(source: object) -> str | None:
    """Read a raw ``service_tier`` off a response body or usage object, dict or pydantic model alike."""
    if isinstance(source, BaseModel):
        return getattr(source, "service_tier", None)
    elif isinstance(source, dict):
        return source.get("service_tier")

    return None


def _get_transcription_usage_duration(completion_response: object) -> float | None:
    response_fields: Final = as_str_mapping(completion_response)
    usage_object: Final = (
        response_fields.get("usage") if response_fields is not None else attribute_or_none(completion_response, "usage")
    )
    usage_fields: Final = as_str_mapping(usage_object)
    usage_type: Final = (
        usage_fields.get("type") if usage_fields is not None else attribute_or_none(usage_object, "type")
    )
    if usage_type != "duration":
        return None
    seconds: Final = (
        usage_fields.get("seconds") if usage_fields is not None else attribute_or_none(usage_object, "seconds")
    )
    return normalized_audio_duration_seconds(seconds)


def infer_call_type(call_type: CallTypesLiteral | None, completion_response: object) -> CallTypesLiteral | None:
    if call_type is not None:
        return call_type

    if completion_response is None:
        return None

    if isinstance(completion_response, ModelResponse) or isinstance(completion_response, ModelResponseStream):
        return "completion"
    elif isinstance(completion_response, EmbeddingResponse):
        return "embedding"
    elif isinstance(completion_response, TranscriptionResponse):
        return "transcription"
    elif isinstance(completion_response, HttpxBinaryResponseContent):
        return "speech"
    elif isinstance(completion_response, RerankResponse):
        return "rerank"
    elif isinstance(completion_response, ImageResponse):
        return "image_generation"
    elif isinstance(completion_response, TextCompletionResponse):
        return "text_completion"
    elif isinstance(completion_response, LiteLLMSendMessageResponse):
        return "send_message"
    elif isinstance(completion_response, (DecisionsResponse, OpenAIDecisionResponse)):
        return "adecisions"
    elif isinstance(completion_response, ExtractionResponse):
        return "aextraction"

    return call_type


_infer_call_type = infer_call_type


def _store_cost_breakdown_in_logging_obj(
    litellm_logging_obj: LitellmLoggingObject | None,
    prompt_tokens_cost_usd_dollar: float | None,
    completion_tokens_cost_usd_dollar: float | None,
    cost_for_built_in_tools_cost_usd_dollar: float | None,
    total_cost_usd_dollar: float,
    additional_costs: dict | None = None,
    original_cost: float | None = None,
    discount_percent: float | None = None,
    discount_amount: float | None = None,
    margin_percent: float | None = None,
    margin_fixed_amount: float | None = None,
    margin_total_amount: float | None = None,
    cache_read_cost: float | None = None,
    cache_creation_cost: float | None = None,
    reasoning_cost: float | None = None,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    billed_token_rates: BilledTokenRates | None = None,
) -> None:
    """
    Helper function to store cost breakdown in the logging object.

    Args:
        litellm_logging_obj: The logging object to store breakdown in
        prompt_tokens_cost_usd_dollar: Cost of input tokens
        completion_tokens_cost_usd_dollar: Cost of completion tokens (includes reasoning if applicable)
        cost_for_built_in_tools_cost_usd_dollar: Cost of built-in tools
        total_cost_usd_dollar: Total cost of request
        additional_costs: Free-form additional costs dict (e.g., {"azure_model_router_flat_cost": 0.00014})
        original_cost: Cost before discount
        discount_percent: Discount percentage applied (0.05 = 5%)
        discount_amount: Discount amount in USD
        margin_percent: Margin percentage applied (0.10 = 10%)
        margin_fixed_amount: Fixed margin amount in USD
        margin_total_amount: Total margin added in USD
        service_tier: Tier the costs above were priced on, already resolved
        data_residency: Region uplift the costs above were priced on, already resolved
        vertex_location: Vertex AI location the costs above were priced on, already resolved
    """
    if litellm_logging_obj is None:
        return

    try:
        # Store the cost breakdown
        litellm_logging_obj.set_cost_breakdown(
            input_cost=prompt_tokens_cost_usd_dollar,
            output_cost=completion_tokens_cost_usd_dollar,
            total_cost=total_cost_usd_dollar,
            cost_for_built_in_tools_cost_usd_dollar=cost_for_built_in_tools_cost_usd_dollar,
            additional_costs=additional_costs,
            original_cost=original_cost,
            discount_percent=discount_percent,
            discount_amount=discount_amount,
            margin_percent=margin_percent,
            margin_fixed_amount=margin_fixed_amount,
            margin_total_amount=margin_total_amount,
            cache_read_cost=cache_read_cost,
            cache_creation_cost=cache_creation_cost,
            reasoning_cost=reasoning_cost,
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=vertex_location,
            billed_token_rates=billed_token_rates,
        )

    except Exception as breakdown_error:
        verbose_logger.debug("Error storing cost breakdown: %s", breakdown_error)
        # Don't fail the main cost calculation if breakdown storage fails


def _without_provider_stated_cost(usage: Usage | None) -> Usage | None:
    if usage is None or getattr(usage, "cost", None) is None:
        return usage
    return usage.model_copy(update=MappingProxyType({"cost": None}))


def _split_responses_ws_logging_object_by_service_tier(
    completion_response: LiteLLMRealtimeStreamLoggingObject,
) -> tuple[LiteLLMRealtimeStreamLoggingObject, ...] | None:
    partition: Final = ResponsesWebSocketTokenUsageProcessor.partition_results_by_service_tier(
        cast(Sequence[Mapping[str, object]], completion_response.results)
    )
    if len(partition) <= 1:
        return None
    return tuple(
        LiteLLMRealtimeStreamLoggingObject(
            results=cast(OpenAIRealtimeStreamList, list(group)),
            usage=ResponsesWebSocketTokenUsageProcessor.collect_and_combine_usage_from_responses_ws_results(group),
            service_tier=tier,
        )
        for tier, group in partition.items()
    )


def _response_time_ms_for_cost(
    completion_response: object,
    litellm_logging_obj: LitellmLoggingObject | None,
    total_time: float | None,
) -> float:
    stamped: Final = getattr(completion_response, "_response_ms", None)
    if isinstance(stamped, (int, float)):
        return float(stamped)
    if total_time:
        return total_time
    if litellm_logging_obj is None:
        return 0.0
    reported_ms: Final = response_ms_since_logging_start(litellm_logging_obj)
    if reported_ms is not None and litellm_logging_obj.stream is not True:
        return reported_ms
    return litellm_logging_obj.get_response_ms()


def _model_is_none_error(completion_response_text: str, litellm_logging_obj: LitellmLoggingObject | None) -> ValueError:
    message: Final = "Model is None and does not exist in passed completion_response."
    model_call_details: Final = {} if litellm_logging_obj is None else litellm_logging_obj.model_call_details
    if should_redact_message_logging(model_call_details) or model_call_details.get("query_image") is not None:
        return ValueError(message)
    return ValueError(f"{message} Passed completion_response={completion_response_text}, model=None")


def _breakdown_total(litellm_logging_obj: LitellmLoggingObject | None) -> float | None:
    breakdown: Final = None if litellm_logging_obj is None else litellm_logging_obj.cost_breakdown
    total: Final[object] = None if breakdown is None else breakdown.get("total_cost")
    return float(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else None


def _cost_map_priced_input_output(
    model: str | None,
    custom_llm_provider: str | None,
    usage: Usage | None,
    call_type: CallTypesLiteral | None,
    service_tier: str | None,
    data_residency: str | None,
) -> tuple[float, float]:
    if not model or usage is None:
        return 0.0, 0.0
    try:
        return cost_per_token(
            model=model,
            custom_llm_provider=custom_llm_provider,
            usage_object=usage,
            call_type=call_type or "completion",
            service_tier=service_tier,
            data_residency=data_residency,
        )
    except Exception as pricing_error:  # noqa: BLE001  # cost_per_token raises plain Exception for an unmapped model
        verbose_logger.debug("No cost map price for reported-cost split of %s: %s", model, pricing_error)
        return 0.0, 0.0


def reported_cost_receipt(
    reported: ReportedCost,
    model: str | None,
    custom_llm_provider: str | None,
    usage: Usage | None,
    call_type: CallTypesLiteral | None,
    service_tier: str | None,
    data_residency: str | None,
    adjust: AdjustFacts | None = None,
) -> NativeReceipt | None:
    """``reported`` billed in Rust by its source's policy, its billed total split between input and output in the
    ratio litellm's own cost map prices the same usage, so a per-input-token model books the whole total on input.
    Unmapped or zero-priced usage keeps the whole total on output."""
    priced_input, priced_output = _cost_map_priced_input_output(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        call_type=call_type,
        service_tier=service_tier,
        data_residency=data_residency,
    )
    receipt: Final = price_reported(reported, PricedSides(prompt=priced_input, completion=priced_output), adjust)
    return receipt if isinstance(receipt, NativeReceipt) else None


def _adjusted_provider_billed_cost(
    reported_cost: float,
    custom_llm_provider: str | None,
    pricing_provider: str | None,
    model: str | None,
    region_name: str | None,
    litellm_logging_obj: LitellmLoggingObject | None,
    usage: Usage | None,
    call_type: CallTypesLiteral | None,
    service_tier: str | None,
    data_residency: str | None,
) -> float:
    receipt: Final = reported_cost_receipt(
        ReportedCost(source="upstream_usage_cost", usd=reported_cost),
        model=model,
        custom_llm_provider=pricing_provider,
        usage=usage,
        call_type=call_type,
        service_tier=service_tier,
        data_residency=data_residency,
        adjust=AdjustFacts(provider=custom_llm_provider, model=model, region=region_name),
    )
    if receipt is None:
        return reported_cost
    _store_cost_breakdown_in_logging_obj(
        litellm_logging_obj=litellm_logging_obj,
        prompt_tokens_cost_usd_dollar=receipt.prompt_cost,
        completion_tokens_cost_usd_dollar=receipt.completion_cost,
        cost_for_built_in_tools_cost_usd_dollar=0.0,
        total_cost_usd_dollar=receipt.total,
        original_cost=receipt.original_cost,
        discount_percent=receipt.discount_percent,
        discount_amount=receipt.discount_amount,
        margin_percent=receipt.margin_percent,
        margin_fixed_amount=receipt.margin_fixed_amount,
        margin_total_amount=receipt.margin_total_amount,
        service_tier=service_tier,
        data_residency=data_residency,
    )
    return receipt.total


_NO_HIDDEN_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


def _hidden_params_of(completion_response: object) -> Mapping[str, object]:
    return as_str_mapping(attribute_or_none(completion_response, "_hidden_params")) or _NO_HIDDEN_PARAMS


def _with_known_usage(completion_response: object) -> None:
    """Swaps a usage model litellm does not know for the ``Usage`` its fields make, as main's candidate loop did
    before it read the usage."""
    if not _loop_reads_hidden_params(completion_response) or isinstance(completion_response, dict):
        return
    usage_obj: Final[object] = getattr(completion_response, "usage", None)
    if isinstance(usage_obj, BaseModel) and not is_known_usage_object(usage_obj=usage_obj):
        setattr(completion_response, "usage", usage_from_model(usage_obj))


@dataclass(frozen=True, slots=True)
class _CandidateExtras:
    """What Python finds for one candidate of a token call: the built-in tools it used, the rows they and the Model
    Router fee read, and whether finding them raised."""

    candidate: int
    tools: ToolUse | None
    rows: tuple[Mapping[str, object], ...]
    fee_at: int | None
    failed: bool

    def facts(self, rows_from: int) -> CandidateFacts:
        return CandidateFacts(
            candidate=self.candidate,
            tools=self.tools,
            router_fee=None if self.fee_at is None else CandidateFee(billed="additional", row=self.fee_at),
            failed=self.failed,
            rows_from=rows_from,
        )


def _candidate_extras(
    row: CostRow,
    completion_response: object,
    usage: Usage,
    standard_built_in_tools_params: StandardBuiltInToolsParams | None,
    request_model: str | None,
) -> _CandidateExtras:
    try:
        tools, tool_rows = StandardBuiltInToolCostTracking.built_in_tool_use(
            model=row.model,
            response_object=completion_response,
            usage=usage,
            standard_built_in_tools_params=standard_built_in_tools_params,
            custom_llm_provider=row.loop_provider,
        )
        fee_row: Final = _router_fee_row(
            model=row.model,
            custom_llm_provider=row.loop_provider,
            request_model=request_model,
            completion_response=completion_response,
        )
    except Exception as e:  # noqa: BLE001  # main's candidate loop moved to the next candidate on any error
        verbose_logger.debug("Error finding the tools and fees of candidate %s - %s", row.model, e)
        return _CandidateExtras(candidate=row.candidate, tools=None, rows=(), fee_at=None, failed=True)
    return _CandidateExtras(
        candidate=row.candidate,
        tools=tools,
        rows=(*tool_rows, *(() if fee_row is None else (fee_row,))),
        fee_at=None if fee_row is None else len(tool_rows),
        failed=False,
    )


def _vertex_characters(messages: Sequence[object], completion_response: object) -> Characters | None:
    """The prompt and response characters ``completion_cost`` counts for a Vertex AI call."""
    prompt: Final = (
        litellm.utils.count_characters(
            text=litellm.utils.get_formatted_prompt(data={"messages": list(messages)}, call_type="completion")
        )
        if len(messages) > 0
        else None
    )
    completion: Final = (
        litellm.utils.count_characters(text=litellm.utils.get_response_string(response_obj=completion_response))
        if isinstance(completion_response, ModelResponse)
        else None
    )
    return None if prompt is None and completion is None else Characters(prompt=prompt, completion=completion)


def rust_priced_cost(
    call: CallFacts,
    completion_response: object,
    call_type: str,
    usage: Usage | None,
    messages: Sequence[object],
    custom_priced_request_model: str | None,
    total_time: float | None,
    service_tier: str | None,
    data_residency: str | None,
    vertex_location: str | None,
    standard_built_in_tools_params: StandardBuiltInToolsParams | None,
    litellm_logging_obj: LitellmLoggingObject | None,
    reported_cost_applies: bool,
) -> float | None:
    """``completion_cost`` of a token call, its candidates walked and priced in Rust in main's order. ``None`` where
    Python prices the call: a call kind priced by its own calculator, a provider-billed cost, or a candidate only
    Python prices coming first."""
    if call_type not in _RUST_PRICED_CALL_TYPES or usage is None or not _loop_reads_hidden_params(completion_response):
        return None
    if reported_cost_applies and billed_usage_cost(completion_response, call["loop_provider"]) is not None:
        return None
    rows, _ = cost_rows(call)
    if not rows:
        return None
    request_model: Final = None if litellm_logging_obj is None else litellm_logging_obj.model
    extras: Final = tuple(
        _candidate_extras(row, completion_response, usage, standard_built_in_tools_params, request_model)
        for row in rows
    )
    offsets: Final = tuple(accumulate((len(extra.rows) for extra in extras), initial=0))
    receipt: Final = price_call(
        call,
        priced_usage(usage, web_search=None),
        candidates=tuple(extra.facts(offset) for extra, offset in zip(extras, offsets, strict=False)),
        rows=tuple(chain.from_iterable(extra.rows for extra in extras)),
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        requested_model=request_model,
        adjust=AdjustFacts(provider=None, model=custom_priced_request_model, region=call["loop_region"]),
        branch=BranchFacts(
            elapsed=Elapsed(
                response_ms=_response_time_ms_for_cost(completion_response, litellm_logging_obj, total_time),
                audio_seconds=0.0,
            ),
            characters=_vertex_characters(messages, completion_response)
            if call["loop_provider"] == "vertex_ai"
            else None,
            embedding=call_type in _EMBEDDING_CALL_TYPES,
            raw=raw_counts(usage),
        ),
        breakdown="absent" if litellm_logging_obj is None else "row",
    )
    if receipt is None:
        return None
    store_receipt_breakdown(litellm_logging_obj, receipt, service_tier, data_residency, vertex_location)
    return receipt.total


def store_receipt_breakdown(
    litellm_logging_obj: LitellmLoggingObject | None,
    receipt: NativeReceipt,
    service_tier: str | None,
    data_residency: str | None,
    vertex_location: str | None,
) -> None:
    """The cost breakdown of a call Rust priced, stored on its logging object as ``completion_cost`` stores it."""
    additional_costs: Final = part_costs(receipt)
    _store_cost_breakdown_in_logging_obj(
        litellm_logging_obj=litellm_logging_obj,
        prompt_tokens_cost_usd_dollar=receipt.prompt_cost,
        completion_tokens_cost_usd_dollar=receipt.completion_cost,
        cost_for_built_in_tools_cost_usd_dollar=receipt.tools_cost,
        total_cost_usd_dollar=receipt.total,
        additional_costs=dict(additional_costs) or None,
        original_cost=receipt.original_cost,
        discount_percent=receipt.discount_percent,
        discount_amount=receipt.discount_amount,
        margin_percent=receipt.margin_percent,
        margin_fixed_amount=receipt.margin_fixed_amount,
        margin_total_amount=receipt.margin_total_amount,
        cache_read_cost=receipt.cache_read_cost,
        cache_creation_cost=receipt.cache_creation_cost,
        reasoning_cost=receipt.reasoning_cost,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        billed_token_rates=billed_token_rates(receipt),
    )


class _PricedTokens(Protocol):
    def __call__(
        self, *, custom_llm_provider: str | None, receipt: ReceiptFacts | None, adjust: AdjustFacts | None
    ) -> NativeReceipt | tuple[float, float]: ...


def _candidate_cost(
    priced_tokens: _PricedTokens,
    *,
    model: str,
    custom_llm_provider: str | None,
    candidate_provider: str | None,
    completion_response: object,
    usage: Usage | None,
    standard_built_in_tools_params: StandardBuiltInToolsParams | None,
    request_model: str | None,
    custom_cost_per_token: CostPerToken | None,
    bedrock_model_info: ModelInfo | None,
    litellm_logging_obj: LitellmLoggingObject | None,
    adjust: AdjustFacts,
    service_tier: str | None,
    data_residency: str | None,
    vertex_location: str | None,
) -> float:
    """One candidate's ``completion_cost``: its token receipt with the tool fees, the Model Router fee and the row the
    stored breakdown reads, stored on the logging object."""
    tools, tool_rows = StandardBuiltInToolCostTracking.built_in_tool_use(
        model=model,
        response_object=completion_response,
        usage=usage,
        standard_built_in_tools_params=standard_built_in_tools_params,
        custom_llm_provider=custom_llm_provider,
    )
    fee_row: Final = _router_fee_row(
        model=model,
        custom_llm_provider=custom_llm_provider,
        request_model=request_model,
        completion_response=completion_response,
    )
    stored_row: Final = _breakdown_row(
        model=model,
        custom_llm_provider=candidate_provider,
        custom_cost_per_token=custom_cost_per_token,
        bedrock_model_info=bedrock_model_info,
        usage=usage,
        stored=litellm_logging_obj is not None,
    )
    priced: Final = priced_tokens(
        custom_llm_provider=candidate_provider,
        receipt=ReceiptFacts(
            breakdown=stored_row[0],
            breakdown_row=stored_row[1],
            tools=tools,
            tool_rows=tool_rows,
            router_fee=None if fee_row is None else "additional",
            fee_row=fee_row,
        ),
        adjust=adjust,
    )
    if not isinstance(priced, NativeReceipt):
        return _unit_kind_cost(
            priced,
            model=model,
            custom_llm_provider=candidate_provider,
            usage=usage,
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=vertex_location,
            stored_row=stored_row,
            tool_fees=price_tools(ToolsOfCall(call=tools), tool_rows) or 0.0,
            fee_row=fee_row,
            adjust=adjust,
            litellm_logging_obj=litellm_logging_obj,
        )
    store_receipt_breakdown(litellm_logging_obj, priced, service_tier, data_residency, vertex_location)
    return priced.total


def completion_cost(
    completion_response: object | None = None,
    model: str | None = None,
    prompt="",
    messages: list = [],
    completion="",
    total_time: float | None = 0.0,  # used for replicate, sagemaker
    call_type: CallTypesLiteral | None = None,
    ### REGION ###
    custom_llm_provider: str | None = None,
    region_name=None,  # used for bedrock pricing
    ### IMAGE GEN ###
    size: str | None = None,
    quality: str | None = None,
    n: int | None = None,  # number of images
    ### CUSTOM PRICING ###
    custom_cost_per_token: CostPerToken | None = None,
    custom_cost_per_second: float | None = None,
    requested_model_info_override: ModelInfo | None = None,
    optional_params: dict | None = None,
    custom_pricing: bool | None = None,
    base_model: str | None = None,
    standard_built_in_tools_params: StandardBuiltInToolsParams | None = None,
    litellm_model_name: str | None = None,
    router_model_id: str | None = None,
    litellm_logging_obj: LitellmLoggingObject | None = None,
    pricing_provider: str | None = None,
    ### SERVICE TIER ###
    service_tier: str | None = None,  # for OpenAI service tier pricing
    ### DATA RESIDENCY ###
    data_residency: str | None = None,  # for OpenAI regional-processing uplift (e.g. "eu", "us")
    ### VERTEX LOCATION ###
    vertex_location: str | None = None,  # for Vertex AI regional-endpoint uplift (e.g. "us-east5", "global")
) -> float:
    """
    Return the USD cost of a LiteLLM call.

    Pass `completion_response` to price a response LiteLLM returned. Without one, pass `model` plus
    `messages` / `prompt` and `completion`, and token counts are computed from the text. Prices come from
    `litellm.model_cost` unless `custom_cost_per_token` or `custom_cost_per_second` is given. `total_time`
    is in milliseconds and only matters for per-second pricing such as Replicate and SageMaker.

    Raises if no candidate model name resolves to pricing. Register the model via `litellm.register_model`
    or add it to model_prices_and_context_window.json.
    """
    try:
        call_type = infer_call_type(call_type, completion_response) or "completion"
        if isinstance(completion_response, DecisionsResponse) and completion_response.usage is None:
            raise ValueError("Cannot calculate Decisions cost: provider did not report usage")
        if pricing_provider is not None:
            custom_llm_provider = pricing_provider  # rebind-ok: model_info.pricing_provider overrides the routing provider for cost-map lookups

        if call_type == CallTypes.aresponses_websocket.value and isinstance(
            completion_response, LiteLLMRealtimeStreamLoggingObject
        ):
            ws_tier_parts: Final = _split_responses_ws_logging_object_by_service_tier(completion_response)
            if ws_tier_parts is not None:
                return sum(
                    completion_cost(
                        completion_response=part,
                        model=model,
                        prompt=prompt,
                        messages=messages,
                        completion=completion,
                        total_time=total_time,
                        call_type=call_type,
                        custom_llm_provider=custom_llm_provider,
                        region_name=region_name,
                        size=size,
                        quality=quality,
                        n=n,
                        custom_cost_per_token=custom_cost_per_token,
                        requested_model_info_override=requested_model_info_override,
                        custom_cost_per_second=custom_cost_per_second,
                        optional_params=optional_params,
                        custom_pricing=custom_pricing,
                        base_model=base_model,
                        pricing_provider=pricing_provider,
                        standard_built_in_tools_params=standard_built_in_tools_params,
                        litellm_model_name=litellm_model_name,
                        router_model_id=router_model_id,
                        litellm_logging_obj=litellm_logging_obj,
                        service_tier=service_tier,
                        data_residency=data_residency,
                        vertex_location=vertex_location,
                    )
                    for part in ws_tier_parts
                )

        # Handle Inputs to completion_cost
        prompt_tokens = 0
        prompt_characters: int | None = None
        completion_tokens = 0
        completion_characters: int | None = None
        cache_creation_input_tokens: int | None = None
        cache_read_input_tokens: int | None = None
        audio_transcription_file_duration: float = 0.0
        provider_usage_object: Final = get_usage_object(completion_response=completion_response)
        cost_per_token_usage_object: Final[Usage | None] = (
            _without_provider_stated_cost(provider_usage_object) if custom_pricing else provider_usage_object
        )
        rerank_billed_units: RerankBilledUnits | None = None

        explicit_tier: Final = _normalize_service_tier(service_tier)
        if explicit_tier is not None:
            service_tier = explicit_tier
        else:
            service_tier = _resolve_billable_service_tier(  # rebind-ok: resolved from request then response
                requested=optional_params.get("service_tier") if optional_params is not None else None,
                served=_served_service_tier(completion_response, cost_per_token_usage_object),
            )

        explicit_pricing: Final = custom_pricing is True or base_model is not None
        caller_set_prices: Final = (
            explicit_pricing or custom_cost_per_token is not None or custom_cost_per_second is not None
        )
        custom_priced_request_model: Final = model if custom_pricing is True else None
        configured_cost_per_query: Final = (
            _configured_search_cost_per_query(litellm_logging_obj) if custom_pricing else None
        )
        invoked_model_id: Final = _invoked_model_id_for_cost(
            _LOGGED_LITELLM_PARAMS.validate_python(optional_params) if optional_params is not None else None,
            litellm_logging_obj,
        )
        call: Final = _call_facts(
            completion_response,
            model,
            custom_llm_provider,
            base_model,
            custom_pricing,
            router_model_id,
            region_name,
            invoked_model_id,
            pricing_provider,
        )
        found: Final = cost_candidates(call)
        potential_model_names: Final = found.names
        selected_model: Final = found.call_selected
        bedrock_model_info: Final = _bedrock_model_info(found.bedrock)
        requested_model: Final = model
        request_pricing_provider: Final = custom_llm_provider if isinstance(custom_llm_provider, str) else None
        _with_known_usage(completion_response)

        rust_priced: Final = (
            rust_priced_cost(
                call,
                completion_response=completion_response,
                call_type=call_type,
                usage=cost_per_token_usage_object,
                messages=messages,
                custom_priced_request_model=custom_priced_request_model,
                total_time=total_time,
                service_tier=service_tier,
                data_residency=data_residency,
                vertex_location=vertex_location,
                standard_built_in_tools_params=standard_built_in_tools_params,
                litellm_logging_obj=litellm_logging_obj,
                reported_cost_applies=not caller_set_prices,
            )
            if custom_cost_per_token is None
            and custom_cost_per_second is None
            and requested_model_info_override is None
            else None
        )
        if rust_priced is not None:
            return rust_priced

        priced_response: Final[object] = completion_response
        for idx, (candidate_model, candidate_pricing_provider) in enumerate(potential_model_names):
            model = candidate_model
            try:
                if verbose_logger.isEnabledFor(logging.DEBUG):
                    verbose_logger.debug("selected model name for cost calculation: %s", model)

                if completion_response is not None and (
                    isinstance(completion_response, (BaseModel, dict))
                    or isinstance(completion_response, HttpxBinaryResponseContent)
                    and completion_response.usage is not None
                ):
                    if isinstance(completion_response, dict):
                        usage_obj: dict | Usage | None = completion_response.get("usage", {})
                    else:
                        usage_obj = getattr(completion_response, "usage", {})
                    if isinstance(usage_obj, (DecisionsUsage, OpenAIDecisionUsage)):
                        _usage = decisions_usage(usage_obj).model_dump()
                    elif isinstance(usage_obj, ExtractionUsage):
                        _usage = extraction_usage(usage_obj).model_dump()
                    elif usage_obj is None:
                        _usage = {}
                    elif isinstance(usage_obj, BaseModel):
                        _usage = cast(BaseModel, usage_obj).model_dump()
                    else:
                        _usage = usage_obj

                    if litellm.AnthropicConfig.is_anthropic_usage_object(_usage):
                        _usage = (
                            litellm.AnthropicConfig()
                            .calculate_usage(usage_object=_usage, reasoning_content=None)
                            .model_dump()
                        )
                    elif ResponseAPILoggingUtils.is_response_api_usage(_usage):
                        _usage = ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(_usage).model_dump()
                    elif TranscriptionUsageObjectTransformation.is_transcription_usage_object(_usage):
                        tr_usage = TranscriptionUsageObjectTransformation.transform_transcription_usage_object(
                            cast(
                                TranscriptionUsageDurationObject | TranscriptionUsageTokensObject,
                                _usage,
                            )
                        )
                        if tr_usage is not None:
                            _usage = tr_usage.model_dump()
                    elif InteractionsUsageObjectTransformation.is_interactions_usage_object(_usage):
                        _usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(
                            _usage
                        ).model_dump()
                    else:
                        _usage = _usage

                    # get input/output tokens from completion_response
                    prompt_tokens = _usage.get("prompt_tokens", 0)
                    completion_tokens = _usage.get("completion_tokens", 0)
                    cache_creation_input_tokens = _usage.get("cache_creation_input_tokens", 0)
                    cache_read_input_tokens = _usage.get("cache_read_input_tokens", 0)
                    if (
                        "prompt_tokens_details" in _usage
                        and _usage["prompt_tokens_details"] != {}
                        and _usage["prompt_tokens_details"]
                    ):
                        prompt_tokens_details = _usage.get("prompt_tokens_details") or {}
                        cache_read_input_tokens = prompt_tokens_details.get("cached_tokens", 0)

                    hidden_params = getattr(completion_response, "_hidden_params", None)
                    if hidden_params is not None:
                        custom_llm_provider = pricing_provider or hidden_params.get(
                            "custom_llm_provider", custom_llm_provider or None
                        )
                        region_name = hidden_params.get("region_name", region_name)

                else:
                    if model is None:
                        raise _model_is_none_error(f"{completion_response}", litellm_logging_obj)
                    if len(messages) > 0:
                        prompt_tokens = token_counter(model=model, messages=messages)
                    elif len(prompt) > 0:
                        prompt_tokens = token_counter(model=model, text=prompt)
                    completion_tokens = token_counter(model=model, text=completion)

                # Handle A2A calls before model check - A2A doesn't require a model
                if call_type in _A2A_CALL_TYPES:
                    from litellm.a2a_protocol.cost_calculator import A2ACostCalculator

                    return A2ACostCalculator.calculate_a2a_cost(litellm_logging_obj=litellm_logging_obj)

                if model is None:
                    raise _model_is_none_error(f"{completion_response}", litellm_logging_obj)
                if custom_llm_provider is None:
                    try:
                        model, custom_llm_provider, _, _ = litellm.get_llm_provider(
                            model=model
                        )  # strip the llm provider from the model name -> for image gen cost calculation
                    except Exception as e:
                        verbose_logger.debug(
                            "litellm.cost_calculator.py::completion_cost() - Error inferring custom_llm_provider - %s",
                            e,
                        )
                if CostCalculatorUtils.call_type_has_image_response(call_type) and isinstance(
                    completion_response, ImageResponse
                ):
                    ### IMAGE GENERATION COST CALCULATION ###
                    return CostCalculatorUtils.route_image_generation_cost_calculator(
                        model=model,
                        custom_llm_provider=custom_llm_provider,
                        completion_response=completion_response,
                        quality=quality,
                        n=n,
                        size=size,
                        optional_params=optional_params,
                        call_type=call_type,
                        model_info=_deployment_model_info(litellm_logging_obj, custom_pricing, router_model_id),
                        vertex_location=vertex_location,
                        service_tier=_served_image_service_tier(completion_response, cost_per_token_usage_object),
                    )
                elif call_type in _VIDEO_CALL_TYPES:
                    ### VIDEO GENERATION COST CALCULATION ###
                    _video_model_info: ModelInfo | None = _deployment_model_info(
                        litellm_logging_obj, custom_pricing, router_model_id
                    )

                    _video_usage_obj: object = attribute_or_none(completion_response, "usage")
                    duration_seconds: float | None = None
                    video_resolution: str | None = None
                    provider_reported_cost: float | None = None
                    if completion_response is not None and _video_usage_obj:
                        duration_seconds = _VIDEO_USAGE_NUMBER.validate_python(
                            _video_usage_value(_video_usage_obj, "duration_seconds")
                        )
                        provider_reported_cost = _VIDEO_USAGE_NUMBER.validate_python(
                            _video_usage_value(_video_usage_obj, "provider_reported_cost_usd")
                        )
                        if (
                            video_resolution_value := _video_usage_value(_video_usage_obj, "video_resolution")
                        ) is not None:
                            video_resolution = str(video_resolution_value).strip().lower()

                        if _video_model_info is None and provider_reported_cost is not None:
                            return float(provider_reported_cost)

                        if duration_seconds is not None:
                            return priced_video(
                                model=model,
                                custom_llm_provider=custom_llm_provider,
                                seconds=duration_seconds,
                                video_resolution=video_resolution,
                                deployment=_video_model_info,
                                catalog=None
                                if _video_model_info is not None
                                else litellm.get_model_info(
                                    model=model, custom_llm_provider=custom_llm_provider or "openai"
                                ),
                                videos=_video_count(_video_usage_obj),
                                input_images=_billable_input_images(_video_usage_obj),
                                requested_row=(
                                    litellm.get_model_info(
                                        model=requested_model or model, custom_llm_provider=custom_llm_provider
                                    )
                                    if _billable_input_images(_video_usage_obj) > 0
                                    and (
                                        _video_model_info is None
                                        or _video_model_info.get("input_cost_per_image") is None
                                    )
                                    else None
                                ),
                            )
                    # Fallback to default video cost calculation if no duration available
                    return default_video_cost_calculator(
                        model=model,
                        duration_seconds=0.0,  # Default to 0 if no duration available
                        custom_llm_provider=custom_llm_provider,
                        model_info=_video_model_info,
                        video_resolution=video_resolution,
                        videos=_video_count(_video_usage_obj),
                    )
                elif call_type in _SPEECH_CALL_TYPES:
                    prompt_characters = _speech_billed_characters(
                        completion_response if isinstance(completion_response, HttpxBinaryResponseContent) else None,
                        prompt,
                    )
                elif call_type in _TRANSCRIPTION_CALL_TYPES:
                    audio_transcription_file_duration = _transcription_duration_for_cost(completion_response)
                elif call_type in _RERANK_CALL_TYPES:
                    if completion_response is not None and isinstance(completion_response, RerankResponse):
                        meta_obj = completion_response.meta
                        if meta_obj is not None:
                            billed_units: RerankBilledUnits = meta_obj.get("billed_units") or {}
                        else:
                            billed_units = {}

                        rerank_billed_units = RerankBilledUnits(
                            search_units=billed_units.get("search_units"),
                            total_tokens=billed_units.get("total_tokens"),
                        )

                        search_units = billed_units.get("search_units") or 1  # cohere charges per request by default.
                        completion_tokens = search_units
                elif call_type in _SEARCH_CALL_TYPES:
                    # Extract number_of_queries from optional_params or default to 1
                    number_of_queries = 1
                    if optional_params is not None:
                        # Check if query is a list (multiple queries)
                        query = optional_params.get("query")
                        if isinstance(query, list):
                            number_of_queries = len(query)
                        elif query is not None:
                            number_of_queries = 1

                    search_model = model or ""
                    if custom_llm_provider and "/" not in search_model:
                        # If model is like "tavily-search", construct "tavily/search" for cost lookup
                        search_model = f"{custom_llm_provider}/search"

                    return _priced_search(
                        search_model=search_model,
                        model=model,
                        custom_llm_provider=custom_llm_provider,
                        number_of_queries=number_of_queries,
                        optional_params=optional_params,
                        region=region_name if isinstance(region_name, str) else None,
                        configured_cost_per_query=configured_cost_per_query,
                        litellm_logging_obj=litellm_logging_obj,
                        service_tier=service_tier,
                        data_residency=data_residency,
                    )
                elif call_type == _AREALTIME_CALL_TYPE and isinstance(
                    completion_response, LiteLLMRealtimeStreamLoggingObject
                ):
                    if cost_per_token_usage_object is None or custom_llm_provider is None:
                        raise ValueError(
                            f"usage object and custom_llm_provider must be provided for realtime stream cost calculation. Got cost_per_token_usage_object={cost_per_token_usage_object}, custom_llm_provider={custom_llm_provider}"
                        )
                    return handle_realtime_stream_cost_calculation(
                        results=completion_response.results,
                        combined_usage_object=cost_per_token_usage_object,
                        custom_llm_provider=custom_llm_provider,
                        litellm_model_name=model,
                        data_residency=data_residency,
                        litellm_logging_obj=litellm_logging_obj,
                        custom_pricing_model=selected_model if custom_pricing else None,
                        base_pricing_model=(selected_model if base_model is not None and not custom_pricing else None),
                        deployment_models=(base_model, requested_model),
                    )
                elif call_type == _MCP_CALL_TYPE:
                    from litellm.proxy.mcp_server.cost_calculator import (
                        MCPCostCalculator,
                    )

                    return MCPCostCalculator.calculate_mcp_tool_call_cost(litellm_logging_obj=litellm_logging_obj)
                response_time_ms = _response_time_ms_for_cost(
                    completion_response=completion_response,
                    litellm_logging_obj=litellm_logging_obj,
                    total_time=total_time,
                )
                model_for_discount = custom_priced_request_model or model
                if (
                    not caller_set_prices
                    and (reported_cost := billed_usage_cost(priced_response, custom_llm_provider)) is not None
                ):
                    return _adjusted_provider_billed_cost(
                        reported_cost=reported_cost,
                        custom_llm_provider=custom_llm_provider,
                        pricing_provider=_pricing_provider_for_cost_candidate(
                            candidate_provider=candidate_pricing_provider,
                            request_provider=request_pricing_provider,
                            current_provider=custom_llm_provider,
                        ),
                        model=model_for_discount,
                        region_name=region_name if isinstance(region_name, str) else None,
                        litellm_logging_obj=litellm_logging_obj,
                        usage=cost_per_token_usage_object,
                        call_type=call_type,
                        service_tier=service_tier,
                        data_residency=data_residency,
                    )
                # Calculate cost based on prompt_tokens, completion_tokens
                if (
                    "togethercomputer" in model or "together_ai" in model or custom_llm_provider == "together_ai"
                ) and not has_together_registry_pricing(model, litellm.model_cost):
                    model = get_model_params_and_category(model, call_type=CallTypes(call_type))

                # replicate llms are calculate based on time for request running
                # see https://replicate.com/pricing
                elif (model in litellm.replicate_models or "replicate" in model) and model not in litellm.model_cost:
                    # for unmapped replicate model, default to replicate's time tracking logic
                    return get_replicate_completion_pricing(completion_response, response_time_ms)

                if model is None:
                    raise _model_is_none_error(f"{completion_response}", litellm_logging_obj)

                if custom_llm_provider is not None and custom_llm_provider == "vertex_ai":
                    # Calculate the prompt characters + response characters
                    if len(messages) > 0:
                        prompt_string = litellm.utils.get_formatted_prompt(
                            data={"messages": messages}, call_type="completion"
                        )

                        prompt_characters = litellm.utils.count_characters(text=prompt_string)
                    if completion_response is not None and isinstance(completion_response, ModelResponse):
                        completion_string = litellm.utils.get_response_string(response_obj=completion_response)
                        completion_characters = litellm.utils.count_characters(text=completion_string)

                # Get the original request model for router detection
                request_model_for_cost = None
                if litellm_logging_obj is not None:
                    request_model_for_cost = litellm_logging_obj.model

                return _candidate_cost(
                    partial(
                        _priced_tokens,
                        model=model,
                        prompt_tokens=prompt_tokens or 0,
                        completion_tokens=completion_tokens or 0,
                        response_time_ms=response_time_ms,
                        region_name=None if explicit_pricing else region_name,
                        custom_cost_per_second=custom_cost_per_second,
                        custom_cost_per_token=custom_cost_per_token,
                        requested_model_info_override=requested_model_info_override,
                        prompt_characters=prompt_characters,
                        completion_characters=completion_characters,
                        cache_creation_input_tokens=cache_creation_input_tokens,
                        cache_read_input_tokens=cache_read_input_tokens,
                        usage_object=cost_per_token_usage_object,
                        call_type=call_type,
                        audio_transcription_file_duration=audio_transcription_file_duration,
                        rerank_billed_units=rerank_billed_units,
                        service_tier=service_tier,
                        data_residency=data_residency,
                        vertex_location=vertex_location,
                        response=completion_response,
                        custom_model_info=_ocr_price_tables(litellm_logging_obj, custom_pricing, router_model_id),
                        bedrock_model_info=bedrock_model_info,
                    ),
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    candidate_provider=_pricing_provider_for_cost_candidate(
                        candidate_provider=candidate_pricing_provider,
                        request_provider=request_pricing_provider,
                        current_provider=custom_llm_provider,
                    ),
                    completion_response=completion_response,
                    usage=cost_per_token_usage_object,
                    standard_built_in_tools_params=standard_built_in_tools_params,
                    request_model=request_model_for_cost,
                    custom_cost_per_token=custom_cost_per_token,
                    bedrock_model_info=bedrock_model_info,
                    litellm_logging_obj=litellm_logging_obj,
                    adjust=AdjustFacts(
                        provider=custom_llm_provider,
                        model=model_for_discount,
                        region=region_name if isinstance(region_name, str) else None,
                    ),
                    service_tier=service_tier,
                    data_residency=data_residency,
                    vertex_location=vertex_location,
                )
            except Exception as e:
                verbose_logger.debug(
                    "litellm.cost_calculator.py::completion_cost() - Error calculating cost for model=%s - %s", model, e
                )
                if idx == len(potential_model_names) - 1:
                    raise e
        raise Exception(f"Unable to calculate cost for received potential model names - {potential_model_names}")
    except Exception as e:
        raise e


_ADDITIONAL_HEADERS: Final = TypeAdapter(Mapping[str, object])


def get_response_cost_from_hidden_params(
    hidden_params: Mapping[str, object] | BaseModel,
) -> float | None:
    params: Final[Mapping[str, object]] = (
        hidden_params.model_dump() if isinstance(hidden_params, BaseModel) else hidden_params
    )
    additional_headers: Final = params.get("additional_headers")
    if not isinstance(additional_headers, Mapping):
        return None
    headers: Final = _ADDITIONAL_HEADERS.validate_python(additional_headers)
    return _billable_header_cost(headers.get("llm_provider-x-litellm-response-cost"))


def _billable_header_cost(value: object) -> float | None:
    if not isinstance(value, (str, int, float)):
        return None
    try:
        cost: Final = float(value)
    except ValueError:
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def response_cost_calculator(
    response_object: ModelResponse
    | EmbeddingResponse
    | ImageResponse
    | TranscriptionResponse
    | TextCompletionResponse
    | HttpxBinaryResponseContent
    | RerankResponse
    | ResponsesAPIResponse
    | LiteLLMRealtimeStreamLoggingObject
    | OpenAIModerationResponse
    | Response
    | SearchResponse
    | DecisionsResponse
    | OpenAIDecisionResponse
    | ExtractionResponse,
    model: str,
    custom_llm_provider: str | None,
    call_type: Literal[
        "embedding",
        "aembedding",
        "completion",
        "acompletion",
        "atext_completion",
        "text_completion",
        "image_generation",
        "aimage_generation",
        "moderation",
        "amoderation",
        "atranscription",
        "transcription",
        "aspeech",
        "speech",
        "rerank",
        "arerank",
        "search",
        "asearch",
        "decisions",
        "adecisions",
        "systemone",
        "asystemone",
        "extraction",
        "aextraction",
    ],
    optional_params: dict,
    cache_hit: bool | None = None,
    base_model: str | None = None,
    custom_pricing: bool | None = None,
    prompt: str = "",
    standard_built_in_tools_params: StandardBuiltInToolsParams | None = None,
    litellm_model_name: str | None = None,
    router_model_id: str | None = None,
    litellm_logging_obj: LitellmLoggingObject | None = None,
    pricing_provider: str | None = None,
    ### SERVICE TIER ###
    service_tier: str | None = None,  # for OpenAI service tier pricing
    ### DATA RESIDENCY ###
    data_residency: str | None = None,  # for OpenAI regional-processing uplift (e.g. "eu", "us")
    ### VERTEX LOCATION ###
    vertex_location: str | None = None,  # for Vertex AI regional-endpoint uplift (e.g. "us-east5", "global")
    region_name: str | None = None,
    custom_cost_per_token: CostPerToken | None = None,
    custom_cost_per_second: float | None = None,
    requested_model_info_override: ModelInfo | None = None,
) -> float:
    """
    Returns
    - float or None: cost of response
    """
    try:
        response_cost: float = 0.0
        if cache_hit is not None and cache_hit is True:
            response_cost = 0.0
        else:
            if isinstance(response_object, BaseModel):
                if hasattr(response_object, HIDDEN_PARAMS_ATTR):
                    hidden_params: Final = cast(  # cast-ok: cost metadata supports dict and Pydantic storage
                        dict[str, object] | BaseModel,
                        getattr(response_object, HIDDEN_PARAMS_ATTR),
                    )
                    provider_response_cost: Final = get_response_cost_from_hidden_params(hidden_params)
                    if provider_response_cost is not None and not custom_pricing:
                        if _breakdown_total(litellm_logging_obj) != provider_response_cost:
                            header_priced_model: Final = select_model_name_for_cost_calc(
                                model=model,
                                completion_response=response_object,
                                custom_llm_provider=custom_llm_provider,
                                custom_pricing=custom_pricing,
                                base_model=base_model,
                                router_model_id=router_model_id,
                                region_name=region_name,
                            )
                            header_receipt: Final = reported_cost_receipt(
                                ReportedCost(source="llm_provider-x-litellm-response-cost", usd=provider_response_cost),
                                model=header_priced_model or model,
                                custom_llm_provider=pricing_provider or custom_llm_provider,
                                usage=get_usage_object(response_object),
                                call_type=call_type,
                                service_tier=service_tier,
                                data_residency=data_residency,
                            )
                            _store_cost_breakdown_in_logging_obj(
                                litellm_logging_obj=litellm_logging_obj,
                                prompt_tokens_cost_usd_dollar=0.0
                                if header_receipt is None
                                else header_receipt.prompt_cost,
                                completion_tokens_cost_usd_dollar=provider_response_cost
                                if header_receipt is None
                                else header_receipt.completion_cost,
                                cost_for_built_in_tools_cost_usd_dollar=0.0,
                                total_cost_usd_dollar=provider_response_cost,
                                original_cost=provider_response_cost,
                            )
                        return provider_response_cost

            response_cost = completion_cost(
                completion_response=response_object,
                model=model,
                call_type=call_type,
                custom_llm_provider=custom_llm_provider,
                optional_params=optional_params,
                custom_pricing=custom_pricing,
                base_model=base_model,
                custom_cost_per_token=custom_cost_per_token,
                requested_model_info_override=requested_model_info_override,
                custom_cost_per_second=custom_cost_per_second,
                prompt=prompt,
                standard_built_in_tools_params=standard_built_in_tools_params,
                litellm_model_name=litellm_model_name,
                router_model_id=router_model_id,
                litellm_logging_obj=litellm_logging_obj,
                pricing_provider=pricing_provider,
                service_tier=service_tier,
                data_residency=data_residency,
                vertex_location=vertex_location,
                region_name=region_name,
            )
        return response_cost
    except Exception as e:
        raise e


def _speech_billed_characters(speech_response: HttpxBinaryResponseContent | None, prompt: str) -> int:
    reported: Final = speech_response.hidden_params.get("billed_characters") if speech_response is not None else None
    return reported if isinstance(reported, int) else litellm.utils.count_characters(text=prompt)


_VIDEO_USAGE_NUMBER: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)


def _video_usage_value(usage: object, field: str) -> object:
    fields: Final = as_str_mapping(usage)
    return fields.get(field) if fields is not None else attribute_or_none(usage, field)


def _video_count(usage: object) -> int:
    count: Final = _video_usage_value(usage, "video_count")
    return count if type(count) is int and count > 1 else 1


def _priced_search(
    search_model: str,
    model: str,
    custom_llm_provider: str | None,
    number_of_queries: int,
    optional_params: Mapping[str, object] | None,
    region: str | None,
    configured_cost_per_query: float | None,
    litellm_logging_obj: LitellmLoggingObject | None,
    service_tier: str | None,
    data_residency: str | None,
) -> float:
    from litellm.search.cost_calculator import search_cost

    searched: Final = search_cost(
        model=search_model,
        custom_llm_provider=custom_llm_provider,
        number_of_queries=number_of_queries,
        optional_params=optional_params,
        adjust=AdjustFacts(provider=custom_llm_provider, model=model, region=region),
        configured_cost_per_query=configured_cost_per_query,
    )
    adjusted: Final = (
        AdjustedCost(
            cost=searched.total,
            discount_percent=searched.discount_percent,
            discount_amount=searched.discount_amount,
            margin_percent=searched.margin_percent,
            margin_fixed_amount=searched.margin_fixed_amount,
            margin_total_amount=searched.margin_total_amount,
        )
        if isinstance(searched, NativeReceipt)
        else adjusted_cost(searched, custom_llm_provider, model, region)
    )
    prompt_cost: Final = searched.prompt_cost if isinstance(searched, NativeReceipt) else searched
    _store_cost_breakdown_in_logging_obj(
        litellm_logging_obj=litellm_logging_obj,
        prompt_tokens_cost_usd_dollar=prompt_cost,
        completion_tokens_cost_usd_dollar=0.0,
        cost_for_built_in_tools_cost_usd_dollar=0.0,
        total_cost_usd_dollar=adjusted.cost,
        original_cost=prompt_cost,
        discount_percent=adjusted.discount_percent,
        discount_amount=adjusted.discount_amount,
        margin_percent=adjusted.margin_percent,
        margin_fixed_amount=adjusted.margin_fixed_amount,
        margin_total_amount=adjusted.margin_total_amount,
        service_tier=service_tier,
        data_residency=data_residency,
    )
    return adjusted.cost


def _billable_input_images(usage: object) -> int:
    match usage:
        case {"billable_input_image_count": int(billable_input_images)} if billable_input_images > 0:
            return billable_input_images
        case _:
            return 0


_LOGGED_LITELLM_PARAMS: Final = TypeAdapter(Mapping[str, object])


def _configured_search_cost_per_query(litellm_logging_obj: LitellmLoggingObject | None) -> float | None:
    logged_params: Final = getattr(litellm_logging_obj, "litellm_params", None)
    if logged_params is None:
        return None
    cost_per_query: Final = _LOGGED_LITELLM_PARAMS.validate_python(logged_params).get("input_cost_per_query")
    return float(cost_per_query) if isinstance(cost_per_query, (int, float)) else None


def _deployment_model_info(
    litellm_logging_obj: LitellmLoggingObject | None,
    custom_pricing: bool | None,
    router_model_id: str | None,
) -> ModelInfo | None:
    if not custom_pricing:
        return None
    registered_deployment_info: Final = _raw_cost_map_entry(router_model_id) if router_model_id is not None else None
    if registered_deployment_info is not None:
        return cast(ModelInfo, registered_deployment_info)  # cast-ok: router registers deployment prices under its id
    if litellm_logging_obj is None:
        return None
    litellm_params: Final = litellm_logging_obj.litellm_params
    if not litellm_params:
        return None
    return next(
        (
            model_info
            for metadata_key in ("metadata", "litellm_metadata")
            if (metadata := litellm_params.get(metadata_key)) and (model_info := metadata.get("model_info")) is not None
        ),
        None,
    )


def _ocr_price_tables(
    litellm_logging_obj: LitellmLoggingObject | None,
    custom_pricing: bool | None,
    router_model_id: str | None,
) -> tuple[Mapping[str, object], ...]:
    deployment_info: Final = _deployment_model_info(litellm_logging_obj, custom_pricing, router_model_id)
    litellm_params: Final = (
        litellm_logging_obj.litellm_params if custom_pricing and litellm_logging_obj is not None else None
    )
    return tuple(table for table in (litellm_params, deployment_info) if table is not None)


def _cost_map_model_info(model: str, custom_llm_provider: str | None) -> ModelInfo | None:
    try:
        return litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        return None


_NO_COST_MAP_ENTRY: Final[Mapping[str, object]] = MappingProxyType({})


def _raw_cost_map_entry(key: str) -> Mapping[str, object] | None:
    raw_entry: Final = litellm.model_cost.get(key)
    return raw_entry if isinstance(raw_entry, Mapping) else None


def pricing_entry_for_cost_calc(
    model: str | None,
    completion_response: object | None,
    custom_llm_provider: str | None,
    custom_pricing: bool | None,
    base_model: str | None,
    router_model_id: str | None,
    region_name: str | None,
    litellm_logging_obj: LitellmLoggingObject | None,
) -> tuple[str, Mapping[str, object]] | None:
    deployment_entry: Final = _deployment_model_info(litellm_logging_obj, custom_pricing, router_model_id)
    deployment_key: Final = router_model_id or model
    if deployment_entry is not None and deployment_key is not None:
        return deployment_key, deployment_entry
    found: Final = cost_candidates(
        _call_facts(
            completion_response,
            model,
            custom_llm_provider,
            base_model,
            custom_pricing,
            router_model_id,
            region_name,
            _invoked_model_id_for_cost(None, litellm_logging_obj),
        )
    )
    selected_model: Final = found.selected
    bedrock_model_info: Final = _bedrock_model_info(found.bedrock)
    if bedrock_model_info is not None and found.bedrock is not None:
        key: Final = bedrock_model_info["key"]
        declared_pricing: Final = declared_cost_map_pricing(key, model or key, custom_llm_provider)
        registered: Final = _registered_fields(found.bedrock[1]) if model else MappingProxyType({})
        declared_keys: Final = frozenset(declared_pricing) | frozenset(registered)
        selected_entry: Final[Mapping[str, object]] = bedrock_model_info
        return key, MappingProxyType({name: value for name, value in selected_entry.items() if name in declared_keys})
    candidates: Final = (selected_model, _get_response_model(completion_response), model)
    resolved: Final = next(
        (info for info in (_cost_map_model_info(name, custom_llm_provider) for name in candidates if name) if info),
        None,
    )
    if resolved is None:
        return None
    return resolved["key"], declared_cost_map_pricing(resolved["key"], model or resolved["key"], custom_llm_provider)


def ocr_cost(
    model: str,
    custom_llm_provider: str | None,
    response: object | None = None,
    model_info: OCRPricing | Sequence[Mapping[str, object]] | None = None,
) -> tuple[float, float]:
    """
    Args:
        model: str - model name
        custom_llm_provider: Optional[str] - custom LLM provider
        response: Optional[Any] - response object
        model_info: Optional[OCRPricing] - deployment-specific OCR pricing, or several such tables in order; each
            rate the first table sets overrides the later ones and the model cost map's

    Returns:
        Tuple[float, float]: cost of OCR processing

        (Parent function requires a tuple, so we return a tuple. Cost is only in the first element.)
    """
    from litellm.llms.base_llm.ocr.transformation import OCRResponse

    if response is None or not isinstance(response, OCRResponse):
        raise ValueError(f"response must be of type OCRResponse got type={type(response)}")

    if response.usage_info is None:
        raise ValueError("OCR response usage_info is None")

    outcome: Final = price_units(
        OcrDocument(
            kind="ocr",
            model=model,
            provider=custom_llm_provider,
            pages=response.usage_info.pages_processed,
            annotation_pages=response.usage_info.pages_processed_annotation or 0,
            credits=response.usage_info.credits,
            batch=False,
            catalog_deferred=False,
        ),
        _price_tables(model_info),
        _cost_map_model_info(model, custom_llm_provider),
    )
    if isinstance(outcome, NativeUnpriced) and outcome.reason == "missing_usage":
        raise ValueError("OCR response pages_processed is None")
    if isinstance(outcome, NativeUnpriced):
        verbose_logger.warning(
            "OCR cost: model=%s custom_llm_provider=%s reported pages_processed=%s credits=%s but no OCR rate "
            "prices it; returning 0.0 cost.",
            _single_log_line(model),
            _single_log_line(custom_llm_provider),
            response.usage_info.pages_processed,
            response.usage_info.credits,
        )
    return prompt_and_completion(outcome)


def _price_tables(
    model_info: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
) -> tuple[Mapping[str, object], ...]:
    if model_info is None:
        return ()
    if isinstance(model_info, Mapping):
        return (model_info,)
    return tuple(model_info)


def ocr_batch_cost(
    model: str,
    custom_llm_provider: str | None,
    usage_info: "OCRUsageInfo",
    model_info: ModelInfo | None = None,
) -> tuple[float, float]:
    """Per-page cost of one OCR result inside a batch output file.

    Batch OCR is billed per page at the ``*_batches`` rate, falling back to the
    synchronous per-page rate when a model has no batch price recorded. Each
    per-page family (OCR pages, annotation pages) belongs to the deployment's
    ``model_info`` when it prices that family at either rate and to the published
    cost map otherwise. Returns ``(prompt_cost, completion_cost)`` with the
    whole cost in the first slot, like ``ocr_cost``.
    """
    document: Final = OcrDocument(
        kind="ocr",
        model=model,
        provider=custom_llm_provider,
        pages=usage_info.pages_processed or 0,
        annotation_pages=usage_info.pages_processed_annotation or 0,
        credits=None,
        batch=True,
        catalog_deferred=True,
    )
    tables: Final = _price_tables(model_info)
    deferred: Final = price_units(document, tables)
    outcome: Final = (
        price_units(
            OcrDocument(
                kind="ocr",
                model=model,
                provider=custom_llm_provider,
                pages=document["pages"],
                annotation_pages=document["annotation_pages"],
                credits=None,
                batch=True,
                catalog_deferred=False,
            ),
            tables,
            _lookup_model_info_or_none(model=model, custom_llm_provider=custom_llm_provider),
        )
        if isinstance(deferred, NativeUnpriced) and deferred.reason == "catalog_deferred"
        else deferred
    )
    if isinstance(outcome, NativeUnpriced):
        verbose_logger.warning(
            "OCR batch cost: model=%s custom_llm_provider=%s reported pages_processed=%s but no "
            "ocr_cost_per_page is configured; returning 0.0 cost for those pages.",
            _single_log_line(model),
            _single_log_line(custom_llm_provider),
            usage_info.pages_processed,
        )
    return prompt_and_completion(outcome)


def _single_log_line(value: str | None) -> str:
    return str(value).replace("\n", "").replace("\r", "")


def _lookup_model_info_or_none(model: str, custom_llm_provider: str | None) -> ModelInfo | None:
    try:
        return litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models; caller logs and bills 0.0
        return None


def vector_store_search_cost(
    model: str | None,
    custom_llm_provider: str,
    response: VectorStoreSearchResponse,
) -> tuple[float, float]:
    """
    Returns
    - float or None: cost of vector store search
    """
    api_type: str | None = None
    if custom_llm_provider is None:
        custom_llm_provider = "openai"

    if model is not None and "/" in model:
        api_type, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=model,
        )

    config: Final = ProviderConfigManager.get_provider_vector_stores_config(
        provider=LlmProviders(custom_llm_provider),
        api_type=api_type,
    )
    if config is None or config.pricing_model is None:
        verbose_logger.debug("Vector store search is not priced for %s", custom_llm_provider)
        return 0.0, 0.0
    return prompt_and_completion(
        price_units(
            VectorStoreSearchDocument(
                kind="vector_store_search", model=config.pricing_model, provider=custom_llm_provider
            ),
            catalog=litellm.get_model_info(model=config.pricing_model),
        )
    )


def rerank_cost(
    model: str,
    custom_llm_provider: str | None,
    billed_units: RerankBilledUnits | None = None,
) -> tuple[float, float]:
    """
    Returns
    - float or None: cost of response OR none if error.
    """
    _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    units: Final = billed_units or RerankBilledUnits()
    return prompt_and_completion(
        price_units(
            RerankDocument(
                kind="rerank",
                model=model,
                provider=LlmProviders(custom_llm_provider).value,
                search_units=units.get("search_units"),
                total_tokens=units.get("total_tokens"),
            ),
            catalog=_lookup_model_info_or_none(model=model, custom_llm_provider=custom_llm_provider),
        )
    )


def transcription_cost(model: str, custom_llm_provider: str | None, duration: float) -> tuple[float, float]:
    return openai_cost_per_second(model=model, custom_llm_provider=custom_llm_provider, duration=duration)


def default_image_row_key(
    model: str, custom_llm_provider: str | None, quality: str | None, size: str | None
) -> tuple[str | None, int, tuple[str | None, ...]]:
    """The ``litellm.model_cost`` key ``default_image_cost_calculator`` reads for an image of ``size`` at ``quality``,
    ``None`` when no name it tries is a key, the image's pixels and the names it tried. Raises on a size that is not
    two numbers."""
    size_text: Final = size or "1024-x-1024"
    size_str: Final = size_text.replace("x", "-x-") if "x" in size_text and "-x-" not in size_text else size_text
    height, width = map(int, size_str.split("-x-"))
    without_provider: Final = (
        model.replace(f"{custom_llm_provider}/", "")
        if custom_llm_provider and model.startswith(f"{custom_llm_provider}/")
        else None
    )
    base_model_name: Final = (
        f"{custom_llm_provider}/{size_str}/{without_provider}"
        if without_provider is not None
        else f"{size_str}/{model}"
    )
    model_name_with_quality: Final = f"{quality}/{base_model_name}" if quality else base_model_name
    provider_first_model_name_with_quality: Final = (
        f"{custom_llm_provider}/{quality}/{size_str}/{without_provider or model}"
        if quality and custom_llm_provider
        else None
    )
    model_name_with_v2_quality: Final = f"{ImageGenerationRequestQuality.HIGH.value}/{base_model_name}"
    model_without_provider: Final = f"{size_str}/{model.split('/')[-1]}"
    model_with_quality_without_provider: Final = (
        f"{quality}/{model_without_provider}" if quality else model_without_provider
    )
    verbose_logger.debug("Looking up cost for models: %s, %s", model_name_with_quality, base_model_name)
    models_to_check: Final = (
        model_name_with_quality,
        provider_first_model_name_with_quality,
        base_model_name,
        model_name_with_v2_quality,
        model_with_quality_without_provider,
        model_without_provider,
        model,
        without_provider,
    )
    matched: Final = next((name for name in models_to_check if name is not None and name in litellm.model_cost), None)
    return matched, height * width, models_to_check


def default_image_cost_calculator(
    model: str,
    custom_llm_provider: str | None = None,
    quality: str | None = None,
    n: int | None = 1,  # Default to 1 image
    size: str | None = "1024-x-1024",  # OpenAI default
    optional_params: dict | None = None,
    model_info: ModelInfo | None = None,
) -> float:
    """
    Default image cost calculator for image generation

    Args:
        model (str): Model name
        image_response (ImageResponse): Response from image generation
        quality (Optional[str]): Image quality setting
        n (Optional[int]): Number of images generated
        size (Optional[str]): Image size (e.g. "1024x1024" or "1024-x-1024")
        model_info (Optional[ModelInfo]): The deployment's own prices, consulted before the cost map

    Returns:
        float: Cost in USD for the image generation

    Raises:
        Exception: If model pricing not found in cost map
    """
    matched_model, pixels, models_to_check = default_image_row_key(model, custom_llm_provider, quality, size)
    if matched_model is None and model_info is None:
        raise Exception(f"Model not found in cost map. Tried checking {models_to_check}")

    image_count: Final = n if n is not None else 1
    outcome: Final = price_units(
        ImageDocument(
            kind="image",
            model=model,
            provider=custom_llm_provider,
            calculator="default",
            images=image_count,
            pixels=pixels * image_count,
            input_images=0,
        ),
        (model_info,) if model_info is not None else (),
        litellm.model_cost[matched_model] if matched_model is not None else None,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # model cost map is dynamically loaded
    )
    if not isinstance(outcome, NativeReceipt):
        raise Exception(f"No pricing information found for model {model}. Tried checking {models_to_check}")
    return outcome.total


def default_video_cost_calculator(
    model: str,
    duration_seconds: float,
    custom_llm_provider: str | None = None,
    model_info: ModelInfo | None = None,
    video_resolution: str | None = None,
    videos: int = 1,
) -> float:
    """
    Default video cost calculator for video generation

    Args:
        model (str): Model name
        duration_seconds (float): Duration of the generated video in seconds
        custom_llm_provider (Optional[str]): Custom LLM provider
        model_info (Optional[ModelInfo]): Deployment-level model info containing
            custom video pricing. When provided, used before falling back to
            the global litellm.model_cost lookup.
        video_resolution (Optional[str]): From usage (e.g. ``720p``, ``1080p``) for tiered per-second pricing.
        videos (int): How many videos the call generated, each billed the same

    Returns:
        float: Cost in USD for the video generation

    Raises:
        Exception: If model pricing not found in cost map
    """
    # Use custom model_info pricing if provided (deployment-specific pricing)
    cost_info: dict | None = None
    if model_info is not None:
        cost_info = dict(model_info)
    else:
        # Build model names for cost lookup
        base_model_name = model
        model_name_without_custom_llm_provider: str | None = None
        if custom_llm_provider and model.startswith(f"{custom_llm_provider}/"):
            model_name_without_custom_llm_provider = model.replace(f"{custom_llm_provider}/", "")
            base_model_name = f"{custom_llm_provider}/{model_name_without_custom_llm_provider}"

        verbose_logger.debug("Looking up cost for video model: %s", base_model_name)

        model_without_provider: Final = model.split("/")[-1]

        # Try model with provider first, fall back to base model name
        models_to_check: Final[list[str | None]] = [
            base_model_name,
            model,
            model_without_provider,
            model_name_without_custom_llm_provider,
        ]
        for _model in models_to_check:
            if _model is not None and _model in litellm.model_cost:
                cost_info = litellm.model_cost[_model]
                break

        # If still not found, try with custom_llm_provider prefix
        if cost_info is None and custom_llm_provider:
            prefixed_model: Final = f"{custom_llm_provider}/{model}"
            if prefixed_model in litellm.model_cost:
                cost_info = litellm.model_cost[prefixed_model]

    if cost_info is None:
        raise Exception(f"Model not found in cost map for model={model}")

    return priced_video(
        model=model,
        custom_llm_provider=custom_llm_provider,
        seconds=duration_seconds,
        video_resolution=video_resolution,
        deployment=model_info,
        catalog=None if model_info is not None else cost_info,
        videos=videos,
    )


_BATCH_TOKEN_RATE_KEYS: Final = (
    "input_cost_per_token_batches",
    "input_cost_per_token",
    "output_cost_per_token_batches",
    "output_cost_per_token",
)


def batch_line_model_info(model: str, custom_llm_provider: str, model_info: ModelInfo | None) -> ModelInfo | None:
    """The row a batch line is priced from: the deployment's own when it sets a token rate, else the cost map's row
    with the deployment's values written over it, ``None`` when neither exists."""
    if model_info is None:
        return _lookup_model_info_or_none(model, custom_llm_provider)
    if any(model_info.get(key) is not None for key in _BATCH_TOKEN_RATE_KEYS):
        return model_info
    global_info: Final = _lookup_model_info_or_none(model, custom_llm_provider)
    if not global_info:
        return model_info
    return cast(  # cast-ok: the deployment's set prices layered over the cost map's ModelInfo row
        ModelInfo, {**global_info, **{key: value for key, value in model_info.items() if value is not None}}
    )


def batch_line_document(
    usage: Usage,
    model: str,
    custom_llm_provider: str,
    data_residency: str | None,
    vertex_location: str | None,
) -> BatchLineDocument:
    return BatchLineDocument(
        kind="batch_line",
        model=model,
        provider=custom_llm_provider,
        usage=batch_usage(usage),
        data_residency=data_residency,
        vertex_location=vertex_location,
    )


def batch_cost_calculator(
    usage: Usage,
    model: str,
    custom_llm_provider: str | None = None,
    model_info: ModelInfo | None = None,
    data_residency: str | None = None,
    *,
    vertex_location: str | None = None,
) -> tuple[float, float]:
    """
    Calculate the cost of a batch job.

    Args:
        model_info: Optional deployment-level model info containing custom
            batch pricing (e.g. input_cost_per_token_batches). When provided,
            skips the global litellm.get_model_info() lookup so that
            deployment-specific pricing is used.
    """
    _, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    row: Final = batch_line_model_info(model, provider, model_info)
    if not row:
        return 0.0, 0.0
    return prompt_and_completion(
        price_units(batch_line_document(usage, model, provider, data_residency, vertex_location), catalog=row)
    )


_TRANSCRIPTION_COMPLETED_EVENT_TYPE: Final = "conversation.item.input_audio_transcription.completed"


class _LiveResponsePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | None = None
    model: str | None = None
    usage: Mapping[str, object] | None = None


class _LiveResponseEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    response: _LiveResponsePayload | None = None


class _LiveResponseEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    event: _LiveResponseEvent | None = None


def _live_response_payload(result: Mapping[str, object]) -> _LiveResponsePayload | None:
    try:
        envelope: Final = _LiveResponseEnvelope.model_validate(result)
    except ValidationError:
        return None
    if (
        envelope.type != "response.event"
        or envelope.event is None
        or envelope.event.type not in OPENAI_LIVE_TERMINAL_RESPONSE_EVENT_TYPES
        or envelope.event.response is None
        or envelope.event.response.usage is None
    ):
        return None
    return envelope.event.response


def _unique_live_response_payloads(results: Sequence[Mapping[str, object]]) -> tuple[_LiveResponsePayload, ...]:
    response_payloads: Final = tuple(
        response for result in results if (response := _live_response_payload(result)) is not None
    )
    first_payload_by_id: Final = {
        response.id: response for response in reversed(response_payloads) if response.id is not None
    }
    return tuple(reversed(first_payload_by_id.values()))


def _live_session_model(result: Mapping[str, object]) -> str | None:
    session: Final = as_str_mapping(result.get("session"))
    if session is None:
        return None
    model: Final = session.get("model")
    return model if isinstance(model, str) else None


def _live_usage_seconds(result: Mapping[str, object]) -> float | None:
    usage: Final = as_str_mapping(result.get("usage"))
    if usage is None:
        return None
    seconds: Final = usage.get("seconds")
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds):
        return float(seconds)
    return None


def _last_live_usage_seconds(results: Sequence[Mapping[str, object]], event_type: str) -> float | None:
    return next(
        (
            seconds
            for result in reversed(results)
            if result.get("type") == event_type and (seconds := _live_usage_seconds(result)) is not None
        ),
        None,
    )


def _live_session_wall_clock_seconds(litellm_logging_obj: LitellmLoggingObject | None) -> float:
    if litellm_logging_obj is None:
        return 0.0
    model_call_details: Final = litellm_logging_obj.model_call_details
    start_time: Final = model_call_details.get("start_time")
    end_time: Final = model_call_details.get("end_time")
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        return max((end_time - start_time).total_seconds(), 0.0)
    return 0.0


_SESSION_RATE_KEYS: Final = (
    "input_cost_per_second",
    "output_cost_per_second",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
    "input_cost_per_token",
    "output_cost_per_token",
)

_SessionRows: TypeAlias = tuple[Mapping[str, object], ...]


def _live_rows(model_names: Sequence[str | None], custom_llm_provider: str) -> _SessionRows:
    """The row ``get_model_info`` finds for each of ``model_names`` that has one, in order."""
    return tuple(
        row
        for model_name in model_names
        if model_name is not None
        and (row := as_str_mapping(_lookup_model_info_or_none(model_name, custom_llm_provider))) is not None
    )


def _declared_rates(info: ModelInfo | None) -> Mapping[str, object]:
    """The rates of ``info`` its own cost map entry declares, since ``get_model_info`` synthesizes zero token rates
    for an entry that omits them."""
    fields: Final = as_str_mapping(info) or {}
    key: Final = fields.get("key")
    declared: Final = as_str_mapping(litellm.model_cost.get(key)) if isinstance(key, str) else None
    if declared is None:
        return MappingProxyType({})
    return MappingProxyType(
        {
            rate_key: value
            for rate_key in _SESSION_RATE_KEYS
            if declared.get(rate_key) is not None and (value := fields.get(rate_key)) is not None
        }
    )


@dataclass(frozen=True, slots=True)
class _SessionUse:
    """What a session used beside its tokens, each part beside the rows main tries for it."""

    clips: tuple[tuple[ClipUsage, _SessionRows], ...] = ()
    translation: tuple[float, float, _SessionRows] | None = None
    live_voice: tuple[float, _SessionRows] | None = None
    audio_duration: tuple[float, _SessionRows] | None = None
    text_messages: tuple[float, _SessionRows] | None = None
    tokens: "_SessionTokens | None" = None
    turns: "tuple[_SessionTurn, ...]" = ()
    provider: str = ""
    data_residency: str | None = None


@dataclass(frozen=True, slots=True)
class _TokenCandidate:
    model: str
    row: Mapping[str, object]
    declares: bool


@dataclass(frozen=True, slots=True)
class _SessionTokens:
    """The session's combined usage and the models main tries for it, each beside the row ``get_model_info`` found."""

    usage: PricedUsage
    candidates: tuple[_TokenCandidate, ...]


@dataclass(frozen=True, slots=True)
class _SessionTurn:
    """One Live ``response.done`` turn, its row ``None`` when ``get_model_info`` has none for its model."""

    model: str
    usage: PricedUsage
    row: Mapping[str, object] | None


def _metered(used: tuple[float, _SessionRows] | None, rows: Sequence[int]) -> MeteredUse | None:
    return None if used is None else MeteredUse(quantity=used[0], rows=rows)


def _session_costs(used: _SessionUse) -> Mapping[str, float]:
    """Each part's cost, priced in one crossing, a part no row prices left out."""
    return part_costs(_priced_session(used))


def _priced_session(used: _SessionUse) -> NativeReceipt | NativeUnpriced:
    """The session's tokens, turns and parts, priced in one crossing."""
    parts: Final = (used.translation, used.live_voice, used.audio_duration, used.text_messages)
    token_rows: Final = () if used.tokens is None else tuple(offered.row for offered in used.tokens.candidates)
    turn_rows: Final = tuple(turn.row for turn in used.turns if turn.row is not None)
    groups: Final = (
        *(rows for _, rows in used.clips),
        *(() if part is None else part[-1] for part in parts),
        token_rows,
        turn_rows,
    )
    starts: Final = tuple(accumulate((len(group) for group in groups), initial=0))
    indexes: Final = tuple(tuple(range(start, start + len(group))) for start, group in zip(starts, groups))
    clip_indexes: Final = indexes[: len(used.clips)]
    translation_rows, live_rows, audio_rows, message_rows, token_indexes, turn_indexes = indexes[len(used.clips) :]
    rows_before: Final = tuple(accumulate((turn.row is not None for turn in used.turns), initial=0))
    document: Final = SessionDocument(
        transcription=tuple(
            ClipUse(rows=rows, usage=usage) for (usage, _), rows in zip(used.clips, clip_indexes, strict=True)
        ),
        translation=None
        if used.translation is None
        else TranslationUse(
            input_seconds=used.translation[0], output_seconds=used.translation[1], rows=translation_rows
        ),
        live_voice=_metered(used.live_voice, live_rows),
        audio_duration=_metered(used.audio_duration, audio_rows),
        text_messages=_metered(used.text_messages, message_rows),
        tokens=None
        if used.tokens is None
        else SessionTokensUse(
            usage=used.tokens.usage,
            candidates=tuple(
                TokenCandidateUse(
                    candidate=SessionCandidate(model=offered.model, provider=used.provider),
                    row=row,
                    declares=offered.declares,
                )
                for offered, row in zip(used.tokens.candidates, token_indexes, strict=True)
            ),
        ),
        turns=tuple(
            SessionTurnUse(
                candidate=SessionCandidate(model=turn.model, provider=used.provider),
                usage=turn.usage,
                row=None if turn.row is None else turn_indexes[before],
            )
            for turn, before in zip(used.turns, rows_before, strict=False)
        ),
        data_residency=used.data_residency,
    )
    return price_session(document, tuple(chain.from_iterable(groups)))


def _xai_metered_usage(litellm_logging_obj: LitellmLoggingObject | None) -> RealtimeMeteredUsage | None:
    if litellm_logging_obj is None:
        return None
    usage: Final = litellm_logging_obj.model_call_details.get(REALTIME_METERED_USAGE_KEY)
    return usage if isinstance(usage, RealtimeMeteredUsage) else None


def _cost_map_entry_declares_pricing(model_name: str, custom_llm_provider: str) -> bool:
    """Whether the entry behind ``model_name`` sets any rate of its own, even a zero one.

    The name is resolved the way ``get_model_info`` resolves it before the raw entry is read,
    because a deployment-scoped name arrives here already carrying its provider prefix. Two raw
    lookups cannot strip that prefix, so a zero-rated override read as declaring nothing, and a
    session that should bill nothing fell through to the public rates instead.
    """
    resolved: Final = _cost_map_model_info(model_name, custom_llm_provider)
    entries: Final = (
        litellm.model_cost.get(resolved.get("key")) if resolved is not None else None,
        litellm.model_cost.get(model_name),
        litellm.model_cost.get(f"{custom_llm_provider}/{model_name}"),
    )
    return any(entry is not None and _cost_map_entry_prices_anything(entry) for entry in entries)


def _token_candidates(model_names: Sequence[str | None], custom_llm_provider: str) -> tuple[_TokenCandidate, ...]:
    """Each model main tries for a session's tokens that ``get_model_info`` has a row for, in order."""
    return tuple(
        _TokenCandidate(
            model=model_name, row=row, declares=_cost_map_entry_declares_pricing(model_name, custom_llm_provider)
        )
        for model_name in model_names
        if model_name is not None
        and (row := as_str_mapping(_lookup_model_info_or_none(model_name, custom_llm_provider))) is not None
    )


def _turn_usage(response: _LiveResponsePayload) -> Usage | None:
    try:
        return get_usage_object(response.model_dump())
    except (ValidationError, TypeError, ValueError, KeyError, AttributeError):
        return None


def _live_turns(results: OpenAIRealtimeStreamList, custom_llm_provider: str) -> tuple[_SessionTurn, ...]:
    """Each distinct Live ``response.done`` turn that names its id, model and usage."""
    return tuple(
        _SessionTurn(
            model=response.model,
            usage=priced_usage(usage, web_search=None),
            row=as_str_mapping(_lookup_model_info_or_none(response.model, custom_llm_provider)),
        )
        for response in _unique_live_response_payloads(results)
        if response.id is not None
        and response.model is not None
        and response.usage is not None
        and (usage := _turn_usage(response)) is not None
    )


def handle_realtime_stream_cost_calculation(
    results: OpenAIRealtimeStreamList,
    combined_usage_object: Usage,
    custom_llm_provider: str,
    litellm_model_name: str,
    data_residency: str | None = None,
    litellm_logging_obj: LitellmLoggingObject | None = None,
    custom_pricing_model: str | None = None,
    base_pricing_model: str | None = None,
    deployment_models: Sequence[str | None] = (),
) -> float:
    """
    Handles the cost calculation for realtime stream responses.

    Pick the 'response.done' events. Calculate total cost across all 'response.done' events.

    Args:
        results: A list of OpenAIRealtimeStreamBaseObject objects
        custom_pricing_model: deployment-scoped pricing key from the deployment's
            custom rates, tried ahead of the session-reported model
        base_pricing_model: the deployment's resolved base_model, tried ahead of the
            session-reported model but after custom rates
    """
    received_model: Final = next(
        (
            model_name
            for result in results
            if result["type"] in ("session.created", "session.started")
            and (model_name := _live_session_model(result)) is not None
        ),
        None,
    )
    potential_model_names: Final = (
        custom_pricing_model,
        base_pricing_model,
        received_model,
        litellm_model_name,
    )
    live_lifecycle_event_types: Final = frozenset({"session.started", "session.usage.updated", "session.closed"})
    live_session_started: Final = any(result["type"] == "session.started" for result in results)
    live_session_closed: Final = any(result["type"] == "session.closed" for result in results)
    live_event_seen: Final = any(result["type"] in live_lifecycle_event_types for result in results)
    closed_seconds: Final = _last_live_usage_seconds(results, "session.closed")
    reported_seconds: Final = _last_live_usage_seconds(results, "session.usage.updated")
    latest_reported_seconds: Final = 0.0 if reported_seconds is None else reported_seconds
    voice_seconds: Final = (
        closed_seconds
        if closed_seconds is not None
        else max(
            latest_reported_seconds,
            _live_session_wall_clock_seconds(litellm_logging_obj),
        )
        if live_session_started and not live_session_closed
        else latest_reported_seconds
    )
    live_model_names: Final = (
        (custom_pricing_model, base_pricing_model, received_model, litellm_model_name) if live_event_seen else ()
    )
    xai_metered: Final = (
        _xai_metered_usage(litellm_logging_obj) if custom_llm_provider == LlmProviders.XAI.value else None
    )
    xai_audio_seconds: Final = (
        xai_metered.audio_seconds_sent + xai_metered.audio_seconds_received
        if xai_metered is not None and xai_metered.push_to_talk
        else _live_session_wall_clock_seconds(litellm_logging_obj)
        if custom_llm_provider == LlmProviders.XAI.value
        else 0.0
    )
    session_rows: Final = _live_rows(potential_model_names, custom_llm_provider)
    priced: Final = _priced_session(
        _SessionUse(
            clips=_transcription_clips(results, custom_llm_provider, litellm_model_name, custom_pricing_model),
            translation=_translation_use(
                results,
                custom_llm_provider,
                (custom_pricing_model, base_pricing_model, *deployment_models, litellm_model_name, received_model),
            ),
            live_voice=(voice_seconds, _live_rows(live_model_names, custom_llm_provider)),
            audio_duration=(xai_audio_seconds, session_rows),
            text_messages=(float(xai_metered.text_messages if xai_metered is not None else 0), session_rows),
            tokens=_SessionTokens(
                usage=priced_usage(combined_usage_object, web_search=None),
                candidates=_token_candidates(potential_model_names, custom_llm_provider),
            ),
            turns=_live_turns(results, custom_llm_provider),
            provider=custom_llm_provider,
            data_residency=data_residency,
        )
    )
    costs: Final = part_costs(priced)
    transcription_cost: Final = costs.get("transcription_cost", 0.0)
    translation_cost: Final = costs.get("translation_cost", 0.0)
    live_voice_cost: Final = costs.get("live_voice_cost", 0.0)
    audio_duration_cost: Final = costs.get("audio_duration_cost", 0.0)
    text_message_cost: Final = costs.get("text_message_cost", 0.0)
    input_cost: Final = priced.prompt_cost if isinstance(priced, NativeReceipt) else 0.0
    output_cost: Final = priced.completion_cost if isinstance(priced, NativeReceipt) else 0.0
    total_cost: Final = total_or_zero(priced)
    additional_costs: Final = tuple(
        (cost_name, cost)
        for cost_name, cost in (
            ("transcription_cost", transcription_cost),
            ("translation_cost", translation_cost),
            ("live_voice_cost", live_voice_cost),
            ("audio_duration_cost", audio_duration_cost),
            ("text_message_cost", text_message_cost),
        )
        if cost > 0
    )

    _store_cost_breakdown_in_logging_obj(
        litellm_logging_obj=litellm_logging_obj,
        prompt_tokens_cost_usd_dollar=input_cost,
        completion_tokens_cost_usd_dollar=output_cost,
        cost_for_built_in_tools_cost_usd_dollar=0.0,
        total_cost_usd_dollar=total_cost,
        additional_costs=dict(additional_costs) if additional_costs else None,
        data_residency=data_residency,
    )

    return total_cost


def _translation_use(
    results: OpenAIRealtimeStreamList, custom_llm_provider: str, model_names: Sequence[str | None]
) -> tuple[float, float, _SessionRows]:
    input_seconds, output_seconds = translation_session_seconds(results)
    rows: Final = (
        tuple(
            _declared_rates(_lookup_model_info_or_none(model, custom_llm_provider))
            for model in model_names
            if model is not None
        )
        if input_seconds > 0 or output_seconds > 0
        else ()
    )
    return input_seconds, output_seconds, rows


def handle_realtime_translation_cost_calculation(
    results: OpenAIRealtimeStreamList,
    custom_llm_provider: str,
    litellm_model_name: str,
    deployment_pricing_models: Sequence[str | None] = (),
    session_model: str | None = None,
) -> float:
    translation: Final = _translation_use(
        results, custom_llm_provider, (*deployment_pricing_models, litellm_model_name, session_model)
    )
    return _session_costs(_SessionUse(translation=translation)).get("translation_cost", 0.0)


def _clip_usage(usage: Mapping[str, object]) -> ClipUsage:
    usage_type: Final = usage.get("type")
    if usage_type == "duration":
        return DurationClip(
            type="duration", seconds=float(normalized_audio_duration_seconds(usage.get("seconds")) or 0.0)
        )
    if usage_type != "tokens":
        return OtherClip(type="other")
    details: Final = as_str_mapping(usage.get("input_token_details")) or {}
    return TokenClip(
        type="tokens",
        audio=float(normalized_audio_duration_seconds(details.get("audio_tokens")) or 0.0),
        text=float(normalized_audio_duration_seconds(details.get("text_tokens")) or 0.0),
        output=float(normalized_audio_duration_seconds(usage.get("output_tokens")) or 0.0),
    )


def _transcription_clips(
    results: OpenAIRealtimeStreamList,
    custom_llm_provider: str,
    litellm_model_name: str,
    custom_pricing_model: str | None,
) -> tuple[tuple[ClipUsage, _SessionRows], ...]:
    """Each completed transcription clip and its rows: the session's pricing override first, then the ASR model the
    session configured for it."""
    configured_models: Final = tuple(accumulate(map(_session_transcription_model, results), _newer_model))
    clips: Final = tuple(
        (as_str_mapping(result.get("usage")) or {}, configured)
        for result, configured in zip(results, configured_models, strict=True)
        if result.get("type") == _TRANSCRIPTION_COMPLETED_EVENT_TYPE
    )
    if not clips:
        return ()
    fallback_model: Final = _get_transcription_model_name_from_results(results) or litellm_model_name
    clip_models: Final = tuple((usage, configured or fallback_model) for usage, configured in clips)
    model_rows: Final = MappingProxyType(
        {
            model: _declared_rates(_cost_map_model_info(model, custom_llm_provider))
            for model in frozenset(m for _, m in clip_models)
        }
    )
    override_rows: Final = (
        _declared_rates(_cost_map_model_info(custom_pricing_model, custom_llm_provider))
        if custom_pricing_model is not None
        else MappingProxyType({})
    )
    return tuple((_clip_usage(usage), (override_rows, model_rows[model])) for usage, model in clip_models)


def handle_realtime_transcription_cost_calculation(
    results: OpenAIRealtimeStreamList,
    custom_llm_provider: str,
    litellm_model_name: str,
    custom_pricing_model: str | None = None,
) -> float:
    """
    Cost for realtime transcription sessions (e.g. gpt-realtime-whisper).

    Transcription sessions emit no `response.done` events; instead each
    `conversation.item.input_audio_transcription.completed` event carries a
    `usage` object billed by the ASR model. The usage is one of:
      - {"type": "duration", "seconds": <float>}  -> priced via input_cost_per_second
      - {"type": "tokens", "input_token_details": ...}  -> priced via input/audio token cost
    """
    clips: Final = _transcription_clips(results, custom_llm_provider, litellm_model_name, custom_pricing_model)
    return _session_costs(_SessionUse(clips=clips)).get("transcription_cost", 0.0)


_TRANSCRIPTION_SESSION_EVENT_TYPES: Final = frozenset(
    {"transcription_session.created", "transcription_session.updated", "session.created", "session.updated"}
)


def _str_field(mapping: Mapping[str, object] | None, key: str) -> str | None:
    value: Final = None if mapping is None else mapping.get(key)
    return value if isinstance(value, str) and value else None


def _configured_transcription_model(session: Mapping[str, object]) -> str | None:
    audio_input: Final = as_str_mapping((as_str_mapping(session.get("audio")) or {}).get("input"))
    nested: Final = _str_field(as_str_mapping((audio_input or {}).get("transcription")), "model")
    return nested or _str_field(as_str_mapping(session.get("input_audio_transcription")), "model")


def _session_of(result: Mapping[str, object]) -> Mapping[str, object] | None:
    return as_str_mapping(result.get("session")) if result.get("type") in _TRANSCRIPTION_SESSION_EVENT_TYPES else None


def _session_transcription_model(result: Mapping[str, object]) -> str | None:
    session: Final = _session_of(result)
    return None if session is None else _configured_transcription_model(session)


def _newer_model(current: str | None, newer: str | None) -> str | None:
    return newer or current


def _get_transcription_model_name_from_results(
    results: OpenAIRealtimeStreamList,
) -> str | None:
    return next(
        (model for result in results if (model := _str_field(_session_of(result), "model"))),
        None,
    )
