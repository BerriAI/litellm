from datetime import date
from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from litellm.types.llms.base_model import LiteLLMBaseModel

UsdPrice = Annotated[float, Field(ge=0)]
Multiplier = Annotated[float, Field(gt=0)]


class CostMapConfig(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class SearchContextCost(CostMapConfig):
    search_context_size_high: UsdPrice
    search_context_size_low: UsdPrice
    search_context_size_medium: UsdPrice


class SharedModelConfig(CostMapConfig):
    model_provider: ClassVar[str]
    inference_provider: ClassVar[str]

    cache_creation_input_token_cost: UsdPrice | None = None
    cache_creation_input_token_cost_batches: UsdPrice | None = None
    cache_read_input_token_cost: UsdPrice | None = None
    cache_read_input_token_cost_batches: UsdPrice | None = None
    deprecation_date: date | None = None
    deprecation_source: str | None = None
    input_cost_per_token: UsdPrice | None = None
    input_cost_per_token_batches: UsdPrice | None = None
    max_input_tokens: NonNegativeInt | None = None
    max_output_tokens: NonNegativeInt | None = None
    mode: str | None = None
    output_cost_per_token: UsdPrice | None = None
    output_cost_per_token_batches: UsdPrice | None = None
    search_context_cost_per_query: SearchContextCost | None = None
    source: str | None = None
    successor_model: str | None = None
    supports_computer_use: bool | None = None
    supports_function_calling: bool | None = None
    supports_max_reasoning_effort: bool | None = None
    supports_pdf_input: bool | None = None
    supports_prompt_caching: bool | None = None
    supports_reasoning: bool | None = None
    supports_response_schema: bool | None = None
    supports_tool_choice: bool | None = None
    supports_vision: bool | None = None
    supports_web_search: bool | None = None
    supports_xhigh_reasoning_effort: bool | None = None


class OpenAIProviderSpecificEntry(CostMapConfig):
    realtime_ga_only: float | None = None
    transcription_json_or_text_only: float | None = None
    transcription_usage_duration: float | None = None


class OpenAIModelOnOpenAIProvider(SharedModelConfig):
    model_provider: ClassVar[str] = "openai"
    inference_provider: ClassVar[str] = "openai"

    cache_creation_input_audio_token_cost: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_above_272k_tokens: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_batches: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_flex: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_priority: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_ultrafast: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_batches: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_flex: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_priority: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_ultrafast: UsdPrice | None = None
    cache_creation_input_token_cost_above_272k_tokens: UsdPrice | None = None
    cache_creation_input_token_cost_above_272k_tokens_batches: UsdPrice | None = None
    cache_creation_input_token_cost_above_272k_tokens_decisions: UsdPrice | None = None
    cache_creation_input_token_cost_above_272k_tokens_flex: UsdPrice | None = None
    cache_creation_input_token_cost_above_272k_tokens_priority: UsdPrice | None = None
    cache_creation_input_token_cost_above_272k_tokens_ultrafast: UsdPrice | None = None
    cache_creation_input_token_cost_decisions: UsdPrice | None = None
    cache_creation_input_token_cost_flex: UsdPrice | None = None
    cache_creation_input_token_cost_priority: UsdPrice | None = None
    cache_creation_input_token_cost_ultrafast: UsdPrice | None = None
    cache_read_input_audio_token_cost: UsdPrice | None = None
    cache_read_input_image_token_cost: UsdPrice | None = None
    cache_read_input_image_token_cost_batches: UsdPrice | None = None
    cache_read_input_token_cost_above_272k_tokens: UsdPrice | None = None
    cache_read_input_token_cost_above_272k_tokens_batches: UsdPrice | None = None
    cache_read_input_token_cost_above_272k_tokens_decisions: UsdPrice | None = None
    cache_read_input_token_cost_above_272k_tokens_flex: UsdPrice | None = None
    cache_read_input_token_cost_above_272k_tokens_priority: UsdPrice | None = None
    cache_read_input_token_cost_above_272k_tokens_ultrafast: UsdPrice | None = None
    cache_read_input_token_cost_decisions: UsdPrice | None = None
    cache_read_input_token_cost_flex: UsdPrice | None = None
    cache_read_input_token_cost_priority: UsdPrice | None = None
    cache_read_input_token_cost_ultrafast: UsdPrice | None = None
    code_interpreter_cost_per_session: UsdPrice | None = None
    default_reasoning_effort: str | None = None
    fedramp_uplift_multiplier: Multiplier | None = None
    input_cost_per_audio_token: UsdPrice | None = None
    input_cost_per_character: UsdPrice | None = None
    input_cost_per_image: UsdPrice | None = None
    input_cost_per_image_token: UsdPrice | None = None
    input_cost_per_image_token_batches: UsdPrice | None = None
    input_cost_per_pixel: UsdPrice | None = None
    input_cost_per_second: UsdPrice | None = None
    input_cost_per_token_above_272k_tokens: UsdPrice | None = None
    input_cost_per_token_above_272k_tokens_batches: UsdPrice | None = None
    input_cost_per_token_above_272k_tokens_decisions: UsdPrice | None = None
    input_cost_per_token_above_272k_tokens_flex: UsdPrice | None = None
    input_cost_per_token_above_272k_tokens_priority: UsdPrice | None = None
    input_cost_per_token_above_272k_tokens_ultrafast: UsdPrice | None = None
    input_cost_per_token_decisions: UsdPrice | None = None
    input_cost_per_token_flex: UsdPrice | None = None
    input_cost_per_token_priority: UsdPrice | None = None
    input_cost_per_token_ultrafast: UsdPrice | None = None
    litellm_provider: Literal["openai"] | None = None
    output_cost_per_audio_token: UsdPrice | None = None
    output_cost_per_image_token: UsdPrice | None = None
    output_cost_per_image_token_batches: UsdPrice | None = None
    output_cost_per_pixel: UsdPrice | None = None
    output_cost_per_second: UsdPrice | None = None
    output_cost_per_token_above_272k_tokens: UsdPrice | None = None
    output_cost_per_token_above_272k_tokens_batches: UsdPrice | None = None
    output_cost_per_token_above_272k_tokens_decisions: UsdPrice | None = None
    output_cost_per_token_above_272k_tokens_flex: UsdPrice | None = None
    output_cost_per_token_above_272k_tokens_priority: UsdPrice | None = None
    output_cost_per_token_above_272k_tokens_ultrafast: UsdPrice | None = None
    output_cost_per_token_decisions: UsdPrice | None = None
    output_cost_per_token_flex: UsdPrice | None = None
    output_cost_per_token_priority: UsdPrice | None = None
    output_cost_per_token_ultrafast: UsdPrice | None = None
    output_cost_per_video_per_second: UsdPrice | None = None
    output_vector_size: NonNegativeInt | None = None
    provider_specific_entry: OpenAIProviderSpecificEntry | None = None
    reasoning_contexts: tuple[str, ...] | None = None
    reasoning_effort_levels: tuple[str, ...] | None = None
    reasoning_modes: tuple[str, ...] | None = None
    regional_processing_uplift_multiplier_eu: Multiplier | None = None
    regional_processing_uplift_multiplier_us: Multiplier | None = None
    supported_endpoints: tuple[str, ...] | None = None
    supported_modalities: tuple[str, ...] | None = None
    supported_output_modalities: tuple[str, ...] | None = None
    supports_audio_input: bool | None = None
    supports_audio_output: bool | None = None
    supports_low_reasoning_effort: bool | None = None
    supports_minimal_reasoning_effort: bool | None = None
    supports_native_streaming: bool | None = None
    supports_none_reasoning_effort: bool | None = None
    supports_parallel_function_calling: bool | None = None
    supports_prompt_cache_breakpoint: bool | None = None
    supports_system_messages: bool | None = None
    supports_tool_search: bool | None = None


class AnthropicProviderSpecificEntry(CostMapConfig):
    fast: float | None = None
    us: float | None = None


class AnthropicModelOnAnthropicProvider(SharedModelConfig):
    model_provider: ClassVar[str] = "anthropic"
    inference_provider: ClassVar[str] = "anthropic"

    cache_creation_input_token_cost_above_100k_tokens: UsdPrice | None = None
    cache_creation_input_token_cost_above_100k_tokens_batches: UsdPrice | None = None
    cache_creation_input_token_cost_above_1hr: UsdPrice | None = None
    cache_creation_input_token_cost_above_1hr_above_100k_tokens: UsdPrice | None = None
    cache_read_input_token_cost_above_100k_tokens: UsdPrice | None = None
    cache_read_input_token_cost_above_100k_tokens_batches: UsdPrice | None = None
    input_cost_per_token_above_100k_tokens: UsdPrice | None = None
    input_cost_per_token_above_100k_tokens_batches: UsdPrice | None = None
    litellm_provider: Literal["anthropic"] | None = None
    output_cost_per_token_above_100k_tokens: UsdPrice | None = None
    output_cost_per_token_above_100k_tokens_batches: UsdPrice | None = None
    prompt_cache_min_tokens: NonNegativeFloat | None = None
    provider_specific_entry: AnthropicProviderSpecificEntry | None = None
    supports_adaptive_thinking: bool | None = None
    supports_anthropic_compaction: bool | None = None
    supports_assistant_prefill: bool | None = None
    supports_fast_mode: bool | None = None
    supports_forced_tool_use: bool | None = None
    supports_legacy_thinking: bool | None = None
    supports_mid_conversation_system: bool | None = None
    supports_native_structured_output: bool | None = None
    supports_output_config: bool | None = None
    supports_sampling_params: bool | None = None
    supports_speed: bool | None = None
    supports_thinking_cache_preservation: bool | None = None
    thinking_always_on: bool | None = None
    thinking_on_by_default: bool | None = None
