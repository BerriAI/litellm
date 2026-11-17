from dataclasses import dataclass, replace
from datetime import date
from typing import Annotated, ClassVar, Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from litellm.types.llms.base_model import LiteLLMBaseModel

UsdPrice: TypeAlias = Annotated[float, Field(ge=0)]
Multiplier: TypeAlias = Annotated[float, Field(gt=0)]


class CostMapConfig(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


def _first(preferred: float | None, fallback: float | None) -> float | None:
    return fallback if preferred is None else preferred


def _or(preferred: float | None, fallback: float) -> float:
    return fallback if preferred is None else preferred


@dataclass(frozen=True, slots=True)
class TokenPrices:
    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None
    cache_write_1h: float | None = None
    cache_write_12h: float | None = None
    reasoning: float | None = None


@dataclass(frozen=True, slots=True)
class LongContextPrices:
    above_prompt_tokens: int
    standard: TokenPrices
    service_tier: TokenPrices = TokenPrices()


@dataclass(frozen=True, slots=True)
class TierPrices:
    above_prompt_tokens: float
    up_to_prompt_tokens: float
    prices: TokenPrices


@dataclass(frozen=True, slots=True)
class ModalityPrices:
    input_audio_token: float | None = None
    input_image_token: float | None = None
    input_character: float | None = None
    input_image: float | None = None
    cache_read_audio_token: float | None = None
    cache_read_image_token: float | None = None
    output_audio_token: float | None = None
    output_image_token: float | None = None


@dataclass(frozen=True, slots=True)
class BatchPrices:
    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None
    cache_write_1h: float | None = None
    cache_write_12h: float | None = None
    cache_read_image: float | None = None

    def or_else(self, fallback: "BatchPrices") -> "BatchPrices":
        return BatchPrices(
            input=_first(self.input, fallback.input),
            output=_first(self.output, fallback.output),
            cache_read=_first(self.cache_read, fallback.cache_read),
            cache_write=_first(self.cache_write, fallback.cache_write),
            cache_write_1h=_first(self.cache_write_1h, fallback.cache_write_1h),
            cache_write_12h=_first(self.cache_write_12h, fallback.cache_write_12h),
            cache_read_image=_first(self.cache_read_image, fallback.cache_read_image),
        )


class OffPeakWindow(CostMapConfig):
    hours_utc: str | tuple[str, ...] | None = None
    weekdays: tuple[str | int, ...] | None = None


class OffPeakPricing(CostMapConfig):
    hours_utc: str | tuple[str, ...] | None = None
    windows: tuple[OffPeakWindow, ...] | None = None
    weekday_timezone: str | None = None
    input_cost_per_token: UsdPrice | None = None
    output_cost_per_token: UsdPrice | None = None
    cache_read_input_token_cost: UsdPrice | None = None
    cache_creation_input_token_cost: UsdPrice | None = None
    output_cost_per_reasoning_token: UsdPrice | None = None


class PricingTier(CostMapConfig):
    range: tuple[float, float] | None = None
    input_cost_per_token: UsdPrice | None = None
    output_cost_per_token: UsdPrice | None = None
    cache_read_input_token_cost: UsdPrice | None = None
    cache_creation_input_token_cost: UsdPrice | None = None
    output_cost_per_reasoning_token: UsdPrice | None = None

    def _prices(self) -> TokenPrices:
        return TokenPrices(
            input=self.input_cost_per_token,
            output=self.output_cost_per_token,
            cache_read=self.cache_read_input_token_cost,
            cache_write=self.cache_creation_input_token_cost,
            reasoning=self.output_cost_per_reasoning_token,
        )


@dataclass(frozen=True, slots=True)
class TokenPricing:
    standard: TokenPrices
    service_tier: TokenPrices | None = None
    long_context: tuple[LongContextPrices, ...] = ()
    tiers: tuple[TierPrices, ...] = ()
    off_peak: OffPeakPricing | None = None
    modality: ModalityPrices = ModalityPrices()
    multiplier: float = 1.0


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
    off_peak_pricing: OffPeakPricing | None = None
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


class OpenAIPricingTier(PricingTier):
    cache_creation_input_token_cost_above_12hr: UsdPrice | None = None

    def tier_prices(self) -> TierPrices:
        above, up_to = self.range or (0.0, 0.0)
        return TierPrices(
            above, up_to, replace(self._prices(), cache_write_12h=self.cache_creation_input_token_cost_above_12hr)
        )


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
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_balanced: UsdPrice | None = None
    cache_creation_input_token_cost_above_12hr_balanced: UsdPrice | None = None
    cache_creation_input_token_cost_above_272k_tokens_balanced: UsdPrice | None = None
    cache_creation_input_token_cost_balanced: UsdPrice | None = None
    cache_read_input_token_cost_above_272k_tokens_balanced: UsdPrice | None = None
    cache_read_input_token_cost_balanced: UsdPrice | None = None
    input_cost_per_token_above_272k_tokens_balanced: UsdPrice | None = None
    input_cost_per_token_balanced: UsdPrice | None = None
    output_cost_per_token_above_272k_tokens_balanced: UsdPrice | None = None
    output_cost_per_token_balanced: UsdPrice | None = None
    litellm_provider: Literal["openai"] | None = None
    output_cost_per_reasoning_token: UsdPrice | None = None
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
    tiered_pricing: tuple[OpenAIPricingTier, ...] | None = None

    def _service_tier_prices(self, service_tier: str | None) -> tuple[TokenPrices, TokenPrices] | None:
        match (service_tier or "").lower():
            case "flex":
                return TokenPrices(
                    input=self.input_cost_per_token_flex,
                    output=self.output_cost_per_token_flex,
                    cache_read=self.cache_read_input_token_cost_flex,
                    cache_write=self.cache_creation_input_token_cost_flex,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr_flex,
                ), TokenPrices(
                    input=self.input_cost_per_token_above_272k_tokens_flex,
                    output=self.output_cost_per_token_above_272k_tokens_flex,
                    cache_read=self.cache_read_input_token_cost_above_272k_tokens_flex,
                    cache_write=self.cache_creation_input_token_cost_above_272k_tokens_flex,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr_above_272k_tokens_flex,
                )
            case "priority" | "fast":
                return TokenPrices(
                    input=self.input_cost_per_token_priority,
                    output=self.output_cost_per_token_priority,
                    cache_read=self.cache_read_input_token_cost_priority,
                    cache_write=self.cache_creation_input_token_cost_priority,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr_priority,
                ), TokenPrices(
                    input=self.input_cost_per_token_above_272k_tokens_priority,
                    output=self.output_cost_per_token_above_272k_tokens_priority,
                    cache_read=self.cache_read_input_token_cost_above_272k_tokens_priority,
                    cache_write=self.cache_creation_input_token_cost_above_272k_tokens_priority,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr_above_272k_tokens_priority,
                )
            case "balanced":
                return TokenPrices(
                    input=self.input_cost_per_token_balanced,
                    output=self.output_cost_per_token_balanced,
                    cache_read=self.cache_read_input_token_cost_balanced,
                    cache_write=self.cache_creation_input_token_cost_balanced,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr_balanced,
                ), TokenPrices(
                    input=self.input_cost_per_token_above_272k_tokens_balanced,
                    output=self.output_cost_per_token_above_272k_tokens_balanced,
                    cache_read=self.cache_read_input_token_cost_above_272k_tokens_balanced,
                    cache_write=self.cache_creation_input_token_cost_above_272k_tokens_balanced,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr_above_272k_tokens_balanced,
                )
            case "ultrafast":
                return TokenPrices(
                    input=self.input_cost_per_token_ultrafast,
                    output=self.output_cost_per_token_ultrafast,
                    cache_read=self.cache_read_input_token_cost_ultrafast,
                    cache_write=self.cache_creation_input_token_cost_ultrafast,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr_ultrafast,
                ), TokenPrices(
                    input=self.input_cost_per_token_above_272k_tokens_ultrafast,
                    output=self.output_cost_per_token_above_272k_tokens_ultrafast,
                    cache_read=self.cache_read_input_token_cost_above_272k_tokens_ultrafast,
                    cache_write=self.cache_creation_input_token_cost_above_272k_tokens_ultrafast,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr_above_272k_tokens_ultrafast,
                )
            case _:
                return None

    def _data_residency_multiplier(self, data_residency: str | None) -> float:
        match (data_residency or "").lower():
            case "eu":
                return _or(self.regional_processing_uplift_multiplier_eu, 1.0)
            case "us":
                return _or(self.regional_processing_uplift_multiplier_us, 1.0)
            case "fedramp":
                return _or(self.fedramp_uplift_multiplier, 1.0)
            case _:
                return 1.0

    def token_pricing(self, service_tier: str | None, data_residency: str | None) -> TokenPricing:
        tier: Final = self._service_tier_prices(service_tier)
        return TokenPricing(
            standard=replace(
                TokenPrices(
                    input=self.input_cost_per_token,
                    output=self.output_cost_per_token,
                    cache_read=self.cache_read_input_token_cost,
                    cache_write=self.cache_creation_input_token_cost,
                    cache_write_12h=self.cache_creation_input_token_cost_above_12hr,
                ),
                reasoning=self.output_cost_per_reasoning_token,
            ),
            service_tier=None if tier is None else tier[0],
            long_context=(
                LongContextPrices(
                    above_prompt_tokens=272_000,
                    standard=TokenPrices(
                        input=self.input_cost_per_token_above_272k_tokens,
                        output=self.output_cost_per_token_above_272k_tokens,
                        cache_read=self.cache_read_input_token_cost_above_272k_tokens,
                        cache_write=self.cache_creation_input_token_cost_above_272k_tokens,
                        cache_write_12h=self.cache_creation_input_token_cost_above_12hr_above_272k_tokens,
                    ),
                    service_tier=TokenPrices() if tier is None else tier[1],
                ),
            ),
            tiers=tuple(tier.tier_prices() for tier in self.tiered_pricing or () if tier.range is not None),
            off_peak=self.off_peak_pricing,
            modality=ModalityPrices(
                input_audio_token=self.input_cost_per_audio_token,
                input_image_token=self.input_cost_per_image_token,
                input_character=self.input_cost_per_character,
                input_image=self.input_cost_per_image,
                cache_read_audio_token=self.cache_read_input_audio_token_cost,
                cache_read_image_token=self.cache_read_input_image_token_cost,
                output_audio_token=self.output_cost_per_audio_token,
                output_image_token=self.output_cost_per_image_token,
            ),
            multiplier=self._data_residency_multiplier(data_residency),
        )

    def batch_prices(self, prompt_tokens: int) -> BatchPrices:
        flat: Final = BatchPrices(
            input=self.input_cost_per_token_batches,
            output=self.output_cost_per_token_batches,
            cache_read=self.cache_read_input_token_cost_batches,
            cache_write=self.cache_creation_input_token_cost_batches,
            cache_write_12h=self.cache_creation_input_token_cost_above_12hr_batches,
            cache_read_image=self.cache_read_input_image_token_cost_batches,
        )
        if prompt_tokens <= 272_000:
            return flat
        return BatchPrices(
            input=self.input_cost_per_token_above_272k_tokens_batches,
            output=self.output_cost_per_token_above_272k_tokens_batches,
            cache_read=self.cache_read_input_token_cost_above_272k_tokens_batches,
            cache_write=self.cache_creation_input_token_cost_above_272k_tokens_batches,
            cache_write_12h=self.cache_creation_input_token_cost_above_12hr_above_272k_tokens_batches,
        ).or_else(flat)


class AnthropicProviderSpecificEntry(CostMapConfig):
    fast: float | None = None
    us: float | None = None


class AnthropicPricingTier(PricingTier):
    cache_creation_input_token_cost_above_1hr: UsdPrice | None = None

    def tier_prices(self) -> TierPrices:
        above, up_to = self.range or (0.0, 0.0)
        return TierPrices(
            above, up_to, replace(self._prices(), cache_write_1h=self.cache_creation_input_token_cost_above_1hr)
        )


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
    cache_creation_input_token_cost_above_100k_tokens_priority: UsdPrice | None = None
    cache_creation_input_token_cost_above_1hr_above_100k_tokens_priority: UsdPrice | None = None
    cache_creation_input_token_cost_above_1hr_priority: UsdPrice | None = None
    cache_creation_input_token_cost_priority: UsdPrice | None = None
    cache_read_input_token_cost_above_100k_tokens_priority: UsdPrice | None = None
    cache_read_input_token_cost_priority: UsdPrice | None = None
    input_cost_per_token_above_100k_tokens_priority: UsdPrice | None = None
    input_cost_per_token_priority: UsdPrice | None = None
    output_cost_per_token_above_100k_tokens_priority: UsdPrice | None = None
    output_cost_per_token_priority: UsdPrice | None = None
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
    tiered_pricing: tuple[AnthropicPricingTier, ...] | None = None

    def _geo_multiplier(self, inference_geo: str | None) -> float:
        if self.provider_specific_entry is None or (inference_geo or "").lower() != "us":
            return 1.0
        return _or(self.provider_specific_entry.us, 1.0)

    def _speed_multiplier(self, speed: str | None) -> float:
        if self.provider_specific_entry is None or speed != "fast":
            return 1.0
        return _or(self.provider_specific_entry.fast, 1.0)

    def token_pricing(self, service_tier: str | None, inference_geo: str | None, speed: str | None) -> TokenPricing:
        is_priority: Final = (service_tier or "").lower() in ("priority", "fast")
        return TokenPricing(
            standard=TokenPrices(
                input=self.input_cost_per_token,
                output=self.output_cost_per_token,
                cache_read=self.cache_read_input_token_cost,
                cache_write=self.cache_creation_input_token_cost,
                cache_write_1h=self.cache_creation_input_token_cost_above_1hr,
            ),
            service_tier=TokenPrices(
                input=self.input_cost_per_token_priority,
                output=self.output_cost_per_token_priority,
                cache_read=self.cache_read_input_token_cost_priority,
                cache_write=self.cache_creation_input_token_cost_priority,
                cache_write_1h=self.cache_creation_input_token_cost_above_1hr_priority,
            )
            if is_priority
            else None,
            long_context=(
                LongContextPrices(
                    above_prompt_tokens=100_000,
                    standard=TokenPrices(
                        input=self.input_cost_per_token_above_100k_tokens,
                        output=self.output_cost_per_token_above_100k_tokens,
                        cache_read=self.cache_read_input_token_cost_above_100k_tokens,
                        cache_write=self.cache_creation_input_token_cost_above_100k_tokens,
                        cache_write_1h=self.cache_creation_input_token_cost_above_1hr_above_100k_tokens,
                    ),
                    service_tier=TokenPrices(
                        input=self.input_cost_per_token_above_100k_tokens_priority,
                        output=self.output_cost_per_token_above_100k_tokens_priority,
                        cache_read=self.cache_read_input_token_cost_above_100k_tokens_priority,
                        cache_write=self.cache_creation_input_token_cost_above_100k_tokens_priority,
                        cache_write_1h=self.cache_creation_input_token_cost_above_1hr_above_100k_tokens_priority,
                    ),
                ),
            ),
            tiers=tuple(tier.tier_prices() for tier in self.tiered_pricing or () if tier.range is not None),
            off_peak=self.off_peak_pricing,
            multiplier=self._speed_multiplier(speed) * self._geo_multiplier(inference_geo),
        )

    def batch_prices(self, prompt_tokens: int) -> BatchPrices:
        flat: Final = BatchPrices(
            input=self.input_cost_per_token_batches,
            output=self.output_cost_per_token_batches,
            cache_read=self.cache_read_input_token_cost_batches,
            cache_write=self.cache_creation_input_token_cost_batches,
        )
        if prompt_tokens <= 100_000:
            return flat
        return BatchPrices(
            input=self.input_cost_per_token_above_100k_tokens_batches,
            output=self.output_cost_per_token_above_100k_tokens_batches,
            cache_read=self.cache_read_input_token_cost_above_100k_tokens_batches,
            cache_write=self.cache_creation_input_token_cost_above_100k_tokens_batches,
        ).or_else(flat)
