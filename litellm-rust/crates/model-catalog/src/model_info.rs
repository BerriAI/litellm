use crate::capabilities::{AudioFormat, InputModality, Mode, OutputModality, VertexAiAudioApi};
use crate::pricing::{OffPeakPricing, SearchContextCostPerQuery, TieredRate, WebSearchBillingUnit};
use litellm_types::llms::openai::ReasoningEffort;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Typed mirror of one catalog model entry.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotation_cost_per_page: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotation_cost_per_page_batches: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_transcription_config: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bedrock_converse_supports_strict_tools: Option<bool>,
    /// Highest reasoning effort the Bedrock output_config accepts for this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bedrock_output_config_effort_ceiling: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_audio_token_cost: Option<f64>,
    /// USD per token written to the provider's prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_32k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_128k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_1hr: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_1hr_above_200k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_200k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_200k_tokens_batches: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_256k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_272k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_272k_tokens_batches: Option<f64>,
    /// Flex service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_272k_tokens_flex: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_above_272k_tokens_priority: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_batches: Option<f64>,
    /// Flex service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_flex: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost_priority: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_audio_token_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_image_token_cost: Option<f64>,
    /// USD per prompt token served from the provider's prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_32k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_128k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_200k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_200k_tokens_batches: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_200k_tokens_priority: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_256k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_272k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_272k_tokens_batches: Option<f64>,
    /// Flex service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_272k_tokens_flex: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_272k_tokens_priority: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_above_512k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_batches: Option<f64>,
    /// Flex service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_flex: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost_priority: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_cost_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_interpreter_cost_per_session: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub computer_use_input_cost_per_1k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub computer_use_output_cost_per_1k_tokens: Option<f64>,
    /// Reasoning effort the provider applies when the request omits reasoning_effort. Gates whether a non-default temperature or the top_p/logprobs sampling params are accepted, which hold only when the effort resolves to 'none'.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_reasoning_effort: Option<ReasoningEffort>,
    /// Date the provider deprecates the model, YYYY-MM-DD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecation_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_search_cost_per_1k_calls: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_search_cost_per_gb_per_day: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gemini_audio_only_live: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gemini_native_audio: Option<bool>,
    /// USD per Grounding with Google Maps request; billed per query or per prompt per web_search_billing_unit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_maps_grounding_cost_per_query: Option<f64>,
    /// USD cost per billable guardrail unit, keyed by the provider's usage counter name (e.g. Bedrock's contentPolicyUnits).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardrail_cost_per_unit: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_audio_per_second: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_audio_per_second_above_128k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_audio_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_audio_token_batches: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_audio_token_priority: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_character: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_character_above_128k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_image: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_image_above_128k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_image_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_image_token_batches: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_pixel: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_query: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_request: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_second: Option<f64>,
    /// USD per prompt token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_32k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_128k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_200k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_200k_tokens_batches: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_200k_tokens_priority: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_256k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_272k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_272k_tokens_batches: Option<f64>,
    /// Flex service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_272k_tokens_flex: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_272k_tokens_priority: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_above_512k_tokens: Option<f64>,
    /// USD per prompt token via the provider's batch API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_batches: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_cache_hit: Option<f64>,
    /// Flex service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_flex: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token_priority: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_video_per_second: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_video_per_second_above_128k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_video_per_second_above_15s_interval: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_video_per_second_above_8s_interval: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_video_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_video_token_batches: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_dbu_cost_per_token: Option<f64>,
    /// LiteLLM provider slug; one of https://docs.litellm.ai/docs/providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub litellm_provider: Option<String>,
    /// Maximum prompt/context tokens the model accepts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u64>,
    /// Maximum tokens the model can generate in one response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Legacy field: max output tokens if the provider specifies it, else max input tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Free-form notes about the entry (e.g. pricing derivation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Value>>,
    /// Primary API surface / task type of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<Mode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_cost_per_credit: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_cost_per_page: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_cost_per_page_batches: Option<f64>,
    /// Rates that replace the same-named base fields while the request falls inside the stated UTC windows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub off_peak_pricing: Option<OffPeakPricing>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_audio_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_character: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_character_above_128k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_image: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_image_1024: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_image_1536: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_image_512: Option<f64>,
    #[serde(
        rename = "output_cost_per_image_0.5K",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_cost_per_image_0_5k: Option<f64>,
    #[serde(
        rename = "output_cost_per_image_1K",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_cost_per_image_1k: Option<f64>,
    #[serde(
        rename = "output_cost_per_image_2K",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_cost_per_image_2k: Option<f64>,
    #[serde(
        rename = "output_cost_per_image_4K",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_cost_per_image_4k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_image_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_pixel: Option<f64>,
    /// USD per reasoning/thinking token, when billed separately.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_reasoning_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_second_1080p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_second_2k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_second_480p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_second_4k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_second_720p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_second_768p: Option<f64>,
    /// USD per generated token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_32k_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_128k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_200k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_200k_tokens_batches: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_200k_tokens_priority: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_256k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_272k_tokens: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_272k_tokens_batches: Option<f64>,
    /// Flex service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_272k_tokens_flex: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_272k_tokens_priority: Option<f64>,
    /// Rate applied once the prompt exceeds the token threshold in the field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_above_512k_tokens: Option<f64>,
    /// USD per generated token via the provider's batch API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_batches: Option<f64>,
    /// Flex service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_flex: Option<f64>,
    /// Priority service-tier rate for the same-named base field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token_priority: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_video_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_video_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dbu_cost_per_token: Option<f64>,
    /// Embedding dimension for embedding models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_vector_size: Option<u64>,
    /// Smallest prefix the provider will actually cache; absent means the provider default applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_min_tokens: Option<u64>,
    /// Provider-internal routing hints (e.g. bedrock_invocation_schema).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_specific_entry: Option<BTreeMap<String, Value>>,
    /// Exact reasoning_effort levels this deployment accepts; wins over supports_* flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort_levels: Option<Vec<ReasoningEffort>>,
    /// Multiplier applied to all token costs when served from a non-global Vertex AI endpoint (e.g. 1.10 = +10%).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regional_endpoint_uplift_multiplier: Option<f64>,
    /// Multiplier applied to all token costs for EU data residency (e.g. 1.10 = +10%).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regional_processing_uplift_multiplier_eu: Option<f64>,
    /// Multiplier applied to all token costs for US data residency (e.g. 1.10 = +10%).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regional_processing_uplift_multiplier_us: Option<f64>,
    /// Provider default requests-per-minute limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rules: Option<Vec<Value>>,
    /// USD cost per web search query, keyed by search context size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_cost_per_query: Option<SearchContextCostPerQuery>,
    /// URL of the provider pricing/model page this entry was taken from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Audio container formats the model can return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_audio_formats: Option<Vec<AudioFormat>>,
    /// OpenAI-style API routes this model can be called through, e.g. /v1/chat/completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_endpoints: Option<Vec<String>>,
    /// Input modalities the model accepts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_modalities: Option<Vec<InputModality>>,
    /// Output modalities the model can produce.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_output_modalities: Option<Vec<OutputModality>>,
    /// Cloud regions the model is available in ('global' or region ids).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_regions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_adaptive_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_anthropic_compaction: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_anthropic_thinking_payload: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_assistant_prefill: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_audio_input: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_audio_output: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_bedrock_runtime_chat_completions_response_format: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_bedrock_runtime_chat_completions_tools_with_reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_computer_use: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_embedding_image_input: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_fast_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_forced_tool_use: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_function_calling: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_image_input: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_image_size: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_legacy_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_low_reasoning_effort: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_max_reasoning_effort: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_mid_conversation_system: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_minimal_reasoning_effort: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_multimodal: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_native_streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_native_structured_output: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_none_reasoning_effort: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_nova_canvas_image_edit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_output_config: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_parallel_function_calling: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_parallel_tool_use_config: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_pdf_input: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_prompt_cache_breakpoint: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_prompt_caching: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_response_schema: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_sampling_params: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_speed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_system_messages: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_thinking_cache_preservation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_choice: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_url_context: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_video_input: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_vision: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_web_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_xhigh_reasoning_effort: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_always_on: Option<bool>,
    /// Context-length or result-count tiered rates; each tier's costs apply within its range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tiered_pricing: Option<Vec<TieredRate>>,
    /// Provider default tokens-per-minute limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_openai_responses_path: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uses_embed_content: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_store_cost_per_gb_per_day: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertex_ai_audio_api: Option<VertexAiAudioApi>,
    /// Whether web search is billed per query or per prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_billing_unit: Option<WebSearchBillingUnit>,
}
