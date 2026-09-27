//! The only module allowed to handle serde_json::Value: it deserializes the
//! wire contract Python sends and exposes the embedded pricing catalog. The
//! JSON shapes are pinned by the wire_usage and wire_response surfaces of
//! tests/generate_python_fixtures.py.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use litellm_python_compat::json as python_json;
use litellm_python_compat::repr::to_str;
use litellm_python_compat::truthy::truthy;
use litellm_python_compat::{Value as PythonValue, number, pydantic};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::pricing::{
    ModelPricing, OffPeakPricing, OffPeakWindow, PtuPricing, Rate, SearchContextCostPerQuery,
    ServiceTier, ThresholdPricing, TieredPricingTier, tokenize,
};
use crate::responses_usage::{
    CacheCreationTokenDetails, CompletionTokenDetails, PromptTokenDetails,
};

static EMBEDDED_CATALOG_JSON: &str =
    include_str!("../../../../model_prices_and_context_window.json");

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RawCatalogEntry(Map<String, Value>);

impl RawCatalogEntry {
    pub fn fields(&self) -> &Map<String, Value> {
        &self.0
    }

    fn merge(&mut self, overlay: RawCatalogEntry) {
        for (key, value) in overlay.0 {
            self.0.insert(key, value);
        }
    }
}

pub static EMBEDDED_CATALOG: LazyLock<BTreeMap<String, RawCatalogEntry>> = LazyLock::new(|| {
    let root: Map<String, Value> =
        serde_json::from_str(EMBEDDED_CATALOG_JSON).expect("embedded catalog JSON parses");
    root.into_iter()
        .filter(|(key, _)| key != "fallback_generalizations")
        .filter_map(|(key, value)| match value {
            Value::Object(entry) => Some((key, RawCatalogEntry(entry))),
            _ => None,
        })
        .collect()
});

pub fn overlay_catalog(
    base: &BTreeMap<String, RawCatalogEntry>,
    overlays: impl IntoIterator<Item = (String, RawCatalogEntry)>,
) -> BTreeMap<String, RawCatalogEntry> {
    let mut catalog: BTreeMap<_, _> = base
        .iter()
        .map(|(key, entry)| (key.clone(), entry.clone()))
        .collect();
    for (key, overlay) in overlays {
        match catalog.get_mut(&key) {
            Some(existing) => existing.merge(overlay),
            None => {
                catalog.insert(key, overlay);
            }
        }
    }
    catalog
}

pub fn catalog_with_overlays(
    overlays: impl IntoIterator<Item = (String, RawCatalogEntry)>,
) -> BTreeMap<String, RawCatalogEntry> {
    overlay_catalog(&EMBEDDED_CATALOG, overlays)
}

fn python_value(value: &Value) -> PythonValue {
    python_json::from_json(value.clone())
}

pub fn is_truthy(value: &Value) -> bool {
    truthy(&python_value(value))
}

pub fn py_str(value: &Value) -> String {
    to_str(&python_value(value))
}

pub fn py_int(value: &Value) -> Option<i64> {
    i64::try_from(&number::int(&python_value(value)).ok()?).ok()
}

pub fn py_float(value: &Value) -> Option<f64> {
    number::float(&python_value(value)).ok()
}

pub fn py_real(value: &Value) -> Option<f64> {
    (value.is_number() || value.is_boolean())
        .then(|| py_float(value))
        .flatten()
}

pub fn py_float_unless_bool(value: &Value) -> Option<f64> {
    (!value.is_boolean()).then(|| py_float(value)).flatten()
}

pub fn lax_float(value: &Value) -> Option<f64> {
    pydantic::lax_float(&python_value(value)).ok()
}

pub fn lax_bool(value: &Value) -> Option<bool> {
    pydantic::lax_bool(&python_value(value)).ok()
}

pub fn lax_int(value: &Value) -> Option<i64> {
    i64::try_from(&pydantic::lax_int(&python_value(value)).ok()?).ok()
}

pub fn lax_count(value: &Value) -> Option<u64> {
    lax_int(value).and_then(|count| u64::try_from(count).ok())
}

pub fn deserialize_lax_count<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    lax_count(&value).ok_or_else(|| serde::de::Error::custom("invalid token count"))
}

pub fn deserialize_lax_count_option<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(value) => lax_count(&value)
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("invalid token count")),
    }
}

pub fn rate(value: Option<&Value>) -> Rate {
    match value {
        None => Rate::Missing,
        Some(Value::Null) => Rate::Null,
        Some(Value::Number(number)) => number.as_f64().map_or(Rate::Invalid, Rate::Value),
        Some(Value::Bool(flag)) => Rate::Value(f64::from(*flag)),
        Some(Value::String(text)) => text
            .trim()
            .parse::<f64>()
            .map_or(Rate::Invalid, Rate::Value),
        Some(_) => Rate::Invalid,
    }
}

pub fn cost_per_unit(
    fields: &Map<String, Value>,
    cost_key: &str,
    default: Option<f64>,
) -> Option<f64> {
    match fields.get(cost_key) {
        None | Some(Value::Null) => {}
        Some(present) => return rate(Some(present)).value().or(default),
    }
    for tier in ServiceTier::SUFFIXES {
        let suffix = format!("_{}", tier.as_str());
        if cost_key.contains(&suffix) {
            let base = cost_key.replace(&suffix, "");
            return rate(fields.get(&base)).value().or(default);
        }
    }
    default
}

const NON_RATE_KEYS: [&str; 76] = [
    "audio_transcription_config",
    "bedrock_converse_supports_strict_tools",
    "bedrock_output_config_effort_ceiling",
    "comment",
    "default_reasoning_effort",
    "deprecation_date",
    "gemini_audio_only_live",
    "gemini_native_audio",
    "key",
    "litellm_provider",
    "max_input_tokens",
    "max_output_tokens",
    "max_tokens",
    "metadata",
    "mode",
    "output_vector_size",
    "prompt_cache_min_tokens",
    "reasoning_effort_levels",
    "rpm",
    "source",
    "supported_audio_formats",
    "supported_endpoints",
    "supported_modalities",
    "supported_output_modalities",
    "supported_regions",
    "supported_openai_params",
    "supports_adaptive_thinking",
    "supports_anthropic_compaction",
    "supports_anthropic_thinking_payload",
    "supports_assistant_prefill",
    "supports_audio_input",
    "supports_audio_output",
    "supports_computer_use",
    "supports_embedding_image_input",
    "supports_fast_mode",
    "supports_forced_tool_use",
    "supports_function_calling",
    "supports_image_input",
    "supports_image_size",
    "supports_legacy_thinking",
    "supports_low_reasoning_effort",
    "supports_max_reasoning_effort",
    "supports_mid_conversation_system",
    "supports_minimal_reasoning_effort",
    "supports_multimodal",
    "supports_native_streaming",
    "supports_native_structured_output",
    "supports_none_reasoning_effort",
    "supports_nova_canvas_image_edit",
    "supports_output_config",
    "supports_parallel_function_calling",
    "supports_parallel_tool_use_config",
    "supports_pdf_input",
    "supports_prompt_cache_breakpoint",
    "supports_prompt_caching",
    "supports_reasoning",
    "supports_response_schema",
    "supports_sampling_params",
    "supports_speed",
    "supports_system_messages",
    "supports_thinking_cache_preservation",
    "supports_tool_choice",
    "supports_tool_search",
    "supports_url_context",
    "supports_video_input",
    "supports_vision",
    "supports_web_search",
    "supports_xhigh_reasoning_effort",
    "thinking_always_on",
    "tpm",
    "use_openai_responses_path",
    "uses_embed_content",
    "vertex_ai_audio_api",
    "web_search_billing_unit",
    "ptu_effective_from",
    "ptu_effective_to",
];

const TYPED_KEYS: [&str; 10] = [
    "off_peak_pricing",
    "tiered_pricing",
    "search_context_cost_per_query",
    "provider_specific_entry",
    "guardrail_cost_per_unit",
    "regional_processing_uplift_multiplier_eu",
    "regional_processing_uplift_multiplier_us",
    "regional_endpoint_uplift_multiplier",
    "ptu_count",
    "cost_per_ptu_per_hour",
];

pub fn recognized_non_rate_key(key: &str) -> bool {
    NON_RATE_KEYS.contains(&key) || TYPED_KEYS.contains(&key)
}

fn hours_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(entries)) => entries
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn weekdays(value: Option<&Value>) -> Option<Vec<String>> {
    match value {
        Some(Value::Array(entries)) => Some(
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
        ),
        Some(Value::String(text)) => Some(vec![text.clone()]),
        _ => None,
    }
}

fn parse_off_peak(block: &Value) -> OffPeakPricing {
    OffPeakPricing {
        input: rate(block.get("input_cost_per_token")),
        output: rate(block.get("output_cost_per_token")),
        cache_read: rate(block.get("cache_read_input_token_cost")),
        cache_creation: rate(block.get("cache_creation_input_token_cost")),
        output_reasoning: rate(block.get("output_cost_per_reasoning_token")),
        hours_utc: hours_list(block.get("hours_utc")),
        weekdays: weekdays(block.get("weekdays")),
        weekday_timezone: block
            .get("weekday_timezone")
            .and_then(Value::as_str)
            .map(str::to_owned),
        windows: block
            .get("windows")
            .and_then(Value::as_array)
            .map_or(Vec::new(), |rules| {
                rules
                    .iter()
                    .map(|rule| OffPeakWindow {
                        hours_utc: hours_list(rule.get("hours_utc")),
                        weekdays: weekdays(rule.get("weekdays")),
                    })
                    .collect()
            }),
    }
}

fn parse_tiered_pricing(tiers: &Value) -> Vec<TieredPricingTier> {
    tiers.as_array().map_or(Vec::new(), |entries| {
        entries
            .iter()
            .map(|entry| {
                let range =
                    entry
                        .get("range")
                        .and_then(Value::as_array)
                        .map_or((0.0, None), |bounds| {
                            let start = bounds.first().and_then(Value::as_f64).unwrap_or(0.0);
                            let end = bounds.get(1).and_then(Value::as_f64);
                            (start, end)
                        });
                TieredPricingTier {
                    range,
                    input: rate(entry.get("input_cost_per_token")),
                    output: rate(entry.get("output_cost_per_token")),
                    output_reasoning: rate(entry.get("output_cost_per_reasoning_token")),
                    cache_read: rate(entry.get("cache_read_input_token_cost")),
                    cache_creation: rate(entry.get("cache_creation_input_token_cost")),
                }
            })
            .collect()
    })
}

fn parse_search_context(block: &Value) -> SearchContextCostPerQuery {
    SearchContextCostPerQuery {
        low: rate(block.get("search_context_size_low")),
        medium: rate(block.get("search_context_size_medium")),
        high: rate(block.get("search_context_size_high")),
    }
}

fn parse_rate_map(block: &Value) -> BTreeMap<String, Rate> {
    block.as_object().map_or(BTreeMap::new(), |entries| {
        entries
            .iter()
            .map(|(key, value)| (key.clone(), rate(Some(value))))
            .collect()
    })
}

fn parse_ptu(fields: &Map<String, Value>) -> Option<PtuPricing> {
    fields
        .get("cost_per_ptu_per_hour")
        .map(|cost_per_hour| PtuPricing {
            count: fields.get("ptu_count").and_then(Value::as_i64).unwrap_or(0),
            cost_per_hour: rate(Some(cost_per_hour)),
            effective_from: fields
                .get("ptu_effective_from")
                .and_then(Value::as_str)
                .map(str::to_owned),
            effective_to: fields
                .get("ptu_effective_to")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
}

impl RawCatalogEntry {
    pub fn pricing(&self) -> ModelPricing {
        let fields = self.fields();
        let mut pricing = ModelPricing {
            web_search_billing_unit: fields
                .get("web_search_billing_unit")
                .and_then(Value::as_str)
                .map(str::to_owned),
            regional_processing_eu: rate(fields.get("regional_processing_uplift_multiplier_eu")),
            regional_processing_us: rate(fields.get("regional_processing_uplift_multiplier_us")),
            regional_endpoint_uplift: rate(fields.get("regional_endpoint_uplift_multiplier")),
            litellm_provider: fields
                .get("litellm_provider")
                .and_then(Value::as_str)
                .map(str::to_owned),
            off_peak: fields.get("off_peak_pricing").map(parse_off_peak),
            tiered_pricing: fields.get("tiered_pricing").map(parse_tiered_pricing),
            search_context_cost_per_query: fields
                .get("search_context_cost_per_query")
                .map(parse_search_context),
            provider_specific_entry: fields
                .get("provider_specific_entry")
                .map(parse_rate_map)
                .unwrap_or_default(),
            guardrail_cost_per_unit: fields
                .get("guardrail_cost_per_unit")
                .map(parse_rate_map)
                .unwrap_or_default(),
            ptu: parse_ptu(fields),
            ..ModelPricing::default()
        };
        let mut thresholds: BTreeMap<u64, ThresholdPricing> = BTreeMap::new();
        for (key, value) in fields {
            if NON_RATE_KEYS.contains(&key.as_str()) || TYPED_KEYS.contains(&key.as_str()) {
                continue;
            }
            let Ok(rate_key) = tokenize(key) else {
                pricing.extra.insert(key.clone(), rate(Some(value)));
                continue;
            };
            let parsed_rate = rate(Some(value));
            match (rate_key.threshold, rate_key.tier, rate_key.batch) {
                (None, None, false) => {
                    pricing.base.insert(rate_key.metric, parsed_rate);
                }
                (None, Some(tier), false) => {
                    pricing.tiers.insert((rate_key.metric, tier), parsed_rate);
                }
                (None, None, true) => {
                    pricing.batches.insert(rate_key.metric, parsed_rate);
                }
                (None, Some(tier), true) => {
                    pricing
                        .batch_tiers
                        .insert((rate_key.metric, tier), parsed_rate);
                }
                (Some(above_tokens), tier, batch) => {
                    let entry =
                        thresholds
                            .entry(above_tokens)
                            .or_insert_with(|| ThresholdPricing {
                                above_tokens,
                                standard: BTreeMap::new(),
                                tiers: BTreeMap::new(),
                                batches: BTreeMap::new(),
                                batch_tiers: BTreeMap::new(),
                            });
                    match (tier, batch) {
                        (None, false) => {
                            entry.standard.insert(rate_key.metric, parsed_rate);
                        }
                        (Some(tier), false) => {
                            entry.tiers.insert((rate_key.metric, tier), parsed_rate);
                        }
                        (None, true) => {
                            entry.batches.insert(rate_key.metric, parsed_rate);
                        }
                        (Some(tier), true) => {
                            entry
                                .batch_tiers
                                .insert((rate_key.metric, tier), parsed_rate);
                        }
                    }
                }
            }
        }
        pricing.thresholds = thresholds.into_values().collect();
        pricing
    }
}

fn count_value(value: &Value) -> Option<u64> {
    match value {
        Value::Null => Some(0),
        Value::Number(number) => number.as_u64().or_else(|| {
            number
                .as_f64()
                .filter(|value| value.is_finite() && *value >= 0.0 && *value < u64::MAX as f64)
                .map(|value| value as u64)
        }),
        Value::Bool(flag) => Some(u64::from(*flag)),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

fn lenient_count<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    count_value(&value).ok_or_else(|| serde::de::Error::custom("invalid token count"))
}

fn lenient_count_option<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => count_value(&value)
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("invalid token count")),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum WireNumber {
    Number(f64),
    Text(String),
    Flag(bool),
}

impl WireNumber {
    pub fn cost(self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(value),
            Self::Text(text) => text.trim().parse().ok(),
            Self::Flag(flag) => Some(f64::from(flag)),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ReportedCostInput {
    Number(f64),
    Detailed { total_cost: Option<f64> },
}

impl ReportedCostInput {
    pub fn cost(self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(value),
            Self::Detailed { total_cost } => total_cost,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ServerToolUse {
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub web_search_requests: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub tool_search_requests: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub browser_open_requests: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ServerSideToolUsageDetails {
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub web_search_calls: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct ChatUsageInput {
    #[serde(default, deserialize_with = "lenient_count")]
    pub prompt_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub completion_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_tokens: u64,
    pub cost: Option<ReportedCostInput>,
    pub prompt_tokens_details: Option<PromptTokenDetails>,
    pub completion_tokens_details: Option<CompletionTokenDetails>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub cache_creation_input_tokens: Option<u64>,
    pub server_tool_use: Option<ServerToolUse>,
    pub speed: Option<String>,
    pub inference_geo: Option<String>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub _cache_read_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub _cache_creation_input_tokens: Option<u64>,
    pub server_side_tool_usage_details: Option<ServerSideToolUsageDetails>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub citation_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ResponsesUsageInput {
    #[serde(default, deserialize_with = "lenient_count")]
    pub input_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub output_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: Option<PromptTokenDetails>,
    #[serde(default)]
    pub input_token_details: Option<PromptTokenDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<CompletionTokenDetails>,
    #[serde(default)]
    pub output_token_details: Option<CompletionTokenDetails>,
    pub cost: Option<ReportedCostInput>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ThinkingTokenDetails {
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub thinking_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicIteration {
    #[serde(default, deserialize_with = "lenient_count")]
    pub input_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub output_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_creation: Option<CacheCreationTokenDetails>,
    pub output_tokens_details: Option<ThinkingTokenDetails>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicUsageInput {
    #[serde(default, deserialize_with = "lenient_count")]
    pub input_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub output_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_creation: Option<CacheCreationTokenDetails>,
    pub output_tokens_details: Option<ThinkingTokenDetails>,
    pub iterations: Option<Vec<AnthropicIteration>>,
    pub server_tool_use: Option<ServerToolUse>,
    pub speed: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ModalityTokens {
    #[serde(default, deserialize_with = "lenient_count")]
    pub tokens: u64,
    pub modality: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GroundingToolCount {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default, deserialize_with = "lenient_count")]
    pub count: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct InteractionsUsageInput {
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_input_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_output_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_tool_use_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_cached_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_reasoning_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_thought_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_tokens: u64,
    #[serde(default)]
    pub input_tokens_by_modality: Vec<ModalityTokens>,
    #[serde(default)]
    pub tool_use_tokens_by_modality: Vec<ModalityTokens>,
    #[serde(default)]
    pub cached_tokens_by_modality: Vec<ModalityTokens>,
    #[serde(default)]
    pub output_tokens_by_modality: Vec<ModalityTokens>,
    #[serde(default)]
    pub grounding_tool_count: Vec<GroundingToolCount>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct TranscriptionTokenDetails {
    #[serde(default, deserialize_with = "lenient_count")]
    pub text_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub audio_tokens: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptionUsageInput {
    Duration {
        seconds: f64,
    },
    Tokens {
        #[serde(default, deserialize_with = "lenient_count")]
        input_tokens: u64,
        #[serde(default, deserialize_with = "lenient_count")]
        output_tokens: u64,
        #[serde(default, deserialize_with = "lenient_count")]
        total_tokens: u64,
        #[serde(default)]
        input_token_details: TranscriptionTokenDetails,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "format", rename_all = "snake_case")]
pub enum UsageInput {
    Chat {
        #[serde(flatten)]
        usage: ChatUsageInput,
    },
    Responses {
        #[serde(flatten)]
        usage: ResponsesUsageInput,
    },
    Anthropic {
        #[serde(flatten)]
        usage: AnthropicUsageInput,
    },
    Interactions {
        #[serde(flatten)]
        usage: InteractionsUsageInput,
    },
    Transcription {
        #[serde(flatten)]
        usage: TranscriptionUsageInput,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BilledUnits {
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub total_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub search_units: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ResponseMeta {
    pub billed_units: Option<BilledUnits>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RealtimeResponse {
    pub usage: Option<UsageInput>,
    pub service_tier: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RealtimeResultInput {
    pub response: Option<RealtimeResponse>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ResponseInput {
    pub model: Option<String>,
    pub usage: Option<UsageInput>,
    pub created: Option<f64>,
    pub ended: Option<f64>,
    #[serde(rename = "_response_ms")]
    pub response_ms: Option<f64>,
    pub provider_reported_cost_usd: Option<f64>,
    pub duration: Option<f64>,
    pub meta: Option<ResponseMeta>,
    #[serde(default)]
    pub results: Vec<RealtimeResultInput>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ProviderSpecificFields {
    pub traffic_type: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct HiddenParamsInput {
    pub custom_llm_provider: Option<String>,
    pub region_name: Option<String>,
    pub model: Option<String>,
    pub litellm_model_name: Option<String>,
    pub additional_headers: Option<BTreeMap<String, WireNumber>>,
    pub audio_transcription_duration: Option<f64>,
    pub provider_specific_fields: Option<ProviderSpecificFields>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryCount(pub u64);

impl QueryCount {
    pub fn get(self) -> u64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for QueryCount {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let count = value.as_array().map_or(1, Vec::len) as u64;
        Ok(Self(count))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OptionalParamsInput {
    pub service_tier: Option<String>,
    pub query: Option<QueryCount>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ImageInputTokenDetails {
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub text_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub image_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ImageOutputTokenDetails {
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub text_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub image_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub audio_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_count_option")]
    pub reasoning_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ImageUsageInput {
    #[serde(default, deserialize_with = "lenient_count")]
    pub input_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub output_tokens: u64,
    #[serde(default, deserialize_with = "lenient_count")]
    pub total_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: Option<ImageInputTokenDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<ImageOutputTokenDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<ImageOutputTokenDetails>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DataLen(pub u64);

impl DataLen {
    pub fn get(self) -> u64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for DataLen {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let count = value.as_array().map_or(0, Vec::len) as u64;
        Ok(Self(count))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ImageResponseInput {
    pub usage: Option<ImageUsageInput>,
    #[serde(default)]
    pub data: DataLen,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn entry(fields: &[(&str, Value)]) -> RawCatalogEntry {
        RawCatalogEntry(Map::from_iter(
            fields
                .iter()
                .map(|(key, value)| (key.to_string(), value.clone())),
        ))
    }

    #[rstest]
    #[case::integral_json_float(serde_json::json!(2.0), Some(2), Some(2), Some(2.0), true)]
    #[case::json_integer_beyond_i64(serde_json::json!(u64::MAX), None, None, Some(u64::MAX as f64), true)]
    #[case::json_string(serde_json::json!(" 3.0 "), None, Some(3), Some(3.0), true)]
    #[case::json_null(serde_json::json!(null), None, None, None, false)]
    fn python_adapters_read_json_as_json_loads_would(
        #[case] value: Value,
        #[case] int: Option<i64>,
        #[case] lax: Option<i64>,
        #[case] float: Option<f64>,
        #[case] truthy: bool,
    ) {
        assert_eq!(py_int(&value), int);
        assert_eq!(lax_int(&value), lax);
        assert_eq!(py_float(&value), float);
        assert_eq!(is_truthy(&value), truthy);
    }

    #[rstest]
    fn overlay_merges_fields_into_existing_entries_and_adds_new_ones() {
        let base = BTreeMap::from([(
            "synthetic-base".to_string(),
            entry(&[
                ("input_cost_per_token", serde_json::json!(3e-6)),
                ("output_cost_per_token", serde_json::json!(15e-6)),
            ]),
        )]);
        let overlaid = overlay_catalog(
            &base,
            [
                (
                    "synthetic-base".to_string(),
                    entry(&[("output_cost_per_token", serde_json::json!(9e-6))]),
                ),
                (
                    "synthetic-new".to_string(),
                    entry(&[("input_cost_per_token", serde_json::json!(1e-6))]),
                ),
            ],
        );
        assert_eq!(overlaid.len(), 2);
        let merged = &overlaid["synthetic-base"];
        assert_eq!(
            merged.fields().get("input_cost_per_token"),
            Some(&serde_json::json!(3e-6))
        );
        assert_eq!(
            merged.fields().get("output_cost_per_token"),
            Some(&serde_json::json!(9e-6))
        );
        assert!(
            overlaid["synthetic-new"]
                .fields()
                .contains_key("input_cost_per_token")
        );
    }

    #[rstest]
    fn embedded_catalog_excludes_generalizations_meta_key() {
        let catalog = LazyLock::force(&EMBEDDED_CATALOG);
        assert!(!catalog.is_empty());
        assert!(!catalog.contains_key("fallback_generalizations"));
    }

    #[rstest]
    fn chat_usage_reads_lenient_counts_and_cost_projection() {
        let usage: ChatUsageInput = serde_json::from_value(serde_json::json!({
            "prompt_tokens": true,
            "completion_tokens": " 42 ",
            "total_tokens": 58.0,
            "cost": {"total_cost": 0.5},
            "cache_read_input_tokens": 7,
        }))
        .unwrap();
        assert_eq!(usage.prompt_tokens, 1);
        assert_eq!(usage.completion_tokens, 42);
        assert_eq!(usage.total_tokens, 58);
        assert_eq!(usage.cost.map(ReportedCostInput::cost), Some(Some(0.5)));
        assert_eq!(usage.cache_read_input_tokens, Some(7));
    }

    #[rstest]
    fn chat_usage_rejects_negative_and_garbage_counts() {
        assert!(
            serde_json::from_value::<ChatUsageInput>(serde_json::json!({
                "prompt_tokens": -1
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<ChatUsageInput>(serde_json::json!({
                "prompt_tokens": "not a number"
            }))
            .is_err()
        );
    }

    #[rstest]
    fn usage_input_dispatches_on_format_tag_not_key_shapes() {
        let chat: UsageInput = serde_json::from_value(serde_json::json!({
            "format": "chat",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "input_tokens": 999,
            "output_tokens": 99,
        }))
        .unwrap();
        let UsageInput::Chat { usage } = chat else {
            panic!("chat tag must select the chat payload");
        };
        assert_eq!(usage.prompt_tokens, 10);
        let transcription: UsageInput = serde_json::from_value(serde_json::json!({
            "format": "transcription",
            "type": "tokens",
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "input_token_details": {"text_tokens": 60, "audio_tokens": 40},
        }))
        .unwrap();
        let UsageInput::Transcription { usage } = transcription else {
            panic!("transcription tag must select the transcription payload");
        };
        let TranscriptionUsageInput::Tokens {
            input_tokens,
            output_tokens,
            ..
        } = usage
        else {
            panic!("transcription type tag must select the tokens variant");
        };
        assert_eq!((input_tokens, output_tokens), (100, 200));
    }

    #[rstest]
    fn hidden_params_read_wire_number_header_costs() {
        let hidden: HiddenParamsInput = serde_json::from_value(serde_json::json!({
            "custom_llm_provider": "openai",
            "additional_headers": {"llm_provider-x-litellm-response-cost": " 0.0123 "},
        }))
        .unwrap();
        let header = hidden
            .additional_headers
            .as_ref()
            .and_then(|headers| headers.get("llm_provider-x-litellm-response-cost"))
            .cloned();
        assert_eq!(header.and_then(WireNumber::cost), Some(0.0123));
    }

    #[rstest]
    fn query_count_counts_arrays_and_defaults_non_arrays_to_one() {
        let params: OptionalParamsInput =
            serde_json::from_value(serde_json::json!({"query": ["a", "b", "c"]})).unwrap();
        assert_eq!(params.query.map(QueryCount::get), Some(3));
        let single: OptionalParamsInput =
            serde_json::from_value(serde_json::json!({"query": "galaxy"})).unwrap();
        assert_eq!(single.query.map(QueryCount::get), Some(1));
    }
}
