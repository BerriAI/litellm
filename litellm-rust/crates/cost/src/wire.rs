//! The only module allowed to handle serde_json::Value: it deserializes the
//! wire contract Python sends and exposes the embedded pricing catalog. The
//! JSON shapes are pinned by the wire_usage and wire_response surfaces of
//! tests/generate_python_fixtures.py.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::Deserialize;
use serde_json::{Map, Value};

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

    fn entry(fields: &[(&str, Value)]) -> RawCatalogEntry {
        RawCatalogEntry(Map::from_iter(
            fields
                .iter()
                .map(|(key, value)| (key.to_string(), value.clone())),
        ))
    }

    #[test]
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

    #[test]
    fn embedded_catalog_excludes_generalizations_meta_key() {
        let catalog = LazyLock::force(&EMBEDDED_CATALOG);
        assert!(!catalog.is_empty());
        assert!(!catalog.contains_key("fallback_generalizations"));
    }

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
    fn query_count_counts_arrays_and_defaults_non_arrays_to_one() {
        let params: OptionalParamsInput =
            serde_json::from_value(serde_json::json!({"query": ["a", "b", "c"]})).unwrap();
        assert_eq!(params.query.map(QueryCount::get), Some(3));
        let single: OptionalParamsInput =
            serde_json::from_value(serde_json::json!({"query": "galaxy"})).unwrap();
        assert_eq!(single.query.map(QueryCount::get), Some(1));
    }
}
