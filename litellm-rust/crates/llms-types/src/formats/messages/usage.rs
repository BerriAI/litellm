use serde_json::{Map, Value};

use crate::recognized::Recognized;

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ServerToolUsage {
    pub web_search_requests: Option<u64>,
    pub web_fetch_requests: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct UsageIteration {
    #[serde(rename = "type")]
    pub iteration_type: Recognized<UsageIterationType>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation: Option<CacheCreationUsage>,
    pub model: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum UsageIterationType {
    Compaction,
    Message,
    AdvisorMessage,
    FallbackMessage,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub server_tool_use: Option<ServerToolUsage>,
    pub cache_creation: Option<CacheCreationUsage>,
    pub output_tokens_details: Option<MessagesOutputTokensDetails>,
    pub service_tier: Option<String>,
    pub inference_geo: Option<String>,
    pub speed: Option<super::Speed>,
    pub iterations: Option<Vec<UsageIteration>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct CacheCreationUsage {
    pub ephemeral_1h_input_tokens: Option<u64>,
    pub ephemeral_5m_input_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesOutputTokensDetails {
    pub thinking_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
