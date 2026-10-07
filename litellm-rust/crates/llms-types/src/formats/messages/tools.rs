use serde_json::{Map, Value};

use super::CacheControl;
use crate::json_schema::JsonSchema;

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ToolDefinition {
    pub name: Option<String>,

    pub description: Option<String>,

    pub input_schema: Option<JsonSchema>,

    pub strict: Option<bool>,

    pub cache_control: Option<CacheControl>,

    pub defer_loading: Option<bool>,

    pub allowed_callers: Option<Vec<String>>,

    pub input_examples: Option<Vec<Map<String, Value>>>,

    pub eager_input_streaming: Option<bool>,

    pub display_width_px: Option<u64>,

    pub display_height_px: Option<u64>,

    pub display_number: Option<u64>,

    pub max_uses: Option<u64>,

    pub max_tokens: Option<u64>,

    pub allowed_domains: Option<Vec<String>>,

    pub blocked_domains: Option<Vec<String>>,

    pub citations: Option<super::CitationsConfig>,

    pub user_location: Option<WebSearchUserLocation>,

    pub model: Option<String>,

    pub caching: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct WebSearchUserLocation {
    #[serde(rename = "type")]
    pub location_type: UserLocationType,

    pub city: Option<String>,

    pub country: Option<String>,

    pub region: Option<String>,

    pub timezone: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum UserLocationType {
    Approximate,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct ToolChoice {
    #[serde(rename = "type")]
    pub choice_type: ToolChoiceType,

    pub name: Option<String>,

    pub disable_parallel_tool_use: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceType {
    Auto,
    Any,
    Tool,
    None,
}
