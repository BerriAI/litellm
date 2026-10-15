use serde::{Deserialize, Serialize};

/// Whether web search is billed per query or per prompt.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum WebSearchBillingUnit {
    PerQuery,
    PerPrompt,
}

/// UTC "HH:MM-HH:MM" window, or a list of them; a window may wrap past midnight.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum UtcHours {
    Single(String),
    Multiple(Vec<String>),
}

/// ISO-8601 weekday number (1 = Monday .. 7 = Sunday) or English day name.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum Weekday {
    Number(u8),
    Name(String),
}

/// One off-peak window entry inside `windows`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct OffPeakWindow {
    pub hours_utc: UtcHours,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weekdays: Option<Vec<Weekday>>,
}

/// Rates that replace the same-named base fields inside the stated UTC windows.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct OffPeakPricing {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hours_utc: Option<UtcHours>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub windows: Option<Vec<OffPeakWindow>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weekday_timezone: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_reasoning_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost: Option<f64>,
}

/// USD cost per web search query, keyed by search context size.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SearchContextCostPerQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size_low: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size_medium: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size_high: Option<f64>,
}

/// One tier of a context-length or result-count tiered rate.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TieredRate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results_range: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_reasoning_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_token_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_token_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_query: Option<f64>,
}
