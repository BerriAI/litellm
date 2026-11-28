use serde_json::{Map, Value};

/// The `hidden_params` of `StandardLoggingHiddenParams` in litellm/types/utils.py, less the
/// `cache_key` the payload carries once for both places it is logged.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HiddenParams {
    pub model_id: Option<String>,
    pub api_base: Option<String>,
    pub response_cost: Option<f64>,
    pub additional_headers: Option<Map<String, Value>>,
    pub litellm_overhead_time_ms: Option<f64>,
    pub batch_models: Option<Vec<String>>,
    pub batch_successful_requests: Option<u64>,
    pub batch_failed_requests: Option<u64>,
    pub litellm_model_name: Option<String>,
    pub usage_object: Option<Map<String, Value>>,
    /// The provider a router such as OpenRouter forwarded the call to, as its response named it.
    /// Python logs the key only when the response carried one.
    pub upstream_provider: Option<String>,
}
