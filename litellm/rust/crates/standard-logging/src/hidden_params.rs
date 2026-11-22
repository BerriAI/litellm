use serde_json::{Map, Value};

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
}
