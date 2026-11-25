use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    Caller, GuardrailStatus, HiddenParams, LoggedErrorInformation, Metadata, Status, wire,
};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInformation {
    pub model_map_key: String,
    pub model_map_value: Option<Value>,
}

/// The `ClassifierAudit` keys of litellm/types/utils.py, which `get_standard_logging_object_payload`
/// logs only for an autorouter classifier call whose messages are not redacted, and each only when
/// it has a value.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClassifierAudit {
    pub classifier_input: Option<Map<String, Value>>,
    pub originating_request_masked: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(into = "wire::Payload")]
pub struct StandardLoggingPayload {
    pub classifier_audit: ClassifierAudit,
    pub id: String,
    pub litellm_call_id: Option<String>,
    pub trace_id: String,
    pub session_id: String,
    pub call_type: String,
    pub cache_hit: Option<bool>,
    pub cache_key: Option<String>,
    pub stream: Option<bool>,
    pub status: Status,
    pub guardrail_status: GuardrailStatus,
    pub custom_llm_provider: Option<String>,
    pub saved_cache_cost: f64,
    pub start_time: f64,
    pub end_time: f64,
    pub completion_start_time: f64,
    pub response_time: f64,
    pub model: String,
    pub caller: Caller,
    pub requester_ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub metadata: Metadata,
    pub cost_finalized: bool,
    pub response_cost: f64,
    pub cost_breakdown: Option<Value>,
    pub autorouter_savings: Option<f64>,
    pub autorouter_savings_estimate: Option<Value>,
    pub autorouter_baseline_observation: Option<String>,
    pub total_tokens: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub request_tags: Vec<Value>,
    pub request_model_access_groups: Vec<String>,
    pub end_user: Option<String>,
    pub api_base: String,
    pub model_group: Option<String>,
    pub model_id: Option<String>,
    pub messages: Value,
    pub response: Value,
    pub model_parameters: Map<String, Value>,
    pub hidden_params: HiddenParams,
    pub model_map_information: ModelInformation,
    pub error_str: Option<String>,
    pub error_information: Option<LoggedErrorInformation>,
    pub response_cost_failure_debug_info: Option<Value>,
    pub zero_cost_diagnostic: Option<Value>,
    pub guardrail_information: Option<Value>,
    pub standard_built_in_tools_params: Option<Value>,
}
