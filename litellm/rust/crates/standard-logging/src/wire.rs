use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    ClaudeSubscriptionUsage, GuardrailStatus, LoggedErrorInformation, ModelInformation, Status,
};

#[derive(Clone, Copy, Serialize)]
pub(crate) struct StatusFields {
    pub(crate) llm_api_status: Status,
    pub(crate) guardrail_status: GuardrailStatus,
}

/// `StandardLoggingMetadata` in the key order Python logs it: the defaults
/// `get_standard_logging_metadata` builds, `is_silent_experiment` when set, the keys it copies from
/// the request that the defaults lack (the app and customer keys, in the order the TypedDict
/// declares them, since Python's own order there follows set iteration), and
/// `claude_subscription_usage`, which `get_standard_logging_object_payload` appends.
#[derive(Serialize)]
pub(crate) struct Metadata {
    pub(crate) user_api_key_hash: Option<String>,
    pub(crate) user_api_key_alias: Option<String>,
    pub(crate) user_api_key_spend: Option<f64>,
    pub(crate) user_api_key_max_budget: Option<f64>,
    pub(crate) user_api_key_budget_reset_at: Option<String>,
    pub(crate) user_api_key_user_spend: Option<f64>,
    pub(crate) user_api_key_user_max_budget: Option<f64>,
    pub(crate) user_api_key_team_spend: Option<f64>,
    pub(crate) user_api_key_team_max_budget: Option<f64>,
    pub(crate) user_api_key_team_id: Option<String>,
    pub(crate) user_api_key_org_id: Option<String>,
    pub(crate) user_api_key_org_alias: Option<String>,
    pub(crate) user_api_key_project_id: Option<String>,
    pub(crate) user_api_key_project_alias: Option<String>,
    pub(crate) user_api_key_user_id: Option<String>,
    pub(crate) user_api_key_team_alias: Option<String>,
    pub(crate) user_api_key_user_email: Option<String>,
    pub(crate) user_api_key_end_user_id: Option<String>,
    pub(crate) user_api_key_customer_member_id: Option<String>,
    pub(crate) user_api_key_customer_team_id: Option<String>,
    pub(crate) user_api_key_customer_org_id: Option<String>,
    pub(crate) user_api_key_request_route: Option<String>,
    pub(crate) spend_logs_metadata: Option<Map<String, Value>>,
    pub(crate) requester_ip_address: Option<String>,
    pub(crate) user_agent: Option<String>,
    pub(crate) requester_metadata: Option<Map<String, Value>>,
    pub(crate) prompt_management_metadata: Option<Value>,
    pub(crate) applied_guardrails: Option<Vec<String>>,
    pub(crate) mcp_tool_call_metadata: Option<Value>,
    pub(crate) vector_store_request_metadata: Option<Value>,
    pub(crate) routing_decision: Option<Value>,
    pub(crate) usage_object: Option<Map<String, Value>>,
    pub(crate) requester_custom_headers: Option<BTreeMap<String, String>>,
    pub(crate) cold_storage_object_key: Option<String>,
    pub(crate) user_api_key_auth_metadata: Option<Map<String, Value>>,
    pub(crate) team_alias: Option<String>,
    pub(crate) team_id: Option<String>,
    pub(crate) kubernetes_pod_routing: Option<Value>,
    pub(crate) used_client_oauth_token: Option<bool>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) is_silent_experiment: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user_api_key_app_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user_api_key_app_group_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user_api_key_app_alias: Option<String>,
    pub(crate) user_api_key_customer_org_alias: Option<String>,
    pub(crate) user_api_key_customer_team_alias: Option<String>,
    pub(crate) user_api_key_customer_member_alias: Option<String>,
    pub(crate) user_api_key_customer_budget_scopes: Option<String>,
    pub(crate) user_api_key_customer_hierarchy: Option<Value>,
    pub(crate) claude_subscription_usage: Option<ClaudeSubscriptionUsage>,
}

#[derive(Serialize)]
pub(crate) struct HiddenParams {
    pub(crate) model_id: Option<String>,
    pub(crate) cache_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) response_cost: Option<f64>,
    pub(crate) additional_headers: Option<Map<String, Value>>,
    pub(crate) litellm_overhead_time_ms: Option<f64>,
    pub(crate) batch_models: Option<Vec<String>>,
    pub(crate) batch_successful_requests: Option<u64>,
    pub(crate) batch_failed_requests: Option<u64>,
    pub(crate) litellm_model_name: Option<String>,
    pub(crate) usage_object: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) upstream_provider: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct Payload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) classifier_input: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) originating_request_masked: Option<Map<String, Value>>,
    pub(crate) id: String,
    pub(crate) litellm_call_id: Option<String>,
    pub(crate) trace_id: String,
    pub(crate) session_id: String,
    pub(crate) call_type: String,
    pub(crate) cache_hit: Option<bool>,
    pub(crate) stream: Option<bool>,
    pub(crate) status: Status,
    pub(crate) status_fields: StatusFields,
    pub(crate) custom_llm_provider: Option<String>,
    pub(crate) saved_cache_cost: f64,
    #[serde(rename = "startTime")]
    pub(crate) start_time: f64,
    #[serde(rename = "endTime")]
    pub(crate) end_time: f64,
    #[serde(rename = "completionStartTime")]
    pub(crate) completion_start_time: f64,
    pub(crate) response_time: f64,
    pub(crate) model: String,
    pub(crate) metadata: Metadata,
    pub(crate) cache_key: Option<String>,
    pub(crate) cost_finalized: bool,
    pub(crate) response_cost: f64,
    pub(crate) cost_breakdown: Option<Value>,
    pub(crate) autorouter_savings: Option<f64>,
    pub(crate) autorouter_savings_estimate: Option<Value>,
    pub(crate) autorouter_baseline_observation: Option<String>,
    pub(crate) total_tokens: u64,
    pub(crate) prompt_tokens: u64,
    pub(crate) completion_tokens: u64,
    pub(crate) request_tags: Vec<Value>,
    pub(crate) request_model_access_groups: Vec<String>,
    pub(crate) end_user: Option<String>,
    pub(crate) api_base: String,
    pub(crate) model_group: Option<String>,
    pub(crate) model_id: Option<String>,
    pub(crate) requester_ip_address: Option<String>,
    pub(crate) user_agent: Option<String>,
    pub(crate) messages: Value,
    pub(crate) response: Value,
    pub(crate) model_parameters: Map<String, Value>,
    pub(crate) hidden_params: HiddenParams,
    pub(crate) model_map_information: ModelInformation,
    pub(crate) error_str: Option<String>,
    pub(crate) error_information: Option<LoggedErrorInformation>,
    pub(crate) response_cost_failure_debug_info: Option<Value>,
    pub(crate) zero_cost_diagnostic: Option<Value>,
    pub(crate) guardrail_information: Option<Value>,
    pub(crate) standard_built_in_tools_params: Option<Value>,
}
