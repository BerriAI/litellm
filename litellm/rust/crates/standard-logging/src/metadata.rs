use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ClaudeSubscriptionUsage {
    pub is_overage: bool,
    pub status: Option<String>,
    pub overage_status: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metadata {
    pub spend_logs_metadata: Option<Map<String, Value>>,
    pub requester_metadata: Option<Map<String, Value>>,
    pub requester_custom_headers: Option<BTreeMap<String, String>>,
    pub prompt_management_metadata: Option<Value>,
    pub applied_guardrails: Option<Vec<String>>,
    pub mcp_tool_call_metadata: Option<Value>,
    pub vector_store_request_metadata: Option<Value>,
    pub routing_decision: Option<Value>,
    pub kubernetes_pod_routing: Option<Value>,
    pub usage_object: Option<Map<String, Value>>,
    pub cold_storage_object_key: Option<String>,
    pub team_alias: Option<String>,
    pub team_id: Option<String>,
    pub used_client_oauth_token: Option<bool>,
    pub claude_subscription_usage: Option<ClaudeSubscriptionUsage>,
    /// Whether the call is a silent experiment's mirrored copy. Python logs the key only when it
    /// is true (`SilentExperimentFlag`).
    pub is_silent_experiment: bool,
}
