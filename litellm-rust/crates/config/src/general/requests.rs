use litellm_auth_types::SecretValue;
use serde::Deserialize;

use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct RequestsSettings {
    pub forward_llm_provider_auth_headers: Option<Spelled<bool>>,
    pub forward_openai_org_id: Option<Spelled<bool>>,
    pub always_include_stream_usage: Option<Spelled<bool>>,
    pub expose_fallback_errors_to_caller: Option<Spelled<bool>>,
    pub include_call_id_in_error_body: Option<Spelled<bool>>,
    pub request_tag_mappings: Option<Box<[RequestTagMapping]>>,
    pub pass_through_endpoints: Option<Box<[PassThroughEndpoint]>>,
    pub pass_through_request_timeout: Option<Spelled<f64>>,
    pub passthrough_managed_object_ids: Option<Spelled<bool>>,
    pub disable_bedrock_agent_runtime_passthrough: Option<Spelled<bool>>,
    pub enable_openai_websocket_passthrough: Option<Spelled<bool>>,
    pub vertex_ai_pass_through_drop_anthropic_betas: Option<Box<[String]>>,
    pub transcribe_media_buckets: Option<Box<[String]>>,
    pub responses_websocket_session_limit_seconds: Option<Spelled<f64>>,
    pub disable_responses_id_security: Option<Spelled<bool>>,
    pub allow_unmanaged_response_ids: Option<Spelled<bool>>,
    pub dangerously_allow_mock_testing_request_params: Option<Spelled<bool>>,
    pub enable_claude_code_gateway: Option<Spelled<bool>>,
    pub claude_code_gateway_managed_settings: Option<Object>,
    pub context_management_summary_model: Option<String>,
    pub context_management_summary_max_tokens: Option<Spelled<i64>>,
    pub safeguards_classifier_model: Option<String>,
    pub user_url_validation: Option<Spelled<bool>>,
    pub user_url_allowed_hosts: Option<Box<[String]>>,
    pub provider_url_destination_allowed_hosts: Option<Box<[String]>>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::IntoStaticStr)]
#[serde(rename_all = "lowercase")]
pub enum RequestTagHeader {
    #[strum(serialize = "baggage")]
    Baggage,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct RequestTagMapping {
    pub header: Option<RequestTagHeader>,
    pub key: Option<String>,
    pub value: Option<String>,
    pub tag: Option<String>,
    pub tag_prefix: Option<String>,
    pub required: Option<Spelled<bool>>,
    pub require_budgeted_tag: Option<Spelled<bool>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PassThroughEndpoint {
    pub id: Option<String>,
    pub path: Option<String>,
    pub target: Option<SecretValue>,
    pub headers: Option<Object>,
    pub default_query_params: Option<Object>,
    pub include_subpath: Option<Spelled<bool>>,
    pub cost_per_request: Option<Spelled<f64>>,
    pub timeout: Option<Spelled<f64>>,
    pub auth: Option<Spelled<bool>>,
    pub guardrails: Option<Object>,
    pub methods: Option<Box<[String]>>,
}
