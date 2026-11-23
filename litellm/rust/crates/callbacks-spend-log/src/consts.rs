//! The field and key tables the row policy copies from Python, in one place.
//!
//! Each table mirrors a named Python constant. The recorder (`scripts/record_goldens.py`) writes
//! those constants to `tests/goldens/constants.json`, and the test at the bottom of this file
//! fails when a Python table changes without its Rust copy: a field added to `SpendLogsMetadata`,
//! a pattern added to `SensitiveDataMasker`, a header added to the client IP list.

/// `SpendLogsMetadata.__annotations__`, in order: the keys of the stored metadata JSON.
pub(crate) const SPEND_LOGS_METADATA_FIELDS: [&str; 66] = [
    "user_api_key_customer_org_id",
    "user_api_key_app_id",
    "user_api_key_app_group_id",
    "user_api_key_app_alias",
    "user_api_key_customer_org_alias",
    "user_api_key_customer_team_id",
    "user_api_key_customer_team_alias",
    "user_api_key_customer_member_id",
    "user_api_key_customer_member_alias",
    "user_api_key_customer_usage",
    "actor_agent_id",
    "target_agent_id",
    "billing_agent_id",
    "agent_execution_mode",
    "verified_human_user_id",
    "autorouter_baseline_observation",
    "additional_usage_values",
    "user_api_key",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_project_id",
    "user_api_key_project_alias",
    "user_api_key_org_id",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_team_alias",
    "spend_logs_metadata",
    "requester_ip_address",
    "user_agent",
    "litellm_call_id",
    "applied_guardrails",
    "mcp_tool_call_metadata",
    "vector_store_request_metadata",
    "routing_decision",
    "kubernetes_pod_routing",
    "internal_call_origin",
    "billed_retry_attempt",
    "route_call_type",
    "upstream_provider",
    "litellm_roi_estimator",
    "is_silent_experiment",
    "guardrail_information",
    "eval_information",
    "status",
    "proxy_server_request",
    "batch_models",
    "batch_successful_requests",
    "batch_failed_requests",
    "error_information",
    "usage_object",
    "model_map_information",
    "cold_storage_object_key",
    "litellm_overhead_time_ms",
    "attempted_retries",
    "max_retries",
    "attempted_fallbacks",
    "original_model_group",
    "cost_breakdown",
    "compression_savings",
    "autorouter_savings",
    "autorouter_savings_estimate",
    "litellm_gateway_injected_cache",
    "router_metadata",
    "azure_spillover",
    "used_client_oauth_token",
    "claude_subscription_usage",
];

/// `_STAMPED_METADATA_KEYS`: metadata keys the row stamps itself instead of copying.
pub(crate) const STAMPED_METADATA_KEYS: [&str; 23] = [
    "router_metadata",
    "azure_spillover",
    "autorouter_savings",
    "autorouter_savings_estimate",
    "autorouter_baseline_observation",
    "used_client_oauth_token",
    "claude_subscription_usage",
    "upstream_provider",
    "kubernetes_pod_routing",
    "litellm_roi_estimator",
    "is_silent_experiment",
    "billed_retry_attempt",
    "route_call_type",
    "user_api_key_user_email",
    "user_api_key_customer_org_id",
    "user_api_key_app_id",
    "user_api_key_app_group_id",
    "user_api_key_app_alias",
    "user_api_key_customer_org_alias",
    "user_api_key_customer_team_id",
    "user_api_key_customer_team_alias",
    "user_api_key_customer_member_id",
    "user_api_key_customer_member_alias",
];

/// `_CustomerIdentityMetadata.__annotations__`.
pub(crate) const CUSTOMER_IDENTITY_FIELDS: [&str; 9] = [
    "user_api_key_app_id",
    "user_api_key_app_group_id",
    "user_api_key_app_alias",
    "user_api_key_customer_org_id",
    "user_api_key_customer_org_alias",
    "user_api_key_customer_team_id",
    "user_api_key_customer_team_alias",
    "user_api_key_customer_member_id",
    "user_api_key_customer_member_alias",
];

/// `_NON_SECRET_KEY_ALIASES`.
pub(crate) const NON_SECRET_KEY_ALIASES: [&str; 3] = [
    "litellm_proxy_master_key",
    "litellm-internal-health-check",
    "litellm-cli",
];

/// `_BATCH_TRACE_CALL_TYPES`.
pub(crate) const BATCH_TRACE_CALL_TYPES: [&str; 4] = [
    "create_batch",
    "acreate_batch",
    "retrieve_batch",
    "aretrieve_batch",
];

/// `BATCH_COST_REQUEST_ID_SUFFIX` from `litellm_core_utils/batch_utils.py`.
pub(crate) const BATCH_COST_REQUEST_ID_SUFFIX: &str = "_batch_cost";

/// `VIDEO_COST_REQUEST_ID_SUFFIX` from `types/videos/utils.py`.
pub(crate) const VIDEO_COST_REQUEST_ID_SUFFIX: &str = "_video_cost";

/// `VIDEO_STATUS_CALL_TYPES` from `types/videos/utils.py`.
pub const VIDEO_STATUS_CALL_TYPES: [&str; 2] = ["video_status", "avideo_status"];

/// `SensitiveDataMasker`'s default patterns plus the `apikey` the request body masker adds.
pub(crate) const SENSITIVE_PATTERNS: [&str; 15] = [
    "password",
    "secret",
    "key",
    "token",
    "auth",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "access",
    "private",
    "certificate",
    "fingerprint",
    "tenancy",
    "apikey",
];

/// `_TOOL_INPUT_BLOCK_TYPES`.
pub(crate) const TOOL_INPUT_BLOCK_TYPES: [&str; 3] =
    ["tool_use", "server_tool_use", "mcp_tool_use"];

/// `_TOOL_OUTPUT_BLOCK_TYPES`.
pub(crate) const TOOL_OUTPUT_BLOCK_TYPES: [&str; 3] =
    ["tool_result", "mcp_tool_result", "function_call_output"];

/// `PROMPT_CARRYING_GUARDRAIL_FIELDS` from `types/utils.py`.
pub(crate) const PROMPT_CARRYING_GUARDRAIL_FIELDS: [&str; 4] = [
    "guardrail_request",
    "guardrail_response",
    "match_details",
    "classification",
];

/// `_NUMERIC_COMPRESSION_STAT_KEYS` from `redact_messages.py`.
pub(crate) const NUMERIC_COMPRESSION_STAT_KEYS: [&str; 4] = [
    "tokens_before",
    "tokens_after",
    "tokens_saved",
    "compression_ratio",
];

/// `_SCRUBBED_ERROR_TEXT_FIELDS`.
pub(crate) const SCRUBBED_ERROR_TEXT_FIELDS: [&str; 2] = ["error_message", "traceback"];

/// The classifier audit fields of the logging payload, which the stored request carries.
pub(crate) const CLASSIFIER_AUDIT_FIELDS: [&str; 2] =
    ["classifier_input", "originating_request_masked"];

/// `_CLIENT_IP_HEADERS`.
pub(crate) const CLIENT_IP_HEADERS: [&str; 18] = [
    "x-forwarded-for",
    "x-real-ip",
    "forwarded",
    "true-client-ip",
    "cf-connecting-ip",
    "x-client-ip",
    "x-cluster-client-ip",
    "fastly-client-ip",
    "x-original-forwarded-for",
    "x-envoy-external-address",
    "x-azure-clientip",
    "cloudfront-viewer-address",
    "fly-client-ip",
    "cf-connecting-ipv6",
    "cf-pseudo-ipv4",
    "x-appengine-user-ip",
    "x-vercel-forwarded-for",
    "x-arr-clientip",
];

/// `MESSAGE_REDACTION_ENABLE_HEADERS` from `redact_messages.py`.
pub(crate) const MESSAGE_REDACTION_ENABLE_HEADERS: [&str; 2] = [
    "litellm-enable-message-redaction",
    "x-litellm-enable-message-redaction",
];

/// `_REDACTED_REQUEST_FIELDS` from `redact_messages.py`.
pub(crate) const REDACTED_REQUEST_FIELDS: [&str; 6] = [
    "instructions",
    "query_image",
    "images",
    "documents",
    "retrievalQuery",
    "retrieval_query",
];

/// `_NESTED_REQUEST_FIELDS` from `redact_messages.py`.
pub(crate) const NESTED_REQUEST_FIELDS: [&str; 3] =
    ["extra_body", "additional_args", "complete_input_dict"];

/// `_WEB_ACTION_TEXT_FIELDS` from `redact_messages.py`.
pub(crate) const WEB_ACTION_TEXT_FIELDS: [&str; 3] = ["query", "url", "pattern"];

/// `_WEB_ACTION_LIST_FIELDS` from `redact_messages.py`.
pub(crate) const WEB_ACTION_LIST_FIELDS: [&str; 2] = ["queries", "sources"];

/// `VERTEX_AI_PROVIDER_METADATA_FIELDS` from `types/llms/vertex_ai.py`.
pub(crate) const VERTEX_AI_PROVIDER_METADATA_FIELDS: [&str; 5] = [
    "vertex_ai_grounding_metadata",
    "vertex_ai_url_context_metadata",
    "vertex_ai_safety_ratings",
    "vertex_ai_safety_results",
    "vertex_ai_citation_metadata",
];

/// `BEDROCK_GUARDRAIL_TRACE_FIELDS` from `redact_messages.py`.
pub(crate) const BEDROCK_GUARDRAIL_TRACE_FIELDS: [&str; 2] = ["trace", "amazon-bedrock-trace"];

/// `_MATCHED_TEXT_KEYS` from `redact_messages.py`.
pub(crate) const MATCHED_TEXT_KEYS: [&str; 2] = ["match", "regex"];

/// `_METADATA_FIELDS` from `accounting/snapshot.py`: the row metadata the accounting payload keeps.
pub(crate) const ACCOUNTING_METADATA_FIELDS: [&str; 9] = [
    "usage_object",
    "status",
    "guardrail_information",
    "standard_logging_guardrail_information",
    "compression_savings",
    "internal_call_origin",
    "billed_retry_attempt",
    "user_api_key_customer_usage",
    "route_call_type",
];

/// `_GUARDRAIL_METADATA_FIELDS` from `accounting/snapshot.py`.
pub(crate) const GUARDRAIL_METADATA_FIELDS: [&str; 2] = [
    "guardrail_information",
    "standard_logging_guardrail_information",
];

/// `_ACCOUNTING_FIELDS` from `accounting/snapshot.py`: the row columns the payload keeps.
pub(crate) const ACCOUNTING_FIELDS: [&str; 24] = [
    "request_id",
    "call_type",
    "api_key",
    "spend",
    "prompt_tokens",
    "completion_tokens",
    "startTime",
    "model",
    "model_group",
    "custom_llm_provider",
    "user",
    "team_id",
    "organization_id",
    "end_user",
    "customer_member_id",
    "customer_team_id",
    "customer_org_id",
    "app_id",
    "app_group_id",
    "agent_id",
    "request_tags",
    "status",
    "request_duration_ms",
    "mcp_namespaced_tool_name",
];

/// `_GUARDRAIL_ACCOUNTING_FIELDS` from `accounting/snapshot.py`.
pub(crate) const GUARDRAIL_ACCOUNTING_FIELDS: [&str; 6] = [
    "guardrail_id",
    "guardrail_name",
    "guardrail_provider",
    "guardrail_usage",
    "guardrail_cost_by_unit",
    "guardrail_cost_in_spend",
];

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use litellm_python_compat::{Value, json};
    use rstest::rstest;

    use super::*;

    /// The Python constants as the recorder wrote them: ordered lists for tuples and annotations,
    /// sorted lists for sets.
    fn recorded(name: &str) -> Vec<String> {
        let text = include_str!("../tests/goldens/constants.json");
        let constants = json::loads(text).expect("constants parse");
        match litellm_python_compat::dict::get(&constants, name) {
            Some(Value::List(items)) => items
                .iter()
                .map(|item| match item {
                    Value::Str(text) => text.clone(),
                    other => panic!("{name} holds {other:?}"),
                })
                .collect(),
            other => panic!("{name} is {other:?}"),
        }
    }

    #[rstest]
    #[case::spend_logs_metadata_fields("SpendLogsMetadata", &SPEND_LOGS_METADATA_FIELDS)]
    #[case::customer_identity_fields("_CustomerIdentityMetadata", &CUSTOMER_IDENTITY_FIELDS)]
    #[case::redacted_request_fields("_REDACTED_REQUEST_FIELDS", &REDACTED_REQUEST_FIELDS)]
    #[case::nested_request_fields("_NESTED_REQUEST_FIELDS", &NESTED_REQUEST_FIELDS)]
    #[case::web_action_text_fields("_WEB_ACTION_TEXT_FIELDS", &WEB_ACTION_TEXT_FIELDS)]
    #[case::web_action_list_fields("_WEB_ACTION_LIST_FIELDS", &WEB_ACTION_LIST_FIELDS)]
    #[case::vertex_ai_provider_metadata_fields(
        "VERTEX_AI_PROVIDER_METADATA_FIELDS",
        &VERTEX_AI_PROVIDER_METADATA_FIELDS
    )]
    #[case::bedrock_guardrail_trace_fields(
        "BEDROCK_GUARDRAIL_TRACE_FIELDS",
        &BEDROCK_GUARDRAIL_TRACE_FIELDS
    )]
    #[case::message_redaction_enable_headers(
        "MESSAGE_REDACTION_ENABLE_HEADERS",
        &MESSAGE_REDACTION_ENABLE_HEADERS
    )]
    fn ordered_tables_match_python(#[case] name: &str, #[case] table: &[&str]) {
        assert_eq!(recorded(name), table);
    }

    #[rstest]
    #[case::stamped_metadata_keys("_STAMPED_METADATA_KEYS", &STAMPED_METADATA_KEYS)]
    #[case::non_secret_key_aliases("_NON_SECRET_KEY_ALIASES", &NON_SECRET_KEY_ALIASES)]
    #[case::batch_trace_call_types("_BATCH_TRACE_CALL_TYPES", &BATCH_TRACE_CALL_TYPES)]
    #[case::batch_cost_suffix("BATCH_COST_REQUEST_ID_SUFFIX", &[BATCH_COST_REQUEST_ID_SUFFIX])]
    #[case::video_cost_suffix("VIDEO_COST_REQUEST_ID_SUFFIX", &[VIDEO_COST_REQUEST_ID_SUFFIX])]
    #[case::video_status_call_types("VIDEO_STATUS_CALL_TYPES", &VIDEO_STATUS_CALL_TYPES)]
    #[case::sensitive_patterns("_REQUEST_BODY_CREDENTIAL_MASKER", &SENSITIVE_PATTERNS)]
    #[case::tool_input_block_types("_TOOL_INPUT_BLOCK_TYPES", &TOOL_INPUT_BLOCK_TYPES)]
    #[case::tool_output_block_types("_TOOL_OUTPUT_BLOCK_TYPES", &TOOL_OUTPUT_BLOCK_TYPES)]
    #[case::prompt_carrying_guardrail_fields(
        "PROMPT_CARRYING_GUARDRAIL_FIELDS",
        &PROMPT_CARRYING_GUARDRAIL_FIELDS
    )]
    #[case::numeric_compression_stat_keys(
        "_NUMERIC_COMPRESSION_STAT_KEYS",
        &NUMERIC_COMPRESSION_STAT_KEYS
    )]
    #[case::scrubbed_error_text_fields("_SCRUBBED_ERROR_TEXT_FIELDS", &SCRUBBED_ERROR_TEXT_FIELDS)]
    #[case::client_ip_headers("_CLIENT_IP_HEADERS", &CLIENT_IP_HEADERS)]
    #[case::matched_text_keys("_MATCHED_TEXT_KEYS", &MATCHED_TEXT_KEYS)]
    #[case::accounting_metadata_fields("accounting._METADATA_FIELDS", &ACCOUNTING_METADATA_FIELDS)]
    #[case::guardrail_metadata_fields(
        "accounting._GUARDRAIL_METADATA_FIELDS",
        &GUARDRAIL_METADATA_FIELDS
    )]
    #[case::accounting_fields("accounting._ACCOUNTING_FIELDS", &ACCOUNTING_FIELDS)]
    #[case::guardrail_accounting_fields(
        "accounting._GUARDRAIL_ACCOUNTING_FIELDS",
        &GUARDRAIL_ACCOUNTING_FIELDS
    )]
    fn set_tables_match_python(#[case] name: &str, #[case] table: &[&str]) {
        let rust: BTreeSet<String> = table.iter().map(|item| (*item).to_owned()).collect();
        assert_eq!(rust.len(), table.len(), "{name} repeats an entry");
        assert_eq!(recorded(name), rust.into_iter().collect::<Vec<_>>());
    }
}
