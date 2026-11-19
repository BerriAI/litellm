use litellm_python_compat::lenient::{IsTrue, LaxBool, StrBool, Truthy};
use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_with::serde_as;
use veil::Redact;

use crate::{
    AdditionalFields, CacheSettings, JwtAuthSettings, LitellmRole, Object, PassThroughEndpoint,
    RedisSettings, RolePermissions, RouteList, Spelled, Value, value::one_or_many,
};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UserHeaderMapping {
    pub header_name: String,
    pub litellm_user_role: LitellmRole,
}

fn user_header_mappings<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Box<[UserHeaderMapping]>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Shape {
        One(UserHeaderMapping),
        Many(Box<[UserHeaderMapping]>),
    }
    Ok(match Option::<Shape>::deserialize(deserializer)? {
        None => Box::default(),
        Some(Shape::One(mapping)) => Box::new([mapping]),
        Some(Shape::Many(mappings)) => mappings,
    })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TracingStoreKind {
    Clickhouse,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClickHouseStoreSettings {
    #[serde(rename = "type")]
    pub kind: TracingStoreKind,
    pub url: Option<SecretValue>,
    pub database: Option<String>,
    pub retention_days: Option<Spelled<f64>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum TracingStoreSettings {
    ClickHouse(ClickHouseStoreSettings),
}

#[derive(Clone, Default, Debug, Deserialize)]
#[serde(default)]
pub struct TracingSettings {
    pub store: Option<TracingStoreSettings>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct GeneralSettings {
    pub completion_model: Option<String>,
    pub max_in_flight_requests_per_worker: Option<u64>,
    pub max_queued_requests_per_worker: Option<u64>,
    pub admission_queue_timeout_seconds: f64,
    pub master_key: Option<SecretValue>,
    pub litellm_license: Option<SecretValue>,
    pub database_url: Option<SecretValue>,
    pub tracing: Option<TracingSettings>,
    pub database_connection_pool_limit: Option<u64>,
    pub database_connection_timeout: Option<f64>,
    pub database_connect_timeout: Option<f64>,
    pub database_socket_timeout: Option<f64>,
    pub database_max_idle_connection_lifetime: Option<f64>,
    pub database_connection_pool_timeout: Option<Spelled<f64>>,
    pub database_statement_timeout: Option<Spelled<f64>>,
    pub database_lock_timeout: Option<Spelled<f64>>,
    pub database_idle_in_transaction_session_timeout: Option<Spelled<f64>>,
    #[serde_as(as = "StrBool")]
    pub database_disable_prepared_statements: bool,
    pub max_parallel_requests: Option<u64>,
    pub global_max_parallel_requests: Option<u64>,
    pub max_request_size_mb: Option<f64>,
    pub max_response_size_mb: Option<u64>,
    pub proxy_config_reload_interval_seconds: u64,
    pub background_health_checks: Option<bool>,
    pub health_check_interval: u64,
    pub health_check_concurrency: Option<u64>,
    pub store_model_in_db: Option<bool>,
    pub forward_client_headers_to_llm_api: Option<bool>,
    pub cancel_on_disconnect: Option<bool>,
    pub infer_model_from_keys: Option<bool>,
    pub enable_public_model_hub: bool,
    pub dangerously_permit_weak_or_unset_master_key: Option<bool>,
    pub plugins: Option<Box<[Object]>>,
    pub coordination_redis: Option<RedisSettings>,
    pub mcp_allowed_hosts: Option<Box<[String]>>,
    pub mcp_allowed_origins: Box<[String]>,
    pub pass_through_endpoints: Option<Box<[PassThroughEndpoint]>>,
    /// `general_settings.get(...) is True`, how the proxy reads each auth flag.
    #[serde_as(as = "IsTrue")]
    pub enable_jwt_auth: bool,
    #[serde_as(as = "IsTrue")]
    pub enable_oauth2_auth: bool,
    #[serde_as(as = "IsTrue")]
    pub enable_oauth2_proxy_auth: bool,
    #[serde_as(as = "IsTrue")]
    pub allow_user_auth: bool,
    #[serde_as(as = "StrBool")]
    pub allow_requests_on_db_unavailable: bool,
    #[serde_as(as = "IsTrue")]
    pub allow_client_side_credentials: bool,
    pub public_routes: Option<RouteList>,
    pub allowed_ips: Option<Box<[String]>>,
    pub allowed_routes: Option<Box<[String]>>,
    pub custom_auth: Option<String>,
    pub litellm_key_header_name: Option<String>,
    pub litellm_jwtauth: Option<JwtAuthSettings>,
    pub role_permissions: Option<Box<[RolePermissions]>>,
    /// `bool(general_settings.get("enforce_rbac"))`
    #[serde_as(as = "Truthy")]
    pub enforce_rbac: bool,
    pub user_header_name: Option<String>,
    #[serde(deserialize_with = "user_header_mappings")]
    pub user_header_mappings: Box<[UserHeaderMapping]>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            completion_model: None,
            max_in_flight_requests_per_worker: None,
            max_queued_requests_per_worker: None,
            admission_queue_timeout_seconds: 1.0,
            master_key: None,
            litellm_license: None,
            database_url: None,
            tracing: None,
            database_connection_pool_limit: Some(10),
            database_connection_timeout: Some(60.0),
            database_connect_timeout: None,
            database_socket_timeout: None,
            database_max_idle_connection_lifetime: Some(60.0),
            database_connection_pool_timeout: None,
            database_statement_timeout: None,
            database_lock_timeout: None,
            database_idle_in_transaction_session_timeout: None,
            database_disable_prepared_statements: false,
            max_parallel_requests: None,
            global_max_parallel_requests: None,
            max_request_size_mb: None,
            max_response_size_mb: None,
            proxy_config_reload_interval_seconds: 30,
            background_health_checks: None,
            health_check_interval: 300,
            health_check_concurrency: None,
            store_model_in_db: None,
            forward_client_headers_to_llm_api: None,
            cancel_on_disconnect: None,
            infer_model_from_keys: None,
            enable_public_model_hub: false,
            dangerously_permit_weak_or_unset_master_key: None,
            plugins: None,
            coordination_redis: None,
            mcp_allowed_hosts: None,
            mcp_allowed_origins: Box::default(),
            pass_through_endpoints: None,
            enable_jwt_auth: false,
            enable_oauth2_auth: false,
            enable_oauth2_proxy_auth: false,
            allow_user_auth: false,
            allow_requests_on_db_unavailable: false,
            allow_client_side_credentials: false,
            public_routes: None,
            allowed_ips: None,
            allowed_routes: None,
            custom_auth: None,
            litellm_key_header_name: None,
            litellm_jwtauth: None,
            role_permissions: None,
            enforce_rbac: false,
            user_header_name: None,
            user_header_mappings: Box::default(),
            additional_fields: AdditionalFields::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct RouterSettings {
    pub routing_strategy: Option<String>,
    pub routing_strategy_args: Option<Object>,
    pub routing_groups: Option<Box<[Object]>>,
    pub retry_policy: Option<Object>,
    pub model_group_retry_policy: Option<Object>,
    pub model_group_affinity_config: Option<Object>,
    pub allowed_fails: Option<u64>,
    pub cooldown_time: Option<f64>,
    pub num_retries: Option<u64>,
    pub timeout: Option<f64>,
    pub max_retries: Option<u64>,
    pub retry_after: Option<f64>,
    pub fallbacks: Option<Box<[Object]>>,
    pub context_window_fallbacks: Option<Box<[Object]>>,
    pub model_group_alias: Option<Object>,
    pub enable_tag_filtering: Option<bool>,
    pub weights: Option<Object>,
    pub tag_routing_prefix: Option<String>,
    pub optional_pre_call_checks: Option<Box<[String]>>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

/// Callback entries can carry credentials inline, so only whether they are set is printed.
#[serde_as]
#[derive(Clone, Default, Deserialize, Redact)]
#[serde(default)]
pub struct LiteLlmSettings {
    pub ssl_verify: Option<Spelled<bool>>,
    pub ssl_certificate: Option<String>,
    pub ssl_security_level: Option<String>,
    pub ssl_ecdh_curve: Option<String>,
    pub force_ipv4: Option<bool>,
    pub http2: Option<bool>,
    pub aiohttp_trust_env: Option<bool>,
    pub disable_aiohttp_trust_env: Option<bool>,
    pub disable_aiohttp_transport: Option<bool>,
    #[serde_as(as = "LaxBool")]
    pub drop_params: Option<bool>,
    #[serde_as(as = "LaxBool")]
    pub modify_params: Option<bool>,
    #[serde_as(as = "LaxBool")]
    pub enable_anthropic_prompt_caching: Option<bool>,
    #[serde_as(as = "LaxBool")]
    pub reasoning_auto_summary: Option<bool>,
    #[serde_as(as = "LaxBool")]
    pub use_chat_completions_url_for_anthropic_messages: Option<bool>,
    #[serde_as(as = "LaxBool")]
    pub route_all_chat_openai_to_responses: Option<bool>,
    #[serde_as(as = "LaxBool")]
    pub gemini_live_defer_setup: Option<bool>,
    pub anthropic_prompt_caching_ttl: Option<String>,
    pub model_cost_map_url: Option<SecretValue>,
    pub blog_posts_url: Option<SecretValue>,
    pub anthropic_beta_headers_url: Option<SecretValue>,
    pub autorouter_presets_url: Option<SecretValue>,
    pub request_timeout: Option<Spelled<f64>>,
    pub num_retries: Option<u64>,
    pub cache: Option<bool>,
    pub cache_params: Option<CacheSettings>,
    #[serde(deserialize_with = "one_or_many")]
    #[redact(fixed = 3)]
    pub callbacks: Option<Vec<Value>>,
    #[serde(deserialize_with = "one_or_many")]
    #[redact(fixed = 3)]
    pub success_callback: Option<Vec<Value>>,
    #[serde(deserialize_with = "one_or_many")]
    #[redact(fixed = 3)]
    pub failure_callback: Option<Vec<Value>>,
    pub json_logs: Option<bool>,
    pub set_verbose: Option<bool>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{GeneralSettings, LiteLlmSettings};
    use crate::JwtAuthSettings;

    #[test]
    fn jwt_auth_settings_hide_unknown_fields() {
        let settings: JwtAuthSettings =
            serde_json::from_value(serde_json::json!({"future_secret": "credential-value"}))
                .unwrap();

        let debug = format!("{settings:?}");

        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    #[rstest]
    #[case::a_word_for_on(serde_json::json!("on"), Some(true))]
    #[case::a_word_that_is_no_flag(serde_json::json!("banana"), None)]
    fn a_flag_key_written_as_text_is_read_by_the_one_flag_rule(
        #[case] written: serde_json::Value,
        #[case] expected: Option<bool>,
    ) {
        let settings: LiteLlmSettings =
            serde_json::from_value(serde_json::json!({"modify_params": written})).unwrap();

        assert_eq!(settings.modify_params, expected);
    }

    #[test]
    fn unset_text_flags_are_false() {
        let general: GeneralSettings = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(!general.allow_requests_on_db_unavailable && !general.enforce_rbac);
    }

    #[rstest]
    #[case::int(serde_json::json!(1), 1.0)]
    #[case::fraction(serde_json::json!(0.5), 0.5)]
    fn a_request_size_limit_is_an_int_or_a_float(
        #[case] limit: serde_json::Value,
        #[case] expected: f64,
    ) {
        let settings: GeneralSettings =
            serde_json::from_value(serde_json::json!({"max_request_size_mb": limit})).unwrap();
        assert_eq!(settings.max_request_size_mb, Some(expected));
    }
}
