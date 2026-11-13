use litellm_python_compat::boolean::str_to_bool;
use litellm_secret_value::{Redacted, SecretValue};
use serde::Deserialize;

use crate::{
    AdditionalFields, CacheSettings, JwtAuthSettings, LitellmRole, Object, RedisSettings,
    RolePermissions, Spelled, Value, value::one_or_many,
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

/// A flag the proxy also accepts as a string, true only for one `str_to_bool` reads as true.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LenientFlag(pub bool);

impl<'de> Deserialize<'de> for LenientFlag {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Shape {
            Flag(bool),
            Text(String),
        }
        Ok(Self(match Option::<Shape>::deserialize(deserializer)? {
            None => false,
            Some(Shape::Flag(flag)) => flag,
            Some(Shape::Text(text)) => str_to_bool(&text) == Some(true),
        }))
    }
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

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct GeneralSettings {
    pub completion_model: Option<String>,
    pub max_in_flight_requests_per_worker: Option<u64>,
    pub max_queued_requests_per_worker: Option<u64>,
    pub admission_queue_timeout_seconds: f64,
    pub master_key: Option<SecretValue>,
    pub database_url: Option<SecretValue>,
    pub tracing: Option<TracingSettings>,
    pub database_connection_pool_limit: Option<u64>,
    pub database_connection_timeout: Option<f64>,
    pub database_connect_timeout: Option<f64>,
    pub database_socket_timeout: Option<f64>,
    pub database_max_idle_connection_lifetime: Option<f64>,
    pub max_parallel_requests: Option<u64>,
    pub global_max_parallel_requests: Option<u64>,
    pub max_request_size_mb: Option<u64>,
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
    pub plugins: Option<Redacted<Box<[Object]>>>,
    pub coordination_redis: Option<RedisSettings>,
    pub mcp_allowed_hosts: Option<Box<[String]>>,
    pub mcp_allowed_origins: Box<[String]>,
    pub enable_jwt_auth: Option<bool>,
    pub enable_oauth2_auth: Option<bool>,
    pub enable_oauth2_proxy_auth: Option<bool>,
    pub allow_user_auth: Option<bool>,
    pub allow_requests_on_db_unavailable: LenientFlag,
    pub public_routes: Option<Box<[String]>>,
    pub allowed_ips: Option<Box<[String]>>,
    pub allowed_routes: Option<Box<[String]>>,
    pub custom_auth: Option<String>,
    pub litellm_key_header_name: Option<String>,
    pub pass_through_endpoints: Option<Redacted<Box<[Object]>>>,
    pub litellm_jwtauth: Option<JwtAuthSettings>,
    pub role_permissions: Option<Box<[RolePermissions]>>,
    pub enforce_rbac: Option<bool>,
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
            database_url: None,
            tracing: None,
            database_connection_pool_limit: Some(10),
            database_connection_timeout: Some(60.0),
            database_connect_timeout: None,
            database_socket_timeout: None,
            database_max_idle_connection_lifetime: Some(60.0),
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
            enable_jwt_auth: None,
            enable_oauth2_auth: None,
            enable_oauth2_proxy_auth: None,
            allow_user_auth: None,
            allow_requests_on_db_unavailable: LenientFlag::default(),
            public_routes: None,
            allowed_ips: None,
            allowed_routes: None,
            custom_auth: None,
            litellm_key_header_name: None,
            pass_through_endpoints: None,
            litellm_jwtauth: None,
            role_permissions: None,
            enforce_rbac: None,
            user_header_name: None,
            user_header_mappings: Box::default(),
            additional_fields: AdditionalFields::default(),
        }
    }
}

#[derive(Clone, Default, Debug, Deserialize)]
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

#[derive(Clone, Default, Debug, Deserialize)]
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
    pub drop_params: Option<Spelled<bool>>,
    pub request_timeout: Option<Spelled<f64>>,
    pub num_retries: Option<u64>,
    pub cache: Option<bool>,
    pub cache_params: Option<CacheSettings>,
    #[serde(deserialize_with = "one_or_many")]
    pub callbacks: Option<Redacted<Vec<Value>>>,
    #[serde(deserialize_with = "one_or_many")]
    pub success_callback: Option<Redacted<Vec<Value>>>,
    #[serde(deserialize_with = "one_or_many")]
    pub failure_callback: Option<Redacted<Vec<Value>>>,
    pub json_logs: Option<bool>,
    pub set_verbose: Option<bool>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

#[cfg(test)]
mod tests {
    use super::{
        ClickHouseStoreSettings, GeneralSettings, LenientFlag, LiteLlmSettings, RouterSettings,
        TracingSettings,
    };
    use rstest::rstest;
    use serde::de::DeserializeOwned;
    use std::fmt::Debug;

    fn debug_of<T: DeserializeOwned + Debug>(value: serde_json::Value) -> String {
        format!("{:?}", serde_json::from_value::<T>(value).unwrap())
    }

    #[rstest]
    #[case::general_master_key(debug_of::<GeneralSettings>(serde_json::json!({"master_key": "credential-value"})))]
    #[case::general_database_url(debug_of::<GeneralSettings>(serde_json::json!({"database_url": "credential-value"})))]
    #[case::general_plugins(debug_of::<GeneralSettings>(serde_json::json!({"plugins": [{"api_key": "credential-value"}]})))]
    #[case::general_pass_through_endpoints(debug_of::<GeneralSettings>(
        serde_json::json!({"pass_through_endpoints": [{"headers": {"Authorization": "credential-value"}}]})
    ))]
    #[case::general_unknown(debug_of::<GeneralSettings>(serde_json::json!({"alerting_secret": "credential-value"})))]
    #[case::jwt_auth_unknown(debug_of::<GeneralSettings>(
        serde_json::json!({"litellm_jwtauth": {"future_secret": "credential-value"}})
    ))]
    #[case::router_unknown(debug_of::<RouterSettings>(serde_json::json!({"redis_password": "credential-value"})))]
    #[case::litellm_callbacks(debug_of::<LiteLlmSettings>(serde_json::json!({"callbacks": {"api_key": "credential-value"}})))]
    #[case::litellm_unknown(debug_of::<LiteLlmSettings>(serde_json::json!({"langfuse_secret": "credential-value"})))]
    #[case::tracing_unknown(debug_of::<TracingSettings>(serde_json::json!({"api_key": "credential-value"})))]
    #[case::clickhouse_url(debug_of::<ClickHouseStoreSettings>(
        serde_json::json!({"type": "clickhouse", "url": "credential-value"})
    ))]
    fn credentials_are_redacted(#[case] debug: String) {
        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    #[rstest]
    #[case::flag(serde_json::json!(true), true)]
    #[case::true_text(serde_json::json!(" True "), true)]
    #[case::python_stripped_true_text(serde_json::json!("\u{1c}true"), true)]
    #[case::false_text(serde_json::json!("false"), false)]
    #[case::other_text(serde_json::json!("yes"), false)]
    #[case::null(serde_json::Value::Null, false)]
    fn lenient_flag_is_true_only_for_true(
        #[case] value: serde_json::Value,
        #[case] expected: bool,
    ) {
        assert_eq!(
            serde_json::from_value::<LenientFlag>(value).unwrap(),
            LenientFlag(expected)
        );
    }

    #[test]
    fn unset_lenient_flag_is_false() {
        let general: GeneralSettings = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(general.allow_requests_on_db_unavailable, LenientFlag(false));
    }
}
