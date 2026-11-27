use std::{num::NonZeroUsize, time::Duration};

use litellm_python_compat::{
    lenient::{IsTrue, LaxBool, StrBool, Truthy},
    number::seconds,
    pydantic::{lax_float, lax_int},
};
use litellm_secret_value::SecretValue;
use serde::{Deserialize, Deserializer, de::Error as _};
use serde_with::{Map, serde_as};
use veil::Redact;

use crate::{
    AdditionalFields, AddressList, CacheSettings, JwtAuthSettings, LitellmRole, Object,
    PassThroughEndpoint, PriorityReservation, PriorityReservationSettings, RedisSettings,
    RolePermissions, RouteList, Spelled, Value, value::one_or_many,
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

/// `general_settings.user_api_key_cache_ttl` as the proxy reads it at startup: `float(value)` seconds
/// handed to the auth cache store, which reads a negative number as zero and refuses infinity and
/// NaN, so the proxy does not start over a value refused here.
fn user_api_key_cache_ttl<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Duration>, D::Error> {
    Option::<litellm_python_compat::Value>::deserialize(deserializer)?
        .map(|value| {
            lax_float(&value)
                .and_then(|value| seconds(value).ok())
                .ok_or_else(|| D::Error::custom("expected a finite number of seconds"))
        })
        .transpose()
}

/// `general_settings.user_api_key_cache_max_size` as `ConfigGeneralSettings` validates it: a lax
/// `int` greater than zero, so the proxy does not start over a value refused here.
fn user_api_key_cache_max_size<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<NonZeroUsize>, D::Error> {
    Option::<litellm_python_compat::Value>::deserialize(deserializer)?
        .map(|value| {
            lax_int(&value)
                .and_then(|number| usize::try_from(number).ok())
                .and_then(NonZeroUsize::new)
                .ok_or_else(|| D::Error::custom("expected an integer greater than zero"))
        })
        .transpose()
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
    /// `None` when neither the config nor `PROXY_CONFIG_RELOAD_INTERVAL_SECONDS` sets it.
    pub proxy_config_reload_interval_seconds: Option<i64>,
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
    pub allowed_ips: Option<AddressList>,
    #[serde_as(as = "IsTrue")]
    pub use_x_forwarded_for: bool,
    #[serde(deserialize_with = "crate::routes::where_the_key_is_written")]
    pub allowed_routes: Option<RouteList>,
    pub custom_auth: Option<String>,
    pub litellm_key_header_name: Option<String>,
    pub required_customer_headers: Box<[String]>,
    pub litellm_jwtauth: Option<JwtAuthSettings>,
    pub role_permissions: Option<Box<[RolePermissions]>>,
    /// `bool(general_settings.get("enforce_rbac"))`
    #[serde_as(as = "Truthy")]
    pub enforce_rbac: bool,
    pub user_header_name: Option<String>,
    /// Which tokens a TPM limit counts: `input`, `output`, or anything else for their total.
    pub token_rate_limit_type: Option<String>,
    #[serde(deserialize_with = "user_header_mappings")]
    pub user_header_mappings: Box<[UserHeaderMapping]>,
    #[serde(deserialize_with = "user_api_key_cache_ttl")]
    pub user_api_key_cache_ttl: Option<Duration>,
    #[serde(deserialize_with = "user_api_key_cache_max_size")]
    pub user_api_key_cache_max_size: Option<NonZeroUsize>,
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
            proxy_config_reload_interval_seconds: None,
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
            use_x_forwarded_for: false,
            allowed_routes: None,
            custom_auth: None,
            litellm_key_header_name: None,
            required_customer_headers: Box::default(),
            litellm_jwtauth: None,
            role_permissions: None,
            enforce_rbac: false,
            user_header_name: None,
            token_rate_limit_type: None,
            user_header_mappings: Box::default(),
            user_api_key_cache_ttl: None,
            user_api_key_cache_max_size: None,
            additional_fields: AdditionalFields::default(),
        }
    }
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
    /// `bool(value)`, as Python's `litellm.reasoning_auto_summary or ...` reads the setting.
    #[serde_as(as = "Truthy")]
    pub reasoning_auto_summary: bool,
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
    #[serde_as(as = "LaxBool")]
    pub enable_redis_auth_cache: Option<bool>,
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
    /// In the order written, which the proxy's dict keeps.
    #[serde_as(as = "Option<Map<_, _>>")]
    pub priority_reservation: Option<Vec<(String, PriorityReservation)>>,
    pub priority_reservation_settings: Option<PriorityReservationSettings>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use std::{num::NonZeroUsize, time::Duration};

    use super::{GeneralSettings, LiteLlmSettings};
    use crate::{JwtAuthSettings, RouteList};

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
    #[case::left_out(serde_json::json!({}), None)]
    #[case::written_with_no_value(
        serde_json::json!({"allowed_routes": null}),
        Some(RouteList::Malformed)
    )]
    #[case::written_as_text(
        serde_json::json!({"allowed_routes": "/health"}),
        Some(RouteList::Malformed)
    )]
    #[case::a_list(
        serde_json::json!({"allowed_routes": ["/health"]}),
        Some(RouteList::Valid(Box::from(["/health".to_owned()])))
    )]
    fn allowed_routes_is_a_list_wherever_the_key_is_written(
        #[case] general: serde_json::Value,
        #[case] expected: Option<RouteList>,
    ) {
        let settings: GeneralSettings = serde_json::from_value(general).unwrap();

        assert_eq!(settings.allowed_routes, expected);
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

    #[rstest]
    #[case::unset(serde_json::json!({}), None)]
    #[case::null(serde_json::json!({"user_api_key_cache_ttl": null}), None)]
    #[case::int(serde_json::json!({"user_api_key_cache_ttl": 600}), Some(Duration::from_secs(600)))]
    #[case::fraction(serde_json::json!({"user_api_key_cache_ttl": 1.5}), Some(Duration::from_millis(1500)))]
    #[case::text(serde_json::json!({"user_api_key_cache_ttl": " 600 "}), Some(Duration::from_secs(600)))]
    #[case::negative_is_zero(serde_json::json!({"user_api_key_cache_ttl": -5}), Some(Duration::ZERO))]
    fn the_auth_cache_ttl_is_read_as_float_reads_it(
        #[case] written: serde_json::Value,
        #[case] expected: Option<Duration>,
    ) {
        let settings: GeneralSettings = serde_json::from_value(written).unwrap();

        assert_eq!(settings.user_api_key_cache_ttl, expected);
    }

    #[rstest]
    #[case::unset(serde_json::json!({}), None)]
    #[case::null(serde_json::json!({"user_api_key_cache_max_size": null}), None)]
    #[case::int(serde_json::json!({"user_api_key_cache_max_size": 500}), NonZeroUsize::new(500))]
    #[case::whole_float(serde_json::json!({"user_api_key_cache_max_size": 500.0}), NonZeroUsize::new(500))]
    #[case::text(serde_json::json!({"user_api_key_cache_max_size": "500"}), NonZeroUsize::new(500))]
    fn the_auth_cache_size_is_read_as_pydantic_reads_an_int(
        #[case] written: serde_json::Value,
        #[case] expected: Option<NonZeroUsize>,
    ) {
        let settings: GeneralSettings = serde_json::from_value(written).unwrap();

        assert_eq!(settings.user_api_key_cache_max_size, expected);
    }

    #[rstest]
    #[case::ttl_text_that_is_no_number(serde_json::json!({"user_api_key_cache_ttl": "soon"}))]
    #[case::ttl_infinity(serde_json::json!({"user_api_key_cache_ttl": "inf"}))]
    #[case::ttl_nan(serde_json::json!({"user_api_key_cache_ttl": "nan"}))]
    #[case::size_zero(serde_json::json!({"user_api_key_cache_max_size": 0}))]
    #[case::size_negative(serde_json::json!({"user_api_key_cache_max_size": -1}))]
    #[case::size_fraction(serde_json::json!({"user_api_key_cache_max_size": 1.5}))]
    #[case::size_text_that_is_no_int(serde_json::json!({"user_api_key_cache_max_size": "many"}))]
    fn an_auth_cache_setting_the_proxy_would_not_start_with_is_refused(
        #[case] written: serde_json::Value,
    ) {
        assert!(serde_json::from_value::<GeneralSettings>(written).is_err());
    }
}
