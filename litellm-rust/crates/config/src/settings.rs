use std::fmt;

use litellm_auth_types::SecretValue;
use serde::Deserialize;

use crate::{AdditionalFields, Object, Spelled, Value, value::one_or_many};

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TracingStoreKind {
    Clickhouse,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClickHouseStoreSettings {
    #[serde(rename = "type")]
    pub kind: TracingStoreKind,
    pub url: Option<SecretValue>,
    pub database: Option<String>,
    pub retention_days: Option<Spelled<f64>>,
}

impl fmt::Debug for ClickHouseStoreSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ClickHouseStoreSettings")
            .field("kind", &self.kind)
            .field("database", &self.database)
            .field("retention_days", &self.retention_days)
            .finish()
    }
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

#[derive(Clone, Deserialize)]
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
    pub plugins: Option<Box<[Object]>>,
    pub coordination_redis: Option<Object>,
    pub mcp_allowed_hosts: Option<Box<[String]>>,
    pub mcp_allowed_origins: Box<[String]>,
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
            additional_fields: AdditionalFields::new(),
        }
    }
}

impl fmt::Debug for GeneralSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeneralSettings")
            .field("completion_model", &self.completion_model)
            .field(
                "max_in_flight_requests_per_worker",
                &self.max_in_flight_requests_per_worker,
            )
            .field(
                "max_queued_requests_per_worker",
                &self.max_queued_requests_per_worker,
            )
            .field(
                "admission_queue_timeout_seconds",
                &self.admission_queue_timeout_seconds,
            )
            .field("master_key", &self.master_key)
            .field("database_url", &self.database_url)
            .field("tracing", &self.tracing)
            .field("store_model_in_db", &self.store_model_in_db)
            .field("additional_fields", &self.additional_fields.keys())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Default, Deserialize)]
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

impl fmt::Debug for RouterSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RouterSettings")
            .field("routing_strategy", &self.routing_strategy)
            .field("routing_strategy_args", &self.routing_strategy_args)
            .field("routing_groups", &self.routing_groups)
            .field("retry_policy", &self.retry_policy)
            .field("model_group_retry_policy", &self.model_group_retry_policy)
            .field(
                "model_group_affinity_config",
                &self.model_group_affinity_config,
            )
            .field("allowed_fails", &self.allowed_fails)
            .field("cooldown_time", &self.cooldown_time)
            .field("num_retries", &self.num_retries)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("retry_after", &self.retry_after)
            .field("fallbacks", &self.fallbacks)
            .field("context_window_fallbacks", &self.context_window_fallbacks)
            .field("model_group_alias", &self.model_group_alias)
            .field("enable_tag_filtering", &self.enable_tag_filtering)
            .field("weights", &self.weights)
            .field("tag_routing_prefix", &self.tag_routing_prefix)
            .field("optional_pre_call_checks", &self.optional_pre_call_checks)
            .field("additional_fields", &self.additional_fields.keys())
            .finish()
    }
}

#[derive(Clone, Default, Deserialize)]
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
    pub cache_params: Option<Object>,
    #[serde(deserialize_with = "one_or_many")]
    pub callbacks: Option<Vec<Value>>,
    #[serde(deserialize_with = "one_or_many")]
    pub success_callback: Option<Vec<Value>>,
    #[serde(deserialize_with = "one_or_many")]
    pub failure_callback: Option<Vec<Value>>,
    pub json_logs: Option<bool>,
    pub set_verbose: Option<bool>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

impl fmt::Debug for LiteLlmSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LiteLlmSettings")
            .field("drop_params", &self.drop_params)
            .field("request_timeout", &self.request_timeout)
            .field("num_retries", &self.num_retries)
            .field("cache", &self.cache)
            .field("cache_params", &self.cache_params)
            .field(
                "callbacks",
                &self.callbacks.as_ref().map(|callbacks| callbacks.len()),
            )
            .field(
                "success_callback",
                &self
                    .success_callback
                    .as_ref()
                    .map(|callbacks| callbacks.len()),
            )
            .field(
                "failure_callback",
                &self
                    .failure_callback
                    .as_ref()
                    .map(|callbacks| callbacks.len()),
            )
            .field("json_logs", &self.json_logs)
            .field("set_verbose", &self.set_verbose)
            .field("additional_fields", &self.additional_fields.keys())
            .finish()
    }
}
