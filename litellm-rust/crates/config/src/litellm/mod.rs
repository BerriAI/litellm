mod budget;
mod caching;
mod fallbacks;
mod logging;
mod mcp;
mod transport;

use serde::Deserialize;

pub use self::{
    budget::{BudgetSettings, CallbackVar, TeamDefaultSettings, UpperboundKeyGenerateParams},
    caching::{
        AzureBlobCacheParams, CacheParams, CachingSettings, DiskCacheParams, GcsCacheParams,
        ProxyCacheTtlParams, QdrantCacheParams, RedisCacheParams, RedisServerCacheParams,
        S3CacheParams, SemanticCacheParams, ValkeyCacheParams,
    },
    fallbacks::FallbackSettings,
    logging::LoggingSettings,
    mcp::McpToolSettings,
    transport::{
        AiohttpSettings, ClientCertificateSettings, Http2Settings, SocketSettings, TlsSettings,
    },
};
use crate::{Object, Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct LiteLlmSettings {
    pub drop_params: Option<Spelled<bool>>,
    pub request_timeout: Option<Spelled<f64>>,
    pub num_retries: Option<u64>,
    pub prometheus_deployment_and_latency_caller_identity: Option<CallerIdentity>,
    pub guardrails: Option<Box<[Object]>>,
    pub prompt_injection_params: Option<Object>,
    pub azure_content_safety_params: Option<Object>,
    pub global_prompt_directory: Option<String>,
    pub global_bitbucket_config: Option<Object>,
    pub global_gitlab_config: Option<Object>,
    pub priority_reservation_settings: Option<PriorityReservationSettings>,
    pub model_group_settings: Option<ModelGroupSettings>,
    pub post_call_rules: Option<String>,
    pub custom_provider_map: Option<Box<[CustomProvider]>>,
    pub responses: Option<ResponsesSettings>,
    pub enable_preview_features: Option<Spelled<bool>>,
    pub public_model_groups: Option<Box<[String]>>,
    pub timezone: Option<String>,
    #[serde(flatten)]
    pub tls: TlsSettings,
    #[serde(flatten)]
    pub client_certificate: ClientCertificateSettings,
    #[serde(flatten)]
    pub socket: SocketSettings,
    #[serde(flatten)]
    pub http2: Http2Settings,
    #[serde(flatten)]
    pub aiohttp: AiohttpSettings,
    #[serde(flatten)]
    pub caching: CachingSettings,
    #[serde(flatten)]
    pub budget: BudgetSettings,
    #[serde(flatten)]
    pub logging: LoggingSettings,
    #[serde(flatten)]
    pub fallbacks: FallbackSettings,
    #[serde(flatten)]
    pub mcp: McpToolSettings,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::IntoStaticStr)]
#[serde(rename_all = "snake_case")]
pub enum CallerIdentity {
    #[strum(serialize = "api_key_alias")]
    ApiKeyAlias,
    #[strum(serialize = "user_email")]
    UserEmail,
    #[strum(serialize = "both")]
    Both,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PriorityReservationSettings {
    pub default_priority: Option<Spelled<f64>>,
    pub saturation_threshold: Option<Spelled<f64>>,
    pub saturation_check_cache_ttl: Option<Spelled<i64>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ModelGroupSettings {
    pub forward_client_headers_to_llm_api: Option<Box<[String]>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct CustomProvider {
    pub provider: Option<String>,
    pub custom_handler: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ResponsesSettings {
    pub background_mode: Option<BackgroundMode>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct BackgroundMode {
    pub polling_via_cache: Option<Raw>,
    pub native_background_mode: Option<Box<[String]>>,
    pub ttl: Option<Spelled<i64>>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{CallerIdentity, Config, LiteLlmSettings, Spelled};

    fn settings(yaml: &str) -> LiteLlmSettings {
        Config::from_yaml(&format!("litellm_settings:\n{yaml}"))
            .unwrap()
            .litellm_settings
    }

    #[rstest]
    #[case::bool_word("  turn_off_message_logging: 'false'", Spelled::Text("false".into()))]
    #[case::bool_value("  turn_off_message_logging: true", Spelled::Value(true))]
    #[case::env_reference("  turn_off_message_logging: os.environ/REDACT", Spelled::Text("os.environ/REDACT".into()))]
    fn bool_settings_keep_python_spellings(#[case] yaml: &str, #[case] expected: Spelled<bool>) {
        assert_eq!(
            settings(yaml).logging.turn_off_message_logging,
            Some(expected)
        );
    }

    #[rstest]
    #[case::number("  max_budget: 10", Spelled::Value(10.0))]
    #[case::string("  max_budget: '10'", Spelled::Text("10".into()))]
    #[case::env_reference("  max_budget: os.environ/MAX_BUDGET", Spelled::Text("os.environ/MAX_BUDGET".into()))]
    fn budget_accepts_number_or_string(#[case] yaml: &str, #[case] expected: Spelled<f64>) {
        assert_eq!(settings(yaml).budget.max_budget, Some(expected));
    }

    #[rstest]
    #[case::api_key_alias("api_key_alias", CallerIdentity::ApiKeyAlias)]
    #[case::user_email("user_email", CallerIdentity::UserEmail)]
    #[case::both("both", CallerIdentity::Both)]
    fn caller_identity_accepts_python_values(
        #[case] value: &str,
        #[case] expected: CallerIdentity,
    ) {
        assert_eq!(
            settings(&format!(
                "  prometheus_deployment_and_latency_caller_identity: {value}"
            ))
            .prometheus_deployment_and_latency_caller_identity,
            Some(expected)
        );
    }

    #[rstest]
    fn caller_identity_rejects_values_python_refuses() {
        assert!(
            Config::from_yaml(
                "litellm_settings:\n  prometheus_deployment_and_latency_caller_identity: team"
            )
            .is_err()
        );
    }

    #[rstest]
    fn cache_params_reads_cache_keywords() {
        let cache = settings(
        "  cache_params:\n    type: redis\n    port: 6379\n    password: os.environ/REDIS_PASSWORD\n    ttl: 600\n    supported_call_types: [acompletion]\n    namespace_extra: x",
    )
    .caching
    .cache_params
    .unwrap();
        assert_eq!(cache.cache_type.as_deref(), Some("redis"));
        assert_eq!(cache.redis_server.port, Some(Spelled::Value(6379)));
        assert_eq!(
            cache.redis_server.password.unwrap().expose(),
            "os.environ/REDIS_PASSWORD"
        );
        assert_eq!(cache.ttl, Some(Spelled::Value(600.0)));
        assert_eq!(
            cache.supported_call_types.as_deref(),
            Some(&["acompletion".to_owned()][..])
        );
        assert!(cache.additional_fields.contains_key("namespace_extra"));
    }

    #[rstest]
    fn cache_params_route_each_key_to_its_backend_group() {
        let cache = settings(
            r#"  cache_params:
    type: s3
    namespace: shared
    port: 6379
    similarity_threshold: 0.8
    default_redis_ttl: 30
    redis_flush_size: 10
    valkey_semantic_cache_index_name: valkey-index
    s3_bucket_name: bucket
    gcs_path: gcs/path
    azure_blob_container: container
    qdrant_collection_name: collection
    disk_cache_dir: /tmp/cache
    namespace_extra: x"#,
        )
        .caching
        .cache_params
        .unwrap();
        assert_eq!(cache.namespace.as_deref(), Some("shared"));
        assert_eq!(cache.redis_server.port, Some(Spelled::Value(6379)));
        assert_eq!(
            cache.semantic.similarity_threshold,
            Some(Spelled::Value(0.8))
        );
        assert_eq!(
            cache.proxy_ttl.default_redis_ttl,
            Some(Spelled::Value(30.0))
        );
        assert_eq!(cache.redis.redis_flush_size, Some(Spelled::Value(10)));
        assert_eq!(
            cache.valkey.valkey_semantic_cache_index_name.as_deref(),
            Some("valkey-index")
        );
        assert_eq!(cache.s3.s3_bucket_name.as_deref(), Some("bucket"));
        assert_eq!(cache.gcs.gcs_path.as_deref(), Some("gcs/path"));
        assert_eq!(
            cache.azure_blob.azure_blob_container.as_deref(),
            Some("container")
        );
        assert_eq!(
            cache.qdrant.qdrant_collection_name.as_deref(),
            Some("collection")
        );
        assert_eq!(cache.disk.disk_cache_dir.as_deref(), Some("/tmp/cache"));
        assert_eq!(
            cache.additional_fields.keys().collect::<Vec<_>>(),
            ["namespace_extra"]
        );
    }

    #[rstest]
    #[case::absent("    ttl: 1", None)]
    #[case::explicit_null("    max_messages: null", Some(None))]
    #[case::limit("    max_messages: 8", Some(Some(Spelled::Value(8))))]
    fn cache_max_messages_tells_null_from_absent(
        #[case] yaml: &str,
        #[case] expected: Option<Option<Spelled<i64>>>,
    ) {
        assert_eq!(
            settings(&format!("  cache_params:\n{yaml}"))
                .caching
                .cache_params
                .unwrap()
                .max_messages,
            expected
        );
    }

    #[rstest]
    fn nested_settings_land_in_typed_fields() {
        let settings = settings(
        "  priority_reservation_settings: {default_priority: 0.5, saturation_check_cache_ttl: 30}
  model_group_settings: {forward_client_headers_to_llm_api: [gpt]}
  custom_provider_map: [{provider: my-llm, custom_handler: handler.instance}]
  responses: {background_mode: {polling_via_cache: all, native_background_mode: [o3], ttl: 60}}
  default_team_settings: [{team_id: team-1, success_callback: [langfuse]}]
  upperbound_key_generate_params: {max_budget: 100, duration: 30d, rpm_limit: os.environ/RPM}",
    );
        let priority = settings.priority_reservation_settings.unwrap();
        assert_eq!(priority.default_priority, Some(Spelled::Value(0.5)));
        assert_eq!(
            priority.saturation_check_cache_ttl,
            Some(Spelled::Value(30))
        );
        assert_eq!(
            settings
                .model_group_settings
                .unwrap()
                .forward_client_headers_to_llm_api
                .as_deref(),
            Some(&["gpt".to_owned()][..])
        );
        let provider = &settings.custom_provider_map.unwrap()[0];
        assert_eq!(provider.provider.as_deref(), Some("my-llm"));
        assert_eq!(provider.custom_handler.as_deref(), Some("handler.instance"));
        let background = settings.responses.unwrap().background_mode.unwrap();
        assert_eq!(
            background.polling_via_cache.unwrap().get().as_str(),
            Some("all")
        );
        assert_eq!(
            background.native_background_mode.as_deref(),
            Some(&["o3".to_owned()][..])
        );
        assert_eq!(background.ttl, Some(Spelled::Value(60)));
        let team = &settings.budget.default_team_settings.unwrap()[0];
        assert_eq!(team.team_id, "team-1");
        assert_eq!(
            team.success_callback.as_deref(),
            Some(&["langfuse".to_owned()][..])
        );
        let upperbound = settings.budget.upperbound_key_generate_params.unwrap();
        assert_eq!(upperbound.max_budget, Some(Spelled::Value(100.0)));
        assert_eq!(upperbound.duration.as_deref(), Some("30d"));
        assert_eq!(
            upperbound.rpm_limit,
            Some(Spelled::Text("os.environ/RPM".into()))
        );
    }

    #[rstest]
    #[case::cache_password(
        r#"
  cache: true
  cache_params: {password: SENTINEL}
"#
    )]
    #[case::cache_endpoint("  cache_params: {s3_endpoint_url: 'https://user:SENTINEL@s3.local'}")]
    #[case::cache_raw("  cache_params: {s3_config: {secret: SENTINEL}}")]
    #[case::gitlab_object("  global_gitlab_config: {access_token: SENTINEL}")]
    #[case::s3_callback_object("  s3_callback_params: {s3_aws_secret_access_key: SENTINEL}")]
    #[case::team_inline_var("  default_team_settings: [{team_id: t, langfuse_secret: SENTINEL}]")]
    #[case::team_callback_vars(
        "  default_team_settings: [{team_id: t, callback_vars: {langfuse_secret: SENTINEL}}]"
    )]
    #[case::fallbacks_raw("  default_fallbacks: [SENTINEL]")]
    #[case::user_params_raw("  default_internal_user_params: {models: [SENTINEL]}")]
    #[case::callback_entry("  callbacks: [SENTINEL]")]
    #[case::success_callback_map("  success_callback: {name: SENTINEL}")]
    #[case::failure_callback_one("  failure_callback: SENTINEL")]
    #[case::audit_object("  s3_audit_callback_params: {secret: SENTINEL}")]
    #[case::context_window_raw("  context_window_fallbacks: [{gpt: [SENTINEL]}]")]
    #[case::mcp_raw("  mcp_aliases: {docs: SENTINEL}")]
    #[case::injection_object("  prompt_injection_params: {key: SENTINEL}")]
    #[case::unknown_setting("  future_setting: SENTINEL")]
    fn debug_output_does_not_expose_settings_values(#[case] yaml: &str) {
        let config = Config::from_yaml(&format!("litellm_settings:\n{yaml}")).unwrap();
        assert!(!format!("{config:?}").contains("SENTINEL"));
    }

    #[rstest]
    fn unknown_setting_stays_in_additional_fields() {
        assert!(
            settings("  future_setting: 1")
                .additional_fields
                .contains_key("future_setting")
        );
    }

    #[rstest]
    fn transport_keys_land_in_their_reader_groups() {
        let settings = settings(
            r#"  ssl_verify: false
  ssl_security_level: '1'
  ssl_certificate: /etc/client.pem
  force_ipv4: true
  http2: true
  aiohttp_trust_env: true"#,
        );
        assert_eq!(settings.tls.ssl_verify, Some(Spelled::Value(false)));
        assert_eq!(settings.tls.ssl_security_level.as_deref(), Some("1"));
        assert_eq!(
            settings.client_certificate.ssl_certificate.as_deref(),
            Some("/etc/client.pem")
        );
        assert_eq!(settings.socket.force_ipv4, Some(true));
        assert_eq!(settings.http2.http2, Some(true));
        assert_eq!(settings.aiohttp.aiohttp_trust_env, Some(true));
        assert!(settings.additional_fields.is_empty());
    }

    #[rstest]
    #[case::tls("  ssl_verify: false", "ssl_verify")]
    #[case::client_certificate("  ssl_certificate: /etc/client.pem", "ssl_certificate")]
    #[case::socket("  force_ipv4: true", "force_ipv4")]
    #[case::http2("  http2: true", "http2")]
    #[case::aiohttp("  disable_aiohttp_transport: true", "disable_aiohttp_transport")]
    #[case::caching("  cache: true", "cache")]
    #[case::budget("  max_budget: 10", "max_budget")]
    #[case::logging("  json_logs: true", "json_logs")]
    #[case::fallbacks("  default_fallbacks: [gpt]", "default_fallbacks")]
    #[case::mcp("  mcp_aliases: {docs: docs_server}", "mcp_aliases")]
    fn grouped_settings_stay_out_of_additional_fields(#[case] yaml: &str, #[case] key: &str) {
        assert!(!settings(yaml).additional_fields.contains_key(key));
    }

    #[rstest]
    #[case::one("callbacks: custom_callbacks.logger", 1)]
    #[case::many("callbacks: [prometheus, otel]", 2)]
    fn accepts_python_callback_shorthand(#[case] setting: &str, #[case] expected_len: usize) {
        let config = Config::from_yaml(&format!("litellm_settings:\n  {setting}")).unwrap();
        assert_eq!(
            config
                .litellm_settings
                .logging
                .callbacks
                .as_ref()
                .unwrap()
                .len(),
            expected_len
        );
    }
}
