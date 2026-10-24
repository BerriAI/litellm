mod alerting;
mod auth;
mod database;
mod files;
mod health;
mod limits;
mod mcp;
mod objects;
mod requests;
mod spend;
mod tracing;
mod ui;

use litellm_auth_types::SecretValue;
use serde::Deserialize;

pub use self::{
    alerting::{Alerting, AlertingArgs, AlertingSettings},
    auth::{
        AuthSettings, KeyManagementSettings, RoleBasedPermissions, ServiceAccountSettings,
        UserHeaderMapping,
    },
    database::DatabaseSettings,
    files::FilesSettings,
    health::HealthSettings,
    limits::LimitsSettings,
    mcp::McpSettings,
    objects::{ObjectsSettings, TeamMetadataField},
    requests::{PassThroughEndpoint, RequestTagHeader, RequestTagMapping, RequestsSettings},
    spend::{
        ScheduledJobStaggerSettings, SpendCaptureRateCheckSettings, SpendLogsMetadataFields,
        SpendSettings,
    },
    tracing::{ClickHouseStoreSettings, TracingSettings, TracingStoreSettings},
    ui::{UiAccessMode, UiAccessSsoGroup, UiSettings},
};
use crate::Object;

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
    pub plugins: Option<Box<[Object]>>,
    pub coordination_redis: Option<Object>,
    #[serde(flatten)]
    pub auth: AuthSettings,
    #[serde(flatten)]
    pub database: DatabaseSettings,
    #[serde(flatten)]
    pub spend: SpendSettings,
    #[serde(flatten)]
    pub limits: LimitsSettings,
    #[serde(flatten)]
    pub health: HealthSettings,
    #[serde(flatten)]
    pub alerting: AlertingSettings,
    #[serde(flatten)]
    pub ui: UiSettings,
    #[serde(flatten)]
    pub objects: ObjectsSettings,
    #[serde(flatten)]
    pub requests: RequestsSettings,
    #[serde(flatten)]
    pub files: FilesSettings,
    #[serde(flatten)]
    pub mcp: McpSettings,
    #[serde(flatten)]
    pub additional_fields: Object,
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
            auth: AuthSettings::default(),
            database: DatabaseSettings::default(),
            spend: SpendSettings::default(),
            limits: LimitsSettings::default(),
            health: HealthSettings::default(),
            alerting: AlertingSettings::default(),
            ui: UiSettings::default(),
            objects: ObjectsSettings::default(),
            requests: RequestsSettings::default(),
            files: FilesSettings::default(),
            mcp: McpSettings::default(),
            additional_fields: Object::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Alerting, Config, GeneralSettings, Spelled, UiAccessMode};

    fn general(yaml: &str) -> GeneralSettings {
        Config::from_yaml(&format!("general_settings:\n{yaml}"))
            .unwrap()
            .general_settings
    }

    fn text<T>(value: &str) -> Option<Spelled<T>> {
        Some(Spelled::Text(value.to_string()))
    }

    #[rstest]
    #[case::yaml_bool("true", Some(Spelled::Value(true)))]
    #[case::yaml_1_1_word("yes", text("yes"))]
    #[case::quoted("'false'", text("false"))]
    #[case::environment_reference("os.environ/ENABLE_JWT", text("os.environ/ENABLE_JWT"))]
    fn bool_settings_keep_every_python_spelling(
        #[case] spelling: &str,
        #[case] expected: Option<Spelled<bool>>,
    ) {
        let settings = general(&format!(
            "  enable_jwt_auth: {spelling}\n  disable_spend_logs: {spelling}\n  use_redis_transaction_buffer: {spelling}\n  model_list_healthy_only: {spelling}\n"
        ));
        assert_eq!(settings.auth.enable_jwt_auth, expected);
        assert_eq!(settings.spend.disable_spend_logs, expected);
        assert_eq!(settings.database.use_redis_transaction_buffer, expected);
        assert_eq!(settings.health.model_list_healthy_only, expected);
    }

    #[rstest]
    #[case::bare_days("7", Some(Spelled::Value(7)))]
    #[case::duration("'60d'", text("60d"))]
    #[case::environment_reference("os.environ/RETENTION", text("os.environ/RETENTION"))]
    fn retention_periods_accept_days_or_a_duration(
        #[case] spelling: &str,
        #[case] expected: Option<Spelled<i64>>,
    ) {
        let settings = general(&format!(
            "  maximum_spend_logs_retention_period: {spelling}\n"
        ));
        assert_eq!(settings.spend.maximum_spend_logs_retention_period, expected);
    }

    #[rstest]
    #[case::integer("5", Some(Spelled::Value(5.0)))]
    #[case::fraction("0.5", Some(Spelled::Value(0.5)))]
    #[case::environment_reference("os.environ/TTL", text("os.environ/TTL"))]
    fn number_settings_accept_integers_fractions_and_references(
        #[case] spelling: &str,
        #[case] expected: Option<Spelled<f64>>,
    ) {
        let settings = general(&format!(
            "  user_api_key_cache_ttl: {spelling}\n  database_statement_timeout: {spelling}\n  health_check_staleness_threshold: {spelling}\n"
        ));
        assert_eq!(settings.limits.user_api_key_cache_ttl, expected);
        assert_eq!(settings.database.database_statement_timeout, expected);
        assert_eq!(settings.health.health_check_staleness_threshold, expected);
    }

    #[test]
    fn nested_settings_are_typed_and_keep_unknown_keys() {
        let settings = general(
            r#"
  allowed_routes: [/chat/completions, /models]
  custom_auth: custom_auth.user_api_key_auth
  user_header_mappings:
    - header_name: X-OpenWebUI-User-Id
      litellm_user_role: internal_user
  role_permissions:
    - role: team
      models: [gpt]
      routes: [/chat/completions]
  service_account_settings:
    enforced_params: [user]
  key_management_settings:
    access_mode: read_and_write
    store_virtual_keys: true
    future_option: 1
  spend_logs_metadata_fields:
    include: [user_api_key_alias]
  spend_capture_rate_check:
    threshold: 0.8
    lookback_days: 3
  scheduled_job_stagger:
    window_seconds: 120
    offsets: {reset_budget_job: 0}
"#,
        );
        assert_eq!(
            settings.auth.allowed_routes.as_deref(),
            Some(["/chat/completions".to_string(), "/models".to_string()].as_slice())
        );
        assert_eq!(
            settings.auth.custom_auth.as_deref(),
            Some("custom_auth.user_api_key_auth")
        );
        let mapping = &settings.auth.user_header_mappings.as_deref().unwrap()[0];
        assert_eq!(mapping.header_name.as_deref(), Some("X-OpenWebUI-User-Id"));
        assert_eq!(mapping.litellm_user_role.as_deref(), Some("internal_user"));
        let permission = &settings.auth.role_permissions.as_deref().unwrap()[0];
        assert_eq!(permission.role.as_deref(), Some("team"));
        assert_eq!(
            permission.routes.as_deref(),
            Some(["/chat/completions".to_string()].as_slice())
        );
        assert_eq!(
            settings
                .auth
                .service_account_settings
                .as_ref()
                .unwrap()
                .enforced_params
                .as_deref(),
            Some(["user".to_string()].as_slice())
        );
        let key_management = settings.auth.key_management_settings.as_ref().unwrap();
        assert_eq!(
            key_management.access_mode.as_deref(),
            Some("read_and_write")
        );
        assert_eq!(
            key_management.store_virtual_keys,
            Some(Spelled::Value(true))
        );
        assert_eq!(
            settings
                .spend
                .spend_logs_metadata_fields
                .as_ref()
                .unwrap()
                .include
                .as_deref(),
            Some(["user_api_key_alias".to_string()].as_slice())
        );
        let capture = settings.spend.spend_capture_rate_check.as_ref().unwrap();
        assert_eq!(capture.threshold, Some(Spelled::Value(0.8)));
        assert_eq!(capture.lookback_days, Some(Spelled::Value(3)));
        let stagger = settings.spend.scheduled_job_stagger.as_ref().unwrap();
        assert_eq!(stagger.window_seconds, Some(Spelled::Value(120)));
        assert!(
            stagger
                .offsets
                .as_ref()
                .unwrap()
                .contains_key("reset_budget_job")
        );
        assert!(settings.additional_fields.is_empty());
    }

    #[rstest]
    #[case::license("litellm_license: SENTINEL")]
    #[case::drain_token("drain_endpoint_token: SENTINEL")]
    #[case::jwt_auth_object("litellm_jwtauth: {team_id_jwt_field: SENTINEL}")]
    #[case::oauth2_mappings("oauth2_config_mappings: {X-Header: SENTINEL}")]
    #[case::database_params("database_extra_connection_params: {sslpassword: SENTINEL}")]
    #[case::key_management_token("key_management_settings: {aws_web_identity_token: SENTINEL}")]
    #[case::key_management_external_id("key_management_settings: {aws_external_id: SENTINEL}")]
    #[case::key_management_hosted_keys("key_management_settings: {hosted_keys: [SENTINEL]}")]
    #[case::key_management_tags("key_management_settings: {tags: {owner: SENTINEL}}")]
    #[case::mcp_session_signing("mcp_session_token_signing: {private_key: SENTINEL}")]
    #[case::master_key("master_key: SENTINEL")]
    #[case::database_url("database_url: postgres://user:SENTINEL@db/litellm")]
    #[case::coordination_redis("coordination_redis: {password: SENTINEL}")]
    #[case::plugins("plugins: [{token: SENTINEL}]")]
    #[case::unknown_setting("future_setting: SENTINEL")]
    fn debug_output_does_not_expose_setting_values(#[case] setting: &str) {
        let config = Config::from_yaml(&format!("general_settings:\n  {setting}\n")).unwrap();
        let printed = format!("{config:?}");
        assert!(!printed.contains("SENTINEL"), "{printed}");
    }

    #[test]
    fn unknown_keys_still_land_in_additional_fields() {
        let settings = general("  future_setting: 1\n  custom_auth: auth.py\n");
        assert!(settings.additional_fields.contains_key("future_setting"));
        assert!(!settings.additional_fields.contains_key("custom_auth"));
    }

    #[rstest]
    #[case("true", Spelled::Value(true))]
    #[case("\"false\"", Spelled::Text("false".to_string()))]
    #[case("yes", Spelled::Text("yes".to_string()))]
    #[case("os.environ/GATEWAY_FLAG", Spelled::Text("os.environ/GATEWAY_FLAG".to_string()))]
    fn bool_spellings_land_in_typed_field(#[case] value: &str, #[case] expected: Spelled<bool>) {
        let settings = general(&format!("  enable_claude_code_gateway: {value}\n"));
        assert_eq!(settings.requests.enable_claude_code_gateway, Some(expected));
    }

    #[rstest]
    #[case("100", Spelled::Value(100))]
    #[case("os.environ/MAX_FILE_MB", Spelled::Text("os.environ/MAX_FILE_MB".to_string()))]
    fn int_spellings_land_in_typed_field(#[case] value: &str, #[case] expected: Spelled<i64>) {
        let settings = general(&format!("  max_file_size_mb: {value}\n"));
        assert_eq!(settings.files.max_file_size_mb, Some(expected));
    }

    #[rstest]
    #[case("600", Spelled::Value(600.0))]
    #[case("2.5", Spelled::Value(2.5))]
    #[case("os.environ/ALERT_THRESHOLD", Spelled::Text("os.environ/ALERT_THRESHOLD".to_string()))]
    fn float_spellings_land_in_typed_field(#[case] value: &str, #[case] expected: Spelled<f64>) {
        let settings = general(&format!("  alerting_threshold: {value}\n"));
        assert_eq!(settings.alerting.alerting_threshold, Some(expected));
    }

    #[rstest]
    fn nested_mappings_land_in_typed_structs() {
        let settings = general(
            r#"
  alerting: [slack, email]
  alerting_args:
    daily_report_frequency: 3600
    log_to_console: true
    spend_anomaly_multiplier: 2.5
    future_arg: 1
  team_metadata_schema:
    - key: cost_center
      label: Cost center
  request_tag_mappings:
    - header: baggage
      key: tenant
      tag_prefix: tenant-
      required: true
  pass_through_endpoints:
    - path: /bria
      target: https://engine.prod.bria-api.com
      include_subpath: true
      cost_per_request: 0.5
      methods: [GET, POST]
      custom_handler: handler.fn
  allowed_file_extensions: [jsonl, csv]
"#,
        );

        assert_eq!(
            settings.alerting.alerting,
            Some(Alerting::List(
                ["slack".to_string(), "email".to_string()].into()
            ))
        );
        let args = settings.alerting.alerting_args.unwrap();
        assert_eq!(args.daily_report_frequency, Some(Spelled::Value(3600)));
        assert_eq!(args.log_to_console, Some(Spelled::Value(true)));
        assert_eq!(args.spend_anomaly_multiplier, Some(Spelled::Value(2.5)));

        let schema = settings.objects.team_metadata_schema.unwrap();
        assert_eq!(schema[0].key.as_deref(), Some("cost_center"));
        assert_eq!(schema[0].label.as_deref(), Some("Cost center"));

        let mapping = &settings.requests.request_tag_mappings.unwrap()[0];
        let header: &'static str = mapping.header.unwrap().into();
        assert_eq!(header, "baggage");
        assert_eq!(mapping.key.as_deref(), Some("tenant"));
        assert_eq!(mapping.tag_prefix.as_deref(), Some("tenant-"));
        assert_eq!(mapping.required, Some(Spelled::Value(true)));

        let endpoint = &settings.requests.pass_through_endpoints.unwrap()[0];
        assert_eq!(endpoint.path.as_deref(), Some("/bria"));
        assert_eq!(
            endpoint.target.as_ref().unwrap().expose(),
            "https://engine.prod.bria-api.com"
        );
        assert_eq!(endpoint.include_subpath, Some(Spelled::Value(true)));
        assert_eq!(endpoint.cost_per_request, Some(Spelled::Value(0.5)));
        assert_eq!(
            endpoint.methods.as_deref(),
            Some(&["GET".to_string(), "POST".to_string()][..])
        );

        assert_eq!(
            settings.files.allowed_file_extensions.as_deref(),
            Some(&["jsonl".to_string(), "csv".to_string()][..])
        );
    }

    #[rstest]
    fn ui_access_mode_accepts_a_mode_string() {
        let settings = general("  ui_access_mode: admin_only\n");
        assert!(
            matches!(settings.ui.ui_access_mode, Some(UiAccessMode::Mode(mode)) if mode == "admin_only")
        );
    }

    #[rstest]
    fn ui_access_mode_accepts_an_sso_group_mapping() {
        let settings = general(
            "  ui_access_mode: {type: restricted_sso_group, restricted_sso_group: ai-admins, sso_group_jwt_field: groups}\n",
        );
        let Some(UiAccessMode::SsoGroup(group)) = settings.ui.ui_access_mode else {
            panic!("expected an SSO group mapping")
        };
        assert_eq!(group.kind.as_deref(), Some("restricted_sso_group"));
        assert_eq!(group.restricted_sso_group.as_deref(), Some("ai-admins"));
        assert_eq!(group.sso_group_jwt_field.as_deref(), Some("groups"));
    }

    #[rstest]
    fn alerting_accepts_a_single_name() {
        let settings = general("  alerting: slack\n");
        assert_eq!(
            settings.alerting.alerting,
            Some(Alerting::One("slack".to_string()))
        );
    }

    #[rstest]
    fn rejects_request_tag_header_python_rejects_at_boot() {
        assert!(
        Config::from_yaml(
            "general_settings:\n  request_tag_mappings: [{header: x-tenant, key: tenant, tag: t}]\n"
        )
        .is_err()
    );
    }

    #[rstest]
    #[case("alert_to_webhook_url: {budget_alerts: SENTINEL}", |settings: &GeneralSettings| format!("{:?}", settings.alerting))]
    #[case("alert_type_config: {budget_alerts: {webhook: SENTINEL}}", |settings: &GeneralSettings| format!("{:?}", settings.alerting))]
    #[case("alerting_args: {webhook: SENTINEL}", |settings: &GeneralSettings| format!("{:?}", settings.alerting))]
    #[case("pass_through_endpoints: [{path: /x, target: SENTINEL}]", |settings: &GeneralSettings| format!("{:?}", settings.requests))]
    #[case("pass_through_endpoints: [{path: /x, target: t, headers: {Authorization: SENTINEL}}]", |settings: &GeneralSettings| format!("{:?}", settings.requests))]
    #[case("claude_code_gateway_managed_settings: {token: SENTINEL}", |settings: &GeneralSettings| format!("{:?}", settings.requests))]
    #[case("search_tools: [{litellm_params: {api_key: SENTINEL}}]", |settings: &GeneralSettings| format!("{:?}", settings.objects))]
    #[case("max_failed_login_attempts_per_source_overrides: {10.0.0.1: SENTINEL}", |settings: &GeneralSettings| format!("{:?}", settings.ui))]
    fn debug_hides_values_that_can_carry_secrets(
        #[case] entry: &str,
        #[case] render: fn(&GeneralSettings) -> String,
    ) {
        let settings = general(&format!("  {entry}\n"));
        assert!(!render(&settings).contains("SENTINEL"));
    }

    #[rstest]
    fn unknown_key_still_lands_in_additional_fields() {
        let settings = general("  future_setting: 1\n  forward_openai_org_id: true\n");
        assert!(settings.additional_fields.contains_key("future_setting"));
        assert!(
            !settings
                .additional_fields
                .contains_key("forward_openai_org_id")
        );
        assert_eq!(
            settings.requests.forward_openai_org_id,
            Some(Spelled::Value(true))
        );
    }

    #[rstest]
    fn mcp_allowed_hosts_and_origins_land_typed() {
        let settings = general(
            "  mcp_allowed_hosts: [localhost]\n  mcp_allowed_origins: ['https://app.test']\n",
        );
        assert_eq!(
            settings.mcp.mcp_allowed_hosts.as_deref(),
            Some(&["localhost".to_string()][..])
        );
        assert_eq!(
            settings.mcp.mcp_allowed_origins.as_ref(),
            ["https://app.test".to_string()]
        );
    }

    #[rstest]
    #[case::literal("secret-master-key")]
    #[case::reference("os.environ/LITELLM_MASTER_KEY")]
    fn loads_and_redacts_the_master_key(#[case] key: &str) {
        let config = Config::from_yaml(&format!(
            "model_list: []\ngeneral_settings:\n  master_key: {key}\n"
        ))
        .unwrap();
        assert_eq!(
            config
                .general_settings
                .master_key
                .as_ref()
                .unwrap()
                .expose(),
            key
        );
        assert!(!format!("{config:?}").contains(key));
    }

    #[rstest]
    fn missing_general_settings_has_no_master_key() {
        let config = Config::from_yaml("model_list: []").unwrap();
        assert!(config.general_settings.master_key.is_none());
    }
}
