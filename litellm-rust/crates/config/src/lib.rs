macro_rules_attribute::attribute_alias! {
    #[apply(config_section)] =
        #[derive(Clone, Debug, Default, ::serde::Deserialize)]
        #[serde(default)];
}

mod agent;
mod callback;
mod credential;
mod error;
mod general;
mod guardrail;
mod litellm;
mod mcp;
mod model;
mod policy;
mod prompt;
mod provider;
mod router;
mod tool;
mod value;
mod vector_store;
pub mod yaml;

use std::collections::BTreeMap;

pub use agent::Agent;
pub use callback::CallbackSettings;
pub use credential::Credential;
pub use error::Error;
pub use general::{
    Alerting, AlertingArgs, AlertingSettings, AuthSettings, ClickHouseStoreSettings,
    DatabaseSettings, FilesSettings, GeneralSettings, HealthSettings, KeyManagementSettings,
    LimitsSettings, McpSettings, ObjectsSettings, PassThroughEndpoint, RequestTagHeader,
    RequestTagMapping, RequestsSettings, RoleBasedPermissions, ScheduledJobStaggerSettings,
    ServiceAccountSettings, SpendCaptureRateCheckSettings, SpendLogsMetadataFields, SpendSettings,
    TeamMetadataField, TracingSettings, TracingStoreSettings, UiAccessMode, UiAccessSsoGroup,
    UiSettings, UserHeaderMapping,
};
pub use guardrail::{Guardrail, GuardrailParams};
pub use litellm::{
    AiohttpSettings, AzureBlobCacheParams, BackgroundMode, BudgetSettings, CacheParams,
    CachingSettings, CallbackVar, CallerIdentity, ClientCertificateSettings, CustomProvider,
    DiskCacheParams, FallbackSettings, GcsCacheParams, Http2Settings, LiteLlmSettings,
    LoggingSettings, McpToolSettings, ModelGroupSettings, PriorityReservationSettings,
    ProxyCacheTtlParams, QdrantCacheParams, RedisCacheParams, RedisServerCacheParams,
    ResponsesSettings, S3CacheParams, SemanticCacheParams, SocketSettings, TeamDefaultSettings,
    TlsSettings, UpperboundKeyGenerateParams, ValkeyCacheParams,
};
pub use litellm_python_compat::read::Spelled;
pub use litellm_router_types::LitellmParams;
pub use mcp::{McpAuth, McpServer, McpTransport};
pub use model::{AccessWindow, Model, ModelId, ModelInfo};
pub use policy::{
    Policy, PolicyAttachment, PolicyCondition, PolicyGuardrailChanges, PolicyGuardrails,
    PolicyPipeline, PolicyPipelineStep,
};
pub use prompt::{Prompt, PromptInfo, PromptParams};
pub use provider::{AssistantSettings, ProviderSettings, VertexConfig, Worker};
pub use router::{AlertingConfig, AllowedFailsPolicy, ProviderBudget, RouterSettings};
pub use tool::{SandboxTool, SandboxToolParams, SearchTool, SearchToolInfo, SearchToolParams};
pub use value::{Object, Raw, Value};
pub use vector_store::{VectorStore, VectorStoreParams};

#[macro_rules_attribute::apply(config_section)]
pub struct Config {
    pub model_list: Box<[Model]>,
    pub general_settings: GeneralSettings,
    pub router_settings: RouterSettings,
    pub litellm_settings: LiteLlmSettings,
    pub environment_variables: Object,
    pub callback_settings: BTreeMap<String, CallbackSettings>,
    pub assistant_settings: Option<AssistantSettings>,
    pub default_vertex_config: Option<VertexConfig>,
    pub mcp_servers: BTreeMap<String, McpServer>,
    pub credential_list: Box<[Credential]>,
    pub guardrails: Box<[Guardrail]>,
    pub prompts: Box<[Prompt]>,
    pub sandbox_tools: Box<[SandboxTool]>,
    pub search_tools: Box<[SearchTool]>,
    pub files_settings: Box<[ProviderSettings]>,
    pub finetune_settings: Box<[ProviderSettings]>,
    pub mcp_tools: Box<[Object]>,
    pub vector_store_registry: Box<[VectorStore]>,
    pub worker_registry: Box<[Worker]>,
    pub agents: Box<[Agent]>,
    pub agent_list: Box<[Agent]>,
    pub policies: BTreeMap<String, Policy>,
    pub policy_attachments: Box<[PolicyAttachment]>,
    pub include: Box<[String]>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

impl Config {
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        Self::from_mapping(yaml::mapping(yaml)?)
    }

    pub fn from_mapping(document: serde_yaml_ng::Mapping) -> Result<Self, Error> {
        Ok(serde_yaml_ng::from_value(serde_yaml_ng::Value::Mapping(
            document,
        ))?)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Config, Error, Spelled};

    #[rstest]
    fn config_debug_redacts_api_keys() {
        let config = Config::from_yaml(
        "model_list: [{model_name: assistant, litellm_params: {model: anthropic/test-model, api_key: secret-value}}]",
    )
    .unwrap();
        assert_eq!(
            config.model_list[0]
                .litellm_params
                .api_key
                .as_ref()
                .unwrap()
                .expose(),
            "secret-value"
        );
        assert!(!format!("{config:?}").contains("secret-value"));
    }

    #[rstest]
    #[case::malformed_yaml("model_list: [")]
    #[case::missing_params("model_list: [{model_name: assistant}]")]
    #[case::missing_model("model_list: [{model_name: assistant, litellm_params: {api_key: key}}]")]
    #[case::provider_param_of_the_wrong_type(
        "model_list: [{model_name: assistant, litellm_params: {model: test, aws_region_name: [eu-central-1]}}]"
    )]
    #[case::negative_model_rpm(
        "model_list: [{model_name: assistant, rpm: -1, litellm_params: {model: test}}]"
    )]
    fn rejects_malformed_and_incomplete_config(#[case] yaml: &str) {
        let error = Config::from_yaml(yaml).unwrap_err();
        assert!(matches!(error, Error::Parse(_)), "{error:?}");
    }

    #[rstest]
    fn empty_config_matches_python_defaults() {
        let config = Config::from_yaml("{}").unwrap();
        assert!(config.model_list.is_empty());
        assert_eq!(config.general_settings.admission_queue_timeout_seconds, 1.0);
        assert_eq!(
            config.general_settings.database_connection_pool_limit,
            Some(10)
        );
        assert_eq!(
            config.general_settings.proxy_config_reload_interval_seconds,
            30
        );
        assert_eq!(config.general_settings.health_check_interval, 300);
    }

    #[rstest]
    fn parses_typed_settings_and_preserves_extension_fields() {
        let config = Config::from_yaml(
            r#"
model_list:
  - model_name: assistant
    litellm_params:
      model: vertex_ai/test-model
      timeout: os.environ/REQUEST_TIMEOUT
      tpm: os.environ/TPM_LIMIT
      rpm: 5
      drop_params: "true"
      vertex_project: test-project
      vertex_credentials:
        type: service_account
      future_param: 1
    model_info:
      mode: chat
    access_groups: [internal]
general_settings:
  master_key: secret-master-key
  store_model_in_db: true
  custom_auth: auth.py
router_settings:
  routing_strategy: simple-shuffle
  allowed_fails: 2
  future_router_setting: cache.internal
litellm_settings:
  drop_params: true
  cache: true
  custom_callback_name: audit
future_section:
  enabled: true
"#,
        )
        .unwrap();

        let model = &config.model_list[0];
        assert_eq!(
            model.litellm_params.timeout,
            Some(Spelled::Text("os.environ/REQUEST_TIMEOUT".to_string()))
        );
        assert_eq!(
            model.litellm_params.tpm,
            Some(Spelled::Text("os.environ/TPM_LIMIT".to_string()))
        );
        assert_eq!(model.litellm_params.rpm, Some(Spelled::Value(5)));
        assert_eq!(
            model.litellm_params.drop_params,
            Some(Spelled::Text("true".to_string()))
        );
        assert_eq!(
            model.litellm_params.vertex.vertex_project.as_deref(),
            Some("test-project")
        );
        assert_eq!(
            model.litellm_params.vertex.vertex_credentials.as_deref(),
            Some(r#"{"type":"service_account"}"#)
        );
        assert!(model.litellm_params.extra.contains_key("future_param"));
        assert!(model.additional_fields.contains_key("access_groups"));
        assert_eq!(config.router_settings.allowed_fails, Some(2));
        assert!(
            config
                .router_settings
                .additional_fields
                .contains_key("future_router_setting")
        );
        assert!(
            config
                .litellm_settings
                .additional_fields
                .contains_key("custom_callback_name")
        );
        assert!(config.additional_fields.contains_key("future_section"));
    }

    #[rstest]
    fn parses_python_config_sections() {
        let config = Config::from_yaml(
            r#"
environment_variables:
  REDIS_PORT: 6379
callback_settings:
  otel:
    message_logging: false
assistant_settings:
  custom_llm_provider: openai
credential_list:
  - credential_name: bedrock
    credential_values:
      aws_region_name: us-east-1
guardrails:
  - guardrail_name: pii
    litellm_params:
      guardrail: presidio
prompts:
  - prompt_id: support
sandbox_tools:
  - sandbox_tool_name: e2b
search_tools:
  - search_tool_name: web
files_settings:
  - custom_llm_provider: openai
finetune_settings:
  - custom_llm_provider: openai
mcp_tools:
  - name: lookup
mcp_servers:
  docs:
    url: https://example.test/mcp
vector_store_registry:
  - vector_store_name: docs
worker_registry:
  - worker_id: regional
agents:
  - agent_name: reviewer
agent_list:
  - agent_name: legacy-reviewer
policies:
  safe:
    guardrails:
      add: [pii]
policy_attachments:
  - policy_id: safe
"#,
        )
        .unwrap();

        assert_eq!(config.environment_variables.len(), 1);
        assert_eq!(config.credential_list.len(), 1);
        assert_eq!(config.guardrails.len(), 1);
        assert_eq!(config.prompts.len(), 1);
        assert_eq!(config.sandbox_tools.len(), 1);
        assert_eq!(config.search_tools.len(), 1);
        assert_eq!(config.files_settings.len(), 1);
        assert_eq!(config.finetune_settings.len(), 1);
        assert_eq!(config.mcp_tools.len(), 1);
        assert_eq!(config.mcp_servers.len(), 1);
        assert_eq!(config.vector_store_registry.len(), 1);
        assert_eq!(config.worker_registry.len(), 1);
        assert_eq!(config.agents.len(), 1);
        assert_eq!(config.agent_list.len(), 1);
        assert!(config.policies.contains_key("safe"));
        assert_eq!(config.policy_attachments.len(), 1);
    }

    #[rstest]
    #[case::api_key("api_key", "provider-secret")]
    #[case::aws_secret("aws_secret_access_key", "aws-secret")]
    #[case::vertex_credentials("vertex_credentials", "vertex-secret")]
    #[case::provider_extension("azure_ad_token", "azure-secret")]
    #[case::general_extension("custom_auth_secret", "auth-secret")]
    #[case::router_extension("future_router_token", "router-secret")]
    #[case::litellm_extension("callback_token", "callback-secret")]
    #[case::root_extension("private_token", "root-secret")]
    fn debug_output_does_not_expose_config_values(#[case] field: &str, #[case] secret: &str) {
        let yaml = match field {
            "api_key" => format!(
                "model_list: [{{model_name: assistant, litellm_params: {{model: test, api_key: {secret}}}}}]"
            ),
            "aws_secret_access_key" => format!(
                "model_list: [{{model_name: assistant, litellm_params: {{model: test, aws_secret_access_key: {secret}}}}}]"
            ),
            "vertex_credentials" => format!(
                "model_list: [{{model_name: assistant, litellm_params: {{model: test, vertex_credentials: {secret}}}}}]"
            ),
            "azure_ad_token" => format!(
                "model_list: [{{model_name: assistant, litellm_params: {{model: test, azure_ad_token: {secret}}}}}]"
            ),
            "custom_auth_secret" => format!("general_settings: {{{field}: {secret}}}"),
            "future_router_token" => format!("router_settings: {{{field}: {secret}}}"),
            "callback_token" => format!("litellm_settings: {{{field}: {secret}}}"),
            "private_token" => format!("{field}: {secret}"),
            _ => unreachable!(),
        };
        let config = Config::from_yaml(&yaml).unwrap();
        assert!(!format!("{config:?}").contains(secret));
    }

    #[rstest]
    fn unknown_top_level_key_lands_in_additional_fields() {
        let config = Config::from_yaml("future_section:\n  a: 1\nenable_jwt_auth: true\n").unwrap();

        assert!(config.additional_fields.contains_key("future_section"));
        assert!(config.additional_fields.contains_key("enable_jwt_auth"));
    }
}
