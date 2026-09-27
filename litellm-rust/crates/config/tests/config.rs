use litellm_config::{Config, Error, Flag, NumberOrString};
use rstest::{fixture, rstest};
use tempfile::TempDir;

#[fixture]
fn directory() -> TempDir {
    tempfile::tempdir().unwrap()
}

#[fixture]
fn model_list_yaml() -> &'static str {
    r#"
model_list:
  - model_name: assistant
    litellm_params:
      model: anthropic/test-model
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: local
    litellm_params:
      model: test-model
      api_base: http://localhost:8000/v1
      custom_llm_provider: openai
"#
}

#[rstest]
fn loads_model_list_from_file(directory: TempDir, model_list_yaml: &str) {
    let path = directory.path().join("config.yaml");
    std::fs::write(&path, model_list_yaml).unwrap();

    let config = Config::load(path).unwrap();
    assert_eq!(config.model_list.len(), 2);
    let anthropic = &config.model_list[0];
    assert_eq!(anthropic.model_name, "assistant");
    assert_eq!(anthropic.litellm_params.model, "anthropic/test-model");
    assert_eq!(
        anthropic.litellm_params.api_key.as_ref().unwrap().expose(),
        "os.environ/ANTHROPIC_API_KEY"
    );
    assert!(anthropic.litellm_params.api_base.is_none());
    assert!(anthropic.litellm_params.custom_llm_provider.is_none());
    let local = &config.model_list[1];
    assert_eq!(local.model_name, "local");
    assert_eq!(local.litellm_params.model, "test-model");
    assert!(local.litellm_params.api_key.is_none());
    assert_eq!(
        local.litellm_params.api_base.as_deref(),
        Some("http://localhost:8000/v1")
    );
    assert_eq!(
        local.litellm_params.custom_llm_provider.as_deref(),
        Some("openai")
    );
}

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
fn rejects_malformed_and_incomplete_config(#[case] yaml: &str) {
    assert!(matches!(Config::from_yaml(yaml), Err(Error::Parse(_))));
}

#[rstest]
fn distinguishes_read_errors_from_parse_errors(directory: TempDir) {
    assert!(matches!(
        Config::load(directory.path().join("missing.yaml")),
        Err(Error::Read(error)) if error.kind() == std::io::ErrorKind::NotFound
    ));
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
      drop_params: "true"
      vertex_project: test-project
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
  redis_host: cache.internal
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
        Some(NumberOrString::String(
            "os.environ/REQUEST_TIMEOUT".to_string()
        ))
    );
    assert_eq!(
        model.litellm_params.drop_params,
        Some(Flag::String("true".to_string()))
    );
    assert!(
        model
            .litellm_params
            .additional_fields
            .contains_key("vertex_project")
    );
    assert!(model.additional_fields.contains_key("access_groups"));
    assert_eq!(config.router_settings.allowed_fails, Some(2));
    assert!(
        config
            .router_settings
            .additional_fields
            .contains_key("redis_host")
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
include:
  - models.yaml
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
    assert_eq!(config.vector_store_registry.len(), 1);
    assert_eq!(config.worker_registry.len(), 1);
    assert_eq!(config.agents.len(), 1);
    assert_eq!(config.agent_list.len(), 1);
    assert!(config.policies.contains_key("safe"));
    assert_eq!(config.policy_attachments.len(), 1);
    assert_eq!(&*config.include, &["models.yaml"]);
}

#[rstest]
#[case::policy_pipeline("../../../litellm/proxy/example_config_yaml/test_pipeline_config.yaml")]
#[case::gateway("../../../tests/e2e/gateway/stage_mirror_ci_config.yml")]
fn parses_representative_python_configs(#[case] relative_path: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative_path);
    let config = Config::load(path).unwrap();
    assert!(!config.model_list.is_empty());
}

#[rstest]
#[case::one("callbacks: custom_callbacks.logger", 1)]
#[case::many("callbacks: [prometheus, otel]", 2)]
fn accepts_python_callback_shorthand(#[case] setting: &str, #[case] expected_len: usize) {
    let config = Config::from_yaml(&format!("litellm_settings:\n  {setting}")).unwrap();
    assert_eq!(
        config.litellm_settings.callbacks.as_ref().unwrap().len(),
        expected_len
    );
}

#[rstest]
#[case::api_key("api_key", "provider-secret")]
#[case::provider_extension("aws_secret_access_key", "aws-secret")]
#[case::general_extension("custom_auth_secret", "auth-secret")]
#[case::router_extension("redis_password", "redis-secret")]
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
        "custom_auth_secret" => format!("general_settings: {{{field}: {secret}}}"),
        "redis_password" => format!("router_settings: {{{field}: {secret}}}"),
        "callback_token" => format!("litellm_settings: {{{field}: {secret}}}"),
        "private_token" => format!("{field}: {secret}"),
        _ => unreachable!(),
    };
    let config = Config::from_yaml(&yaml).unwrap();
    assert!(!format!("{config:?}").contains(secret));
}

#[rstest]
fn resolves_nested_includes_once_in_breadth_first_order() {
    let directory = TempDir::new().unwrap();
    let root = directory.path();
    std::fs::create_dir(root.join("nested")).unwrap();
    std::fs::write(root.join("config.yaml"), "include: [nested/first.yaml, second.yaml]\nmodel_list: [{model_name: root, litellm_params: {model: root}}]\n").unwrap();
    std::fs::write(root.join("nested/first.yaml"), "include: [third.yaml]\nmodel_list: [{model_name: first, litellm_params: {model: first}}]\n").unwrap();
    std::fs::write(root.join("second.yaml"), "general_settings: {master_key: second}\nmodel_list: [{model_name: second, litellm_params: {model: second}}]\n").unwrap();
    std::fs::write(root.join("nested/third.yaml"), "include: [../config.yaml]\ngeneral_settings: {master_key: third}\nmodel_list: [{model_name: third, litellm_params: {model: third}}]\n").unwrap();

    let config = Config::load(root.join("config.yaml")).unwrap();
    assert_eq!(
        config
            .model_list
            .iter()
            .map(|model| model.model_name.as_str())
            .collect::<Vec<_>>(),
        ["root", "first", "second", "third"]
    );
    assert_eq!(
        config.general_settings.master_key.unwrap().expose(),
        "third"
    );
    assert!(config.include.is_empty());
}
