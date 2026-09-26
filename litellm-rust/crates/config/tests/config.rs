use litellm_config::{Config, Error};
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
#[case::missing_model_list("{}")]
#[case::missing_params("model_list: [{model_name: assistant}]")]
#[case::missing_model("model_list: [{model_name: assistant, litellm_params: {api_key: key}}]")]
#[case::unsupported_settings("model_list: []\ngeneral_settings: {unknown: true}")]
#[case::misspelled_param(
    "model_list: [{model_name: assistant, litellm_params: {model: test, api_bsae: url}}]"
)]
fn rejects_malformed_incomplete_and_unsupported_config(#[case] yaml: &str) {
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
