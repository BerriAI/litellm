use litellm_config_load::Error;
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

    let config = litellm_config_load::load(path).unwrap();
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
fn distinguishes_read_errors_from_parse_errors(directory: TempDir) {
    let error = litellm_config_load::load(directory.path().join("missing.yaml")).unwrap_err();
    assert!(
        litellm_error::Report::new(&error)
            .to_string()
            .contains("missing.yaml")
    );
    assert!(matches!(
        error,
        Error::Read { source, .. } if source.kind() == std::io::ErrorKind::NotFound
    ));
}

#[rstest]
#[case::policy_pipeline("../../../proxy/example_config_yaml/test_pipeline_config.yaml")]
#[case::gateway("../../../../tests/e2e/gateway/stage_mirror_ci_config.yml")]
#[case::pass_through("../../../proxy/example_config_yaml/pass_through_config.yaml")]
fn parses_representative_python_configs(#[case] relative_path: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../config")
        .join(relative_path);
    let config = litellm_config_load::load(path).unwrap();
    assert!(!config.model_list.is_empty());
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

    let config = litellm_config_load::load(root.join("config.yaml")).unwrap();
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
