use std::sync::Arc;

use litellm_config::Spelled;
use litellm_config_load::Error;
use litellm_secrets::{OidcResolver, SecretManagerState, SecretResolver};
use rstest::{fixture, rstest};
use tempfile::TempDir;

#[fixture]
fn secrets() -> SecretResolver {
    SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::default()),
        Arc::new(|name: &str| (name == "ANTHROPIC_API_KEY").then(|| "sk-from-env".to_owned())),
        OidcResolver::new(litellm_http::Client::for_test()),
    )
}

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
#[tokio::test]
async fn loads_model_list_from_file(
    directory: TempDir,
    model_list_yaml: &str,
    secrets: SecretResolver,
) {
    let path = directory.path().join("config.yaml");
    std::fs::write(&path, model_list_yaml).unwrap();

    let config = litellm_config_load::load(path, secrets).await.unwrap();
    assert_eq!(config.model_list.len(), 2);
    let anthropic = &config.model_list[0];
    assert_eq!(anthropic.model_name, "assistant");
    assert_eq!(anthropic.litellm_params.model, "anthropic/test-model");
    assert_eq!(
        anthropic.litellm_params.api_key.as_ref().unwrap().expose(),
        "sk-from-env"
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
#[tokio::test]
async fn distinguishes_read_errors_from_parse_errors(directory: TempDir, secrets: SecretResolver) {
    let error = litellm_config_load::load(directory.path().join("missing.yaml"), secrets)
        .await
        .unwrap_err();
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
#[tokio::test]
async fn parses_representative_python_configs(
    #[case] relative_path: &str,
    secrets: SecretResolver,
) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../config")
        .join(relative_path);
    let config = litellm_config_load::load(path, secrets).await.unwrap();
    assert!(!config.model_list.is_empty());
}

#[rstest]
#[tokio::test]
async fn resolves_nested_includes_once_in_breadth_first_order(secrets: SecretResolver) {
    let directory = TempDir::new().unwrap();
    let root = directory.path();
    std::fs::create_dir(root.join("nested")).unwrap();
    std::fs::write(root.join("config.yaml"), "include: [nested/first.yaml, second.yaml]\nmodel_list: [{model_name: root, litellm_params: {model: root}}]\n").unwrap();
    std::fs::write(root.join("nested/first.yaml"), "include: [third.yaml]\nmodel_list: [{model_name: first, litellm_params: {model: first}}]\n").unwrap();
    std::fs::write(root.join("second.yaml"), "general_settings: {master_key: second}\nmodel_list: [{model_name: second, litellm_params: {model: second}}]\n").unwrap();
    std::fs::write(root.join("nested/third.yaml"), "include: [../config.yaml]\ngeneral_settings: {master_key: third}\nmodel_list: [{model_name: third, litellm_params: {model: third}}]\n").unwrap();

    let config = litellm_config_load::load(root.join("config.yaml"), secrets)
        .await
        .unwrap();
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

#[rstest]
#[tokio::test]
async fn a_key_the_config_leaves_unset_is_read_from_a_variable_it_exports(
    directory: TempDir,
    secrets: SecretResolver,
) {
    let path = directory.path().join("config.yaml");
    std::fs::write(
        &path,
        "environment_variables:\n  LITELLM_DROP_PARAMS: \"true\"\n",
    )
    .unwrap();

    let config = litellm_config_load::load(path, secrets).await.unwrap();

    assert_eq!(
        (
            config.litellm_settings.drop_params,
            config.origins.variable("litellm_settings.drop_params")
        ),
        (Some(true), Some("LITELLM_DROP_PARAMS"))
    );
}

#[rstest]
#[case::only_the_config_exports_it(None, None)]
#[case::the_process_sets_it_too(Some("5"), Some(5.0))]
#[tokio::test]
async fn a_request_timeout_is_read_from_the_process_and_never_from_what_the_config_exports(
    directory: TempDir,
    #[case] process: Option<&'static str>,
    #[case] seconds: Option<f64>,
) {
    let path = directory.path().join("config.yaml");
    std::fs::write(&path, "environment_variables:\n  REQUEST_TIMEOUT: \"2\"\n").unwrap();
    let secrets = SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::default()),
        Arc::new(move |name: &str| {
            (name == "REQUEST_TIMEOUT")
                .then_some(process)
                .flatten()
                .map(str::to_owned)
        }),
        OidcResolver::new(litellm_http::Client::for_test()),
    );

    let config = litellm_config_load::load(path, secrets).await.unwrap();

    assert_eq!(
        config.litellm_settings.request_timeout,
        seconds.map(Spelled::Value)
    );
}
