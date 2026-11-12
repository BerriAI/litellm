use std::time::Duration;

use litellm_config::Config;
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_model_catalog::Catalog;
use litellm_router::{Deployment, Router};
use rstest::rstest;

const MODELS: &str = r#"{
    "no-sampling": {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": false},
    "vertex_ai/routed-no-sampling": {"litellm_provider": "vertex_ai", "mode": "chat", "supports_sampling_params": false},
    "sampling": {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": true},
    "unflagged": {"litellm_provider": "anthropic", "mode": "chat"}
}"#;

fn models() -> Catalog {
    Catalog::parse(MODELS.as_bytes()).unwrap().value
}

#[rstest]
#[case::provider_prefixed("model: anthropic/no-sampling", false)]
#[case::custom_provider("model: no-sampling\n      custom_llm_provider: anthropic", false)]
#[case::routed_key("model: vertex_ai/routed-no-sampling", false)]
#[case::flagged_with_sampling("model: anthropic/sampling", true)]
#[case::unflagged("model: anthropic/unflagged", true)]
#[case::not_in_the_map("model: anthropic/not-in-the-map", true)]
fn sampling_support_comes_from_the_model_map(#[case] parameters: &str, #[case] expected: bool) {
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      {parameters}"
    ))
    .unwrap();

    assert_eq!(
        Router::from_config(&config, &|_: &str| None, &models())
            .get("public-model")
            .unwrap()
            .shaping
            .capabilities
            .supports_sampling_params,
        expected
    );
}

#[rstest]
#[case::unset("", &[])]
#[case::paths("additional_drop_params: [temperature, metadata.user_id]", &["temperature", "metadata.user_id"])]
#[case::only_text_entries("additional_drop_params: [temperature, 5, null]", &["temperature"])]
#[case::not_a_list("additional_drop_params: temperature", &[])]
fn deployment_additional_drop_params_reach_the_shaping(
    #[case] parameters: &str,
    #[case] expected: &[&str],
) {
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/no-sampling\n      {parameters}"
    ))
    .unwrap();

    assert_eq!(
        Router::from_config(&config, &|_: &str| None, &models())
            .get("public-model")
            .unwrap()
            .shaping
            .settings
            .additional_drop_params,
        expected
    );
}

#[rstest]
#[case::minimal("")]
#[case::configured(
    "api_key: test-key\n      api_base: https://provider.example/v1\n      custom_llm_provider: test-provider"
)]
#[case::secret_reference("api_key: os.environ/ROUTER_TEST_API_KEY")]
fn configuration_preserves_deployment_parameters(#[case] parameters: &str) {
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: provider/model\n      {parameters}"
    ))
    .unwrap();
    let router = Router::from_config(&config, &|_: &str| None, &models());
    let deployment = router.get(&config.model_list[0].model_name).unwrap();
    let params = &config.model_list[0].litellm_params;

    assert_eq!(deployment.model, params.model);
    assert_eq!(
        deployment.api_key.as_deref(),
        params.api_key.as_ref().map(|key| key.expose())
    );
    assert_eq!(deployment.api_base, params.api_base);
    assert_eq!(deployment.custom_llm_provider, params.custom_llm_provider);
    assert_eq!(deployment.timeout, Deployment::default().timeout);
    assert_eq!(deployment.shaping, Deployment::default().shaping);
    assert_eq!(deployment.litellm_params, *params);
}

#[rstest]
fn configuration_types_the_provider_params_of_a_deployment() {
    let config = Config::from_yaml(
        "model_list:
  - model_name: public-model
    litellm_params:
      model: bedrock/anthropic.claude-3
      aws_region_name: eu-central-1
      aws_bedrock_runtime_endpoint: https://runtime.example
      azure_ad_token: ignored-by-every-group",
    )
    .unwrap();
    let router = Router::from_config(&config, &|_: &str| None, &models());
    let params = &router.get("public-model").unwrap().litellm_params;

    assert_eq!(
        (
            params.aws.aws_region_name.as_deref(),
            params.aws.aws_bedrock_runtime_endpoint.as_deref(),
            params.aws.aws_access_key_id.as_deref(),
        ),
        (Some("eu-central-1"), Some("https://runtime.example"), None)
    );
}

#[rstest]
#[case::first("public-a", Some("provider/a"))]
#[case::second("public-b", Some("provider/b"))]
#[case::unknown("missing", None)]
#[case::provider_name_is_not_an_alias("provider/a", None)]
#[case::case_sensitive("PUBLIC-A", None)]
fn lookup_uses_public_names(#[case] name: &str, #[case] expected: Option<&str>) {
    let config = Config::from_yaml(
        "model_list:
  - model_name: public-a
    litellm_params:
      model: provider/a
  - model_name: public-b
    litellm_params:
      model: provider/b",
    )
    .unwrap();
    let router = Router::from_config(&config, &|_: &str| None, &models());

    assert_eq!(router.get(name).map(|entry| entry.model.as_str()), expected);
}

#[rstest]
fn empty_configuration_has_no_deployment() {
    let config = Config::from_yaml("model_list: []").unwrap();

    assert!(
        Router::from_config(&config, &|_: &str| None, &models())
            .get("")
            .is_none()
    );
    assert!(Router::default().get("unknown").is_none());
}

#[rstest]
fn programmatic_deployments_preserve_overrides_and_last_entry_wins() {
    let deployment = Deployment {
        model: "provider/selected".into(),
        api_key: Some("test-key".into()),
        api_base: Some("https://provider.example/v1".into()),
        custom_llm_provider: Some("test-provider".into()),
        litellm_params: Default::default(),
        timeout: Some(Duration::from_secs(7)),
        shaping: MessagesShaping {
            settings: MessagesSettings {
                drop_params: true,
                additional_drop_params: vec!["metadata.test".into()],
                ..MessagesSettings::default()
            },
            ..Default::default()
        },
    };
    let router = Router::from_iter([
        ("public-model".into(), Deployment::default()),
        ("public-model".into(), deployment.clone()),
    ]);
    let selected = router.get("public-model").unwrap();

    assert_eq!(selected.model, deployment.model);
    assert_eq!(selected.api_key, deployment.api_key);
    assert_eq!(selected.api_base, deployment.api_base);
    assert_eq!(selected.custom_llm_provider, deployment.custom_llm_provider);
    assert_eq!(selected.timeout, deployment.timeout);
    assert_eq!(selected.shaping, deployment.shaping);
}
