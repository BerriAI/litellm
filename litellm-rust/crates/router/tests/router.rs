use std::time::Duration;

use litellm_config::Config;
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_router::{Deployment, Error, Router};
use rstest::rstest;

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
    let router = Router::from_model_list(&config.model_list).unwrap();
    let deployment = only(&router, &config.model_list[0].model_name);
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
    let router = Router::from_model_list(&config.model_list).unwrap();
    let params = &only(&router, "public-model").litellm_params;

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
    let router = Router::from_model_list(&config.model_list).unwrap();

    assert_eq!(
        router
            .group(name)
            .map(|group| group.deployments().next().unwrap().model.as_str()),
        expected
    );
}

#[rstest]
fn empty_configuration_has_no_deployment() {
    let config = Config::from_yaml("model_list: []").unwrap();

    assert!(
        Router::from_model_list(&config.model_list)
            .unwrap()
            .group("")
            .is_none()
    );
    assert!(Router::default().group("unknown").is_none());
}

#[rstest]
fn programmatic_deployments_preserve_overrides_and_share_their_group() {
    let deployment = Deployment {
        model: "provider/selected".into(),
        api_key: Some("test-key".into()),
        api_base: Some("https://provider.example/v1".into()),
        custom_llm_provider: Some("test-provider".into()),
        litellm_params: Default::default(),
        timeout: Some(Duration::from_secs(7)),
        stream_timeout: Some(Duration::from_secs(3)),
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
    let group = router.group("public-model").unwrap();
    let selected = group.deployments().last().unwrap();

    assert_eq!(group.deployments().len(), 2);
    assert_eq!(selected.model, deployment.model);
    assert_eq!(selected.api_key, deployment.api_key);
    assert_eq!(selected.api_base, deployment.api_base);
    assert_eq!(selected.custom_llm_provider, deployment.custom_llm_provider);
    assert_eq!(selected.timeout, deployment.timeout);
    assert_eq!(selected.stream_timeout, deployment.stream_timeout);
    assert_eq!(selected.shaping, deployment.shaping);
}

#[rstest]
fn configuration_keeps_every_deployment_of_a_model_name_in_order() {
    let config = Config::from_yaml(
        "model_list:
  - model_name: shared
    litellm_params: {model: provider/a}
  - model_name: other
    litellm_params: {model: provider/x}
  - model_name: shared
    litellm_params: {model: provider/b}
  - model_name: shared
    litellm_params: {model: provider/c}",
    )
    .unwrap();
    let router = Router::from_config(&config).unwrap();

    assert_eq!(
        router
            .group("shared")
            .unwrap()
            .deployments()
            .map(|deployment| deployment.model.as_str())
            .collect::<Vec<_>>(),
        ["provider/a", "provider/b", "provider/c"]
    );
}

#[rstest]
#[case::deployment_timeouts(
    "timeout: 300, stream_timeout: 1800",
    "",
    Some(Duration::from_secs(300)),
    Some(Duration::from_secs(1800))
)]
#[case::router_timeout_fills_in(
    "",
    "router_settings: {timeout: 60}",
    Some(Duration::from_secs(60)),
    None
)]
#[case::deployment_timeout_wins(
    "timeout: 5",
    "router_settings: {timeout: 60}",
    Some(Duration::from_secs(5)),
    None
)]
fn configuration_reads_timeouts(
    #[case] params: &str,
    #[case] settings: &str,
    #[case] timeout: Option<Duration>,
    #[case] stream_timeout: Option<Duration>,
) {
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: m\n    litellm_params: {{model: provider/m, {params}}}\n{settings}"
    ))
    .unwrap();
    let router = Router::from_config(&config).unwrap();
    let deployment = only(&router, "m");

    assert_eq!(
        (deployment.timeout, deployment.stream_timeout),
        (timeout, stream_timeout)
    );
}

#[rstest]
#[case::unsupported_strategy(
    "model_list: []\nrouter_settings: {routing_strategy: usage-based-routing-v2}",
    Error::UnsupportedStrategy("usage-based-routing-v2".into())
)]
#[case::bad_timeout(
    "model_list:\n  - model_name: m\n    litellm_params: {model: provider/m, timeout: soon}",
    Error::InvalidSeconds { setting: "litellm_params.timeout", value: "soon".into() }
)]
fn configuration_rejects_what_the_router_cannot_honor(#[case] yaml: &str, #[case] error: Error) {
    let config = Config::from_yaml(yaml).unwrap();

    assert_eq!(Router::from_config(&config).err(), Some(error));
}

fn only<'a>(router: &'a Router, name: &str) -> &'a Deployment {
    let group = router.group(name).unwrap();
    assert_eq!(group.deployments().len(), 1);
    group.deployments().next().unwrap()
}
