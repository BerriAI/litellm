use std::time::Duration;

use litellm_config::Config;
use litellm_core::messages::MessagesShaping;
use litellm_router::{Deployment, Router};
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
    let router = Router::from_model_list(&config.model_list);
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
    let router = Router::from_model_list(&config.model_list);

    assert_eq!(router.get(name).map(|entry| entry.model.as_str()), expected);
}

#[rstest]
fn empty_configuration_has_no_deployment() {
    let config = Config::from_yaml("model_list: []").unwrap();

    assert!(
        Router::from_model_list(&config.model_list)
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
        timeout: Some(Duration::from_secs(7)),
        shaping: MessagesShaping {
            drop_params: true,
            additional_drop_params: vec!["metadata.test".into()],
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

use litellm_router::{
    Error,
    call::{Endpoint, RoutingOptions, RoutingRequest},
    config::RouterConfig,
    selection::{SelectionContext, Selector},
};
use rstest::fixture;

#[fixture]
fn grouped_config() -> RouterConfig {
    let config = Config::from_yaml(
        "model_list:
  - model_name: primary
    model_info: {id: first}
    litellm_params: {model: provider/a, api_key: private-key}
  - model_name: primary
    model_info: {id: second}
    litellm_params: {model: provider/b}
  - model_name: backup
    model_info: {id: third}
    litellm_params: {model: provider/c}",
    )
    .unwrap();
    RouterConfig {
        model_list: config.model_list.into_vec(),
        ..RouterConfig::default()
    }
}

fn request(model: &str) -> RoutingRequest {
    RoutingRequest {
        model: model.into(),
        endpoint: Endpoint::ChatCompletions,
        options: RoutingOptions::default(),
    }
}

struct Pinned<'a>(&'a str);

impl Selector for Pinned<'_> {
    fn select(&self, _context: &SelectionContext) -> Result<String, Error> {
        Ok(self.0.to_owned())
    }
}

#[rstest]
fn selection_retains_every_deployment_in_the_requested_group(grouped_config: RouterConfig) {
    let router = Router::new(grouped_config).unwrap();
    let call = router.start(request("primary")).unwrap();
    let context = call.selection_context();

    assert_eq!(
        context
            .candidates
            .iter()
            .map(|candidate| candidate.deployment_id.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(router.snapshot().deployments.len(), 3);
    assert_eq!(router.get("primary").unwrap().model, "provider/b");
    assert_eq!(
        call.select(&Pinned("first"))
            .unwrap()
            .deployment()
            .api_key
            .as_deref(),
        Some("private-key")
    );
}

#[rstest]
#[case::eligible("second", true)]
#[case::other_group("third", false)]
#[case::unknown("missing", false)]
fn selectors_cannot_escape_the_candidate_group(
    grouped_config: RouterConfig,
    #[case] id: &str,
    #[case] allowed: bool,
) {
    let router = Router::new(grouped_config).unwrap();
    let call = router.start(request("primary")).unwrap();
    let result = call.select(&Pinned(id));

    if allowed {
        assert_eq!(result.unwrap().deployment_id(), id);
    } else {
        assert!(matches!(result, Err(Error::InvalidSelection { id: selected }) if selected == id));
    }
}

#[rstest]
fn reconfiguration_and_close_preserve_started_calls(grouped_config: RouterConfig) {
    let mut router = Router::new(grouped_config).unwrap();
    let call = router.start(request("primary")).unwrap();
    let original = call.selection_context();
    router.reconfigure(RouterConfig::default()).unwrap();
    router.close();
    router.close();

    assert_eq!(call.selection_context().candidates, original.candidates);
    assert_eq!(router.snapshot().generation, call.generation() + 1);
    assert!(router.snapshot().closed);
    assert!(matches!(
        router.start(request("primary")),
        Err(Error::Closed)
    ));
    assert!(matches!(
        router.reconfigure(RouterConfig::default()),
        Err(Error::Closed)
    ));
    assert_eq!(
        call.select(&Pinned("first")).unwrap().deployment().model,
        "provider/a"
    );
}

#[rstest]
fn invalid_reconfiguration_does_not_replace_the_catalog(grouped_config: RouterConfig) {
    let mut router = Router::new(grouped_config.clone()).unwrap();
    let original = router.snapshot();
    let duplicate = RouterConfig {
        model_list: vec![
            grouped_config.model_list[0].clone(),
            grouped_config.model_list[0].clone(),
        ],
        ..RouterConfig::default()
    };

    assert!(matches!(
        router.reconfigure(duplicate),
        Err(Error::DuplicateDeployment { .. })
    ));
    assert_eq!(router.snapshot().generation, original.generation);
    assert_eq!(router.snapshot().deployments, original.deployments);
}

#[rstest]
fn starting_an_unknown_group_reports_the_requested_alias(grouped_config: RouterConfig) {
    let router = Router::new(grouped_config).unwrap();

    assert!(
        matches!(router.start(request("missing")), Err(Error::UnknownModel { model }) if model == "missing")
    );
}
