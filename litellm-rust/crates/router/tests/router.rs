use std::time::Duration;

use litellm_core::messages::MessagesShaping;
use litellm_router::{
    Deployment, Error, Router,
    call::{Endpoint, RoutingOptions, RoutingRequest},
    config::{DeploymentConfig, RouterConfig},
    selection::{SelectionContext, Selector},
};
use rstest::{fixture, rstest};

#[rstest]
#[case::first("public-a", Some("provider/a"))]
#[case::second("public-b", Some("provider/b"))]
#[case::unknown("missing", None)]
#[case::provider_name_is_not_an_alias("provider/a", None)]
#[case::case_sensitive("PUBLIC-A", None)]
fn lookup_uses_public_names(#[case] name: &str, #[case] expected: Option<&str>) {
    let router = Router::new(RouterConfig {
        deployments: vec![
            deployment_config(None, "public-a", "provider/a"),
            deployment_config(None, "public-b", "provider/b"),
        ],
        ..RouterConfig::default()
    })
    .unwrap();

    assert_eq!(router.get(name).map(|entry| entry.model.as_str()), expected);
}

#[rstest]
fn empty_configuration_has_no_deployment() {
    assert!(
        Router::new(RouterConfig::default())
            .unwrap()
            .get("")
            .is_none()
    );
    assert!(Router::default().get("unknown").is_none());
}

#[rstest]
#[case::owned_config(true)]
#[case::compatibility_lookup(false)]
fn programmatic_deployments_preserve_overrides_and_last_entry_wins(#[case] owned_config: bool) {
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
    let router = if owned_config {
        Router::new(RouterConfig {
            deployments: vec![
                deployment_config(None, "public-model", "provider/first"),
                DeploymentConfig {
                    deployment: deployment.clone(),
                    ..deployment_config(None, "public-model", "provider/selected")
                },
            ],
            ..RouterConfig::default()
        })
        .unwrap()
    } else {
        Router::from_iter([
            ("public-model".into(), Deployment::default()),
            ("public-model".into(), deployment.clone()),
        ])
    };
    let selected = router.get("public-model").unwrap();

    assert_eq!(selected.model, deployment.model);
    assert_eq!(selected.api_key, deployment.api_key);
    assert_eq!(selected.api_base, deployment.api_base);
    assert_eq!(selected.custom_llm_provider, deployment.custom_llm_provider);
    assert_eq!(selected.timeout, deployment.timeout);
    assert_eq!(selected.shaping, deployment.shaping);
}

fn deployment_config(id: Option<&str>, name: &str, model: &str) -> DeploymentConfig {
    DeploymentConfig {
        deployment_id: id.map(str::to_owned),
        model_name: name.into(),
        deployment: Deployment {
            model: model.into(),
            ..Deployment::default()
        },
    }
}

#[fixture]
fn grouped_config() -> RouterConfig {
    RouterConfig {
        deployments: vec![
            DeploymentConfig {
                deployment: Deployment {
                    model: "provider/a".into(),
                    api_key: Some("private-key".into()),
                    ..Deployment::default()
                },
                ..deployment_config(Some("first"), "primary", "provider/a")
            },
            deployment_config(Some("second"), "primary", "provider/b"),
            deployment_config(Some("third"), "backup", "provider/c"),
        ],
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
        deployments: vec![
            grouped_config.deployments[0].clone(),
            grouped_config.deployments[0].clone(),
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

#[rstest]
fn generated_ids_identify_distinct_deployments_in_one_group() {
    let router = Router::new(RouterConfig {
        deployments: vec![
            deployment_config(None, "primary", "provider/a"),
            deployment_config(None, "primary", "provider/b"),
        ],
        ..RouterConfig::default()
    })
    .unwrap();
    let call = router.start(request("primary")).unwrap();
    let context = call.selection_context();

    assert_ne!(
        context.candidates[0].deployment_id,
        context.candidates[1].deployment_id
    );
    for candidate in &context.candidates {
        assert_eq!(
            call.select(&Pinned(&candidate.deployment_id))
                .unwrap()
                .deployment()
                .model,
            candidate.model
        );
    }
}
