use std::time::Duration;

use litellm_core::messages::MessagesShaping;
use litellm_router::{Deployment, Router};
use rstest::rstest;

#[rstest]
#[case::first("public-a", Some("provider/a"))]
#[case::second("public-b", Some("provider/b"))]
#[case::unknown("missing", None)]
#[case::provider_name_is_not_an_alias("provider/a", None)]
#[case::case_sensitive("PUBLIC-A", None)]
fn lookup_uses_public_names(#[case] name: &str, #[case] expected: Option<&str>) {
    let router = Router::from_iter([
        (
            "public-a".into(),
            Deployment {
                model: "provider/a".into(),
                ..Default::default()
            },
        ),
        (
            "public-b".into(),
            Deployment {
                model: "provider/b".into(),
                ..Default::default()
            },
        ),
    ]);

    assert_eq!(router.get(name).map(|entry| entry.model.as_str()), expected);
}

#[rstest]
fn empty_router_has_no_deployment() {
    assert!(Router::from_iter([]).get("").is_none());
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
