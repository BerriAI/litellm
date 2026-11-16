//! The affinity checks the core runs itself for a host that does not select.

use std::{collections::BTreeSet, sync::Arc};

use litellm_router_core::{
    Affinity, Attempt, AttemptError, AttemptFailure, ConfigError, Deployment, Error, FailureKind,
    HostFacts, InMemoryRouterCache, Op, Router, RouterConfig, RouterDynamicConfig, RouterHost,
    RouterRequest, RoutingContext, SelectStage, Served, Unsupported,
};
use parking_lot::Mutex;
use rstest::rstest;
use serde_json::{Value, json};

const NO_STEPS: HostFacts = HostFacts {
    capabilities: &[],
    supports: |_, _| true,
};

/// Serves every attempt, failing the first one when asked to.
#[derive(Default)]
struct Host {
    cache: InMemoryRouterCache,
    fail_first: Mutex<bool>,
    attempts: Mutex<Vec<String>>,
}

impl RouterHost for Host {
    type Response = ();
    type Failure = ();
    type Cache = InMemoryRouterCache;

    async fn attempt(&self, attempt: &Attempt) -> Result<Served<()>, AttemptFailure<()>> {
        self.attempts.lock().push(attempt.deployment.id.clone());
        if std::mem::take(&mut *self.fail_first.lock()) {
            return Err(AttemptFailure::Failed {
                facts: AttemptError::new(FailureKind::InternalServer, Some(500)),
                raised: (),
            });
        }
        Ok(Served::new((), 1))
    }

    async fn select(
        &self,
        _stage: &SelectStage,
        _ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        Ok(candidates)
    }

    async fn allow_fallback(&self, _target: &str) -> Result<bool, AttemptError> {
        Ok(true)
    }

    fn cache(&self) -> &InMemoryRouterCache {
        &self.cache
    }

    async fn sleep(&self, _duration: std::time::Duration) {}

    fn observe(&self, _event: &litellm_router_core::RouterEvent) {}
}

fn deployment(group: &str, id: &str, params: Value, model_info: Value) -> Value {
    let mut model_info = model_info;
    model_info["id"] = json!(id);
    let mut params = params;
    params["model"] = json!(format!("azure/{id}"));
    json!({"model_name": group, "litellm_params": params, "model_info": model_info})
}

fn account(name: &str) -> Value {
    json!({"api_base": format!("https://{name}.example"), "api_key": format!("key-{name}")})
}

fn router(checks: Value, model_list: Vec<Value>) -> Router {
    let config: RouterConfig = serde_json::from_value(json!({
        "pre_call_checks": checks,
        "model_list": model_list,
    }))
    .unwrap();
    let dynamic = RouterDynamicConfig {
        num_retries: Some(1),
        ..RouterDynamicConfig::default()
    };
    Router::new(config, dynamic, NO_STEPS).unwrap()
}

fn pair() -> Vec<Value> {
    vec![
        deployment("primary", "a1", json!({}), json!({})),
        deployment("primary", "a2", json!({}), json!({})),
    ]
}

fn request(affinity: Affinity) -> RouterRequest {
    RouterRequest {
        model: "primary".to_owned(),
        affinity,
        ..RouterRequest::default()
    }
}

fn minted_by(origin: &str) -> Affinity {
    Affinity {
        encrypted_content: Some(origin.to_owned()),
        ..Affinity::default()
    }
}

fn continuing(previous: &str) -> Affinity {
    Affinity {
        previous_response: Some(previous.to_owned()),
        ..Affinity::default()
    }
}

/// Every deployment 40 picks for `affinity` land on.
async fn picks(router: &Router, affinity: Affinity) -> BTreeSet<String> {
    let host = Host::default();
    let request = request(affinity);
    let mut picked = BTreeSet::new();
    for _ in 0..40 {
        picked.insert(
            router
                .select(&host, &Op::RESPONSES, &request)
                .await
                .unwrap()
                .id
                .clone(),
        );
    }
    picked
}

fn ids(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[rstest]
#[case::encrypted_content_pins_its_origin(
    json!({"global": ["encrypted_content_affinity"]}),
    minted_by("a2"),
    &["a2"]
)]
#[case::encrypted_content_off_for_the_group(
    json!({"groups": {"other": ["encrypted_content_affinity"]}}),
    minted_by("a2"),
    &["a1", "a2"]
)]
#[case::encrypted_content_on_for_the_group(
    json!({"groups": {"primary": ["encrypted_content_affinity"]}}),
    minted_by("a2"),
    &["a2"]
)]
#[case::an_unknown_origin_keeps_the_pool(
    json!({"global": ["encrypted_content_affinity"]}),
    minted_by("gone"),
    &["a1", "a2"]
)]
#[case::previous_response_pins_its_deployment(
    json!({"groups": {"primary": ["responses_api_deployment_check"]}}),
    continuing("a1"),
    &["a1"]
)]
#[case::a_group_list_replaces_the_global_one(
    json!({
        "global": ["responses_api_deployment_check"],
        "groups": {"primary": ["encrypted_content_affinity"]},
    }),
    continuing("a1"),
    &["a1", "a2"]
)]
#[tokio::test]
async fn an_affinity_check_pins_a_follow_up_to_the_deployment_its_ids_name(
    #[case] checks: Value,
    #[case] affinity: Affinity,
    #[case] expected: &[&str],
) {
    assert_eq!(
        picks(&router(checks, pair()), affinity).await,
        ids(expected)
    );
}

#[rstest]
#[case::explicit_account(account("east"), json!({}), account("west"), json!({}), &["p1"])]
#[case::declared_boundary(
    json!({}),
    json!({"encrypted_content_boundary": "org"}),
    account("west"),
    json!({"encrypted_content_boundary": "org"}),
    &["p2"]
)]
#[case::no_account_no_peer(json!({}), json!({}), json!({}), json!({}), &["p1", "p2"])]
#[tokio::test]
async fn an_origin_outside_the_group_pins_the_deployments_that_decrypt_its_reasoning(
    #[case] origin: Value,
    #[case] origin_info: Value,
    #[case] other: Value,
    #[case] other_info: Value,
    #[case] expected: &[&str],
) {
    let router = router(
        json!({"global": ["encrypted_content_affinity"]}),
        vec![
            deployment("elsewhere", "origin", origin.clone(), origin_info),
            deployment("primary", "p1", origin, json!({})),
            deployment("primary", "p2", other, other_info),
        ],
    );
    assert_eq!(picks(&router, minted_by("origin")).await, ids(expected));
}

#[rstest]
#[case::the_retry_leaves_the_origin(minted_by("a2"), "a1")]
#[case::a_response_that_binds_the_origin_keeps_it(
    Affinity {
        previous_response: Some("a2".to_owned()),
        encrypted_content: Some("a2".to_owned()),
    },
    "a2"
)]
#[tokio::test]
async fn a_retry_after_the_origin_failed_leaves_it_unless_the_response_binds_it(
    #[case] affinity: Affinity,
    #[case] retried_on: &str,
) {
    let router = router(json!({"global": ["encrypted_content_affinity"]}), pair());
    let host = Host {
        fail_first: Mutex::new(true),
        ..Host::default()
    };
    router
        .call(&host, Op::RESPONSES, request(affinity))
        .await
        .unwrap();
    assert_eq!(*host.attempts.lock(), ["a2", retried_on]);
}

#[rstest]
#[case::global(json!({"global": ["prompt_caching"]}), "`optional_pre_call_checks`")]
#[case::group(
    json!({"groups": {"primary": ["session_affinity"]}}),
    "`model_group_affinity_config`"
)]
fn a_check_the_core_does_not_run_still_needs_a_host_that_selects(
    #[case] checks: Value,
    #[case] named: &str,
) {
    let config: RouterConfig = serde_json::from_value(json!({"pre_call_checks": checks})).unwrap();
    let refused = match Router::new(config, RouterDynamicConfig::default(), NO_STEPS) {
        Err(ConfigError::Unsupported(unsupported @ Unsupported::Setting { .. })) => {
            unsupported.to_string()
        }
        other => panic!("expected an unsupported setting, got {:?}", other.err()),
    };
    assert!(refused.contains(named), "{refused}");
}

#[rstest]
#[case::itself("a1", "a1", true)]
#[case::same_account("east", "a1", true)]
#[case::other_account("west", "a1", false)]
#[case::removed("gone", "a1", false)]
fn a_deployment_decrypts_what_its_own_account_minted(
    #[case] origin: &str,
    #[case] target: &str,
    #[case] expected: bool,
) {
    let router = router(
        json!({}),
        vec![
            deployment("primary", "a1", account("east"), json!({})),
            deployment("other", "east", account("east"), json!({})),
            deployment("other", "west", account("west"), json!({})),
        ],
    );
    let target = router.registry().deployment(target).unwrap();
    assert_eq!(router.decrypts(origin, &target), expected);
}
