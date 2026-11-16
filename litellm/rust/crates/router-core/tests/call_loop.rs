//! A call through the router end to end: selection, retries, fallbacks and cooldowns against a
//! scripted host.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use async_trait::async_trait;
use litellm_router_core::{
    Attempt, AttemptError, CallbackFiltered, Deployment, Error, FailureKind, FailureReport, Host,
    InMemoryRouterCache, Op, Output, Payload, Router, RouterCache, RouterConfig,
    RouterDynamicConfig, RouterEvent, RouterRequest, RoutingContext, SeededRandomness, SelectStage,
    SuccessReport, UsageWindow, call::StreamOrigin, fallback::Fallbacks, health::current_minute,
    host::UncountedRetries, select::PassThrough,
};
use parking_lot::Mutex;
use rstest::{fixture, rstest};
use serde_json::{Value, json};

#[derive(Default)]
struct ScriptedHost {
    /// Failures handed out in order, keyed by model group; a group with none left succeeds.
    script: Mutex<Vec<(String, VecDeque<AttemptError>)>>,
    events: Mutex<Vec<RouterEvent>>,
    cache: InMemoryRouterCache,
    /// Answer every success as a stream whose outcome the host reports when it ends.
    deferred: bool,
    /// [`Output::tracks_usage`] on every success.
    tracks_usage: Option<bool>,
    /// Refuse every fallback chain with a dict target.
    refuses_dict_targets: bool,
    /// Every fallback check raises this exception class, as a host's sync check does.
    fallback_check_raises: Option<&'static str>,
    /// Failures the tag stage raises in turn, before it answers.
    pick_failures: Mutex<VecDeque<AttemptError>>,
    /// The next attempt's failure, whatever group it serves.
    first_failure: Mutex<Option<AttemptError>>,
    /// The callback filters pin to this deployment until an attempt of the request failed on it,
    /// as `encrypted_content_affinity` does.
    pinned: Option<&'static str>,
}

impl ScriptedHost {
    fn failing(group: &str, failures: Vec<AttemptError>) -> Self {
        Self::scripted(vec![(group, failures)])
    }

    fn scripted(groups: Vec<(&str, Vec<AttemptError>)>) -> Self {
        Self {
            script: Mutex::new(
                groups
                    .into_iter()
                    .map(|(group, failures)| (group.to_owned(), failures.into()))
                    .collect(),
            ),
            ..Self::default()
        }
    }

    fn selected(&self) -> Vec<String> {
        self.events
            .lock()
            .iter()
            .filter_map(|event| match event {
                RouterEvent::Selected { deployment_id, .. } => Some(deployment_id.clone()),
                _ => None,
            })
            .collect()
    }

    fn retries(&self) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|event| matches!(event, RouterEvent::Retry { .. }))
            .count()
    }
}

#[async_trait]
impl Host for ScriptedHost {
    async fn invoke(&self, attempt: &Attempt) -> Result<Output, AttemptError> {
        if let Some(failure) = self.first_failure.lock().take() {
            return Err(failure);
        }
        let next = self
            .script
            .lock()
            .iter_mut()
            .find(|(group, _)| *group == attempt.model_group)
            .and_then(|(_, failures)| failures.pop_front());
        match next {
            Some(failure) => Err(failure),
            None => Ok(Output {
                value: Payload(Arc::new(attempt.deployment.id.clone())),
                total_tokens: 10,
                blocked: None,
                deferred: self.deferred,
                tracks_usage: self.tracks_usage,
            }),
        }
    }

    async fn select(
        &self,
        _stage: &SelectStage,
        _ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        match self.pick_failures.lock().pop_front() {
            Some(failure) => Err(failure.into()),
            None => Ok(candidates),
        }
    }

    async fn callback_filters(
        &self,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<CallbackFiltered, Error> {
        let kept = match self.pinned {
            Some(pin) if !ctx.skipped_ids.iter().any(|id| id == pin) => {
                candidates.into_iter().filter(|d| d.id == pin).collect()
            }
            Some(pin) => candidates.into_iter().filter(|d| d.id != pin).collect(),
            None => candidates,
        };
        Ok(CallbackFiltered {
            kept,
            preferred: None,
        })
    }

    async fn allow_fallback(
        &self,
        _request: &RouterRequest,
        _target: &str,
    ) -> Result<bool, AttemptError> {
        match self.fallback_check_raises {
            Some(class) => Err(AttemptError::host_raised(&[class])),
            None => Ok(true),
        }
    }

    async fn refuses_fallbacks(
        &self,
        _level: u32,
        targets: &[serde_json::Map<String, serde_json::Value>],
    ) -> bool {
        self.refuses_dict_targets && !targets.is_empty()
    }

    async fn sleep(&self, _duration: Duration) {}

    fn observe(&self, event: &RouterEvent) {
        self.events.lock().push(event.clone());
    }

    fn cache(&self) -> Arc<dyn RouterCache> {
        Arc::new(self.cache.clone())
    }
}

#[fixture]
fn config() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "a2"}},
            {"model_name": "backup", "litellm_params": {"model": "anthropic/c"}, "model_info": {"id": "b1"}},
        ]
    }))
    .unwrap()
}

fn router(config: RouterConfig, dynamic: serde_json::Value, host: Arc<ScriptedHost>) -> Router {
    let dynamic: RouterDynamicConfig = serde_json::from_value(dynamic).unwrap();
    Router::builder(config, dynamic, host)
        .randomness(Arc::new(SeededRandomness::new(7)))
        .build()
        .unwrap()
}

fn request() -> RouterRequest {
    RouterRequest {
        model: "primary".to_owned(),
        ..RouterRequest::default()
    }
}

/// `primary` at order 1, its second deployment at order 2 when `tiered`, and two other groups.
fn tiers(tiered: bool) -> RouterConfig {
    let second = if tiered { "primary" } else { "unused" };
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/p1", "order": 1}, "model_info": {"id": "p1"}},
            {"model_name": second, "litellm_params": {"model": "openai/p2", "order": 2}, "model_info": {"id": "p2"}},
            {"model_name": "backup", "litellm_params": {"model": "anthropic/b1"}, "model_info": {"id": "b1"}},
            {"model_name": "context", "litellm_params": {"model": "anthropic/c1"}, "model_info": {"id": "c1"}},
        ]
    }))
    .unwrap()
}

fn failure(kind: FailureKind, status: u16) -> AttemptError {
    AttemptError::new(kind, Some(status))
}

fn context_window() -> AttemptError {
    failure(FailureKind::ContextWindowExceeded, 400)
}

fn server_error() -> AttemptError {
    failure(FailureKind::InternalServer, 500)
}

fn agentic_followup() -> AttemptError {
    let mut failed = server_error();
    failed.agentic_followup = true;
    failed
}

fn over_every_tpm_limit() -> AttemptError {
    let mut failed = failure(FailureKind::RateLimit, 429);
    failed.classes = vec![
        "RequestOverDeploymentTpmError".into(),
        "RateLimitError".into(),
    ];
    failed
}

#[rstest]
#[tokio::test]
async fn a_rate_limit_retries_in_the_same_group(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![AttemptError::new(FailureKind::RateLimit, Some(429))],
    ));
    let router = router(config, json!({"num_retries": 2}), host.clone());
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(
        (
            response.model_group.as_str(),
            response.attempted_retries,
            response.attempted_fallbacks
        ),
        ("primary", 1, 0)
    );
}

#[rstest]
#[case::server_error(server_error())]
#[case::rate_limit(failure(FailureKind::RateLimit, 429))]
#[case::connection_error(AttemptError::new(FailureKind::ApiConnection, None))]
#[tokio::test]
async fn a_retry_leaves_a_pinned_deployment_that_failed_the_request(
    mut config: RouterConfig,
    #[case] origin_failure: AttemptError,
) {
    config.callback_filters = true;
    let host = Arc::new(ScriptedHost {
        first_failure: Mutex::new(Some(origin_failure)),
        pinned: Some("a1"),
        ..ScriptedHost::default()
    });
    let router = router(config, json!({"num_retries": 3}), host.clone());
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(
        (host.selected(), response.attempted_retries),
        (vec!["a1".to_owned(), "a2".to_owned()], 1)
    );
}

#[rstest]
#[tokio::test]
async fn a_bad_request_falls_back_without_retrying(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![AttemptError::new(FailureKind::BadRequest, Some(400))],
    ));
    let router = router(
        config,
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let retried = host
        .events
        .lock()
        .iter()
        .any(|event| matches!(event, RouterEvent::Retry { .. }));
    assert_eq!(
        (
            response.deployment_id.as_str(),
            response.attempted_fallbacks,
            retried
        ),
        ("b1", 1, false)
    );
}

#[rstest]
#[tokio::test]
async fn a_rate_limit_cools_the_deployment_down(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![AttemptError::new(FailureKind::RateLimit, Some(429))],
    ));
    let router = router(config, json!({}), host.clone());
    let error = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap_err();
    let cooled = router
        .health()
        .active_cooldowns(&["a1".to_owned(), "a2".to_owned()])
        .await
        .unwrap();
    assert_eq!(
        (error.attempt().map(|failure| failure.kind), cooled.len()),
        (Some(FailureKind::RateLimit), 1)
    );
}

#[rstest]
#[case::now(false, (1, (10, 1)))]
#[case::deferred(true, (0, (10, 1)))]
#[tokio::test]
async fn a_deferred_attempt_counts_its_request_and_tokens_but_no_success(
    #[case] deferred: bool,
    #[case] expected: (u64, (u64, u64)),
) {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a", "rpm": 100}, "model_info": {"id": "a1"}},
        ]
    }))
    .unwrap();
    let host = Arc::new(ScriptedHost {
        deferred,
        ..ScriptedHost::default()
    });
    let router = router(config, json!({}), host.clone());
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let deployment = router.registry().deployment("a1").unwrap();
    let successes = host.cache.outcomes("a1").await.unwrap().successes;
    let usage = router.health().usage(&deployment).await.unwrap();
    assert_eq!((successes, usage), expected);
}

/// `litellm_params`, `model_info` and a limit only the host knows of all count; a realtime
/// session (`Some(false)` on a deferred answer) counts nothing, whatever limits it has.
#[rstest]
#[case::in_params(json!({"rpm": 100}), json!({}), false, None, (10, 1))]
#[case::in_model_info(json!({}), json!({"rpm": 100}), false, None, (10, 1))]
#[case::in_the_catalog(json!({}), json!({}), false, Some(true), (10, 1))]
#[case::no_limit(json!({}), json!({}), false, None, (0, 0))]
#[case::a_stream_in_model_info(json!({}), json!({"rpm": 100}), true, None, (10, 1))]
#[case::a_realtime_session(json!({"rpm": 100}), json!({}), true, Some(false), (0, 0))]
#[tokio::test]
async fn usage_counts_for_a_deployment_that_tracks_it(
    #[case] params: serde_json::Value,
    #[case] info: serde_json::Value,
    #[case] deferred: bool,
    #[case] tracks_usage: Option<bool>,
    #[case] expected: (u64, u64),
) {
    let mut litellm_params = json!({"model": "openai/a"});
    litellm_params
        .as_object_mut()
        .unwrap()
        .extend(params.as_object().unwrap().clone());
    let mut model_info = json!({"id": "a1"});
    model_info
        .as_object_mut()
        .unwrap()
        .extend(info.as_object().unwrap().clone());
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": litellm_params, "model_info": model_info},
        ]
    }))
    .unwrap();
    let host = Arc::new(ScriptedHost {
        deferred,
        tracks_usage,
        ..ScriptedHost::default()
    });
    let router = router(config, json!({}), host.clone());
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let deployment = router.registry().deployment("a1").unwrap();
    assert_eq!(router.health().usage(&deployment).await.unwrap(), expected);
}

/// A stream end's report of a wildcard deployment counts under the model the route filled, the
/// key its request was counted under, not the configured `*`.
#[rstest]
#[tokio::test]
async fn a_reported_success_keeps_the_filled_wildcard_model() {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*", "rpm": 100}, "model_info": {"id": "w1"}},
        ]
    }))
    .unwrap();
    let host = Arc::new(ScriptedHost::default());
    let router = router(config, json!({}), host.clone());
    let report = SuccessReport {
        usage_model: Some("openai/gpt-4o".to_owned()),
        ..SuccessReport::default()
    };
    router.report_success("w1", 7.0, 0.0, report).await.unwrap();
    let minute = current_minute(std::time::SystemTime::now());
    let tokens = |model: &str| {
        host.cache
            .counted(&UsageWindow {
                deployment_id: "w1".to_owned(),
                model: model.to_owned(),
                minute: minute.clone(),
            })
            .tokens
    };
    assert_eq!(
        (tokens("openai/gpt-4o"), tokens("openai/*")),
        (Some(7.0), None),
    );
}

/// `async_deployment_callback_on_failure`'s request is written only when it runs: a failure the
/// host says ran no failure callback (`recorded`) writes none, nor does a sync call's.
#[rstest]
#[case::the_first_logged_failure(false, false, 1, 1)]
#[case::a_later_failure_on_the_shared_logging_object(true, false, 0, 0)]
#[case::a_sync_call(false, true, 0, 1)]
#[tokio::test]
async fn a_failed_attempt_counts_its_request_only_when_its_callback_ran(
    #[case] recorded: bool,
    #[case] uncounted_request: bool,
    #[case] requests: u64,
    #[case] fails: u64,
) {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
        ]
    }))
    .unwrap();
    let mut failure = AttemptError::new(FailureKind::InternalServer, Some(500));
    failure.recorded = recorded;
    failure.uncounted_request = uncounted_request;
    let host = Arc::new(ScriptedHost::failing("primary", vec![failure]));
    let router = router(config, json!({"num_retries": 0}), host.clone());
    let _ = router.call(Op::from("acompletion"), request()).await;
    let deployment = router.registry().deployment("a1").unwrap();
    let usage = router.health().usage(&deployment).await.unwrap();
    let counted = host.cache.outcomes("a1").await.unwrap().fails;
    assert_eq!((usage.1, counted), (requests, fails));
}

#[rstest]
#[case::counted(true, 2)]
#[case::uncounted(false, 1)]
#[tokio::test]
async fn a_reported_failure_counts_its_request_when_asked(
    #[case] counts_request: bool,
    #[case] requests: u64,
) {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a", "rpm": 100}, "model_info": {"id": "a1"}},
        ]
    }))
    .unwrap();
    let host = Arc::new(ScriptedHost {
        deferred: true,
        ..ScriptedHost::default()
    });
    let router = router(config, json!({}), host.clone());
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let report = FailureReport {
        counts_request,
        ..FailureReport::default()
    };
    router
        .report_failure(
            "a1",
            &AttemptError::new(FailureKind::RateLimit, Some(429)),
            report,
        )
        .await
        .unwrap();
    let deployment = router.registry().deployment("a1").unwrap();
    let fails = host.cache.outcomes("a1").await.unwrap().fails;
    let usage = router.health().usage(&deployment).await.unwrap();
    assert_eq!((fails, usage.1), (1, requests));
}

#[rstest]
#[tokio::test]
async fn a_raising_retry_policy_lookup_ends_the_hop_and_falls_back(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![AttemptError::new(FailureKind::RateLimit, Some(429))],
    ));
    let router = router(
        config,
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let request = RouterRequest {
        retry_policy_raises: vec!["primary".to_owned()],
        ..request()
    };
    let response = router.call(Op::from("acompletion"), request).await.unwrap();
    let events = host.events.lock();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RouterEvent::Retry { .. }))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        RouterEvent::RetryPolicyRaised { level: 0, model, .. } if model == "primary"
    )));
    assert_eq!(response.model_group, "backup");
}

#[rstest]
#[tokio::test]
async fn a_num_retries_that_raises_ends_the_hop_and_falls_back(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(
        config,
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let request = RouterRequest {
        uncounted_retries: Some(UncountedRetries::Raises),
        ..request()
    };
    let response = router.call(Op::from("acompletion"), request).await.unwrap();
    let events = host.events.lock();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RouterEvent::Retry { .. }))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        RouterEvent::RetryCountRaised { level: 0, model, .. } if model == "primary"
    )));
    assert_eq!(response.model_group, "backup");
}

#[rstest]
#[case::negative(UncountedRetries::Negative)]
#[case::not_positive(UncountedRetries::NotPositive)]
#[tokio::test]
async fn a_num_retries_below_one_leaves_the_failure_standing(
    config: RouterConfig,
    #[case] uncounted: UncountedRetries,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(config, json!({"num_retries": 3}), host.clone());
    let request = RouterRequest {
        uncounted_retries: Some(uncounted),
        ..request()
    };
    let refused = router.call(Op::from("acompletion"), request).await;
    assert!(matches!(refused, Err(Error::Attempt(failed)) if failed.status == Some(500)));
    assert_eq!(host.retries(), 0);
    assert!(
        !host
            .events
            .lock()
            .iter()
            .any(|event| matches!(event, RouterEvent::RetryCountRaised { .. }))
    );
}

#[rstest]
#[tokio::test]
async fn a_retry_policy_that_grants_retries_outranks_a_num_retries_that_raises(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(
        RouterConfig {
            retry_policy: serde_json::from_value(json!({"DefaultRetries": 1})).unwrap(),
            ..config
        },
        json!({}),
        host.clone(),
    );
    let request = RouterRequest {
        uncounted_retries: Some(UncountedRetries::Raises),
        ..request()
    };
    let response = router.call(Op::from("acompletion"), request).await.unwrap();
    assert_eq!(response.model_group, "primary");
    assert_eq!(host.retries(), 1);
}

#[rstest]
#[tokio::test]
async fn a_name_nothing_serves_routes_to_the_default_deployment(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::default());
    let router = router(
        config,
        json!({"default_deployment": {
            "model_name": "dflt", "litellm_params": {"model": "openai/x"}, "model_info": {"id": "dd"}
        }}),
        host.clone(),
    );
    let request = RouterRequest {
        model: "other".to_owned(),
        ..RouterRequest::default()
    };
    let response = router
        .call(Op::from("acompletion"), request.clone())
        .await
        .unwrap();
    assert_eq!(
        (
            response.deployment_id.as_str(),
            response.model_group.as_str()
        ),
        ("dd", "other")
    );
    let picked = router
        .select(&Op::from("acompletion"), &request)
        .await
        .unwrap();
    assert_eq!(picked.params.model, "other");
}

#[rstest]
#[tokio::test]
async fn a_stale_member_map_naming_a_dropped_group_raises_on_every_pick(mut config: RouterConfig) {
    config.routing_group_members = Some([("primary".to_owned(), "gone".to_owned())].into());
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let refused = router.select(&Op::from("acompletion"), &request()).await;
    assert!(matches!(refused, Err(Error::MissingRoutingGroup { group }) if group == "gone"));
    let other = RouterRequest {
        model: "backup".to_owned(),
        ..RouterRequest::default()
    };
    assert!(
        router
            .select(&Op::from("acompletion"), &other)
            .await
            .is_ok()
    );
}

#[rstest]
#[tokio::test]
async fn written_indexes_replace_the_ones_the_deployments_build(config: RouterConfig) {
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let index: litellm_router_core::registry::IndexOverrides = serde_json::from_value(json!({
        "model_names": ["backup"],
        "model_name": {"primary": [1], "backup": [2]},
    }))
    .unwrap();
    router
        .apply(litellm_router_core::ChangeSet {
            index: Some(index),
            ..litellm_router_core::ChangeSet::default()
        })
        .unwrap();
    for _ in 0..4 {
        let picked = router
            .select(&Op::from("acompletion"), &request())
            .await
            .unwrap();
        assert_eq!(picked.id, "a2");
    }
    router
        .apply(litellm_router_core::ChangeSet {
            index: Some(litellm_router_core::registry::IndexOverrides::default()),
            ..litellm_router_core::ChangeSet::default()
        })
        .unwrap();
    assert_eq!(router.registry().group("primary").unwrap().len(), 2);
}

#[rstest]
#[tokio::test]
async fn a_retry_policy_retries_a_pick_that_found_no_deployment(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::default());
    let mut config = config;
    config.retry_policy = serde_json::from_value(json!({"DefaultRetries": 2})).unwrap();
    let router = router(config, json!({"num_retries": 0}), host.clone());
    let request = RouterRequest {
        model: "nothing".to_owned(),
        ..RouterRequest::default()
    };
    let refused = router.call(Op::from("acompletion"), request).await;
    assert!(matches!(refused, Err(Error::UnknownModel { .. })));
    let events = host.events.lock();
    let retries = events
        .iter()
        .filter(|event| matches!(event, RouterEvent::Retry { .. }))
        .count();
    assert_eq!(retries, 2);
    assert!(events.iter().any(|event| matches!(
        event,
        RouterEvent::RetriesExhausted {
            picked: true,
            level: 0,
            ..
        }
    )));
}

#[rstest]
#[tokio::test]
async fn a_default_deployment_python_cannot_copy_raises_for_a_name_nothing_serves(
    config: RouterConfig,
) {
    let router = router(
        config,
        json!({"default_deployment_raises": true}),
        Arc::new(ScriptedHost::default()),
    );
    let request = RouterRequest {
        model: "other".to_owned(),
        ..RouterRequest::default()
    };
    let refused = router.select(&Op::from("acompletion"), &request).await;
    assert!(matches!(refused, Err(Error::DefaultDeploymentRaises { model }) if model == "other"));
    assert!(
        router
            .select(&Op::from("acompletion"), &self::request())
            .await
            .is_ok()
    );
}

#[rstest]
#[tokio::test]
async fn a_written_index_position_past_the_list_raises_for_that_name(config: RouterConfig) {
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let index: litellm_router_core::registry::IndexOverrides =
        serde_json::from_value(json!({"model_name": {"primary": [0, 9], "backup": [2]}})).unwrap();
    router
        .apply(litellm_router_core::ChangeSet {
            index: Some(index),
            ..litellm_router_core::ChangeSet::default()
        })
        .unwrap();
    let refused = router.select(&Op::from("acompletion"), &request()).await;
    assert!(matches!(refused, Err(Error::IndexOutOfRange { model }) if model == "primary"));
    let backup = RouterRequest {
        model: "backup".to_owned(),
        ..RouterRequest::default()
    };
    assert!(
        router
            .select(&Op::from("acompletion"), &backup)
            .await
            .is_ok()
    );
}

#[rstest]
#[case::no_list_takes_the_order_tier(json!(null), true, false, "p2")]
#[case::unmatched_list_takes_the_order_tier(json!([{"unrelated": ["context"]}]), true, false, "p2")]
#[case::unmatched_list_falls_through_to_regular(json!([{"unrelated": ["context"]}]), false, false, "b1")]
#[case::unmatched_list_leaves_the_client_side_list(json!([{"unrelated": ["context"]}]), false, true, "b1")]
#[case::matched_list_beats_the_order_tier(json!([{"primary": ["context"]}]), true, false, "c1")]
#[case::matched_star_beats_the_order_tier(json!([{"*": ["context"]}]), true, false, "c1")]
#[case::matched_list_beats_the_client_side_list(json!([{"primary": ["context"]}]), false, true, "c1")]
#[case::matched_list_beats_both(json!([{"primary": ["context"]}]), true, true, "c1")]
#[tokio::test]
async fn a_context_window_error_takes_its_own_chain_only_when_the_group_matches(
    #[case] context_window_fallbacks: serde_json::Value,
    #[case] tiered: bool,
    #[case] client_side: bool,
    #[case] expected: &str,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(
        tiers(tiered),
        json!({
            "fallbacks": [{"primary": ["backup"]}],
            "context_window_fallbacks": context_window_fallbacks,
        }),
        host.clone(),
    );
    let call = RouterRequest {
        fallbacks: client_side.then(|| Some(serde_json::from_value(json!(["backup"])).unwrap())),
        ..request()
    };
    let response = router.call(Op::from("acompletion"), call).await.unwrap();
    assert_eq!(response.deployment_id, expected);
}

#[rstest]
#[case::context_window_keeps_its_own_error(
    context_window(),
    FailureKind::ContextWindowExceeded,
    false
)]
#[case::any_other_raises_the_last_hop(server_error(), FailureKind::ServiceUnavailable, true)]
#[tokio::test]
async fn a_failed_order_chain_raises_by_the_error_that_started_it(
    #[case] first: AttemptError,
    #[case] raised: FailureKind,
    #[case] last_hop: bool,
) {
    let host = Arc::new(ScriptedHost::scripted(vec![
        ("primary", vec![first, failure(FailureKind::RateLimit, 429)]),
        (
            "backup",
            vec![failure(FailureKind::ServiceUnavailable, 503)],
        ),
    ]));
    let router = router(
        tiers(true),
        json!({
            "fallbacks": [{"primary": ["backup"]}],
            "context_window_fallbacks": [{"unrelated": ["context"]}],
        }),
        host.clone(),
    );
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(matches!(refused, Err(Error::Attempt(failed)) if failed.kind == raised));
    assert_eq!(host.selected(), ["p1", "p2", "b1"]);
    assert!(
        host.events
            .lock()
            .contains(&RouterEvent::OrderChainFailed { level: 0, last_hop })
    );
}

#[rstest]
#[case::context_window_skips_it(context_window(), "backup")]
#[case::a_rate_limit_takes_it(failure(FailureKind::RateLimit, 429), "primary")]
#[tokio::test]
async fn weighted_failover_serves_every_failure_but_a_context_window_error(
    #[case] first: AttemptError,
    #[case] served_by: &str,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![first]));
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "a2"}},
            {"model_name": "backup", "litellm_params": {"model": "anthropic/c"}, "model_info": {"id": "b1"}},
        ],
        "enable_weighted_failover": true,
    }))
    .unwrap();
    let router = router(
        config,
        json!({
            "fallbacks": [{"primary": ["backup"]}],
            "context_window_fallbacks": [{"unrelated": ["backup"]}],
        }),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(response.model_group, served_by);
}

#[rstest]
#[case::agentic_followup(agentic_followup())]
#[case::over_every_tpm_limit(over_every_tpm_limit())]
#[tokio::test]
async fn a_failure_no_retry_clears_is_not_retried_but_still_falls_back(
    config: RouterConfig,
    #[case] refusal: AttemptError,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![refusal]));
    let router = router(
        config,
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(response.deployment_id, "b1");
    assert_eq!(host.retries(), 0);
    assert_eq!(host.selected().len(), 2);
}

#[rstest]
#[case::agentic_followup(agentic_followup())]
#[case::over_every_tpm_limit(over_every_tpm_limit())]
#[tokio::test]
async fn a_failure_no_retry_clears_skips_the_retry_policy_lookup(
    config: RouterConfig,
    #[case] refusal: AttemptError,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![refusal]));
    let router = router(config, json!({"num_retries": 3}), host.clone());
    let call = RouterRequest {
        retry_policy_raises: vec!["primary".to_owned()],
        ..request()
    };
    let refused = router.call(Op::from("acompletion"), call).await;
    assert!(matches!(refused, Err(Error::Attempt(failed)) if failed.ends_retries()));
    assert!(
        !host
            .events
            .lock()
            .iter()
            .any(|event| matches!(event, RouterEvent::RetryPolicyRaised { .. }))
    );
}

#[rstest]
#[case::agentic_followup(agentic_followup())]
#[case::over_every_tpm_limit(over_every_tpm_limit())]
#[tokio::test]
async fn a_retry_that_fails_beyond_any_retry_ends_the_retries(
    config: RouterConfig,
    #[case] refusal: AttemptError,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![server_error(), refusal],
    ));
    let router = router(config, json!({"num_retries": 3}), host.clone());
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(matches!(refused, Err(Error::Attempt(failed)) if failed.ends_retries()));
    assert_eq!((host.retries(), host.selected().len()), (1, 2));
}

fn stream_origin() -> StreamOrigin {
    StreamOrigin {
        group: "primary".to_owned(),
        ..StreamOrigin::default()
    }
}

fn stream_request() -> RouterRequest {
    RouterRequest {
        num_retries: Some(3),
        ..request()
    }
}

#[rstest]
#[tokio::test]
async fn a_stream_that_failed_its_agentic_followup_is_not_retried(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::default());
    let router = router(config, json!({}), host.clone());
    let refused = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            stream_origin(),
            agentic_followup(),
        )
        .await;
    assert!(matches!(refused, Err(Error::Attempt(failed)) if failed.agentic_followup));
    assert_eq!((host.retries(), host.selected().len()), (0, 0));
}

#[rstest]
#[case::agentic_followup(agentic_followup())]
#[case::over_every_tpm_limit(over_every_tpm_limit())]
#[tokio::test]
async fn a_stream_retry_that_fails_beyond_any_retry_ends_the_retries(
    config: RouterConfig,
    #[case] refusal: AttemptError,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![refusal]));
    let router = router(config, json!({}), host.clone());
    let refused = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            stream_origin(),
            server_error(),
        )
        .await;
    assert!(matches!(refused, Err(Error::Attempt(failed)) if failed.ends_retries()));
    assert_eq!((host.retries(), host.selected().len()), (1, 1));
}

#[rstest]
#[tokio::test]
async fn a_stream_retry_is_reported_before_its_attempt(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::default());
    let router = router(config, json!({}), host.clone());
    let response = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            stream_origin(),
            server_error(),
        )
        .await
        .unwrap();
    assert_eq!((response.attempted_retries, host.retries()), (1, 1));
}

fn pinned_to_a1() -> Arc<ScriptedHost> {
    Arc::new(ScriptedHost {
        pinned: Some("a1"),
        ..ScriptedHost::default()
    })
}

#[rstest]
#[tokio::test]
async fn a_stream_retry_leaves_the_pinned_deployment_the_stream_failed_on(
    mut config: RouterConfig,
) {
    config.callback_filters = true;
    let host = pinned_to_a1();
    let router = router(config, json!({}), host.clone());
    let origin = StreamOrigin {
        deployment_id: Some("a1".to_owned()),
        ..stream_origin()
    };
    router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin,
            server_error(),
        )
        .await
        .unwrap();
    assert_eq!(host.selected(), vec!["a2".to_owned()]);
}

#[rstest]
#[tokio::test]
async fn a_stream_retry_leaves_a_pinned_deployment_a_retry_failed_on(mut config: RouterConfig) {
    config.callback_filters = true;
    let host = pinned_to_a1();
    *host.first_failure.lock() = Some(server_error());
    let router = router(config, json!({}), host.clone());
    router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            stream_origin(),
            server_error(),
        )
        .await
        .unwrap();
    assert_eq!(host.selected(), vec!["a1".to_owned(), "a2".to_owned()]);
}

fn fallback_list(value: serde_json::Value) -> Fallbacks {
    serde_json::from_value(value).unwrap()
}

#[rstest]
#[case::names(json!(["backup"]))]
#[case::keyed(json!([{"primary": ["backup"]}]))]
#[case::star(json!([{"*": ["backup"]}]))]
fn a_request_fallback_list_can_take_over_a_stream(
    config: RouterConfig,
    #[case] list: serde_json::Value,
) {
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let request = RouterRequest {
        fallbacks: Some(Some(fallback_list(list))),
        ..request()
    };
    let policy = router.stream_policy(&request, &stream_origin(), None);
    assert!(policy.can_fall_back);
    assert!(policy.refusal_fallback);
}

#[rstest]
fn a_request_content_policy_list_can_take_over_a_stream(config: RouterConfig) {
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let request = RouterRequest {
        content_policy_fallbacks: Some(Some(fallback_list(json!([{"primary": ["backup"]}])))),
        ..request()
    };
    let policy = router.stream_policy(&request, &stream_origin(), None);
    assert!(policy.can_fall_back);
    assert!(policy.refusal_fallback);
}

#[rstest]
#[case::cleared(None)]
#[case::another_group(Some(json!([{"other": ["backup"]}])))]
fn a_request_fallback_list_replaces_the_router_list_for_a_stream(
    config: RouterConfig,
    #[case] list: Option<serde_json::Value>,
) {
    let router = router(
        config,
        json!({"fallbacks": [{"primary": ["backup"]}], "default_fallbacks": ["backup"]}),
        Arc::new(ScriptedHost::default()),
    );
    let request = RouterRequest {
        fallbacks: Some(list.map(fallback_list)),
        ..request()
    };
    let policy = router.stream_policy(&request, &stream_origin(), None);
    assert!(!policy.can_fall_back);
}

#[rstest]
#[case::readable(vec![], 5, true)]
#[case::malformed(vec!["primary".to_owned()], 1, false)]
fn a_group_policy_that_does_not_read_governs_no_stream_retry(
    mut config: RouterConfig,
    #[case] retry_policy_raises: Vec<String>,
    #[case] budget: u32,
    #[case] applies: bool,
) {
    config.retry_policy = serde_json::from_value(json!({"InternalServerErrorRetries": 5})).unwrap();
    let router = router(
        config,
        json!({"num_retries": 1}),
        Arc::new(ScriptedHost::default()),
    );
    let request = RouterRequest {
        retry_policy_raises,
        ..request()
    };
    let policy = router.stream_policy(
        &request,
        &stream_origin(),
        Some(FailureKind::InternalServer),
    );
    assert_eq!(policy.retry_budget, Some((budget, applies)));
}

#[rstest]
#[tokio::test]
async fn a_level_reports_its_fallback_handling_before_any_hop(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let events = host.events.lock();
    let falling_back = RouterEvent::FallingBack {
        level: 0,
        model: "primary".to_owned(),
    };
    let reported = events.iter().position(|e| *e == falling_back).unwrap();
    let hopped = events
        .iter()
        .position(|e| matches!(e, RouterEvent::Fallback { .. }))
        .unwrap();
    assert!(reported < hopped);
}

#[rstest]
#[tokio::test]
async fn the_order_path_reports_no_fallback_handling() {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(tiers(true), json!({"num_retries": 0}), host.clone());
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert!(
        !host
            .events
            .lock()
            .iter()
            .any(|e| matches!(e, RouterEvent::FallingBack { .. }))
    );
}

fn fallbacks_failed(host: &ScriptedHost) -> Vec<(u32, bool, bool)> {
    host.events
        .lock()
        .iter()
        .filter_map(|event| match event {
            RouterEvent::FallbacksFailed {
                level,
                chain,
                last_hop,
            } => Some((*level, *chain, *last_hop)),
            _ => None,
        })
        .collect()
}

#[rstest]
#[tokio::test]
async fn a_chain_that_fails_reports_every_level_innermost_first(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::scripted(vec![
        ("primary", vec![server_error()]),
        ("backup", vec![server_error()]),
    ]));
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap_err();
    assert_eq!(
        fallbacks_failed(&host),
        vec![(1, true, false), (0, true, true)]
    );
}

#[rstest]
#[tokio::test]
async fn a_chain_past_max_fallbacks_reports_the_levels_own_failure(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(
        config,
        json!({"num_retries": 0, "max_fallbacks": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap_err();
    assert_eq!(fallbacks_failed(&host), vec![(0, true, false)]);
}

#[rstest]
#[tokio::test]
async fn a_served_fallback_and_a_disabled_walk_report_no_failure(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let dynamic = json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]});
    let served = router(config.clone(), dynamic.clone(), host.clone());
    served
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(fallbacks_failed(&host), vec![]);

    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let disabled = router(config, dynamic, host.clone());
    let request = RouterRequest {
        disable_fallbacks: true,
        ..request()
    };
    disabled
        .call(Op::from("acompletion"), request)
        .await
        .unwrap_err();
    assert_eq!(fallbacks_failed(&host), vec![]);
}

fn azure_chat_and_batch_tiers() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "azure/chat", "order": 1}, "model_info": {"id": "c1", "mode": "chat"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/batch-a", "order": 2}, "model_info": {"id": "b2", "mode": "batch"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/batch-b", "order": 3}, "model_info": {"id": "b3", "mode": "batch"}},
        ]
    }))
    .unwrap()
}

fn azure_chat_tiers() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "azure/chat-a", "order": 1}, "model_info": {"id": "c1", "mode": "chat"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/chat-b", "order": 2}, "model_info": {"id": "c2", "mode": "chat"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::batch_create_moves_past_the_level_it_used(azure_chat_and_batch_tiers(), true, vec!["b2", "b3"])]
#[case::other_calls_walk_every_level(azure_chat_and_batch_tiers(), false, vec!["c1", "b2"])]
#[case::batch_create_without_a_batch_deployment_walks_every_level(azure_chat_tiers(), true, vec!["c1", "c2"])]
#[tokio::test]
async fn a_failed_order_level_falls_back_to_the_next_level_its_pick_can_use(
    #[case] config: RouterConfig,
    #[case] batch_create: bool,
    #[case] expected: Vec<&str>,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(config, json!({"num_retries": 0}), host.clone());
    let request = RouterRequest {
        batch_create,
        ..request()
    };
    router
        .call(Op::from("acreate_batch"), request)
        .await
        .unwrap();
    assert_eq!(host.selected(), expected);
}

fn azure_chat_then_batch() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "azure/chat", "order": 1}, "model_info": {"id": "c1", "mode": "chat"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/batch", "order": 2}, "model_info": {"id": "b2", "mode": "batch"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::a_batch_upload_finds_no_deployment(true, vec![])]
#[case::a_batch_create_falls_back_to_chat(false, vec!["c1"])]
#[tokio::test]
async fn while_the_batch_deployment_cools_down_a_batch_upload_finds_no_healthy_deployment(
    #[case] batch_upload: bool,
    #[case] expected: Vec<&str>,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![failure(FailureKind::RateLimit, 429)],
    ));
    let router = router(
        azure_chat_then_batch(),
        json!({"num_retries": 0}),
        host.clone(),
    );
    let create = RouterRequest {
        batch_create: true,
        ..request()
    };
    router
        .call(Op::from("acreate_file"), create.clone())
        .await
        .unwrap_err();
    let upload = RouterRequest {
        batch_upload,
        ..create
    };
    let healthy: Vec<String> = router
        .healthy(&Op::from("acreate_file"), &upload)
        .await
        .map(|candidates| {
            candidates
                .healthy
                .iter()
                .map(|d| d.id.to_string())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(host.selected(), vec!["b2"]);
    assert_eq!(healthy, expected);
}

#[rstest]
#[case::a_batch_upload_finds_no_deployment(true, vec![])]
#[case::a_batch_create_falls_back_to_chat(false, vec!["c1"])]
#[tokio::test]
async fn while_the_batch_deployment_is_paused_a_batch_upload_finds_no_healthy_deployment(
    #[case] batch_upload: bool,
    #[case] expected: Vec<&str>,
) {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "azure/chat", "order": 1}, "model_info": {"id": "c1", "mode": "chat"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/batch", "order": 2}, "model_info": {"id": "b2", "mode": "batch", "blocked": true}},
        ]
    }))
    .unwrap();
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let upload = RouterRequest {
        batch_create: true,
        batch_upload,
        ..request()
    };
    let healthy: Vec<String> = router
        .healthy(&Op::from("acreate_file"), &upload)
        .await
        .map(|candidates| {
            candidates
                .healthy
                .iter()
                .map(|d| d.id.to_string())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(healthy, expected);
}

#[rstest]
#[case::a_rate_limit(FailureKind::RateLimit, 429)]
#[case::a_wrong_key(FailureKind::Authentication, 401)]
#[tokio::test]
async fn a_batch_upload_the_batch_deployment_refuses_keeps_its_error_with_retries_on(
    #[case] kind: FailureKind,
    #[case] status: u16,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        (0..3).map(|_| failure(kind, status)).collect(),
    ));
    let router = router(
        azure_chat_then_batch(),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let upload = RouterRequest {
        batch_create: true,
        batch_upload: true,
        ..request()
    };
    let error = router
        .call(Op::from("acreate_file"), upload)
        .await
        .unwrap_err();
    assert_eq!(host.selected(), vec!["b2"]);
    assert!(matches!(error, Error::Attempt(refused) if refused.kind == kind));
}

#[tokio::test]
async fn the_order_path_reports_no_failed_fallbacks() {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(tiers(true), json!({"num_retries": 0}), host.clone());
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(fallbacks_failed(&host), vec![]);
}

fn refusing_host(group: &str) -> Arc<ScriptedHost> {
    Arc::new(ScriptedHost {
        refuses_dict_targets: true,
        ..ScriptedHost::failing(group, vec![server_error()])
    })
}

#[rstest]
#[case::dict_target_first(json!([{"model": "backup", "api_key": "x"}, "backup"]))]
#[case::dict_target_last(json!(["backup", {"model": "backup", "api_key": "x"}]))]
#[tokio::test]
async fn a_refused_chain_tries_none_of_its_targets(
    config: RouterConfig,
    #[case] chain: serde_json::Value,
) {
    let host = refusing_host("primary");
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": chain}]}),
        host.clone(),
    );
    let raised = router.call(Op::from("acompletion"), request()).await;
    assert!(matches!(raised, Err(Error::Attempt(failed)) if failed.is_a("InternalServerError")));
    assert_eq!(host.selected().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_chain_of_names_is_never_refused(config: RouterConfig) {
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        refusing_host("primary"),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(response.deployment_id, "b1");
}

#[rstest]
#[tokio::test]
async fn a_refused_chain_on_the_order_path_raises_the_refusal() {
    let host = refusing_host("primary");
    let router = router(
        tiers(true),
        json!({"num_retries": 0, "fallbacks": [{"primary": [{"model": "backup", "api_key": "x"}]}]}),
        host.clone(),
    );
    let raised = router.call(Op::from("acompletion"), request()).await;
    assert!(matches!(raised, Err(Error::Attempt(failed)) if failed.is_a("ValueError")));
    assert_eq!(host.selected(), ["p1"]);
}

#[rstest]
#[tokio::test]
async fn a_refused_chain_fails_only_its_own_level(mut config: RouterConfig) {
    let host = Arc::new(ScriptedHost {
        refuses_dict_targets: true,
        ..ScriptedHost::scripted(vec![
            ("primary", vec![server_error()]),
            ("backup", vec![server_error()]),
        ])
    });
    let dynamic = json!({"num_retries": 0, "fallbacks": [
        {"primary": ["backup", "third"]},
        {"backup": [{"model": "third", "api_key": "x"}]},
    ]});
    config.model_list.push(
        serde_json::from_value(json!({
            "model_name": "third", "litellm_params": {"model": "openai/t"}, "model_info": {"id": "t1"},
        }))
        .unwrap(),
    );
    let response = router(config, dynamic, host.clone())
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(response.deployment_id, "t1");
}

fn shared_public_name() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "backup", "litellm_params": {"model": "openai/g"}, "model_info": {"id": "b1"}},
            {"model_name": "t1-pub", "litellm_params": {"model": "openai/m"},
             "model_info": {"id": "p1", "team_id": "T1", "team_public_model_name": "pub"}},
            {"model_name": "t2-pub", "litellm_params": {"model": "openai/m"},
             "model_info": {"id": "p2", "team_id": "T2", "team_public_model_name": "pub"}},
        ]
    }))
    .unwrap()
}

fn proxy_admin_of_pub() -> RouterRequest {
    RouterRequest {
        model: "pub".to_owned(),
        proxy_admin: true,
        admin_model: Some("pub".to_owned()),
        ..RouterRequest::default()
    }
}

#[rstest]
#[tokio::test]
async fn a_name_several_teams_share_falls_back() {
    let router = router(
        shared_public_name(),
        json!({"fallbacks": [{"pub": ["backup"]}]}),
        Arc::new(ScriptedHost::default()),
    );
    let response = router
        .call(Op::from("acompletion"), proxy_admin_of_pub())
        .await
        .unwrap();
    assert_eq!(
        (
            response.deployment_id.as_str(),
            response.attempted_fallbacks
        ),
        ("b1", 1)
    );
}

#[rstest]
#[tokio::test]
async fn a_name_several_teams_share_raises_its_refusal_when_the_fallback_fails() {
    let router = router(
        shared_public_name(),
        json!({"fallbacks": [{"pub": ["backup"]}]}),
        Arc::new(ScriptedHost::failing("backup", vec![server_error()])),
    );
    let raised = router
        .call(Op::from("acompletion"), proxy_admin_of_pub())
        .await;
    assert!(matches!(raised, Err(Error::MultipleTeams { model }) if model == "pub"));
}

/// A group whose second order level belongs to another team.
fn team_tiers() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/p1", "order": 1},
             "model_info": {"id": "p1", "team_id": "t1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/p2", "order": 2},
             "model_info": {"id": "p2", "team_id": "t2"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::listed_for_the_team(Some("t1"), false)]
#[case::listed_for_no_team(None, true)]
#[tokio::test]
async fn a_failed_level_lists_its_order_levels_for_the_metadata_team(
    #[case] listing_team_id: Option<&str>,
    #[case] walks_the_order_chain: bool,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(team_tiers(), json!({"num_retries": 0}), host.clone());
    let request = RouterRequest {
        team_id: Some("t1".to_owned()),
        listing_team_id: listing_team_id.map(str::to_owned),
        ..request()
    };
    let raised = router.call(Op::from("acompletion"), request).await;
    assert_eq!(
        matches!(raised, Err(Error::NoHealthyDeployments { .. })),
        walks_the_order_chain
    );
    assert_eq!(host.selected(), ["p1"]);
}

fn tagged_request() -> RouterRequest {
    RouterRequest {
        filter_tags: true,
        ..request()
    }
}

#[rstest]
#[tokio::test]
async fn a_failed_pick_retries_like_a_failed_attempt(config: RouterConfig) {
    let host = Arc::new(ScriptedHost {
        pick_failures: Mutex::new(VecDeque::from([failure(FailureKind::RateLimit, 429)])),
        ..ScriptedHost::default()
    });
    let response = router(config, json!({"num_retries": 1}), host.clone())
        .call(Op::from("acompletion"), tagged_request())
        .await
        .unwrap();
    assert_eq!(
        (response.model_group.as_str(), response.attempted_retries),
        ("primary", 1)
    );
}

#[rstest]
#[tokio::test]
async fn a_failed_pick_falls_back_with_no_deployment_to_blame(config: RouterConfig) {
    let host = Arc::new(ScriptedHost {
        pick_failures: Mutex::new(VecDeque::from([failure(FailureKind::RateLimit, 429)])),
        ..ScriptedHost::default()
    });
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), tagged_request())
        .await
        .unwrap();
    assert_eq!(
        (
            response.deployment_id.as_str(),
            response.attempted_fallbacks
        ),
        ("b1", 1)
    );
    assert_eq!(host.selected(), ["b1"]);
}

#[rstest]
#[tokio::test]
async fn a_fallback_check_that_raises_ends_the_chain_with_the_levels_own_error(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost {
        fallback_check_raises: Some("TypeError"),
        ..ScriptedHost::failing("primary", vec![failure(FailureKind::RateLimit, 429)])
    });
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let raised = router.call(Op::from("acompletion"), request()).await;
    assert!(matches!(raised, Err(Error::Attempt(failed)) if failed.kind == FailureKind::RateLimit));
    assert_eq!(host.selected().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_fallback_check_that_raises_on_the_order_path_raises_its_own_error() {
    let host = Arc::new(ScriptedHost {
        fallback_check_raises: Some("TypeError"),
        ..ScriptedHost::scripted(vec![("primary", vec![server_error(), server_error()])])
    });
    let router = router(
        tiers(true),
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let raised = router.call(Op::from("acompletion"), request()).await;
    assert!(matches!(raised, Err(Error::Attempt(failed)) if failed.is_a("TypeError")));
}

#[rstest]
#[case::an_int(json!(5))]
#[case::none(json!(null))]
#[tokio::test]
async fn a_dict_target_with_a_non_string_model_is_a_hop_that_attempts_nothing(
    config: RouterConfig,
    #[case] model: serde_json::Value,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": [{"model": model}, "backup"]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(
        (
            response.deployment_id.as_str(),
            response.attempted_fallbacks
        ),
        ("b1", 2)
    );
    assert_eq!(host.selected().len(), 2);
}

#[rstest]
#[tokio::test]
async fn a_dict_target_without_a_model_serves_the_group_that_failed_again(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": [{"temperature": 0.5}]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(
        (response.model_group.as_str(), response.attempted_fallbacks),
        ("primary", 1)
    );
}

fn weighted_failover() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "a2"}},
        ],
        "enable_weighted_failover": true,
    }))
    .unwrap()
}

#[rstest]
#[case::a_bug_propagates(AttemptError::host_raised(&["TypeError", "Exception"]), "TypeError")]
#[case::a_provider_error_defers(failure(FailureKind::RateLimit, 429), "InternalServerError")]
#[case::no_deployment_defers(
    AttemptError::host_raised(&["RouterRateLimitError", "ValueError"]),
    "InternalServerError"
)]
#[tokio::test]
async fn weighted_failover_swallows_only_api_and_rate_limit_errors(
    #[case] hop_failure: AttemptError,
    #[case] raised_class: &str,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![server_error(), hop_failure],
    ));
    let raised = router(weighted_failover(), json!({"num_retries": 0}), host.clone())
        .call(Op::from("acompletion"), request())
        .await;
    assert!(matches!(raised, Err(Error::Attempt(failed)) if failed.is_a(raised_class)));
    assert_eq!(host.selected().len(), 2);
}

fn wildcard_tiers() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*", "order": 1}, "model_info": {"id": "w1"}},
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*", "order": 2}, "model_info": {"id": "w2"}},
        ]
    }))
    .unwrap()
}

fn alias_tiers() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "gg", "litellm_params": {"model": "openai/a", "order": 1}, "model_info": {"id": "w1"}},
            {"model_name": "gg", "litellm_params": {"model": "openai/b", "order": 2}, "model_info": {"id": "w2"}},
        ]
    }))
    .unwrap()
}

fn team_public_tiers() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "t1a", "litellm_params": {"model": "openai/a", "order": 1},
             "model_info": {"id": "w1", "team_id": "T1", "team_public_model_name": "tp"}},
            {"model_name": "t1b", "litellm_params": {"model": "openai/b", "order": 2},
             "model_info": {"id": "w2", "team_id": "T1", "team_public_model_name": "tp"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::a_wildcard(wildcard_tiers(), json!({"num_retries": 0}), "openai/gpt-4o", None)]
#[case::an_alias(
    alias_tiers(),
    json!({"num_retries": 0, "model_group_alias": {"al": "gg"}}),
    "al",
    None
)]
#[case::a_team_public_name(team_public_tiers(), json!({"num_retries": 0}), "tp", Some("T1"))]
#[tokio::test]
async fn order_levels_walk_through_any_name_the_model_list_lists(
    #[case] config: RouterConfig,
    #[case] dynamic: serde_json::Value,
    #[case] model: &str,
    #[case] team: Option<&str>,
) {
    let host = Arc::new(ScriptedHost::default());
    *host.first_failure.lock() = Some(failure(FailureKind::RateLimit, 429));
    let request = RouterRequest {
        model: model.to_owned(),
        team_id: team.map(str::to_owned),
        listing_team_id: team.map(str::to_owned),
        ..RouterRequest::default()
    };
    let response = router(config, dynamic, host.clone())
        .call(Op::from("acompletion"), request)
        .await
        .unwrap();
    assert_eq!(response.deployment_id, "w2");
}

#[rstest]
#[tokio::test]
async fn a_team_wildcard_public_name_routes_for_its_team() {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "t1-w", "litellm_params": {"model": "openai/*"},
             "model_info": {"id": "tw1", "team_id": "T1", "team_public_model_name": "tw-*"}},
            {"model_name": "g", "litellm_params": {"model": "openai/g"}, "model_info": {"id": "gp"}},
        ]
    }))
    .unwrap();
    let request = RouterRequest {
        model: "tw-gpt-4o".to_owned(),
        team_id: Some("T1".to_owned()),
        ..RouterRequest::default()
    };
    let response = router(config, json!({}), Arc::new(ScriptedHost::default()))
        .call(Op::from("acompletion"), request)
        .await
        .unwrap();
    assert_eq!(response.deployment_id, "tw1");
}

#[rstest]
#[tokio::test]
async fn a_paused_deployment_named_by_id_falls_back() {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"},
             "model_info": {"id": "a1", "blocked": true}},
            {"model_name": "backup", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "b1"}},
        ]
    }))
    .unwrap();
    let request = RouterRequest {
        model: "a1".to_owned(),
        ..RouterRequest::default()
    };
    let response = router(
        config,
        json!({"fallbacks": [{"a1": ["backup"]}]}),
        Arc::new(ScriptedHost::default()),
    )
    .call(Op::from("acompletion"), request)
    .await
    .unwrap();
    assert_eq!(response.deployment_id, "b1");
}

/// `primary` with `a1`'s `model_info` as given, and an unremarkable `a2`.
fn with_first_info(info: serde_json::Value) -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": info},
            {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "a2"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[tokio::test]
async fn a_web_search_request_skips_the_deployments_that_declare_no_support() {
    let config = with_first_info(json!({"id": "a1", "supports_web_search": false}));
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let request = RouterRequest {
        web_search: true,
        ..request()
    };
    for _ in 0..16 {
        let picked = router
            .select(&Op::from("acompletion"), &request)
            .await
            .unwrap();
        assert_eq!(picked.id, "a2");
    }
}

#[rstest]
#[tokio::test]
async fn an_endpoint_a_deployment_refuses_routes_to_the_others() {
    let config = with_first_info(json!({
        "id": "a1",
        "enforce_supported_endpoints": true,
        "supported_endpoints": ["/v1/embeddings"],
    }));
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let request = RouterRequest {
        endpoint: Some("/v1/chat/completions".to_owned()),
        ..request()
    };
    for _ in 0..16 {
        let picked = router
            .select(&Op::from("acompletion"), &request)
            .await
            .unwrap();
        assert_eq!(picked.id, "a2");
    }
}

#[rstest]
#[tokio::test]
async fn an_endpoint_every_deployment_refuses_fails_the_pick() {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [{
            "model_name": "primary",
            "litellm_params": {"model": "openai/a"},
            "model_info": {
                "id": "a1",
                "enforce_supported_endpoints": true,
                "supported_endpoints": ["/v1/embeddings"],
            },
        }]
    }))
    .unwrap();
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let request = RouterRequest {
        endpoint: Some("/v1/chat/completions".to_owned()),
        ..request()
    };
    let refused = router
        .call(Op::from("acompletion"), request)
        .await
        .unwrap_err();
    assert!(matches!(
        refused,
        Error::EndpointNotSupported { model, message }
            if model == "primary" && message
                == "/v1/chat/completions is not supported by model=primary. \
                    Supported endpoints: /v1/embeddings"
    ));
}

#[rstest]
#[tokio::test]
async fn an_order_in_model_info_narrows_the_pick() {
    let mut config = with_first_info(json!({"id": "a1", "order": 2}));
    config.model_list[1]
        .model_info
        .insert("order".to_owned(), json!("1"));
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    for _ in 0..16 {
        let picked = router
            .select(&Op::from("acompletion"), &request())
            .await
            .unwrap();
        assert_eq!(picked.id, "a2");
    }
}

#[rstest]
#[tokio::test]
async fn a_fallback_hop_drops_the_order_the_request_named() {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![failure(FailureKind::InternalServer, 500)],
    ));
    let router = router(
        tiers(false),
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host,
    );
    let request = RouterRequest {
        target_order: Some(1),
        ..request()
    };
    let response = router.call(Op::from("acompletion"), request).await.unwrap();
    assert_eq!(response.deployment_id, "b1");
}

#[rstest]
#[tokio::test]
async fn a_listed_pass_through_pick_names_the_group_an_alias_resolves_to(config: RouterConfig) {
    let router = router(
        config,
        json!({"model_alias_map": {"al": "primary"}}),
        Arc::new(ScriptedHost::default()),
    );
    let request = RouterRequest {
        model: "al".to_owned(),
        pass_through: Some(PassThrough::Listed),
        ..RouterRequest::default()
    };
    let refused = router
        .select(&Op::from("acompletion"), &request)
        .await
        .unwrap_err();
    assert!(matches!(
        refused,
        Error::NoPassThroughDeployments { model } if model == "primary"
    ));
}

struct StoppedClock(u64);

impl litellm_router_core::health::Clock for StoppedClock {
    fn now(&self) -> std::time::SystemTime {
        std::time::UNIX_EPOCH + Duration::from_secs(self.0)
    }
}

fn utc_hour(hour: u64) -> u64 {
    1_767_225_600 + hour * 3_600
}

#[rstest]
#[case::inside_the_window(utc_hour(10), vec!["open"])]
#[case::outside_the_window(utc_hour(20), vec!["open", "reserved"])]
#[tokio::test]
async fn access_windows_follow_the_routers_clock(#[case] now: u64, #[case] expected: Vec<&str>) {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "chat", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "open"}},
            {"model_name": "chat", "litellm_params": {"model": "openai/b"}, "model_info": {
                "id": "reserved",
                "access_windows": [{"start": "09:00", "end": "17:00", "timezone": "UTC", "team_ids": ["batch"]}],
            }},
        ]
    }))
    .unwrap();
    let router = Router::builder(
        config,
        RouterDynamicConfig::default(),
        Arc::new(ScriptedHost::default()),
    )
    .randomness(Arc::new(SeededRandomness::new(7)))
    .clock(Arc::new(StoppedClock(now)))
    .build()
    .unwrap();
    let mut picked = std::collections::BTreeSet::new();
    for _ in 0..40 {
        let deployment = router
            .select(&Op::from("acompletion"), &request_for("chat"))
            .await
            .unwrap();
        picked.insert(deployment.id.clone());
    }
    assert_eq!(picked.into_iter().collect::<Vec<_>>(), expected);
}

fn request_for(model: &str) -> RouterRequest {
    RouterRequest {
        model: model.to_owned(),
        ..RouterRequest::default()
    }
}

#[rstest]
#[tokio::test]
async fn every_event_of_a_retried_and_fallen_back_call_serializes_tagged_by_its_name(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![failure(FailureKind::RateLimit, 429); 3],
    ));
    let router = router(
        config,
        json!({"num_retries": 1, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let tags: Value = host
        .events
        .lock()
        .iter()
        .map(|event| serde_json::to_value(event).unwrap()["event"].clone())
        .collect();
    assert_eq!(
        tags,
        json!([
            "RetriesOpened",
            "PickStarted",
            "Selected",
            "AttemptFailed",
            "CooldownSet",
            "RetryBudget",
            "RetryLogged",
            "Retry",
            "PickStarted",
            "Selected",
            "AttemptFailed",
            "CooldownSet",
            "RetryLogged",
            "FallingBack",
            "RetryLogged",
            "Fallback",
            "RetriesOpened",
            "PickStarted",
            "Selected",
            "Succeeded",
            "FallbackDone",
        ])
    );
}
