//! A call through the router end to end: selection, retries, fallbacks and cooldowns against a
//! scripted host.

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use litellm_router_core::{
    Attempt, AttemptError, AttemptFailure, CallbackFiltered, Deployment, Error, FailureKind,
    FailureReport, Health, HealthState, HostStep, InMemoryRouterCache, LimitCounter, LimitKind, Op,
    Outcomes, PreRoute, PreRouted, RaisedBy, Router, RouterCache, RouterConfig,
    RouterDynamicConfig, RouterEvent, RouterHost, RouterRequest, RouterResponse, RoutingContext,
    SeededRandomness, SelectStage, Served, SuccessReport, UsageCounts, UsageWindow,
    call::{StreamOrigin, Unanswered},
    fallback::Fallbacks,
    health::CooldownValue,
    health::current_minute,
    host::UncountedRetries,
    mock::MockFailure,
    precall::{DeploymentFacts, PreCallFacts},
    select,
    select::PassThrough,
};
use parking_lot::Mutex;
use rstest::{fixture, rstest};
use serde_json::{Value, json};

#[derive(Default)]
struct ScriptedHost {
    /// Failures handed out in order, keyed by model group; a group with none left succeeds.
    script: Mutex<Vec<(String, VecDeque<AttemptError>)>>,
    events: Mutex<Vec<RouterEvent>>,
    cache: RecordingCache,
    /// Answer every success as a stream whose outcome the host reports when it ends.
    deferred: bool,
    /// [`Served::tracks_usage`] on every success.
    tracks_usage: Option<bool>,
    /// Refuse every fallback chain with a dict target.
    refuses_dict_targets: bool,
    /// Every fallback check raises this exception class, as a host's sync check does.
    fallback_check_raises: Option<&'static str>,
    /// Failures the tag stage raises in turn, before it answers.
    pick_failures: Mutex<VecDeque<AttemptError>>,
    /// The next attempt's failure, whatever group it serves.
    first_failure: Mutex<Option<AttemptError>>,
    /// Every attempt ends the call from inside it.
    refuses: bool,
    /// The callback filters pin to this deployment until an attempt of the request failed on it,
    /// as `encrypted_content_affinity` does.
    pinned: Option<&'static str>,
    /// A pick of this name is answered this way, as a pre-routing hook reroutes it.
    pre_routes: Option<(&'static str, PreRouted)>,
    /// The params each attempt carried, in the order the attempts ran.
    params_sent: Mutex<Vec<Option<Value>>>,
    /// What every pre-call check is answered with.
    pre_call_facts: PreCallFacts,
    callback_filter_runs: Mutex<usize>,
    text_only_ids: &'static [&'static str],
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

/// An [`InMemoryRouterCache`] that keeps the ids of every cooldown read, in order, and answers
/// the cooldowns a test holds with no expiry of their own, so only the router's clock ends them.
#[derive(Default)]
struct RecordingCache {
    inner: InMemoryRouterCache,
    cooldown_reads: Mutex<Vec<Vec<String>>>,
    held: Mutex<HashMap<String, CooldownValue>>,
    health_reads: Mutex<u32>,
}

impl RecordingCache {
    fn hold_cooldown(&self, id: &str, timestamp: f64, cooldown_time: f64) {
        let cooldown = CooldownValue {
            exception_received: "rate limited".to_owned(),
            status_code: "429".to_owned(),
            timestamp,
            cooldown_time,
        };
        self.held.lock().insert(id.to_owned(), cooldown);
    }

    /// Every deployment id any cooldown read asked for.
    fn cooldown_ids_read(&self) -> BTreeSet<String> {
        self.cooldown_reads
            .lock()
            .iter()
            .flatten()
            .cloned()
            .collect()
    }
}

impl std::ops::Deref for RecordingCache {
    type Target = InMemoryRouterCache;

    fn deref(&self) -> &InMemoryRouterCache {
        &self.inner
    }
}

impl RouterCache for RecordingCache {
    async fn cooldowns(
        &self,
        ids: &[String],
    ) -> Result<Vec<Option<CooldownValue>>, litellm_cache::Error> {
        self.cooldown_reads.lock().push(ids.to_vec());
        let stored = self.inner.cooldowns(ids).await?;
        let held = self.held.lock();
        Ok(ids
            .iter()
            .zip(stored)
            .map(|(id, stored)| held.get(id).cloned().or(stored))
            .collect())
    }

    async fn put_cooldown(
        &self,
        id: &str,
        cooldown: &CooldownValue,
        ttl: Duration,
    ) -> Result<(), litellm_cache::Error> {
        self.inner.put_cooldown(id, cooldown, ttl).await
    }

    async fn increment_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<f64, litellm_cache::Error> {
        self.inner.increment_allowed_fails(id, suffix, ttl).await
    }

    async fn add_usage(
        &self,
        window: &UsageWindow,
        tokens: f64,
        requests: f64,
    ) -> Result<(), litellm_cache::Error> {
        self.inner.add_usage(window, tokens, requests).await
    }

    async fn add_request(&self, window: &UsageWindow) -> Result<(), litellm_cache::Error> {
        self.inner.add_request(window).await
    }

    async fn usage(&self, window: &UsageWindow) -> Result<UsageCounts, litellm_cache::Error> {
        self.inner.usage(window).await
    }

    async fn health_states(&self) -> Result<HashMap<String, HealthState>, litellm_cache::Error> {
        *self.health_reads.lock() += 1;
        self.inner.health_states().await
    }

    async fn outcomes(&self, id: &str) -> Result<Outcomes, litellm_cache::Error> {
        self.inner.outcomes(id).await
    }

    async fn count_success(&self, id: &str) -> Result<(), litellm_cache::Error> {
        self.inner.count_success(id).await
    }

    async fn count_failure(&self, id: &str) -> Result<(), litellm_cache::Error> {
        self.inner.count_failure(id).await
    }

    async fn limit_usage(
        &self,
        counter: &LimitCounter,
        limit: f64,
    ) -> Result<Option<f64>, litellm_cache::Error> {
        self.inner.limit_usage(counter, limit).await
    }

    async fn add_limit_usage(
        &self,
        counter: &LimitCounter,
        amount: i64,
        ttl: Duration,
    ) -> Result<Option<f64>, litellm_cache::Error> {
        self.inner.add_limit_usage(counter, amount, ttl).await
    }

    async fn add_limit_usages(
        &self,
        increments: &[(LimitCounter, i64, Duration)],
    ) -> Result<Option<Vec<f64>>, litellm_cache::Error> {
        self.inner.add_limit_usages(increments).await
    }
}

/// The host error behind a scripted failure: the deployment it was raised on.
#[derive(Debug, PartialEq)]
struct Raised(String);

impl ScriptedHost {
    fn raised(attempt: &Attempt, facts: AttemptError) -> AttemptFailure<Raised> {
        AttemptFailure::Failed {
            facts,
            raised: Raised(attempt.deployment.id.clone()),
        }
    }
}

impl RouterHost for ScriptedHost {
    type Response = String;
    type Failure = Raised;
    type Cache = RecordingCache;

    async fn attempt(&self, attempt: &Attempt) -> Result<Served<String>, AttemptFailure<Raised>> {
        let params = attempt.params.as_deref().cloned().map(Value::Object);
        self.params_sent.lock().push(params);
        if let Some(failure) = self.first_failure.lock().take() {
            return Err(Self::raised(attempt, failure));
        }
        if self.refuses {
            return Err(AttemptFailure::Refused(Raised(
                attempt.deployment.id.clone(),
            )));
        }
        let next = self
            .script
            .lock()
            .iter_mut()
            .find(|(group, _)| *group == attempt.model_group)
            .and_then(|(_, failures)| failures.pop_front());
        match next {
            Some(failure) => Err(Self::raised(attempt, failure)),
            None => Ok(Served {
                deferred: self.deferred,
                tracks_usage: self.tracks_usage,
                ..Served::new(attempt.deployment.id.clone(), 10)
            }),
        }
    }

    async fn pre_route(&self, ask: &PreRoute<'_>) -> Result<Option<PreRouted>, Error> {
        Ok(self
            .pre_routes
            .as_ref()
            .filter(|(name, _)| *name == ask.name)
            .map(|(_, routed)| routed.clone()))
    }

    async fn select(
        &self,
        stage: &SelectStage,
        _ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        if matches!(stage, SelectStage::TextOnly) {
            return Ok(candidates
                .into_iter()
                .filter(|deployment| self.text_only_ids.contains(&deployment.id.as_str()))
                .collect());
        }
        match self.pick_failures.lock().pop_front() {
            Some(failure) => Err(failure.into()),
            None => Ok(candidates),
        }
    }

    async fn pre_call_facts(
        &self,
        _ctx: &RoutingContext<'_>,
        _candidates: &[Arc<Deployment>],
    ) -> Result<PreCallFacts, Error> {
        Ok(self.pre_call_facts.clone())
    }

    async fn callback_filters(
        &self,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<CallbackFiltered, Error> {
        *self.callback_filter_runs.lock() += 1;
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

    async fn allow_fallback(&self, _target: &str) -> Result<bool, AttemptError> {
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

    fn cache(&self) -> &RecordingCache {
        &self.cache
    }
}

/// A router with the host its calls run on.
struct Hosted {
    router: Router,
    host: Arc<ScriptedHost>,
}

impl std::ops::Deref for Hosted {
    type Target = Router;

    fn deref(&self) -> &Router {
        &self.router
    }
}

impl Hosted {
    async fn call(&self, op: Op, request: RouterRequest) -> Result<RouterResponse<String>, Error> {
        self.router
            .call(&*self.host, op, request)
            .await
            .map_err(|ended| ended.error)
    }

    async fn select(&self, op: &Op, request: &RouterRequest) -> Result<Arc<Deployment>, Error> {
        self.router.select(&*self.host, op, request).await
    }

    async fn healthy(&self, op: &Op, request: &RouterRequest) -> Result<select::Candidates, Error> {
        self.router.healthy(&*self.host, op, request).await
    }

    fn health(&self) -> Health<'_, RecordingCache> {
        self.router.health(self.host.cache())
    }

    async fn report_success(
        &self,
        id: &str,
        total_tokens: f64,
        requests: f64,
        report: SuccessReport,
    ) -> Result<Option<UsageWindow>, Error> {
        self.router
            .report_success(self.host.cache(), id, total_tokens, requests, report)
            .await
    }

    async fn report_failure(
        &self,
        id: &str,
        failure: &AttemptError,
        report: FailureReport,
    ) -> Result<Option<Duration>, Error> {
        self.router
            .report_failure(self.host.cache(), id, failure, report)
            .await
    }

    async fn retry_stream(
        &self,
        op: Op,
        request: RouterRequest,
        origin: StreamOrigin,
        failure: AttemptError,
    ) -> Result<RouterResponse<String>, Error> {
        self.router
            .retry_stream(&*self.host, op, request, origin, failure)
            .await
            .map_err(|ended| ended.error)
    }

    /// `None` for a stream no attempt may answer in place of, else how its take-over went.
    async fn take_over(
        &self,
        request: RouterRequest,
        origin: StreamOrigin,
        failure: AttemptError,
    ) -> Option<Result<RouterResponse<String>, Unanswered<Raised>>> {
        let take_over = self.router.take_over_stream(&request, &origin, &failure)?;
        Some(take_over.run(&*self.host, Op::from("acompletion")).await)
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

fn pre_call_checked(config: RouterConfig) -> RouterConfig {
    RouterConfig {
        enable_pre_call_checks: true,
        ..config
    }
}

fn router(config: RouterConfig, dynamic: serde_json::Value, host: Arc<ScriptedHost>) -> Hosted {
    let dynamic: RouterDynamicConfig = serde_json::from_value(dynamic).unwrap();
    let router = Router::builder(config, dynamic)
        .randomness(Arc::new(SeededRandomness::new(7)))
        .build()
        .unwrap();
    Hosted { router, host }
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

fn guardrail_block() -> AttemptError {
    failure(FailureKind::GuardrailIntervention, 400)
}

fn agentic_followup() -> AttemptError {
    let mut failed = server_error();
    failed.agentic_followup = true;
    failed
}

fn create_may_exist() -> AttemptError {
    let mut failed = failure(FailureKind::Timeout, 408);
    failed.create_may_exist = true;
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
async fn a_failed_call_hands_back_the_host_error_of_the_attempt_it_ends_on(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::scripted(vec![
        ("primary", vec![failure(FailureKind::BadRequest, 400)]),
        ("backup", vec![server_error()]),
    ]));
    let hosted = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let ended = hosted
        .router
        .call(&*host, Op::from("acompletion"), request())
        .await
        .unwrap_err();
    let selected = host.selected();
    assert_eq!(
        (
            ended.error.attempt().map(|failure| failure.kind),
            ended.raised,
            selected.len(),
        ),
        (
            Some(FailureKind::BadRequest),
            Some(litellm_router_core::Raised {
                failure: Raised(selected[0].clone()),
                deployment_id: Some(selected[0].clone()),
            }),
            2,
        )
    );
}

#[rstest]
#[tokio::test]
async fn a_refused_attempt_ends_the_call_without_retries_or_fallbacks(config: RouterConfig) {
    let host = Arc::new(ScriptedHost {
        refuses: true,
        ..ScriptedHost::default()
    });
    let hosted = router(
        config,
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let ended = hosted
        .router
        .call(&*host, Op::from("acompletion"), request())
        .await
        .unwrap_err();
    let selected = host.selected();
    assert!(
        matches!(ended.error, Error::Refused(_)),
        "{:?}",
        ended.error
    );
    assert_eq!(
        (ended.raised, selected.len(), host.retries()),
        (
            Some(litellm_router_core::Raised {
                failure: Raised(selected[0].clone()),
                deployment_id: Some(selected[0].clone()),
            }),
            1,
            0
        )
    );
}

#[rstest]
#[tokio::test]
async fn an_error_the_router_made_itself_hands_back_no_host_error(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::default());
    let hosted = router(config, json!({}), host.clone());
    let ended = hosted
        .router
        .call(&*host, Op::from("acompletion"), request_for("unconfigured"))
        .await
        .unwrap_err();
    assert!(
        matches!(&ended.error, Error::UnknownModel { model } if model == "unconfigured"),
        "{:?}",
        ended.error
    );
    assert_eq!(ended.raised, None);
}

fn rate_limited() -> AttemptError {
    failure(FailureKind::RateLimit, 429)
}

/// A rate limit cools its deployment down, so the next attempt runs on the group's other one and
/// the two host errors differ.
#[rstest]
#[tokio::test]
async fn a_call_whose_retry_fails_too_hands_back_the_host_error_of_the_retry(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![rate_limited(), rate_limited()],
    ));
    let hosted = router(config, json!({"num_retries": 1}), host.clone());
    let ended = hosted
        .router
        .call(&*host, Op::from("acompletion"), request())
        .await
        .unwrap_err();
    let selected = host.selected();
    assert_eq!(
        (
            selected.len(),
            selected[0] == selected[1],
            ended.raised.map(|raised| raised.failure)
        ),
        (2, false, Some(Raised(selected[1].clone())))
    );
}

#[rstest]
#[tokio::test]
async fn a_retry_the_host_refuses_hands_back_the_host_error_of_the_retry(config: RouterConfig) {
    let host = Arc::new(ScriptedHost {
        first_failure: Mutex::new(Some(rate_limited())),
        refuses: true,
        ..ScriptedHost::default()
    });
    let hosted = router(config, json!({"num_retries": 1}), host.clone());
    let ended = hosted
        .router
        .call(&*host, Op::from("acompletion"), request())
        .await
        .unwrap_err();
    let selected = host.selected();
    assert!(
        matches!(ended.error, Error::Refused(_)),
        "{:?}",
        ended.error
    );
    assert_eq!(
        (
            selected.len(),
            selected[0] == selected[1],
            ended.raised.map(|raised| raised.failure)
        ),
        (2, false, Some(Raised(selected[1].clone())))
    );
}

#[rstest]
#[tokio::test]
async fn stream_retries_that_all_fail_hand_back_the_host_error_of_the_last_one(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![rate_limited(), rate_limited()],
    ));
    let hosted = router(config, json!({}), host.clone());
    let ended = hosted
        .router
        .retry_stream(
            &*host,
            Op::from("acompletion"),
            RouterRequest {
                num_retries: Some(2),
                ..request()
            },
            stream_origin(),
            server_error(),
        )
        .await
        .unwrap_err();
    let selected = host.selected();
    assert_eq!(
        (
            selected.len(),
            selected[0] == selected[1],
            ended.raised.map(|raised| raised.failure)
        ),
        (2, false, Some(Raised(selected[1].clone())))
    );
}

fn to_primary() -> PreRouted {
    PreRouted::Model("primary".to_owned())
}

/// `primary`, served for the group `unused` a session bound the request to.
fn to_primary_bound() -> PreRouted {
    PreRouted::Bound {
        model: "primary".to_owned(),
        group: "unused".to_owned(),
    }
}

/// A pre-routing hook reroutes `context` to `primary`, and the pick of `primary` then fails.
#[rstest]
#[case::nothing_in_it_is_healthy(to_primary(), "primary", true, None, false)]
#[case::a_filter_raises(to_primary(), "primary", false, Some(rate_limited()), false)]
#[case::its_routing_group_is_gone(to_primary(), "primary", false, None, true)]
#[case::by_the_chain_of_the_group_it_is_bound_to(to_primary_bound(), "unused", true, None, false)]
#[tokio::test]
async fn a_pre_routed_group_whose_pick_fails_falls_back_by_its_own_chain(
    #[case] routed: PreRouted,
    #[case] chain_of: &str,
    #[case] cooling: bool,
    #[case] filter_failure: Option<AttemptError>,
    #[case] stale: bool,
) {
    let mut config = tiers(false);
    if stale {
        config.routing_group_members = Some([("primary".to_owned(), "gone".to_owned())].into());
    }
    let host = Arc::new(ScriptedHost {
        pre_routes: Some(("context", routed)),
        pick_failures: Mutex::new(filter_failure.into_iter().collect()),
        ..ScriptedHost::default()
    });
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{chain_of: ["backup"]}]}),
        host.clone(),
    );
    if cooling {
        router
            .health()
            .add_cooldown("p1", &server_error(), Duration::from_secs(60))
            .await
            .unwrap();
    }
    let response = router
        .call(
            Op::from("acompletion"),
            RouterRequest {
                model: "context".to_owned(),
                pre_routes: true,
                filter_tags: true,
                ..RouterRequest::default()
            },
        )
        .await
        .map(|response| (response.deployment_id, response.attempted_fallbacks));
    assert_eq!(response.ok(), Some(("b1".to_owned(), 1)));
}

#[rstest]
#[case::the_group_it_was_routed_to(to_primary(), "primary")]
#[case::the_group_it_is_bound_to(to_primary_bound(), "unused")]
#[tokio::test]
async fn a_pre_routed_group_whose_attempt_fails_falls_back_by_the_chain_of(
    #[case] routed: PreRouted,
    #[case] chain_of: &str,
) {
    let host = Arc::new(ScriptedHost {
        pre_routes: Some(("context", routed)),
        ..ScriptedHost::failing("primary", vec![server_error()])
    });
    let router = router(
        tiers(false),
        json!({"num_retries": 0, "fallbacks": [{chain_of: ["backup"]}]}),
        host.clone(),
    );
    let response = router
        .call(
            Op::from("acompletion"),
            RouterRequest {
                model: "context".to_owned(),
                pre_routes: true,
                ..RouterRequest::default()
            },
        )
        .await
        .map(|response| (response.deployment_id, response.attempted_fallbacks));
    assert_eq!(response.ok(), Some(("b1".to_owned(), 1)));
    assert_eq!(host.selected(), ["p1", "b1"]);
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
async fn a_fallback_the_host_refuses_ends_the_call_before_the_next_target() {
    let host = Arc::new(ScriptedHost {
        first_failure: Mutex::new(Some(rate_limited())),
        refuses: true,
        ..ScriptedHost::default()
    });
    let hosted = router(
        tiers(false),
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup", "context"]}]}),
        host.clone(),
    );
    let ended = hosted
        .router
        .call(&*host, Op::from("acompletion"), request())
        .await
        .unwrap_err();

    assert!(
        matches!(ended.error, Error::Refused(_)),
        "{:?}",
        ended.error
    );
    assert_eq!(host.selected(), ["p1", "b1"]);
}

/// A [`ScriptedHost`] that keeps the group of every attempt and every host step a call makes
/// beyond its attempts, named as [`step_name`] names what `Router::reach` reports.
#[derive(Default)]
struct SteppedHost {
    scripted: ScriptedHost,
    attempted: Mutex<BTreeSet<String>>,
    steps: Mutex<BTreeSet<String>>,
}

fn step_name(step: &HostStep) -> String {
    match step {
        HostStep::Pick(strategy, scope) => format!("pick {strategy} {scope:?}"),
        HostStep::PreCallFacts => "pre-call facts".to_owned(),
        HostStep::HealthStates => "health states".to_owned(),
        HostStep::FallbackParams(_) => "fallback params".to_owned(),
    }
}

impl SteppedHost {
    fn made(&self, step: &HostStep) {
        self.steps.lock().insert(step_name(step));
    }
}

impl RouterHost for SteppedHost {
    type Response = String;
    type Failure = Raised;
    type Cache = RecordingCache;

    async fn attempt(&self, attempt: &Attempt) -> Result<Served<String>, AttemptFailure<Raised>> {
        self.attempted
            .lock()
            .insert(attempt.deployment.model_name.clone());
        self.scripted.attempt(attempt).await
    }

    async fn select(
        &self,
        stage: &SelectStage,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        match stage {
            SelectStage::Strategy(strategy) => {
                self.made(&HostStep::Pick(strategy.clone(), ctx.scope.clone()));
            }
            other => {
                self.steps.lock().insert(format!("{other:?}"));
            }
        }
        Ok(candidates)
    }

    async fn pre_call_facts(
        &self,
        _ctx: &RoutingContext<'_>,
        _candidates: &[Arc<Deployment>],
    ) -> Result<PreCallFacts, Error> {
        self.made(&HostStep::PreCallFacts);
        Ok(PreCallFacts::default())
    }

    async fn allow_fallback(&self, _target: &str) -> Result<bool, AttemptError> {
        Ok(true)
    }

    async fn refuses_fallbacks(
        &self,
        _level: u32,
        _targets: &[serde_json::Map<String, serde_json::Value>],
    ) -> bool {
        self.made(&HostStep::FallbackParams("fallbacks"));
        false
    }

    async fn sleep(&self, _duration: Duration) {}

    fn observe(&self, _event: &RouterEvent) {}

    fn cache(&self) -> &RecordingCache {
        &self.scripted.cache
    }
}

/// `primary` is asked for and each group in `failing` fails once, so the call walks its fallbacks.
#[rstest]
#[case::every_step_behind_a_dict_fallback(
    json!({
        "routing_groups": [
            {"group_name": "pool", "models": ["backup"], "routing_strategy": "least-busy"},
        ],
        "enable_pre_call_checks": true,
        "enable_health_check_routing": true,
    }),
    json!({"fallbacks": [{"primary": [{"model": "backup", "temperature": 0}]}]}),
    vec![("primary", server_error())],
    &["primary", "backup"],
)]
#[case::the_router_strategy_through_an_alias_and_its_own_chain(
    json!({"routing_strategy": "least-busy"}),
    json!({
        "model_group_alias": {"spare": "context"},
        "context_window_fallbacks": [{"primary": ["spare"]}],
        "fallbacks": [{"spare": ["backup"]}],
    }),
    vec![("primary", context_window()), ("context", server_error())],
    &["primary", "context", "backup"],
)]
#[case::the_fallbacks_of_every_group(
    json!({}),
    json!({"fallbacks": [{"*": ["backup"]}]}),
    vec![("primary", server_error())],
    &["primary", "backup"],
)]
#[case::no_fallback(json!({}), json!({}), vec![("primary", server_error())], &["primary"])]
#[tokio::test]
async fn a_call_stays_inside_what_reach_reports_for_its_name(
    #[case] settings: Value,
    #[case] dynamic: Value,
    #[case] failing: Vec<(&str, AttemptError)>,
    #[case] walked: &[&str],
) {
    let config = RouterConfig {
        model_list: tiers(false).model_list,
        ..serde_json::from_value(settings).unwrap()
    };
    let dynamic: RouterDynamicConfig = serde_json::from_value(
        dynamic
            .as_object()
            .into_iter()
            .flatten()
            .chain(json!({"num_retries": 0}).as_object().into_iter().flatten())
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
    .unwrap();
    let router = Router::new(config, dynamic).unwrap();
    let host = SteppedHost {
        scripted: ScriptedHost::scripted(
            failing
                .into_iter()
                .map(|(group, failure)| (group, vec![failure]))
                .collect(),
        ),
        ..SteppedHost::default()
    };

    let _ = router.call(&host, Op::from("acompletion"), request()).await;

    let reached = router.reach("primary").unwrap();
    let served: BTreeSet<String> = reached
        .iter()
        .filter(|reached| reached.served)
        .map(|reached| reached.group.clone())
        .collect();
    let reported: BTreeSet<String> = reached
        .iter()
        .flat_map(|reached| &reached.steps)
        .map(step_name)
        .collect();
    let attempted = host.attempted.lock().clone();
    let made: BTreeSet<String> = host
        .steps
        .lock()
        .iter()
        .cloned()
        .chain(
            (*host.scripted.cache.health_reads.lock() > 0)
                .then(|| step_name(&HostStep::HealthStates)),
        )
        .collect();
    assert_eq!(
        attempted,
        ids(walked),
        "the call did not walk its fallbacks"
    );
    assert!(attempted.is_subset(&served), "{attempted:?} of {served:?}");
    assert_eq!(made, reported);
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

/// `model_list` and `settings`, with a group no request in these tests names.
fn beside_an_unrelated_group(model_list: Value, settings: Value) -> RouterConfig {
    let Value::Array(mut deployments) = model_list else {
        panic!("a model list is a list");
    };
    deployments.push(json!(
        {"model_name": "unrelated", "litellm_params": {"model": "openai/u"}, "model_info": {"id": "u1"}}
    ));
    let Value::Object(mut config) = settings else {
        panic!("settings are a mapping");
    };
    config.insert("model_list".to_owned(), Value::Array(deployments));
    serde_json::from_value(Value::Object(config)).unwrap()
}

fn primary_and_backup() -> Value {
    json!([
        {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
        {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "a2"}},
        {"model_name": "backup", "litellm_params": {"model": "anthropic/c"}, "model_info": {"id": "b1"}},
    ])
}

fn ids(ids: &[&str]) -> BTreeSet<String> {
    ids.iter().map(|id| (*id).to_owned()).collect()
}

/// What [`stopped_router`]'s clock reads, in Unix seconds.
fn stopped_now() -> f64 {
    utc_hour(10) as f64
}

fn stopped_router(config: RouterConfig, dynamic: Value, host: Arc<ScriptedHost>) -> Hosted {
    let dynamic: RouterDynamicConfig = serde_json::from_value(dynamic).unwrap();
    let router = Router::builder(config, dynamic)
        .randomness(Arc::new(SeededRandomness::new(7)))
        .clock(Arc::new(StoppedClock(utc_hour(10))))
        .build()
        .unwrap();
    Hosted { router, host }
}

#[rstest]
#[case::a_group(primary_and_backup(), json!({}), "primary", None, &["a1", "a2"])]
#[case::an_alias(
    json!([
        {"model_name": "gg", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "w1"}},
        {"model_name": "gg", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "w2"}},
    ]),
    json!({"model_group_alias": {"al": "gg"}}),
    "al",
    None,
    &["w1", "w2"]
)]
#[case::a_wildcard(
    json!([{"model_name": "openai/*", "litellm_params": {"model": "openai/*"}, "model_info": {"id": "w1"}}]),
    json!({}),
    "openai/gpt-4o",
    None,
    &["w1"]
)]
#[case::a_team_public_name(
    json!([
        {"model_name": "t1a", "litellm_params": {"model": "openai/a"},
         "model_info": {"id": "w1", "team_id": "T1", "team_public_model_name": "tp"}},
        {"model_name": "t1b", "litellm_params": {"model": "openai/b"},
         "model_info": {"id": "w2", "team_id": "T1", "team_public_model_name": "tp"}},
    ]),
    json!({}),
    "tp",
    Some("T1"),
    &["w1", "w2"]
)]
#[case::a_deployment_id(primary_and_backup(), json!({}), "a1", None, &[])]
#[tokio::test]
async fn a_pick_reads_only_the_cooldowns_of_the_deployments_it_could_route_to(
    #[case] model_list: Value,
    #[case] dynamic: Value,
    #[case] model: &str,
    #[case] team: Option<&str>,
    #[case] expected: &[&str],
) {
    let host = Arc::new(ScriptedHost::default());
    let router = router(
        beside_an_unrelated_group(model_list, json!({})),
        dynamic,
        host.clone(),
    );
    let request = RouterRequest {
        model: model.to_owned(),
        team_id: team.map(str::to_owned),
        listing_team_id: team.map(str::to_owned),
        ..RouterRequest::default()
    };
    router
        .select(&Op::from("acompletion"), &request)
        .await
        .unwrap();
    assert_eq!(host.cache.cooldown_ids_read(), ids(expected));
}

#[rstest]
#[tokio::test]
async fn a_cooling_candidate_is_still_skipped_when_the_read_is_scoped_to_the_candidates() {
    let host = Arc::new(ScriptedHost::default());
    let router = stopped_router(
        beside_an_unrelated_group(primary_and_backup(), json!({})),
        json!({}),
        host.clone(),
    );
    host.cache.hold_cooldown("a2", stopped_now(), 60.0);
    let picked = router
        .select(&Op::from("acompletion"), &request())
        .await
        .unwrap();
    assert_eq!(
        (picked.id.as_str(), host.cache.cooldown_ids_read()),
        ("a1", ids(&["a1", "a2"]))
    );
}

#[rstest]
#[tokio::test]
async fn a_group_cooling_down_falls_back_reading_only_each_hops_candidates() {
    let host = Arc::new(ScriptedHost::default());
    let router = stopped_router(
        beside_an_unrelated_group(primary_and_backup(), json!({})),
        json!({"fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    host.cache.hold_cooldown("a1", stopped_now(), 60.0);
    host.cache.hold_cooldown("a2", stopped_now(), 60.0);
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(
        (
            response.deployment_id.as_str(),
            host.cache.cooldown_ids_read()
        ),
        ("b1", ids(&["a1", "a2", "b1"]))
    );
}

#[rstest]
#[tokio::test]
async fn weighted_failover_reads_only_the_failed_groups_cooldowns() {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![failure(FailureKind::RateLimit, 429)],
    ));
    let router = stopped_router(
        beside_an_unrelated_group(
            primary_and_backup(),
            json!({"enable_weighted_failover": true}),
        ),
        json!({"num_retries": 0}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(host.selected().len(), 2);
    assert_eq!(
        (
            response.model_group.as_str(),
            host.cache.cooldown_ids_read()
        ),
        ("primary", ids(&["a1", "a2"]))
    );
}

#[rstest]
#[tokio::test]
async fn a_pick_left_with_no_candidates_reads_no_cooldowns_and_names_none() {
    let host = Arc::new(ScriptedHost::default());
    let unsupported = json!([
        {"model_name": "primary", "litellm_params": {"model": "openai/a"},
         "model_info": {"id": "a1", "supports_web_search": false}},
        {"model_name": "primary", "litellm_params": {"model": "openai/b"},
         "model_info": {"id": "a2", "supports_web_search": false}},
    ]);
    let router = router(
        beside_an_unrelated_group(unsupported, json!({})),
        json!({}),
        host.clone(),
    );
    let request = RouterRequest {
        web_search: true,
        ..request()
    };
    let error = router
        .select(&Op::from("acompletion"), &request)
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::NoHealthyDeployments { model, request_ids: Some(listed) }
            if model == "primary" && listed.is_empty()),
        "{error:?}"
    );
    assert!(host.cache.cooldown_reads.lock().is_empty());
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
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.status == Some(500))
    );
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

/// A 100 token prompt for `primary`, whose only deployment has a 50 token window.
#[rstest]
#[case::by_its_own_chain(json!({"context_window_fallbacks": [{"primary": ["context"]}]}), Ok("c1"))]
#[case::by_the_regular_chain(json!({"fallbacks": [{"primary": ["backup"]}]}), Ok("b1"))]
#[case::nowhere_without_a_chain(json!({}), Err(true))]
#[tokio::test]
async fn a_prompt_no_deployment_of_its_group_fits_falls_back_as_a_context_window_error(
    #[case] dynamic: Value,
    #[case] expected: Result<&str, bool>,
) {
    let window = |limit| DeploymentFacts {
        max_input_tokens: Some(limit),
        input_tokens: Some(100),
        ..DeploymentFacts::default()
    };
    let host = Arc::new(ScriptedHost {
        pre_call_facts: PreCallFacts {
            has_input: true,
            countable: true,
            deployments: [
                ("p1".to_owned(), window(50)),
                ("c1".to_owned(), window(1000)),
            ]
            .into(),
            ..PreCallFacts::default()
        },
        ..ScriptedHost::default()
    });
    let config = RouterConfig {
        enable_pre_call_checks: true,
        ..tiers(false)
    };
    let router = router(config, dynamic, host.clone());

    let answered = router.call(Op::from("acompletion"), request()).await;

    assert_eq!(
        answered
            .as_ref()
            .map(|response| response.deployment_id.as_str())
            .map_err(|error| matches!(error, Error::ContextWindowExceeded { .. })),
        expected
    );
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
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.kind == raised)
    );
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
#[tokio::test]
async fn a_context_window_error_retries_on_a_deployment_that_has_not_refused(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let selected = host.selected();
    assert_eq!(
        (response.model_group.as_str(), response.attempted_retries),
        ("primary", 1)
    );
    assert_eq!(selected.len(), 2);
    assert_ne!(selected[0], selected[1]);
    assert_eq!(response.deployment_id, selected[1]);
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_is_not_retried_without_pre_call_checks(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(config, json!({"num_retries": 2}), host.clone());
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!((host.retries(), host.selected().len()), (0, 1));
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_without_pre_call_checks_runs_the_callback_filters_only_for_its_pick(
    mut config: RouterConfig,
) {
    config.callback_filters = true;
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(config, json!({"num_retries": 2}), host.clone());
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(refused.is_err());
    assert_eq!(
        (host.selected().len(), *host.callback_filter_runs.lock()),
        (1, 1)
    );
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_on_a_call_by_deployment_id_is_not_retried(config: RouterConfig) {
    let host = refusing_once();
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let by_id = RouterRequest {
        model: "a1".to_owned(),
        ..RouterRequest::default()
    };
    let refused = router.call(Op::from("acompletion"), by_id).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!(host.selected(), ["a1"]);
}

#[rstest]
#[tokio::test]
async fn a_retry_after_a_context_window_error_never_returns_to_the_refusing_deployment(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![context_window(), rate_limited()],
    ));
    let hosted = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let ended = hosted
        .router
        .call(&*host, Op::from("acompletion"), request())
        .await
        .unwrap_err();
    let selected = host.selected();
    assert!(
        matches!(&ended.error, Error::Attempt { failure, .. } if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!(
        (
            selected.len(),
            selected[0] == selected[1],
            ended.raised.map(|raised| raised.failure)
        ),
        (2, false, Some(Raised(selected[0].clone())))
    );
    assert_eq!(
        context_window_raised(&host),
        [(Some(selected[0].clone()), false)]
    );
}

fn context_window_raised(host: &ScriptedHost) -> Vec<(Option<String>, bool)> {
    host.events
        .lock()
        .iter()
        .filter_map(|event| match event {
            RouterEvent::ContextWindowRaised {
                deployment_id,
                stream_origin,
                ..
            } => Some((deployment_id.clone(), *stream_origin)),
            _ => None,
        })
        .collect()
}

#[rstest]
#[tokio::test]
async fn a_pre_routed_context_window_error_retries_in_the_group_it_was_routed_to(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost {
        pre_routes: Some(("context", to_primary())),
        first_failure: Mutex::new(Some(context_window())),
        ..ScriptedHost::default()
    });
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let response = router
        .call(
            Op::from("acompletion"),
            RouterRequest {
                model: "context".to_owned(),
                pre_routes: true,
                ..RouterRequest::default()
            },
        )
        .await
        .unwrap();
    let selected = host.selected();
    assert_eq!(selected.len(), 2);
    assert_ne!(selected[0], selected[1]);
    assert_eq!(response.deployment_id, selected[1]);
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_every_deployment_refused_falls_back(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![context_window(), context_window()],
    ));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let selected = host.selected();
    assert_eq!(response.deployment_id, "b1");
    assert_eq!(selected.len(), 3);
    assert_ne!(selected[0], selected[1]);
    assert_eq!(host.retries(), 1);
}

fn same_model_group(second: Value) -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            second,
            {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "b2"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::same_model(json!(
    {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a2"}}
))]
#[case::same_base_model(json!(
    {"model_name": "primary", "litellm_params": {"model": "openai/alias"}, "model_info": {"id": "a2", "base_model": "openai/a"}}
))]
#[case::same_model_from_another_provider(json!(
    {"model_name": "primary", "litellm_params": {"model": "azure_ai/a"}, "model_info": {"id": "a2"}}
))]
#[tokio::test]
async fn a_context_window_error_skips_every_deployment_serving_the_refused_model(
    #[case] second: Value,
) {
    let host = Arc::new(ScriptedHost {
        first_failure: Mutex::new(Some(context_window())),
        ..ScriptedHost::default()
    });
    let router = router(
        pre_call_checked(same_model_group(second)),
        json!({"num_retries": 3}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let selected = host.selected();
    let serves_a = |id: &str| id != "b2";
    assert_eq!(selected.len(), 2);
    assert_ne!(serves_a(&selected[0]), serves_a(&selected[1]));
    assert_eq!(response.deployment_id, selected[1]);
}

#[rstest]
#[case::a_larger_window(vec![("p1", 200_000), ("p2", 1_000_000)], &["p1", "p2"])]
#[case::the_same_window(vec![("p1", 200_000), ("p2", 200_000)], &["p1"])]
#[case::a_smaller_window(vec![("p1", 200_000), ("p2", 100_000)], &["p1"])]
#[case::no_known_window(vec![("p1", 200_000)], &["p1"])]
#[case::no_known_window_on_the_refused_one(vec![("p2", 1_000_000)], &["p1"])]
#[tokio::test]
async fn a_context_window_error_retries_a_copy_of_the_model_only_with_a_larger_window(
    #[case] windows: Vec<(&'static str, i64)>,
    #[case] tried: &[&str],
) {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a", "weight": 1}, "model_info": {"id": "p1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/a", "weight": 0}, "model_info": {"id": "p2"}},
        ]
    }))
    .unwrap();
    let deployments = windows
        .into_iter()
        .map(|(id, window)| {
            let facts = DeploymentFacts {
                max_input_tokens: Some(window),
                ..DeploymentFacts::default()
            };
            (id.to_owned(), facts)
        })
        .collect();
    let host = Arc::new(ScriptedHost {
        first_failure: Mutex::new(Some(context_window())),
        pre_call_facts: PreCallFacts {
            deployments,
            ..PreCallFacts::default()
        },
        ..ScriptedHost::default()
    });
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let outcome = router.call(Op::from("acompletion"), request()).await;
    assert_eq!(host.selected(), tried);
    assert_eq!(outcome.is_ok(), tried.len() == 2);
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_is_not_retried_when_every_deployment_serves_that_model() {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a2"}},
        ]
    }))
    .unwrap();
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 3}),
        host.clone(),
    );
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!((host.retries(), host.selected().len()), (0, 1));
    assert!(context_window_raised(&host).is_empty());
}

#[rstest]
#[tokio::test]
async fn a_retry_policy_still_retries_a_context_window_error_on_the_same_model() {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a2"}},
        ],
        "retry_policy": {"BadRequestErrorRetries": 2},
    }))
    .unwrap();
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 0}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let selected = host.selected();
    assert_eq!(response.attempted_retries, 1);
    assert_ne!(selected[0], selected[1]);
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_no_deployment_raised_is_not_retried(config: RouterConfig) {
    let host = Arc::new(ScriptedHost {
        pick_failures: Mutex::new(VecDeque::from([context_window()])),
        ..ScriptedHost::default()
    });
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 3}),
        host.clone(),
    );
    let refused = router.call(Op::from("acompletion"), tagged_request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!((host.retries(), host.selected().len()), (0, 0));
}

fn refusing_once() -> Arc<ScriptedHost> {
    Arc::new(ScriptedHost {
        first_failure: Mutex::new(Some(context_window())),
        ..ScriptedHost::default()
    })
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_takes_the_next_order_tier_instead_of_the_refusing_one() {
    let host = refusing_once();
    let router = router(
        pre_call_checked(tiers(true)),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(
        (
            response.deployment_id.as_str(),
            host.selected(),
            host.retries()
        ),
        ("p2", vec!["p1".to_owned(), "p2".to_owned()], 0)
    );
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_on_a_pinned_deployment_is_not_retried(config: RouterConfig) {
    let host = refusing_once();
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let pinned = RouterRequest {
        pinned_deployment_id: Some("a1".to_owned()),
        ..request()
    };
    let refused = router.call(Op::from("acompletion"), pinned).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!(host.selected(), ["a1"]);
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_skips_the_wildcard_deployments_serving_that_model() {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}, "model_info": {"id": "w1"}},
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}, "model_info": {"id": "w2"}},
        ]
    }))
    .unwrap();
    let host = refusing_once();
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let wildcard = RouterRequest {
        model: "openai/gpt-4o".to_owned(),
        ..RouterRequest::default()
    };
    let refused = router.call(Op::from("acompletion"), wildcard).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!(host.selected().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_context_window_list_for_another_group_leaves_the_retry_in_place(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2, "context_window_fallbacks": [{"backup": ["primary"]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(
        (response.model_group.as_str(), response.attempted_retries),
        ("primary", 1)
    );
}

fn provider_bad_request() -> AttemptError {
    let mut failed = AttemptError::new(FailureKind::BadRequest, Some(400));
    failed.names_provider = true;
    failed
}

#[rstest]
#[tokio::test]
async fn a_retry_that_fails_after_a_context_window_error_falls_back_on_that_error(
    config: RouterConfig,
) {
    let config = RouterConfig {
        fallback_on_unsupported_params: Some(false),
        ..config
    };
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![context_window(), provider_bad_request()],
    ));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(
        (response.deployment_id.as_str(), host.selected().len()),
        ("b1", 3)
    );
}

#[rstest]
#[tokio::test]
async fn a_guardrail_block_on_a_retry_after_a_context_window_error_ends_the_call(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![context_window(), guardrail_block()],
    ));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::GuardrailIntervention)
    );
    assert_eq!(host.selected().len(), 2);
}

#[rstest]
#[tokio::test]
async fn a_retry_that_fails_after_a_context_window_error_raises_that_error(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![context_window(), server_error(), server_error()],
    ));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    let selected = host.selected();
    assert_eq!(selected.len(), 3);
    assert_eq!(
        context_window_raised(&host),
        [(Some(selected[0].clone()), false)]
    );
}

#[rstest]
#[tokio::test]
async fn a_context_window_error_on_a_later_retry_is_raised_over_the_last_failure(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![server_error(), context_window(), server_error()],
    ));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!(host.selected().len(), 3);
}

#[rstest]
#[tokio::test]
async fn a_num_retries_that_raises_leaves_a_context_window_error_standing(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(
        pre_call_checked(config),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let request = RouterRequest {
        uncounted_retries: Some(UncountedRetries::Raises),
        ..request()
    };
    let refused = router.call(Op::from("acompletion"), request).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert!(
        !host
            .events
            .lock()
            .iter()
            .any(|event| matches!(event, RouterEvent::RetryCountRaised { .. }))
    );
    assert_eq!(host.selected().len(), 1);
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
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.ends_retries())
    );
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
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.ends_retries())
    );
    assert_eq!((host.retries(), host.selected().len()), (1, 2));
}

#[rstest]
#[tokio::test]
async fn a_create_that_may_exist_is_neither_retried_nor_sent_to_a_fallback(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![create_may_exist()]));
    let router = router(
        config,
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let refused = router.call(Op::from("acreate_batch"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.create_may_exist)
    );
    assert_eq!((host.retries(), host.selected().len()), (0, 1));
}

#[rstest]
#[tokio::test]
async fn a_retry_whose_create_may_exist_ends_the_retries_and_the_fallbacks(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![server_error(), create_may_exist()],
    ));
    let router = router(
        config,
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let refused = router.call(Op::from("acreate_batch"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.create_may_exist)
    );
    assert_eq!((host.retries(), host.selected().len()), (1, 2));
}

#[rstest]
#[tokio::test]
async fn a_fallback_whose_create_may_exist_is_the_last_fallback_tried() {
    let host = Arc::new(ScriptedHost::scripted(vec![
        ("primary", vec![server_error()]),
        ("backup", vec![create_may_exist()]),
    ]));
    let router = router(
        tiers(false),
        json!({"fallbacks": [{"primary": ["backup", "context"]}]}),
        host.clone(),
    );
    let refused = router.call(Op::from("acreate_batch"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.create_may_exist)
    );
    assert_eq!(host.selected(), ["p1", "b1"]);
}

#[rstest]
#[tokio::test]
async fn a_weighted_failover_whose_create_may_exist_skips_the_regular_fallbacks() {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![server_error(), create_may_exist()],
    ));
    let config = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "a2"}},
            {"model_name": "backup", "litellm_params": {"model": "anthropic/b1"}, "model_info": {"id": "b1"}},
        ],
        "enable_weighted_failover": true,
    }))
    .unwrap();
    let router = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let refused = router.call(Op::from("acreate_batch"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.create_may_exist)
    );
    assert_eq!(host.selected().len(), 2);
}

/// Where a stream of `primary` came from, the rest as `named` says. An origin is only ever built
/// by the attempt that served the stream, or read back from the document a host kept of it.
fn origin(named: Value) -> StreamOrigin {
    let Value::Object(named) = named else {
        panic!("an origin is named by an object: {named}");
    };
    let primary = [("group".to_owned(), json!("primary"))];
    serde_json::from_value(Value::Object(primary.into_iter().chain(named).collect())).unwrap()
}

fn stream_origin() -> StreamOrigin {
    origin(json!({}))
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
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.agentic_followup)
    );
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
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.ends_retries())
    );
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

fn stand_in_overflows(ids: &[&str]) -> PreCallFacts {
    serde_json::from_value(json!({
        "has_input": true,
        "countable": true,
        "deployments": ids
            .iter()
            .map(|id| ((*id).to_owned(), json!({"max_input_tokens": 100, "input_tokens": 150, "estimated": true})))
            .collect::<serde_json::Map<_, _>>(),
    }))
    .unwrap()
}

#[rstest]
#[case::the_provider_refused_the_window(context_window(), json!({}), 0)]
#[case::a_context_window_fallback_applies(
    server_error(),
    json!({"context_window_fallbacks": [{"primary": ["backup"]}]}),
    0
)]
#[case::another_failure(server_error(), json!({}), 1)]
#[tokio::test]
async fn a_stream_retry_resends_a_stand_in_overflow_only_without_a_window_refusal_or_context_window_fallback(
    config: RouterConfig,
    #[case] stream_failure: AttemptError,
    #[case] dynamic: Value,
    #[case] resent: usize,
) {
    let host = Arc::new(ScriptedHost {
        pre_call_facts: stand_in_overflows(&["a1", "a2"]),
        ..ScriptedHost::default()
    });
    let router = router(
        RouterConfig {
            enable_pre_call_checks: true,
            retry_policy: serde_json::from_value(json!({"BadRequestErrorRetries": 2})).unwrap(),
            ..config
        },
        dynamic,
        host.clone(),
    );
    let _ = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin(json!({"deployment_id": "a1"})),
            stream_failure,
        )
        .await;
    assert_eq!(host.selected().len(), resent);
}

#[rstest]
#[case::the_vision_deployment_may_fit(stand_in_overflows(&["a2"]), "a2")]
#[case::the_vision_deployment_fits(stand_in_overflows(&[]), "a2")]
#[tokio::test]
async fn an_image_request_goes_to_the_vision_deployment_over_a_fitting_text_only_one(
    config: RouterConfig,
    #[case] pre_call_facts: PreCallFacts,
    #[case] picked: &str,
) {
    let host = Arc::new(ScriptedHost {
        pre_call_facts,
        text_only_ids: &["a1"],
        ..ScriptedHost::default()
    });
    let router = router(
        RouterConfig {
            enable_pre_call_checks: true,
            ..config
        },
        json!({}),
        host.clone(),
    );
    let request = RouterRequest {
        has_image: true,
        ..request()
    };
    let chosen = router
        .select(&Op::from("acompletion"), &request)
        .await
        .unwrap();
    assert_eq!(chosen.id, picked);
}

#[rstest]
#[case::the_provider_refused_the_window(context_window(), 0)]
#[case::another_failure(server_error(), 1)]
#[tokio::test]
async fn a_stream_fallback_sends_a_stand_in_overflow_only_when_no_provider_refused_the_window(
    config: RouterConfig,
    #[case] stream_failure: AttemptError,
    #[case] sent: usize,
) {
    let host = Arc::new(ScriptedHost {
        pre_call_facts: stand_in_overflows(&["b1"]),
        ..ScriptedHost::default()
    });
    let hosted = router(
        RouterConfig {
            enable_pre_call_checks: true,
            ..config
        },
        json!({"fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let _ = hosted
        .router
        .fallback_stream(
            &*host,
            Op::from("acompletion"),
            stream_request(),
            origin(json!({"deployment_id": "a1"})),
            stream_failure,
        )
        .await;
    assert_eq!(host.selected().len(), sent);
}

#[rstest]
#[case::a_retry_got_a_window_refusal(context_window(), false)]
#[case::a_retry_got_another_refusal(failure(FailureKind::BadRequest, 400), true)]
#[tokio::test]
async fn a_stream_take_over_falls_back_to_a_stand_in_overflow_only_when_no_retry_was_refused_the_window(
    config: RouterConfig,
    #[case] retry_failure: AttemptError,
    #[case] sent_to_backup: bool,
) {
    let host = Arc::new(ScriptedHost {
        pre_call_facts: stand_in_overflows(&["b1"]),
        ..ScriptedHost::failing("primary", vec![retry_failure])
    });
    let hosted = router(
        RouterConfig {
            enable_pre_call_checks: true,
            ..config
        },
        json!({"fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let _ = hosted
        .take_over(
            stream_request(),
            origin(json!({"deployment_id": "a1"})),
            server_error(),
        )
        .await;
    assert_eq!(host.selected().contains(&"b1".to_owned()), sent_to_backup);
}

#[rstest]
#[tokio::test]
async fn a_stream_context_window_error_retries_on_another_deployment(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::default());
    let router = router(pre_call_checked(config), json!({}), host.clone());
    let origin = origin_on("a1");
    let response = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin,
            context_window(),
        )
        .await
        .unwrap();
    assert_eq!(
        (response.deployment_id.as_str(), host.selected()),
        ("a2", vec!["a2".to_owned()])
    );
}

/// `primary` with `a1` picked first and `a2` only once `a1` is out.
fn a1_first() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a", "weight": 1}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/b", "weight": 0}, "model_info": {"id": "a2"}},
        ]
    }))
    .unwrap()
}

#[tokio::test]
async fn a_stream_a_retry_opened_after_a_window_refusal_carries_the_refusal() {
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(
        pre_call_checked(a1_first()),
        json!({"num_retries": 1}),
        host,
    );

    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();

    assert_eq!(
        response.origin,
        origin(json!({
            "deployment_id": "a2",
            "attempted_retries": 1,
            "refused_ids": ["a1"],
            "provider_refused_window": true,
        }))
    );
}

#[rstest]
#[case::a_deployment_an_earlier_attempt_refused(json!({"refused_ids": ["a1"]}), PreCallFacts::default(), &["a2"])]
#[case::nothing_refused(json!({}), PreCallFacts::default(), &["a1"])]
#[case::stand_in_overflows_once_the_provider_refused_the_window(
    json!({"provider_refused_window": true}),
    stand_in_overflows(&["a1", "a2"]),
    &[]
)]
#[case::stand_in_overflows_no_provider_refused(json!({}), stand_in_overflows(&["a1", "a2"]), &["a1"])]
#[tokio::test]
async fn a_stream_retry_skips_what_the_request_refused_before_the_stream_opened(
    #[case] carried: Value,
    #[case] pre_call_facts: PreCallFacts,
    #[case] picked: &[&str],
) {
    let host = Arc::new(ScriptedHost {
        pre_call_facts,
        ..ScriptedHost::default()
    });
    let router = router(pre_call_checked(a1_first()), json!({}), host.clone());
    let Value::Object(carried) = carried else {
        panic!("the carried state is an object");
    };
    let named = [("deployment_id".to_owned(), json!("a2"))];
    let _ = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin(Value::Object(named.into_iter().chain(carried).collect())),
            server_error(),
        )
        .await;
    assert_eq!(host.selected(), picked);
}

#[rstest]
#[tokio::test]
async fn a_stream_context_window_error_is_not_retried_without_pre_call_checks(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::default());
    let router = router(config, json!({}), host.clone());
    let refused = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin_on("a1"),
            context_window(),
        )
        .await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert!(host.selected().is_empty());
}

#[rstest]
#[tokio::test]
async fn a_stream_context_window_error_keeps_retrying_past_each_model_that_refuses() {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "a2"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/c"}, "model_info": {"id": "a3"}},
        ]
    }))
    .unwrap();
    let host = Arc::new(ScriptedHost::failing("primary", vec![context_window()]));
    let router = router(pre_call_checked(config), json!({}), host.clone());
    let response = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin_on("a1"),
            context_window(),
        )
        .await
        .unwrap();
    let selected = host.selected();
    assert_eq!(selected.len(), 2);
    assert!(!selected.contains(&"a1".to_owned()));
    assert_ne!(selected[0], selected[1]);
    assert_eq!(response.deployment_id, selected[1]);
}

#[rstest]
#[tokio::test]
async fn a_pre_routed_stream_context_window_error_retries_in_the_group_it_was_routed_to(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost {
        pre_routes: Some(("context", to_primary())),
        ..ScriptedHost::default()
    });
    let router = router(pre_call_checked(config), json!({}), host.clone());
    let origin = origin(json!({"group": "context", "routed": "primary", "deployment_id": "a1"}));
    let request = RouterRequest {
        model: "context".to_owned(),
        pre_routes: true,
        ..stream_request()
    };
    let response = router
        .retry_stream(Op::from("acompletion"), request, origin, context_window())
        .await
        .unwrap();
    assert_eq!(
        (response.deployment_id.as_str(), host.selected()),
        ("a2", vec!["a2".to_owned()])
    );
}

#[rstest]
#[tokio::test]
async fn a_stream_context_window_error_in_an_order_tier_is_not_retried() {
    let host = Arc::new(ScriptedHost::default());
    let router = router(pre_call_checked(tiers(true)), json!({}), host.clone());
    let origin = origin_on("p1");
    let refused = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin,
            context_window(),
        )
        .await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert!(host.selected().is_empty());
}

#[rstest]
#[tokio::test]
async fn a_stream_guardrail_block_on_a_retry_after_a_context_window_error_ends_the_call(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![guardrail_block()]));
    let router = router(
        pre_call_checked(config),
        json!({"fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let refused = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin_on("a1"),
            context_window(),
        )
        .await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::GuardrailIntervention)
    );
    assert_eq!(host.selected().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_stream_retry_that_fails_after_a_context_window_error_raises_that_error(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![server_error(), server_error(), server_error()],
    ));
    let router = router(pre_call_checked(config), json!({}), host.clone());
    let origin = origin_on("a1");
    let refused = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin,
            context_window(),
        )
        .await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure, .. }) if failure.kind == FailureKind::ContextWindowExceeded)
    );
    assert_eq!(host.selected().len(), 3);
    assert_eq!(
        context_window_raised(&host),
        [(Some("a1".to_owned()), true)]
    );
}

#[rstest]
#[tokio::test]
async fn a_stream_fallback_that_serves_nothing_hands_back_no_error_of_another_attempt(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::scripted(vec![
        ("primary", vec![server_error()]),
        ("backup", vec![server_error(), server_error()]),
    ]));
    let hosted = router(
        config,
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let earlier = hosted
        .router
        .call(&*host, Op::from("acompletion"), request())
        .await
        .unwrap_err();
    let carried = earlier.error.attempt().cloned().unwrap();

    let ended = hosted
        .router
        .fallback_stream(
            &*host,
            Op::from("acompletion"),
            request(),
            stream_origin(),
            carried,
        )
        .await
        .unwrap_err();

    assert_eq!(
        (
            earlier.raised.is_some(),
            ended.raised,
            host.selected().len()
        ),
        (true, None, 3)
    );
}

#[rstest]
#[tokio::test]
async fn a_stream_fallback_looks_its_targets_up_by_the_group_a_hook_routed_the_stream_to(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::default());
    let hosted = router(
        config,
        json!({"fallbacks": [{"tier": ["backup"]}]}),
        host.clone(),
    );
    let origin = origin(json!({"routed": "tier"}));

    let served = hosted
        .router
        .fallback_stream(
            &*host,
            Op::from("acompletion"),
            request(),
            origin,
            server_error(),
        )
        .await
        .map(|response| response.output.response);

    assert_eq!(served.ok(), Some("b1".to_owned()));
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
    router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            origin_on("a1"),
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

/// `primary`, whose one deployment has `params`, and `backup`, with one deployment too.
fn lone(params: Value) -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": params, "model_info": {"id": "p1"}},
            {"model_name": "backup", "litellm_params": {"model": "anthropic/b1"}, "model_info": {"id": "b1"}},
        ]
    }))
    .unwrap()
}

fn plain() -> Value {
    json!({"model": "openai/p1"})
}

fn falling_back(num_retries: u32) -> Value {
    json!({"num_retries": num_retries, "fallbacks": [{"primary": ["backup"]}]})
}

fn origin_on(deployment: &str) -> StreamOrigin {
    origin(json!({"deployment_id": deployment}))
}

fn answer(
    taken: Option<Result<RouterResponse<String>, Unanswered<Raised>>>,
) -> RouterResponse<String> {
    match taken {
        Some(Ok(answer)) => answer,
        other => panic!("no attempt answered: {other:?}"),
    }
}

#[rstest]
#[case::neither_a_retry_nor_a_fallback(json!({"num_retries": 0}), json!(null), request(), server_error())]
#[case::a_failure_no_retry_clears(falling_back(1), json!(null), request(), failure(FailureKind::BadRequest, 400))]
#[case::fallbacks_the_request_disabled(
    falling_back(0),
    json!(null),
    RouterRequest { disable_fallbacks: true, ..request() },
    server_error()
)]
#[case::a_retry_policy_the_request_opted_out_of(
    json!({"num_retries": 0}),
    json!({"RateLimitErrorRetries": 2}),
    RouterRequest { num_retries: Some(0), ..request() },
    failure(FailureKind::RateLimit, 429)
)]
#[tokio::test]
async fn a_failed_stream_no_attempt_may_answer_for_is_declined_before_any_attempt(
    #[case] dynamic: Value,
    #[case] retry_policy: Value,
    #[case] request: RouterRequest,
    #[case] failed: AttemptError,
) {
    let mut config = lone(plain());
    config.retry_policy = serde_json::from_value(retry_policy).unwrap();
    let host = Arc::new(ScriptedHost::default());
    let router = router(config, dynamic, host.clone());

    let taken = router.take_over(request, origin_on("p1"), failed).await;

    assert!(taken.is_none(), "{taken:?}");
    assert_eq!(host.selected(), Vec::<String>::new());
}

#[rstest]
#[case::the_routers_num_retries(plain(), json!({"num_retries": 2}), json!(null), server_error(), 2)]
#[case::a_retry_policy_when_num_retries_is_zero(
    plain(),
    json!({"num_retries": 0}),
    json!({"RateLimitErrorRetries": 2}),
    failure(FailureKind::RateLimit, 429),
    2
)]
#[case::the_deployments_own_num_retries(
    json!({"model": "openai/p1", "num_retries": 1}),
    json!({"num_retries": 0}),
    json!(null),
    server_error(),
    1
)]
#[tokio::test]
async fn a_failed_stream_is_answered_by_a_retry_in_its_group_under_the_budget_that_governs_it(
    #[case] params: Value,
    #[case] dynamic: Value,
    #[case] retry_policy: Value,
    #[case] failed: AttemptError,
    #[case] budget: u32,
) {
    let mut config = lone(params);
    config.retry_policy = serde_json::from_value(retry_policy).unwrap();
    let host = Arc::new(ScriptedHost::default());
    let router = router(config, dynamic, host.clone());

    let retried = answer(router.take_over(request(), origin_on("p1"), failed).await);

    assert_eq!(
        (
            retried.output.response.as_str(),
            retried.attempted_retries,
            retried.max_retries,
            retried.origin,
            host.retries(),
        ),
        (
            "p1",
            1,
            Some(budget),
            origin(json!({
                "deployment_id": "p1",
                "attempted_retries": 1,
                "committed_budget": budget,
            })),
            1,
        )
    );
}

/// Take over a stream of `primary`, then the stream of every answer `primary` gives in turn,
/// each failing with a server error before content: the deployments that answered, and how the
/// take-over `primary` did not answer went.
async fn taken_over_while_primary_answers(
    router: &Hosted,
) -> (
    Vec<String>,
    Option<Result<RouterResponse<String>, Unanswered<Raised>>>,
) {
    let mut origin = origin_on("p1");
    let mut answered = Vec::new();
    loop {
        match router.take_over(request(), origin, server_error()).await {
            Some(Ok(answer)) if answer.origin.group() == "primary" => {
                answered.push(answer.output.response);
                origin = answer.origin;
            }
            ended => return (answered, ended),
        }
    }
}

#[tokio::test]
async fn stream_retries_whose_streams_fail_too_stop_at_the_budget() {
    let host = Arc::new(ScriptedHost::default());
    let router = router(lone(plain()), json!({"num_retries": 2}), host.clone());

    let (answered, ended) = taken_over_while_primary_answers(&router).await;

    assert!(ended.is_none(), "{ended:?}");
    assert_eq!(answered, ["p1", "p1"]);
    assert_eq!(host.retries(), 2);
}

#[tokio::test]
async fn spent_stream_retries_hand_the_stream_over_to_the_fallbacks() {
    let host = Arc::new(ScriptedHost::default());
    let router = router(lone(plain()), falling_back(1), host.clone());

    let (answered, ended) = taken_over_while_primary_answers(&router).await;

    let fallback = answer(ended);
    assert_eq!(answered, ["p1"]);
    assert_eq!(
        (
            fallback.output.response.as_str(),
            fallback.attempted_fallbacks,
            fallback.origin,
            host.retries(),
        ),
        (
            "b1",
            1,
            origin(json!({"group": "backup", "depth": 1, "level": 1, "deployment_id": "b1"})),
            1,
        )
    );
}

/// Python's mid-stream retry stamps its budget on the request metadata the fallback hop copies
/// (`record_retry_attempt`), so the fallback's answer reports it though no retry served it.
#[rstest]
#[case::after_a_retry_that_failed_before_its_stream(1, false, Some(1))]
#[case::after_a_retry_whose_stream_failed_too(1, true, Some(1))]
#[case::with_no_retry_run(0, false, None)]
#[tokio::test]
async fn a_fallback_that_takes_over_a_stream_reports_the_retry_budget_its_group_spent(
    #[case] num_retries: u32,
    #[case] retry_answers: bool,
    #[case] max_retries: Option<u32>,
) {
    let failures = if retry_answers {
        vec![]
    } else {
        vec![server_error()]
    };
    let host = Arc::new(ScriptedHost::failing("primary", failures));
    let router = router(lone(plain()), falling_back(num_retries), host.clone());

    let (_, ended) = taken_over_while_primary_answers(&router).await;

    let fallback = answer(ended);
    assert_eq!(
        (
            fallback.output.response.as_str(),
            fallback.attempted_retries,
            fallback.max_retries,
            host.retries(),
        ),
        ("b1", 0, max_retries, usize::try_from(num_retries).unwrap())
    );
}

#[tokio::test]
async fn a_stream_retry_that_fails_spends_a_retry_and_the_next_one_answers() {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(lone(plain()), json!({"num_retries": 2}), host.clone());

    let retried = answer(
        router
            .take_over(request(), origin_on("p1"), server_error())
            .await,
    );

    assert_eq!(
        (retried.attempted_retries, retried.max_retries),
        (2, Some(2))
    );
    assert_eq!(host.selected(), ["p1", "p1"]);
}

#[tokio::test]
async fn a_stream_that_failed_its_agentic_followup_skips_the_retries_for_the_fallback() {
    let host = Arc::new(ScriptedHost::default());
    let router = router(lone(plain()), falling_back(2), host.clone());

    let fallback = answer(
        router
            .take_over(request(), origin_on("p1"), agentic_followup())
            .await,
    );

    assert_eq!(
        (fallback.output.response.as_str(), host.retries()),
        ("b1", 0)
    );
    assert_eq!(host.selected(), ["b1"]);
}

/// A fallback that serves nothing ends the take-over on the failure it began with, so the
/// stream's own error is what the caller hears, not the fallback's.
#[rstest]
#[case::a_fallback_that_fails_leaves_the_streams_own_failure(falling_back(0), "backup", "b1", None)]
#[case::a_retry_that_fails_with_no_fallback_hands_back_its_error(
    json!({"num_retries": 1}),
    "primary",
    "p1",
    Some("p1")
)]
#[tokio::test]
async fn a_take_over_that_serves_nothing_says_which_error_it_ends_on(
    #[case] dynamic: Value,
    #[case] failing: &str,
    #[case] attempted_on: &str,
    #[case] raised_on: Option<&str>,
) {
    let host = Arc::new(ScriptedHost::failing(failing, vec![server_error()]));
    let router = router(lone(plain()), dynamic, host.clone());

    let taken = router
        .take_over(request(), origin_on("p1"), server_error())
        .await;

    let raised = match taken {
        Some(Err(Unanswered { raised })) => raised.map(|Raised(deployment)| deployment),
        other => panic!("the take-over did not serve nothing: {other:?}"),
    };
    assert_eq!(raised.as_deref(), raised_on);
    assert_eq!(host.selected(), [attempted_on]);
}

/// The response of a call `primary` failed and `backup` answered, each group retrying
/// `num_retries` times.
async fn answered_by_the_fallback(num_retries: u32) -> (Hosted, RouterResponse<String>) {
    let failures = (0..=num_retries).map(|_| server_error()).collect();
    let host = Arc::new(ScriptedHost::failing("primary", failures));
    let router = router(lone(plain()), falling_back(num_retries), host);
    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    (router, response)
}

#[tokio::test]
async fn a_response_says_where_its_stream_came_from() {
    let (_, response) = answered_by_the_fallback(1).await;

    assert_eq!(
        response.origin,
        origin(json!({"group": "backup", "depth": 1, "level": 1, "deployment_id": "b1"}))
    );
}

#[tokio::test]
async fn a_retried_response_says_how_many_retries_its_stream_came_after() {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(lone(plain()), json!({"num_retries": 1}), host);

    let response = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();

    assert_eq!(
        response.origin,
        origin(json!({"deployment_id": "p1", "attempted_retries": 1}))
    );
}

#[rstest]
#[case::retried_in_its_own_group_under_that_groups_budget(1, Some("b1"))]
#[case::declined_with_no_retry_left(0, None)]
#[tokio::test]
async fn a_fallbacks_stream_that_fails_too_falls_back_no_further(
    #[case] num_retries: u32,
    #[case] answered_by: Option<&str>,
) {
    let (router, fallback) = answered_by_the_fallback(num_retries).await;

    let taken = router
        .take_over(request(), fallback.origin, server_error())
        .await;

    let answered = taken.map(|taken| taken.map(|answer| answer.output.response));
    assert_eq!(answered, answered_by.map(|id| Ok(id.to_owned())));
}

/// Python wraps a fallback's stream over the fallback's own kwargs, so the retry that takes the
/// stream over sends what the dict target merged in.
#[tokio::test]
async fn a_dict_fallbacks_stream_is_retried_with_the_params_the_fallback_merged() {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![server_error(), server_error()],
    ));
    let router = router(
        lone(plain()),
        json!({
            "num_retries": 1,
            "fallbacks": [{"primary": [{"model": "backup", "max_tokens": 64}]}],
        }),
        host.clone(),
    );
    let fallback = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();

    let retried = answer(
        router
            .take_over(request(), fallback.origin, server_error())
            .await,
    );

    let merged = Some(json!({"model": "backup", "max_tokens": 64}));
    assert_eq!(
        (
            retried.output.response.as_str(),
            host.params_sent.lock().as_slice()
        ),
        ("b1", [None, None, merged.clone(), merged].as_slice())
    );
}

/// A host that ends the call from inside a retry ends the take-over there, as it ends a call
/// from inside any attempt.
#[tokio::test]
async fn a_retry_the_host_refuses_ends_the_take_over_before_any_fallback() {
    let host = Arc::new(ScriptedHost {
        refuses: true,
        ..ScriptedHost::default()
    });
    let router = router(lone(plain()), falling_back(1), host.clone());

    let taken = router
        .take_over(request(), origin_on("p1"), server_error())
        .await;

    let ended = taken.map(|taken| taken.map(|answer| answer.output.response));
    let refused = Unanswered {
        raised: Some(Raised("p1".to_owned())),
    };
    assert_eq!(
        (ended, host.selected()),
        (Some(Err(refused)), vec!["p1".to_owned()])
    );
}

/// The Python router never ran weighted failover for a stream, since the error it built from
/// the stream named no deployment. The stream's origin names it here.
#[tokio::test]
async fn a_failed_stream_fails_over_to_another_deployment_of_its_group() {
    let host = Arc::new(ScriptedHost::default());
    let router = router(weighted_failover(), json!({"num_retries": 0}), host.clone());
    let taken = router
        .take_over(request(), origin_on("a1"), server_error())
        .await;

    let served = answer(taken);

    assert_eq!(served.output.response, "a2");
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
#[case::batch_create_moves_past_the_level_it_used(azure_chat_and_batch_tiers(), true, false, vec!["b2", "b3"])]
#[case::other_batch_calls_walk_every_level(azure_chat_and_batch_tiers(), false, true, vec!["c1", "b2"])]
#[case::batch_create_without_a_batch_deployment_walks_every_level(azure_chat_tiers(), true, false, vec!["c1", "c2"])]
#[tokio::test]
async fn a_failed_order_level_falls_back_to_the_next_level_its_pick_can_use(
    #[case] config: RouterConfig,
    #[case] batch_create: bool,
    #[case] batch_api: bool,
    #[case] expected: Vec<&str>,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(config, json!({"num_retries": 0}), host.clone());
    let request = RouterRequest {
        batch_create,
        batch_api,
        ..request()
    };
    router
        .call(Op::from("acreate_batch"), request)
        .await
        .unwrap();
    assert_eq!(host.selected(), expected);
}

#[rstest]
#[tokio::test]
async fn a_failed_chat_level_never_falls_back_to_a_batch_only_level() {
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let router = router(
        azure_chat_and_batch_tiers(),
        json!({"num_retries": 0}),
        host.clone(),
    );
    let error = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap_err();
    assert_eq!(host.selected(), vec!["c1"]);
    assert!(
        matches!(error, Error::Attempt { failure: failed, .. } if failed.kind == FailureKind::InternalServer)
    );
}

fn azure_chat_beside_batch() -> RouterConfig {
    azure_chat_beside_batch_named("primary")
}

fn azure_chat_beside_batch_named(model_name: &str) -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": model_name, "litellm_params": {"model": "azure/chat"}, "model_info": {"id": "c1", "mode": "chat"}},
            {"model_name": model_name, "litellm_params": {"model": "azure/batch"}, "model_info": {"id": "b2", "mode": "batch"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::a_chat_call("primary", "primary", false, vec!["c1"])]
#[case::a_batch_or_file_call("primary", "primary", true, vec!["c1", "b2"])]
#[case::a_chat_call_through_a_wildcard("azure/*", "azure/gpt-4o", false, vec!["c1"])]
#[case::a_batch_or_file_call_through_a_wildcard("azure/*", "azure/gpt-4o", true, vec!["c1", "b2"])]
#[tokio::test]
async fn only_batch_and_file_calls_pick_among_azure_batch_only_deployments(
    #[case] model_name: &str,
    #[case] model: &str,
    #[case] batch_api: bool,
    #[case] expected: Vec<&str>,
) {
    let router = router(
        azure_chat_beside_batch_named(model_name),
        json!({}),
        Arc::new(ScriptedHost::default()),
    );
    let request = RouterRequest {
        model: model.to_owned(),
        batch_api,
        ..request()
    };
    let candidates = router
        .healthy(&Op::from("acompletion"), &request)
        .await
        .unwrap();
    let healthy: Vec<&str> = candidates.healthy.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(healthy, expected);
}

#[rstest]
#[case::to_the_batch_only_deployment("b2")]
#[case::to_the_chat_deployment("c1")]
#[tokio::test]
async fn a_call_pinned_to_a_deployment_of_a_mixed_azure_group_picks_it(#[case] pinned: &str) {
    let router = router(
        azure_chat_beside_batch(),
        json!({}),
        Arc::new(ScriptedHost::default()),
    );
    let request = RouterRequest {
        pinned_deployment_id: Some(pinned.to_owned()),
        ..request()
    };
    let candidates = router
        .healthy(&Op::from("avector_store_file_create"), &request)
        .await
        .unwrap();
    let healthy: Vec<&str> = candidates.healthy.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(healthy, vec![pinned]);
}

fn azure_chat_with_retries_beside_batch() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "azure/chat", "num_retries": 3}, "model_info": {"id": "c1", "mode": "chat"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/batch"}, "model_info": {"id": "b2", "mode": "batch"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::a_chat_call(false, Some(3))]
#[case::a_batch_or_file_call(true, None)]
#[tokio::test]
async fn a_rate_limit_mock_carries_the_num_retries_of_the_only_deployment_its_call_could_use(
    #[case] batch_api: bool,
    #[case] expected: Option<u32>,
) {
    let host = Arc::new(ScriptedHost::default());
    let router = router(
        azure_chat_with_retries_beside_batch(),
        json!({}),
        host.clone(),
    );
    let request = RouterRequest {
        batch_api,
        mock_failure: Some(MockFailure::RateLimit),
        ..request()
    };
    let _ = router.call(Op::from("acompletion"), request).await;
    let carried = host.events.lock().iter().find_map(|event| match event {
        RouterEvent::MockFailed { num_retries, .. } => Some(*num_retries),
        _ => None,
    });
    assert_eq!(carried, Some(expected));
}

fn azure_chat_alone() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "azure/chat"}, "model_info": {"id": "c1", "mode": "chat"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::alone(azure_chat_alone())]
#[case::beside_a_batch_only_one(azure_chat_beside_batch())]
#[tokio::test]
async fn a_rate_limited_chat_deployment_stays_up_and_is_retried(#[case] config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![failure(FailureKind::RateLimit, 429)],
    ));
    let router = router(config, json!({"num_retries": 2}), host.clone());
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let cooled = router
        .health()
        .active_cooldowns(&["c1".to_owned()])
        .await
        .unwrap();
    assert_eq!(
        (host.selected(), cooled.len()),
        (vec!["c1".to_owned(), "c1".to_owned()], 0)
    );
}

fn azure_team_chat_beside_batch() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "t1a", "litellm_params": {"model": "azure/chat"},
             "model_info": {"id": "c1", "mode": "chat", "team_id": "T1", "team_public_model_name": "tp"}},
            {"model_name": "t1b", "litellm_params": {"model": "azure/batch"},
             "model_info": {"id": "b2", "mode": "batch", "team_id": "T1", "team_public_model_name": "tp"}},
        ]
    }))
    .unwrap()
}

fn azure_chat_and_batch_groups_in_a_routing_group() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "azure/chat"}, "model_info": {"id": "c1", "mode": "chat"}},
            {"model_name": "batches", "litellm_params": {"model": "azure/batch"}, "model_info": {"id": "b2", "mode": "batch"}},
        ],
        "routing_groups": [
            {"group_name": "rg", "models": ["primary", "batches"], "routing_strategy": "simple-shuffle"},
        ],
    }))
    .unwrap()
}

fn azure_chat_beside_chat_and_batch() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "azure/chat"}, "model_info": {"id": "c1", "mode": "chat"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/chat-2"}, "model_info": {"id": "c2", "mode": "chat"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/batch"}, "model_info": {"id": "b3", "mode": "batch"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[case::alone(azure_chat_alone(), None, false)]
#[case::beside_a_batch_only_one(azure_chat_beside_batch(), None, false)]
#[case::beside_a_batch_only_one_under_a_team_name(azure_team_chat_beside_batch(), None, false)]
#[case::beside_a_batch_only_group_in_a_routing_group(
    azure_chat_and_batch_groups_in_a_routing_group(),
    Some("rg"),
    false
)]
#[case::beside_another_chat_one(azure_chat_beside_chat_and_batch(), None, true)]
#[tokio::test]
async fn a_reported_rate_limit_cools_a_chat_deployment_only_beside_another_that_serves_chat(
    #[case] config: RouterConfig,
    #[case] requested_group: Option<&str>,
    #[case] cooled: bool,
) {
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let cooldown = router
        .report_failure(
            "c1",
            &failure(FailureKind::RateLimit, 429),
            FailureReport {
                requested_group: requested_group.map(str::to_owned),
                ..FailureReport::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(cooldown.is_some(), cooled);
}

#[rstest]
#[case::a_chat_call(false, ("c1", "c1"))]
#[case::a_batch_or_file_call(true, ("c1", "b2"))]
#[tokio::test]
async fn a_rate_limit_keeps_chat_on_its_only_deployment_and_moves_only_batch_calls_off_it(
    #[case] batch_api: bool,
    #[case] expected: (&str, &str),
) {
    let host = Arc::new(ScriptedHost::default());
    *host.first_failure.lock() = Some(failure(FailureKind::RateLimit, 429));
    let router = router(
        azure_chat_then_batch(),
        json!({"num_retries": 0}),
        host.clone(),
    );
    let rate_limited = RouterRequest {
        batch_api,
        ..request()
    };
    let _ = router.call(Op::from("acompletion"), rate_limited).await;
    let after_failure = host.selected().len();
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let file_call = RouterRequest {
        batch_api: true,
        ..request()
    };
    router
        .call(Op::from("afile_delete"), file_call)
        .await
        .unwrap();
    let picked = host.selected()[after_failure..].to_vec();
    assert_eq!((picked[0].as_str(), picked[1].as_str()), expected);
}

fn openai_batch_then_azure_batch_only() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/gpt", "order": 1}, "model_info": {"id": "o1", "mode": "batch"}},
            {"model_name": "primary", "litellm_params": {"model": "azure/batch", "order": 2}, "model_info": {"id": "b2", "mode": "batch"}},
        ]
    }))
    .unwrap()
}

#[rstest]
#[tokio::test]
async fn a_rate_limited_batch_create_moves_batch_creates_to_the_azure_batch_only_peer_and_keeps_chat()
 {
    let host = Arc::new(ScriptedHost::default());
    *host.first_failure.lock() = Some(failure(FailureKind::RateLimit, 429));
    let router = router(
        openai_batch_then_azure_batch_only(),
        json!({"num_retries": 0}),
        host.clone(),
    );
    let create = RouterRequest {
        batch_create: true,
        ..request()
    };
    let _ = router.call(Op::from("acreate_batch"), create.clone()).await;
    let after_failure = host.selected().len();
    router
        .call(Op::from("acreate_batch"), create)
        .await
        .unwrap();
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    assert_eq!(host.selected()[after_failure..], ["b2", "o1"]);
}

#[rstest]
#[case::a_chat_call(false, false)]
#[case::a_batch_or_file_call(true, true)]
#[tokio::test]
async fn a_reported_rate_limit_moves_only_batch_calls_off_a_chat_deployment_beside_a_batch_only_one(
    #[case] batch_api: bool,
    #[case] batch_calls_move: bool,
) {
    let router = router(
        azure_chat_beside_batch(),
        json!({}),
        Arc::new(ScriptedHost::default()),
    );
    router
        .report_failure(
            "c1",
            &failure(FailureKind::RateLimit, 429),
            FailureReport {
                batch_api,
                ..FailureReport::default()
            },
        )
        .await
        .unwrap();
    let picks = |batch_api: bool| {
        let router = &router;
        async move {
            let request = RouterRequest {
                batch_api,
                ..request()
            };
            router
                .healthy(&Op::from("acompletion"), &request)
                .await
                .map(|c| {
                    c.healthy
                        .iter()
                        .map(|d| d.id.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        }
    };
    let batch_picks = if batch_calls_move {
        vec!["b2".to_owned()]
    } else {
        vec!["c1".to_owned(), "b2".to_owned()]
    };
    assert_eq!(
        (picks(false).await, picks(true).await),
        (vec!["c1".to_owned()], batch_picks)
    );
}

#[rstest]
#[tokio::test]
async fn a_rate_limited_team_chat_deployment_beside_a_batch_only_sibling_stays_up_and_is_retried() {
    let host = Arc::new(ScriptedHost::default());
    *host.first_failure.lock() = Some(failure(FailureKind::RateLimit, 429));
    let router = router(
        azure_team_chat_beside_batch(),
        json!({"num_retries": 2}),
        host.clone(),
    );
    let request = RouterRequest {
        model: "tp".to_owned(),
        team_id: Some("T1".to_owned()),
        listing_team_id: Some("T1".to_owned()),
        ..RouterRequest::default()
    };
    let response = router.call(Op::from("acompletion"), request).await.unwrap();
    let cooled = router
        .health()
        .active_cooldowns(&["c1".to_owned()])
        .await
        .unwrap();
    assert_eq!(
        (host.selected(), response.deployment_id, cooled.len()),
        (vec!["c1".to_owned(), "c1".to_owned()], "c1".to_owned(), 0)
    );
}

#[rstest]
#[case::a_chat_call(false, false)]
#[case::a_batch_or_file_call(true, true)]
fn a_stream_can_fall_back_only_to_an_order_level_its_call_could_use(
    #[case] batch_api: bool,
    #[case] can_fall_back: bool,
) {
    let router = router(
        azure_chat_then_batch(),
        json!({}),
        Arc::new(ScriptedHost::default()),
    );
    let request = RouterRequest {
        batch_api,
        ..request()
    };
    let policy = router.stream_policy(&request, &origin(json!({"order": 1})), None);
    assert_eq!(policy.can_fall_back, can_fall_back);
}

#[rstest]
#[case::alone(azure_chat_alone())]
#[case::beside_a_batch_only_one(azure_chat_beside_batch())]
#[tokio::test]
async fn a_chat_deployment_refusing_the_key_returns_its_own_error_without_retries(
    #[case] mut config: RouterConfig,
) {
    config.disable_cooldowns = true;
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![failure(FailureKind::Authentication, 401)],
    ));
    let router = router(config, json!({"num_retries": 2}), host.clone());
    let error = router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap_err();
    assert_eq!(
        (host.selected(), error.attempt().map(|failed| failed.kind)),
        (vec!["c1".to_owned()], Some(FailureKind::Authentication))
    );
}

#[rstest]
#[case::alone(azure_chat_alone())]
#[case::beside_a_batch_only_one(azure_chat_beside_batch())]
#[tokio::test]
async fn a_stream_a_chat_deployment_refused_the_key_for_is_not_retried(
    #[case] config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::default());
    let router = router(config, json!({}), host.clone());
    let refused = router
        .retry_stream(
            Op::from("acompletion"),
            stream_request(),
            stream_origin(),
            failure(FailureKind::Authentication, 401),
        )
        .await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.kind == FailureKind::Authentication)
    );
    assert_eq!(host.selected().len(), 0);
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
    assert!(matches!(error, Error::Attempt { failure: refused, .. } if refused.kind == kind));
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
    assert!(
        matches!(raised, Err(Error::Attempt { failure: failed, .. }) if failed.is_a("InternalServerError"))
    );
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
    assert!(
        matches!(raised, Err(Error::Attempt { failure: failed, .. }) if failed.is_a("ValueError"))
    );
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
    assert!(
        matches!(raised, Err(Error::Attempt { failure: failed, .. }) if failed.kind == FailureKind::RateLimit)
    );
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
    assert!(
        matches!(raised, Err(Error::Attempt { failure: failed, .. }) if failed.is_a("TypeError"))
    );
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
    assert!(
        matches!(raised, Err(Error::Attempt { failure: failed, .. }) if failed.is_a(raised_class))
    );
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
#[case::beside_a_batch_only_one(json!([
    {"model_name": "primary", "litellm_params": {"model": "azure/chat"},
     "model_info": {"id": "c1", "mode": "chat", "supports_web_search": false}},
    {"model_name": "primary", "litellm_params": {"model": "azure/batch"}, "model_info": {"id": "b2", "mode": "batch"}},
]))]
#[case::alone(json!([
    {"model_name": "primary", "litellm_params": {"model": "azure/chat"},
     "model_info": {"id": "c1", "mode": "chat", "supports_web_search": false}},
]))]
#[tokio::test]
async fn a_web_search_call_the_azure_chat_deployment_cannot_take_never_goes_to_its_batch_only_peer(
    #[case] model_list: serde_json::Value,
) {
    let config: RouterConfig = serde_json::from_value(json!({"model_list": model_list})).unwrap();
    let router = router(config, json!({}), Arc::new(ScriptedHost::default()));
    let request = RouterRequest {
        web_search: true,
        ..request()
    };
    let error = router
        .select(&Op::from("acompletion"), &request)
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::NoHealthyDeployments { model, .. } if model == "primary"),
        "{error:?}"
    );
}

#[rstest]
#[case::beside_a_batch_only_one(json!([
    {"model_name": "primary", "litellm_params": {"model": "azure/chat"}, "model_info": {"id": "c1", "mode": "chat"}},
    {"model_name": "primary", "litellm_params": {"model": "azure/batch"}, "model_info": {"id": "b2", "mode": "batch"}},
]))]
#[case::alone(json!([
    {"model_name": "primary", "litellm_params": {"model": "azure/chat"}, "model_info": {"id": "c1", "mode": "chat"}},
]))]
#[tokio::test]
async fn a_chat_call_keeps_an_unhealthy_azure_chat_deployment_as_it_would_alone(
    #[case] model_list: serde_json::Value,
) {
    let config: RouterConfig = serde_json::from_value(
        json!({"model_list": model_list, "enable_health_check_routing": true}),
    )
    .unwrap();
    let host = Arc::new(ScriptedHost::default());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    host.cache.inner.set_health_states(HashMap::from([
        (
            "c1".to_owned(),
            HealthState {
                is_healthy: false,
                timestamp: now,
            },
        ),
        (
            "b2".to_owned(),
            HealthState {
                is_healthy: true,
                timestamp: now,
            },
        ),
    ]));
    let picked = router(config, json!({}), host)
        .select(&Op::from("acompletion"), &request())
        .await;
    assert!(
        matches!(&picked, Ok(deployment) if deployment.id == "c1"),
        "{picked:?}"
    );
}

#[rstest]
#[case::never_probed(None)]
#[case::last_failure_gone_stale(Some((false, 3600.0)))]
#[case::last_probe_healthy(Some((true, 0.0)))]
#[tokio::test]
async fn health_check_routing_falls_back_to_the_unhealthy_deployments_beside_a_paused_one(
    #[case] paused_probe: Option<(bool, f64)>,
) {
    let config: RouterConfig = serde_json::from_value(json!({
        "model_list": [
            {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
            {"model_name": "primary", "litellm_params": {"model": "openai/p"}, "model_info": {"id": "p2", "blocked": true}},
        ],
        "enable_health_check_routing": true,
    }))
    .unwrap();
    let host = Arc::new(ScriptedHost::default());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let active = HealthState {
        is_healthy: false,
        timestamp: now,
    };
    host.cache.inner.set_health_states(
        std::iter::once(("a1".to_owned(), active))
            .chain(paused_probe.map(|(is_healthy, age)| {
                (
                    "p2".to_owned(),
                    HealthState {
                        is_healthy,
                        timestamp: now - age,
                    },
                )
            }))
            .collect(),
    );
    let picked = router(config, json!({}), host)
        .select(&Op::from("acompletion"), &request())
        .await;
    assert!(
        matches!(&picked, Ok(deployment) if deployment.id == "a1"),
        "{picked:?}"
    );
}

#[rstest]
#[case::beside_a_batch_only_one(json!([
    {"model_name": "primary", "litellm_params": {"model": "azure/chat"}, "model_info": {"id": "c1", "mode": "chat"}},
    {"model_name": "primary", "litellm_params": {"model": "azure/batch"}, "model_info": {"id": "b2", "mode": "batch"}},
]))]
#[case::alone(json!([
    {"model_name": "primary", "litellm_params": {"model": "azure/chat"}, "model_info": {"id": "c1", "mode": "chat"}},
]))]
#[tokio::test]
async fn weighted_failover_of_a_chat_call_never_counts_the_batch_only_peer_as_left(
    #[case] model_list: serde_json::Value,
) {
    let config: RouterConfig =
        serde_json::from_value(json!({"model_list": model_list, "enable_weighted_failover": true}))
            .unwrap();
    let host = Arc::new(ScriptedHost::failing("primary", vec![server_error()]));
    let raised = router(config, json!({"num_retries": 0}), host.clone())
        .call(Op::from("acompletion"), request())
        .await;
    assert!(
        matches!(&raised, Err(Error::Attempt { failure, .. }) if failure.is_a("InternalServerError")),
        "{raised:?}"
    );
    assert_eq!(host.selected(), vec!["c1".to_owned()]);
    assert!(
        !host
            .events
            .lock()
            .iter()
            .any(|e| matches!(e, RouterEvent::Fallback { .. })),
    );
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
    let host = ScriptedHost::default();
    let router = Router::builder(config, RouterDynamicConfig::default())
        .randomness(Arc::new(SeededRandomness::new(7)))
        .clock(Arc::new(StoppedClock(now)))
        .build()
        .unwrap();
    let mut picked = std::collections::BTreeSet::new();
    for _ in 0..40 {
        let deployment = router
            .select(&host, &Op::from("acompletion"), &request_for("chat"))
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

fn queue_rejected() -> AttemptError {
    let mut rejected = AttemptError::new(FailureKind::QueueRejected, Some(503));
    rejected.classes = vec![
        "DynamicQueueFullError".into(),
        "ServiceUnavailableError".into(),
    ];
    rejected
}

fn event_count(host: &ScriptedHost, name: &str) -> usize {
    host.events
        .lock()
        .iter()
        .filter(|event| serde_json::to_value(event).unwrap()["event"] == name)
        .count()
}

#[rstest]
#[tokio::test]
async fn a_queue_rejection_ends_the_request_with_no_retry_fallback_or_cooldown(
    config: RouterConfig,
) {
    let host = Arc::new(ScriptedHost::failing("primary", vec![queue_rejected()]));
    let router = router(
        config,
        json!({"num_retries": 3, "fallbacks": [{"primary": ["backup"]}]}),
        host.clone(),
    );
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.queue_rejected())
    );
    assert_eq!((host.retries(), host.selected().len()), (0, 1));
    assert_eq!(event_count(&host, "AttemptFailed"), 0);
    assert_eq!(event_count(&host, "CooldownSet"), 0);
}

#[rstest]
#[tokio::test]
async fn a_queue_rejection_on_a_fallback_hop_ends_the_request_with_that_rejection() {
    let host = Arc::new(ScriptedHost::scripted(vec![
        ("primary", vec![server_error()]),
        ("backup", vec![queue_rejected()]),
    ]));
    let router = router(
        tiers(false),
        json!({"num_retries": 0, "fallbacks": [{"primary": ["backup", "context"]}]}),
        host.clone(),
    );
    let refused = router.call(Op::from("acompletion"), request()).await;
    assert!(
        matches!(refused, Err(Error::Attempt { failure: failed, .. }) if failed.queue_rejected())
    );
    assert_eq!(host.selected(), vec!["p1".to_owned(), "b1".to_owned()]);
}

/// `primary`'s one deployment with `limits`, checked by `enforce_model_rate_limits` at 10:00 UTC
/// on 2026-01-01 when the core is `enforcing`.
fn rate_limiting(limits: Value, enforcing: bool, host: Arc<ScriptedHost>) -> Hosted {
    rate_limiting_among(
        &["p1"],
        limits,
        json!({"pre_call_checks": match enforcing {
            true => vec!["enforce_model_rate_limits"],
            false => vec![],
        }}),
        host,
    )
}

/// [`rate_limiting`] with `primary` served by `ids`, each with `limits`, under `settings`.
fn rate_limiting_among(
    ids: &[&str],
    limits: Value,
    settings: Value,
    host: Arc<ScriptedHost>,
) -> Hosted {
    let deployment = |id: &&str| {
        let params = json!({"model": format!("openai/{id}")})
            .as_object()
            .into_iter()
            .chain(limits.as_object())
            .flatten()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<serde_json::Map<_, _>>();
        json!({"model_name": "primary", "litellm_params": params, "model_info": {"id": id}})
    };
    let config = json!({"model_list": ids.iter().map(deployment).collect::<Vec<_>>()})
        .as_object()
        .into_iter()
        .chain(settings.as_object())
        .flatten()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    let config: RouterConfig = serde_json::from_value(Value::Object(config)).unwrap();
    let dynamic: RouterDynamicConfig = serde_json::from_value(json!({"num_retries": 2})).unwrap();
    let router = Router::builder(config, dynamic)
        .randomness(Arc::new(SeededRandomness::new(7)))
        .clock(Arc::new(StoppedClock(utc_hour(10))))
        .build()
        .unwrap();
    Hosted { router, host }
}

fn limit_counter(kind: LimitKind, window: &str) -> LimitCounter {
    LimitCounter {
        deployment_id: "p1".to_owned(),
        model: "openai/p1".to_owned(),
        kind,
        window: window.to_owned(),
    }
}

/// What the router refused the call with: its status, `retry-after`, retries and text.
fn refusal(
    ended: Result<RouterResponse<String>, Error>,
) -> (Option<u16>, u64, Option<u32>, String) {
    match ended {
        Err(Error::Attempt {
            failure,
            raised_by: RaisedBy::PreCallCheck(refused),
        }) => {
            assert_eq!(
                (refused.deployment_id.as_str(), failure.retry_after),
                ("p1", None)
            );
            (
                failure.status,
                refused.resets_in.as_secs(),
                failure.num_retries,
                failure.detail.as_deref().unwrap_or_default().to_owned(),
            )
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// Each success is one request and 10 tokens, so the second call is past every limit here. A
/// daily limit is tried again when its window starts over, an hour from the clock.
#[rstest]
#[case::rpm(json!({"rpm": 1}), 60, "RPM limit=1, current usage=2")]
#[case::tpm(json!({"tpm": 10}), 60, "TPM limit=10, current usage=10")]
#[case::rpd(json!({"rpd": 1, "daily_reset_time": "11:00"}), 3600, "RPD limit=1, current usage=2")]
#[case::tpd(json!({"tpd": 10, "daily_reset_time": "11:00"}), 3600, "TPD limit=10, current usage=10")]
#[case::rpm_as_text(json!({"rpm": "1"}), 60, "RPM limit=1, current usage=2")]
#[case::tpd_as_text(json!({"tpd": "10"}), 50_400, "TPD limit=10, current usage=10")]
#[tokio::test]
async fn a_deployment_past_a_declared_limit_is_refused_without_an_attempt_or_a_retry(
    #[case] limits: Value,
    #[case] retry_after: u64,
    #[case] over: &str,
) {
    let host = Arc::new(ScriptedHost::default());
    let router = rate_limiting(limits, true, host.clone());

    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let refused = refusal(router.call(Op::from("acompletion"), request()).await);

    assert_eq!(
        refused,
        (
            Some(429),
            retry_after,
            Some(0),
            format!("Model rate limit exceeded. {over}"),
        )
    );
    assert_eq!(host.cache.outcomes("p1").await.unwrap().successes, 1);
    assert_eq!(host.retries(), 0);
}

#[tokio::test]
async fn a_refused_request_is_refunded_from_both_request_windows() {
    let host = Arc::new(ScriptedHost::default());
    let router = rate_limiting(json!({"rpm": 1, "rpd": 5}), true, host.clone());

    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap_err();

    assert_eq!(
        (
            host.cache
                .limit_usage(&limit_counter(LimitKind::Rpm, "10-00"), 1.0)
                .await
                .unwrap(),
            host.cache
                .limit_usage(&limit_counter(LimitKind::Rpd, "2026-01-01"), 5.0)
                .await
                .unwrap(),
        ),
        (Some(1.0), Some(1.0))
    );
}

/// `_refused` reports the refusal with `uncounted=True`, so the group's remaining requests do not
/// run one low for each request the limit turned away.
#[tokio::test]
async fn a_refused_request_counts_against_no_usage_window() {
    let host = Arc::new(ScriptedHost::default());
    let router = rate_limiting(json!({"rpm": 1}), true, host.clone());

    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap_err();

    let deployment = router.registry().deployment("p1").unwrap();
    assert_eq!(router.health().usage(&deployment).await.unwrap(), (10, 1));
}

/// `async_routing_strategy_pre_call_checks` cools a refused deployment down with
/// `set_cooldown_deployments` alone: for the router's `cooldown_time`, however far off the limit's
/// window starts over, and with no failure counted.
#[rstest]
#[case::per_minute(json!({"rpm": 0, "weight": 1}))]
#[case::per_day(json!({"rpd": 0, "weight": 1}))]
#[tokio::test]
async fn a_refused_deployment_cools_down_for_the_routers_cooldown_time_and_counts_no_failure(
    #[case] limits: Value,
) {
    let host = Arc::new(ScriptedHost::default());
    let ids = ["p1", "p2"];
    let router = rate_limiting_among(
        &ids,
        limits,
        json!({"pre_call_checks": ["enforce_model_rate_limits"], "cooldown_time": 7}),
        host.clone(),
    );

    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap_err();

    let cooled = router
        .health()
        .active_cooldowns(&ids.map(str::to_owned))
        .await
        .unwrap();
    let fails = host.cache.outcomes("p1").await.unwrap().fails
        + host.cache.outcomes("p2").await.unwrap().fails;
    assert_eq!(
        (
            cooled
                .iter()
                .map(|(_, cooldown)| cooldown.cooldown_time)
                .collect::<Vec<_>>(),
            fails,
        ),
        (vec![7.0, 7.0], 0)
    );
}

/// Puts deployment `id` of [`rate_limiting_among`] at a `rpm` of 1 for the clock's minute.
async fn at_its_rpm(host: &ScriptedHost, id: &str) {
    let counter = LimitCounter {
        deployment_id: id.to_owned(),
        model: format!("openai/{id}"),
        kind: LimitKind::Rpm,
        window: "10-00".to_owned(),
    };
    host.cache
        .add_limit_usage(&counter, 1, Duration::from_secs(60))
        .await
        .unwrap();
}

/// A refused deployment with a sibling cools down, so Python's retry loop moves on to the sibling
/// at once (`_refused`, then `should_retry_this_error` with a healthy deployment left).
#[tokio::test]
async fn a_refused_deployment_with_a_sibling_is_retried_on_the_sibling() {
    let host = Arc::new(ScriptedHost::default());
    let ids = ["p1", "p2"];
    let router = rate_limiting_among(
        &ids,
        json!({"rpm": 1}),
        json!({"pre_call_checks": ["enforce_model_rate_limits"]}),
        host.clone(),
    );
    at_its_rpm(&host, "p1").await;

    let served = router.call(Op::from("acompletion"), request()).await;

    let cooled = router
        .health()
        .active_cooldowns(&ids.map(str::to_owned))
        .await
        .unwrap();
    assert_eq!(
        (
            served.is_ok(),
            host.cache.outcomes("p2").await.unwrap().successes,
            cooled.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            host.retries(),
        ),
        (true, 1, vec!["p1"], 1)
    );
}

/// When every deployment is past its limit, each is refused once and the call ends on the last
/// refusal, with no healthy deployment left to retry on.
#[tokio::test]
async fn a_group_past_every_limit_is_refused_once_per_deployment() {
    let host = Arc::new(ScriptedHost::default());
    let ids = ["p1", "p2"];
    let router = rate_limiting_among(
        &ids,
        json!({"rpm": 1}),
        json!({"pre_call_checks": ["enforce_model_rate_limits"]}),
        host.clone(),
    );
    at_its_rpm(&host, "p1").await;
    at_its_rpm(&host, "p2").await;

    let ended = router.call(Op::from("acompletion"), request()).await;

    let cooled = router
        .health()
        .active_cooldowns(&ids.map(str::to_owned))
        .await
        .unwrap();
    let refused_with = match ended {
        Err(error) if error.refused().is_some() => {
            error.attempt().and_then(|failure| failure.status)
        }
        _ => None,
    };
    assert_eq!(
        (refused_with, cooled.len(), host.retries()),
        (Some(429), 2, 1)
    );
}

#[tokio::test]
async fn a_streams_tokens_count_once_the_host_reports_its_end() {
    let host = Arc::new(ScriptedHost {
        deferred: true,
        ..ScriptedHost::default()
    });
    let router = rate_limiting(json!({"tpm": 10}), true, host.clone());

    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let before = router.call(Op::from("acompletion"), request()).await;
    router
        .report_success("p1", 10.0, 0.0, SuccessReport::default())
        .await
        .unwrap();
    let after = router.call(Op::from("acompletion"), request()).await;

    assert!(before.is_ok(), "{before:?}");
    assert_eq!(
        refusal(after).3,
        "Model rate limit exceeded. TPM limit=10, current usage=10"
    );
}

#[tokio::test]
async fn a_router_not_told_to_enforce_the_limits_counts_nothing() {
    let host = Arc::new(ScriptedHost::default());
    let router = rate_limiting(json!({"rpm": 1}), false, host.clone());

    router
        .call(Op::from("acompletion"), request())
        .await
        .unwrap();
    let second = router.call(Op::from("acompletion"), request()).await;

    assert!(second.is_ok(), "{second:?}");
    assert_eq!(
        host.cache
            .limit_usage(&limit_counter(LimitKind::Rpm, "10-00"), 1.0)
            .await
            .unwrap(),
        None
    );
}
