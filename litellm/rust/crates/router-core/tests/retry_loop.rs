//! The unary retry loop and the stream retry loop end to end against a scripted host: what each
//! sleeps, draws and falls back to around the retry step they share.

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_router_core::{
    Attempt, AttemptError, AttemptFailure, Deployment, Error, FailureKind, InMemoryRouterCache, Op,
    PreRoute, PreRouted, Randomness, Router, RouterConfig, RouterDynamicConfig, RouterEvent,
    RouterHost, RouterRequest, RoutingContext, SelectStage, Served, call::StreamOrigin,
};
use parking_lot::Mutex;
use rstest::rstest;
use serde_json::{Value, json};

/// Fails the scripted attempts of each model group in turn and keeps every sleep it was asked for.
#[derive(Default)]
struct SleepingHost {
    script: Mutex<Vec<(String, VecDeque<AttemptError>)>>,
    sleeps: Mutex<Vec<Duration>>,
    cache: InMemoryRouterCache,
    /// A pick of this name is answered this way, as a pre-routing hook reroutes it.
    pre_routes: Option<(&'static str, PreRouted)>,
}

impl SleepingHost {
    fn failing(group: &str, failures: Vec<AttemptError>) -> Self {
        Self {
            script: Mutex::new(vec![(group.to_owned(), failures.into())]),
            ..Self::default()
        }
    }
}

impl RouterHost for SleepingHost {
    type Response = String;
    type Failure = ();
    type Cache = InMemoryRouterCache;

    async fn attempt(&self, attempt: &Attempt) -> Result<Served<String>, AttemptFailure<()>> {
        let next = self
            .script
            .lock()
            .iter_mut()
            .find(|(group, _)| *group == attempt.model_group)
            .and_then(|(_, failures)| failures.pop_front());
        match next {
            Some(facts) => Err(AttemptFailure::Failed { facts, raised: () }),
            None => Ok(Served::new(attempt.deployment.id.clone(), 10)),
        }
    }

    async fn select(
        &self,
        _stage: &SelectStage,
        _ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        Ok(candidates)
    }

    async fn pre_route(&self, ask: &PreRoute<'_>) -> Result<Option<PreRouted>, Error> {
        Ok(self
            .pre_routes
            .as_ref()
            .filter(|(name, _)| *name == ask.name)
            .map(|(_, routed)| routed.clone()))
    }

    async fn allow_fallback(&self, _target: &str) -> Result<bool, AttemptError> {
        Ok(true)
    }

    async fn sleep(&self, duration: Duration) {
        self.sleeps.lock().push(duration);
    }

    fn observe(&self, _event: &RouterEvent) {}

    fn cache(&self) -> &InMemoryRouterCache {
        &self.cache
    }
}

/// Answers every draw with a half and counts them.
#[derive(Default)]
struct CountedDraws(AtomicUsize);

impl Randomness for CountedDraws {
    fn next_f64(&self) -> f64 {
        self.0.fetch_add(1, Ordering::Relaxed);
        0.5
    }
}

/// Three groups of one deployment each.
fn lone_groups() -> RouterConfig {
    serde_json::from_value(json!({
        "model_list": [
            {"model_name": "solo", "litellm_params": {"model": "openai/s"}, "model_info": {"id": "s1"}},
            {"model_name": "backup", "litellm_params": {"model": "anthropic/b"}, "model_info": {"id": "b1"}},
            {"model_name": "context", "litellm_params": {"model": "anthropic/c"}, "model_info": {"id": "c1"}},
        ]
    }))
    .unwrap()
}

fn router(config: RouterConfig, dynamic: Value, draws: Arc<CountedDraws>) -> Router {
    let dynamic: RouterDynamicConfig = serde_json::from_value(dynamic).unwrap();
    Router::builder(config, dynamic)
        .randomness(draws)
        .build()
        .unwrap()
}

fn asking_for(model: &str) -> RouterRequest {
    RouterRequest {
        model: model.to_owned(),
        ..RouterRequest::default()
    }
}

/// The origin of a stream `solo` served, as the host hands it back.
fn stream_from_solo() -> StreamOrigin {
    serde_json::from_value(json!({"group": "solo"})).unwrap()
}

fn server_error() -> AttemptError {
    AttemptError::new(FailureKind::InternalServer, Some(500))
}

fn unsupported_endpoint() -> AttemptError {
    let mut failure = AttemptError::new(FailureKind::Declined, Some(400));
    failure.classes = vec!["UnsupportedEndpointError".into(), "BadRequestError".into()];
    failure
}

/// A half of the 0.75 second jitter, which every sleep of these tests takes.
const HALF_JITTER: Duration = Duration::from_millis(375);

#[tokio::test]
async fn a_lone_deployments_retry_waits_at_least_the_routers_retry_after() {
    let host = SleepingHost::failing("solo", vec![server_error()]);
    let config = RouterConfig {
        retry_after: 3.0,
        ..lone_groups()
    };
    let router = router(config, json!({"num_retries": 1}), Arc::default());
    let response = router
        .call(&host, Op::from("acompletion"), asking_for("solo"))
        .await
        .unwrap();
    assert_eq!(response.attempted_retries, 1);
    assert_eq!(*host.sleeps.lock(), [Duration::from_secs(3) + HALF_JITTER]);
}

#[tokio::test]
async fn the_unary_loop_sleeps_once_more_after_its_last_failed_retry() {
    let host = SleepingHost::failing("solo", vec![server_error(), server_error()]);
    let router = router(lone_groups(), json!({"num_retries": 1}), Arc::default());
    let failed = router
        .call(&host, Op::from("acompletion"), asking_for("solo"))
        .await;
    assert!(failed.is_err());
    assert_eq!(
        *host.sleeps.lock(),
        [
            Duration::from_millis(500) + HALF_JITTER,
            Duration::from_secs(1) + HALF_JITTER
        ]
    );
}

#[tokio::test]
async fn a_streams_retries_end_without_a_last_sleep() {
    let host = SleepingHost::failing("solo", vec![server_error()]);
    let router = router(lone_groups(), json!({"num_retries": 1}), Arc::default());
    let origin = stream_from_solo();
    let failed = router
        .retry_stream(
            &host,
            Op::from("acompletion"),
            asking_for("solo"),
            origin,
            server_error(),
        )
        .await;
    assert!(failed.is_err());
    assert_eq!(
        *host.sleeps.lock(),
        [Duration::from_millis(500) + HALF_JITTER]
    );
}

/// Nothing but a retry policy retries a wrong key on a lone deployment.
#[rstest]
#[case::a_policy_that_names_the_failure(json!({"AuthenticationErrorRetries": 1}), Some(1))]
#[case::no_policy(Value::Null, None)]
#[tokio::test]
async fn a_retry_policy_retries_a_stream_nothing_else_would(
    #[case] retry_policy: Value,
    #[case] retried: Option<u32>,
) {
    let host = SleepingHost::default();
    let config = RouterConfig {
        retry_policy: serde_json::from_value(retry_policy).unwrap(),
        ..lone_groups()
    };
    let router = router(config, json!({"num_retries": 1}), Arc::default());
    let origin = stream_from_solo();
    let answer = router
        .retry_stream(
            &host,
            Op::from("acompletion"),
            asking_for("solo"),
            origin,
            AttemptError::new(FailureKind::Authentication, Some(401)),
        )
        .await
        .map(|response| response.attempted_retries);
    assert_eq!(answer.ok(), retried);
}

/// A hook reroutes `context` to `solo`, whose chain names itself first: the level skips the group
/// that just failed and goes on to the next target.
#[tokio::test]
async fn a_rerouted_level_skips_a_fallback_to_the_group_it_was_routed_to() {
    let host = SleepingHost {
        pre_routes: Some(("context", PreRouted::Model("solo".to_owned()))),
        ..SleepingHost::failing("solo", vec![server_error()])
    };
    let router = router(
        lone_groups(),
        json!({"num_retries": 0, "fallbacks": [{"solo": ["solo", "backup"]}]}),
        Arc::default(),
    );
    let request = RouterRequest {
        pre_routes: true,
        ..asking_for("context")
    };
    let response = router
        .call(&host, Op::from("acompletion"), request)
        .await
        .map(|response| (response.deployment_id, response.attempted_fallbacks));
    assert_eq!(response.ok(), Some(("b1".to_owned(), 1)));
}

/// The draws of a call on `solo` that falls back to `backup`, and of plain calls on each.
async fn draws_falling_back_and_served(failure: AttemptError, num_retries: u32) -> (usize, usize) {
    let dynamic = json!({"num_retries": num_retries, "fallbacks": [{"solo": ["backup"]}]});
    let op = || Op::from("acompletion");

    let falling_back = Arc::new(CountedDraws::default());
    let host = SleepingHost::failing("solo", vec![failure]);
    let fell_back = router(lone_groups(), dynamic.clone(), falling_back.clone())
        .call(&host, op(), asking_for("solo"))
        .await
        .unwrap();
    assert_eq!(
        (fell_back.deployment_id.as_str(), host.sleeps.lock().len()),
        ("b1", 0)
    );

    let served = Arc::new(CountedDraws::default());
    let host = SleepingHost::default();
    let router = router(lone_groups(), dynamic, served.clone());
    router.call(&host, op(), asking_for("solo")).await.unwrap();
    router
        .call(&host, op(), asking_for("backup"))
        .await
        .unwrap();

    (
        falling_back.0.load(Ordering::Relaxed),
        served.0.load(Ordering::Relaxed),
    )
}

/// A level that gives up before any sleep draws no jitter, so the picks after it draw what they
/// would have drawn without the failure.
#[rstest]
#[case::its_budget_grants_no_retry(server_error(), 0)]
#[case::every_deployment_refused_the_endpoint(unsupported_endpoint(), 2)]
#[tokio::test]
async fn a_level_that_gives_up_without_sleeping_draws_only_what_its_picks_draw(
    #[case] failure: AttemptError,
    #[case] num_retries: u32,
) {
    let (falling_back, served) = draws_falling_back_and_served(failure, num_retries).await;
    assert_eq!(falling_back, served);
}
