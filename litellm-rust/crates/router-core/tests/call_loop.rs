//! A call through the router end to end: selection, retries, fallbacks and cooldowns against a
//! scripted host.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use async_trait::async_trait;
use litellm_router_core::{
    Attempt, AttemptError, Deployment, Error, FailureKind, Host, InMemoryRouterCache, Op, Output,
    Payload, Router, RouterCache, RouterConfig, RouterDynamicConfig, RouterEvent, RouterRequest,
    RoutingContext, SeededRandomness, SelectStage,
};
use parking_lot::Mutex;
use rstest::{fixture, rstest};
use serde_json::json;

#[derive(Default)]
struct ScriptedHost {
    /// Failures handed out in order, keyed by model group; a group with none left succeeds.
    script: Mutex<Vec<(String, VecDeque<AttemptError>)>>,
    events: Mutex<Vec<RouterEvent>>,
    cache: InMemoryRouterCache,
}

impl ScriptedHost {
    fn failing(group: &str, failures: Vec<AttemptError>) -> Self {
        Self {
            script: Mutex::new(vec![(group.to_owned(), failures.into())]),
            ..Self::default()
        }
    }
}

#[async_trait]
impl Host for ScriptedHost {
    async fn invoke(&self, attempt: &Attempt) -> Result<Output, AttemptError> {
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
            }),
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

    async fn allow_fallback(&self, _request: &RouterRequest, _target: &str) -> bool {
        true
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

#[rstest]
#[tokio::test]
async fn a_rate_limit_retries_in_the_same_group(config: RouterConfig) {
    let host = Arc::new(ScriptedHost::failing(
        "primary",
        vec![AttemptError::new(FailureKind::RateLimit, Some(429))],
    ));
    let router = router(config, json!({"num_retries": 2}), host.clone());
    let response = router.call(Op::Completion, request()).await.unwrap();
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
    let response = router.call(Op::Completion, request()).await.unwrap();
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
    let error = router.call(Op::Completion, request()).await.unwrap_err();
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
