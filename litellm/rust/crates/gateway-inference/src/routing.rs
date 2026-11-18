//! How a route is served through router-core. [`call`] authorizes the name the request asks for,
//! then runs it through [`litellm_router_core::Router::call`], where the core picks a deployment,
//! retries and falls back, and each attempt runs the request's own [`RouteAttempt`]. Every other
//! hostcall the core makes is answered by the [`Host`] the gateway was built with. The route's
//! typed outcome comes back with the routing facts, and the route renders it once. A served
//! stream is held until it sends content, so one the provider fails before that is answered by
//! a retry or a fallback instead.

use std::{future::Future, time::Duration};

use futures_util::{StreamExt, future, stream};
use litellm_gateway_auth::{AccessRequest, AuthenticatedRequest, Permissions};
use litellm_inference::Error as InferenceError;
use litellm_router_core::{
    Attempt, AttemptError, AttemptFailure, CallError, Deployment, FailureKind, Op, Router,
    RouterHost, RouterRequest, RouterResponse, Served, call::Unanswered, resolve,
};
use litellm_router_types::Spelled;
use serde_json::{Map, Value};

use crate::{
    Error, Gateway,
    error::{Ended, NoDeployments},
    headers::Routing,
    stream_event::Read,
    stream_hold::{self, Cause, Chunks, Held},
    stream_outcome,
};

/// The chunks of a route's stream.
pub type RouteChunks = Chunks<litellm_host_http::Error<InferenceError>>;

/// The host the routes are served through. It answers router-core's hostcalls for one call, and
/// the gateway crate attaches the native one.
pub trait Host: Send + Sync + Sized + 'static {
    type RouterHost<'a, A: RouteAttempt + 'a>: RouterHost<Response = A::Outcome, Failure = Error>
    where
        Self: 'a;

    /// router-core's host for one call that runs `route` on each attempt for `identity`.
    fn router_host<'a, A: RouteAttempt>(
        &'a self,
        gateway: &'a Gateway<Self>,
        identity: &'a AuthenticatedRequest,
        route: &'a A,
    ) -> Self::RouterHost<'a, A>;
}

/// What one route does for one attempt: build its call for the picked deployment, run it, and
/// return its typed outcome with the tokens it used.
pub trait RouteAttempt: Send + Sync {
    type Outcome: Send;

    fn run(
        &self,
        attempt: &Attempt,
    ) -> impl Future<Output = Result<Served<Self::Outcome>, Error>> + Send;

    /// The stream of a served outcome and the reader of its format, for a route whose stream
    /// another attempt may answer in place of while it has sent no content. `None` for a
    /// complete answer, and for a route whose streams are never taken over.
    fn held(outcome: &mut Self::Outcome) -> Option<(&mut RouteChunks, Read)> {
        let _ = outcome;
        None
    }
}

/// The litellm model of the first deployment the request's name resolves to for its caller,
/// default targets included, which main authorizes the name against.
fn first_model<H: Host>(gateway: &Gateway<H>, request: &RouterRequest) -> Option<String> {
    let routable = resolve::routable(
        &gateway.router.registry(),
        &gateway.router.settings().dynamic,
        request,
    );
    Some(routable.first()?.params.model.clone())
}

async fn authorize(
    identity: &AuthenticatedRequest,
    name: &str,
    model: String,
) -> Result<(), Error> {
    let access = AccessRequest::Model {
        name: name.to_owned(),
        deployment: model,
    };
    identity
        .authorize(access.clone())
        .await?
        .consume(identity.caller(), &access)?;
    Ok(())
}

fn retry_after(headers: &[(String, String)]) -> Option<Duration> {
    let (_, seconds) = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))?;
    Duration::try_from_secs_f64(seconds.trim().parse().ok()?).ok()
}

/// A call the provider never answered. The route reports it as a 500, and Python's router reads
/// it as an `InternalServerError`: retried, and counted toward the deployment's cooldown.
fn unanswered() -> AttemptError {
    AttemptError::new(FailureKind::InternalServer, Some(500))
}

/// What a route's failure says about the deployment it called, `None` for one that says
/// nothing about it. `unanswered` is what a call that got no answer says where it happened.
pub(crate) fn route_failure(
    route: &InferenceError,
    unanswered: fn() -> AttemptError,
) -> Option<AttemptError> {
    match route.upstream_response() {
        Some((status, body)) => Some(AttemptError::from_provider(
            status,
            body,
            retry_after(route.upstream_headers()),
        )),
        None if route.timeout().is_some() => {
            Some(AttemptError::new(FailureKind::Timeout, Some(408)))
        }
        None if route.is_missing_credential() => {
            Some(AttemptError::new(FailureKind::Authentication, Some(401)))
        }
        None if route.declined().is_some() => None,
        None if route.is_request() => Some(AttemptError::new(FailureKind::BadRequest, Some(400))),
        None => Some(unanswered()),
    }
}

/// A route's failed `attempt` as router-core reads it to decide retries and cooldowns. The
/// failure carries its deployment's own `num_retries`, as Python's exception does, which is the
/// retry budget when the request names none.
pub fn attempt_failure(attempt: &Attempt, error: Error) -> AttemptFailure<Error> {
    let facts = match &error {
        Error::Route(route) => route_failure(route, unanswered),
        Error::Ocr(ocr) => ocr
            .upstream_response()
            .map(|(status, body, ..)| AttemptError::from_provider(status, body, None)),
        _ => None,
    };
    match facts {
        Some(mut facts) => {
            facts.num_retries = attempt.deployment.num_retries();
            AttemptFailure::Failed {
                facts,
                raised: error,
            }
        }
        None => AttemptFailure::Refused(error),
    }
}

/// Whether `identity` may be served `target`, which a fallback is only taken to when it may.
pub async fn allows<H: Host>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    target: &str,
) -> bool {
    match first_model(gateway, &caller_request(identity, target)) {
        Some(model) => authorize(identity, target, model).await.is_ok(),
        None => true,
    }
}

/// The model names a caller's grants name, which the core narrows access groups by.
fn allowed_models(identity: &AuthenticatedRequest) -> Vec<String> {
    match identity.caller().permissions() {
        Permissions::Only(grants) => grants
            .iter()
            .filter_map(|grant| match grant {
                AccessRequest::Model { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect(),
        Permissions::All | Permissions::None => Vec::new(),
    }
}

/// A request for `model` carrying what the caller's grants say.
fn caller_request(identity: &AuthenticatedRequest, model: &str) -> RouterRequest {
    RouterRequest {
        model: model.to_owned(),
        proxy_admin: matches!(identity.caller().permissions(), Permissions::All),
        allowed_models: allowed_models(identity),
        ..RouterRequest::default()
    }
}

/// Authorize `model`, then serve it through the router: pick, run `route` on each attempt,
/// retry and fall back. A call that ends on an attempt's error names the deployment the router
/// says raised it.
pub(crate) async fn call<H: Host, A: RouteAttempt>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    op: Op,
    path: &'static str,
    model: &str,
    route: &A,
) -> Result<RouterResponse<A::Outcome>, Ended> {
    let request = caller_request(identity, model);
    let first = first_model(gateway, &request).ok_or_else(|| Error::UnknownModel {
        route: path,
        model: model.to_owned(),
    })?;
    authorize(identity, model, first).await?;
    let host = gateway.host.router_host(gateway, identity, route);
    let ended = gateway
        .router
        .call(&host, op.clone(), request.clone())
        .await;
    let routed = answered(gateway, path, ended).await?;
    taken_over_while_held::<A, _>(&gateway.router, &host, &op, &request, routed).await
}

/// How a call the router ran ended, as its route answers it.
async fn answered<H: Host, O>(
    gateway: &Gateway<H>,
    path: &'static str,
    ended: Result<RouterResponse<O>, Box<CallError<Error>>>,
) -> Result<RouterResponse<O>, Ended> {
    let ended = match ended {
        Ok(routed) => return Ok(routed),
        Err(ended) => ended,
    };
    Err(match *ended {
        CallError {
            raised: Some(raised),
            ..
        } => Ended {
            error: Box::new(raised.failure),
            routing: raised.deployment_id.as_deref().map(Routing::failed_on),
        },
        CallError {
            error: litellm_router_core::Error::UnknownModel { model },
            ..
        } => Error::UnknownModel { route: path, model }.into(),
        CallError {
            error: litellm_router_core::Error::NoHealthyDeployments { model, .. },
            ..
        } => no_deployments(gateway, model).await.into(),
        CallError { error, .. } => Error::Routing(error).into(),
    })
}

/// `routed`, or the answer of the attempt that took its stream over (`wrap_anthropic_stream`).
/// A stream is held while it has sent no content and the core says a retry or a fallback could
/// still answer in its place. When the provider fails it there, the core decides whether an
/// attempt may take it over and runs the take-over, and the stream that answers is held the
/// same way. A stream no attempt may answer for goes out as the provider sent it, and one
/// whose take-over served nothing ends the call on an error, with the routing headers of the
/// stream that failed.
async fn taken_over_while_held<A, RH>(
    router: &Router,
    host: &RH,
    op: &Op,
    request: &RouterRequest,
    routed: RouterResponse<A::Outcome>,
) -> Result<RouterResponse<A::Outcome>, Ended>
where
    A: RouteAttempt,
    RH: RouterHost<Response = A::Outcome, Failure = Error>,
{
    let mut routed = routed;
    loop {
        let Some((chunks, read)) = A::held(&mut routed.output.response) else {
            return Ok(routed);
        };
        let policy = router.stream_policy(request, &routed.origin, None);
        if !policy.can_retry && !policy.can_fall_back {
            return Ok(routed);
        }
        let unsent = std::mem::replace(chunks, stream::empty().boxed());
        let (failed, sent) = match stream_hold::hold(unsent, read, stream_outcome::broke).await {
            Held::Committed(sent) => {
                *chunks = sent;
                return Ok(routed);
            }
            Held::Failed { failed, sent } => (failed, sent),
        };
        let Some(take_over) = router.take_over_stream(request, &routed.origin, &failed.failure)
        else {
            *chunks = sent;
            return Ok(routed);
        };
        // The failed stream is closed before another attempt runs, as Python closes the attempt
        // a take-over supersedes.
        let own = own_error(failed.cause, sent).await;
        match take_over.run(host, op.clone()).await {
            Ok(answer) => routed = answer,
            Err(Unanswered { raised }) => {
                return Err(Ended {
                    error: Box::new(raised.unwrap_or(own)),
                    routing: Some(Routing::served(&routed)),
                });
            }
        }
    }
}

/// The failure of a stream that failed before content, as the error its call ends on when the
/// take-over serves nothing and no later attempt raised one (`_raise_stream_failure`). Reading
/// it out of `sent` closes the stream.
async fn own_error(cause: Cause, sent: RouteChunks) -> Error {
    match cause {
        Cause::ErrorEvent { status, body } => {
            Error::Route(litellm_http::Error::upstream(status, body).into())
        }
        Cause::Broke => sent
            .filter_map(|chunk| future::ready(chunk.err()))
            .next()
            .await
            .map_or(Error::Internal, Error::from),
    }
}

/// `async_raise_no_deployment_exception`: when `model`'s deployments can be tried again, and
/// every deployment that is cooling down.
async fn no_deployments<H: Host>(gateway: &Gateway<H>, model: String) -> NoDeployments {
    let registry = gateway.router.registry();
    let settings = gateway.router.settings();
    let health = gateway.router.health(gateway.routing());
    let group: Vec<String> = registry
        .group(&model)
        .unwrap_or_default()
        .iter()
        .map(|deployment| deployment.id.clone())
        .collect();
    let cooldown = health
        .active_cooldowns(&group)
        .await
        .unwrap_or_default()
        .iter()
        .map(|(_, cooldown)| cooldown.cooldown_time)
        .min_by(f64::total_cmp)
        .unwrap_or(settings.cooldown_time.as_secs_f64());
    let cooling: Vec<String> = health
        .active_cooldowns(&registry.ids())
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    NoDeployments {
        all_cooling: !group.is_empty() && group.iter().all(|id| cooling.contains(id)),
        model,
        cooldown,
        pre_call_checks: settings.enable_pre_call_checks,
        cooling,
    }
}

/// The request as one attempt sends it: the client's params, then the params the fallback chain's
/// dict targets merged in.
pub(crate) fn attempt_body(body: &Map<String, Value>, attempt: &Attempt) -> Map<String, Value> {
    body.iter()
        .chain(attempt.params.iter().flat_map(|params| params.iter()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// Whether the client asked for a stream.
pub(crate) fn streamed(body: &Map<String, Value>) -> bool {
    body.get("stream").and_then(Value::as_bool) == Some(true)
}

/// `resolve_llm_passthrough_timeout` over the settings the gateway reads: a streamed call's
/// `stream_timeout` on the deployment, then the deployment's `timeout`, then the router's.
pub(crate) fn timeout(attempt: &Attempt, streamed: bool) -> Option<Duration> {
    let seconds = |written: &Option<Spelled<f64>>| {
        let seconds = match written.as_ref()? {
            Spelled::Value(seconds) => *seconds,
            Spelled::Text(text) => text.trim().parse().ok()?,
        };
        Duration::try_from_secs_f64(seconds).ok()
    };
    let params = &attempt.deployment.params;
    streamed
        .then(|| seconds(&params.stream_timeout))
        .flatten()
        .or_else(|| seconds(&params.timeout))
        .or(attempt.timeout)
}

pub(crate) fn api_key(deployment: &Deployment) -> Option<String> {
    deployment
        .params
        .api_key
        .as_ref()
        .map(|key| key.expose().to_owned())
}

/// The tokens a response's `usage` counts: its `total_tokens`, else its input plus output.
pub(crate) fn total_tokens(usage: Option<&Value>) -> u64 {
    let count = |name: &str| {
        usage
            .and_then(|usage| usage.get(name))
            .and_then(Value::as_u64)
    };
    count("total_tokens").unwrap_or_else(|| {
        count("input_tokens")
            .or(count("prompt_tokens"))
            .unwrap_or(0)
            + count("output_tokens")
                .or(count("completion_tokens"))
                .unwrap_or(0)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_router_core::DeploymentSpec;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn attempt(params: Value, router_timeout: Option<u64>) -> Attempt {
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "primary", "litellm_params": params, "model_info": {"id": "d1"},
        }))
        .unwrap();
        Attempt {
            op: Op::ANTHROPIC_MESSAGES,
            request: Arc::default(),
            deployment: Arc::new(Deployment::from_spec(spec).unwrap()),
            name: "primary".to_owned(),
            model_group: "primary".to_owned(),
            routed: None,
            target_order: None,
            candidate_ids: vec!["d1".to_owned()],
            retry: 0,
            fallback_depth: 0,
            fallback_level: 0,
            timeout: router_timeout.map(Duration::from_secs),
            params: None,
            scope: 0,
            max_retries: 0,
            model_group_size: 1,
        }
    }

    #[rstest]
    #[case::stream_timeout_wins_for_a_stream(
        json!({"model": "a/b", "timeout": 30, "stream_timeout": 5}), Some(600), true, Some(5)
    )]
    #[case::stream_timeout_is_not_read_for_a_unary_call(
        json!({"model": "a/b", "timeout": 30, "stream_timeout": 5}), Some(600), false, Some(30)
    )]
    #[case::a_stream_without_its_own_timeout_takes_the_deployments(
        json!({"model": "a/b", "timeout": 30}), Some(600), true, Some(30)
    )]
    #[case::the_router_timeout_is_last(json!({"model": "a/b"}), Some(600), true, Some(600))]
    #[case::a_timeout_written_as_text_is_read(
        json!({"model": "a/b", "timeout": "45"}), Some(600), false, Some(45)
    )]
    #[case::unset_everywhere(json!({"model": "a/b"}), None, false, None)]
    fn an_attempt_times_out_by_pythons_precedence(
        #[case] params: Value,
        #[case] router_timeout: Option<u64>,
        #[case] streamed: bool,
        #[case] expected: Option<u64>,
    ) {
        assert_eq!(
            timeout(&attempt(params, router_timeout), streamed),
            expected.map(Duration::from_secs)
        );
    }

    const START: &str =
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n";
    const OVERLOADED: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n";

    /// A route whose every stream fails with an overloaded error before content. Each run notes
    /// how many of the streams served before it were still open.
    #[derive(Default)]
    struct Overloaded {
        open: Arc<()>,
        open_at_each_run: parking_lot::Mutex<Vec<usize>>,
    }

    impl RouteAttempt for Overloaded {
        type Outcome = RouteChunks;

        async fn run(&self, _attempt: &Attempt) -> Result<Served<RouteChunks>, Error> {
            self.open_at_each_run
                .lock()
                .push(Arc::strong_count(&self.open) - 1);
            let open = self.open.clone();
            let sent = stream::iter([START, OVERLOADED]).map(move |event| {
                let _open = &open;
                Ok(bytes::Bytes::from_static(event.as_bytes()))
            });
            Ok(Served {
                deferred: true,
                ..Served::new(litellm_framer::sse::blocks(sent).boxed(), 0)
            })
        }

        fn held(outcome: &mut RouteChunks) -> Option<(&mut RouteChunks, Read)> {
            Some((outcome, crate::stream_event::messages_event))
        }
    }

    struct RouteHost<'a> {
        route: &'a Overloaded,
        cache: litellm_router_core::InMemoryRouterCache,
    }

    impl RouterHost for RouteHost<'_> {
        type Response = RouteChunks;
        type Failure = Error;
        type Cache = litellm_router_core::InMemoryRouterCache;

        async fn attempt(
            &self,
            attempt: &Attempt,
        ) -> Result<Served<RouteChunks>, AttemptFailure<Error>> {
            self.route
                .run(attempt)
                .await
                .map_err(|error| attempt_failure(attempt, error))
        }

        async fn select(
            &self,
            _stage: &litellm_router_core::SelectStage,
            _ctx: &litellm_router_core::RoutingContext<'_>,
            candidates: Vec<Arc<Deployment>>,
        ) -> Result<Vec<Arc<Deployment>>, litellm_router_core::Error> {
            Ok(candidates)
        }

        async fn allow_fallback(&self, _target: &str) -> Result<bool, AttemptError> {
            Ok(true)
        }

        fn cache(&self) -> &Self::Cache {
            &self.cache
        }

        async fn sleep(&self, _duration: Duration) {}

        fn observe(&self, _event: &litellm_router_core::RouterEvent) {}
    }

    /// Python closes the attempt a take-over supersedes before the take-over runs, so a retry
    /// never runs while the stream it replaces still holds its connection.
    #[tokio::test]
    async fn a_failed_stream_is_closed_before_the_attempt_that_takes_it_over_runs() {
        let router = Router::builder(
            serde_json::from_value(json!({"model_list": [{
                "model_name": "primary",
                "litellm_params": {"model": "anthropic/test-model"},
                "model_info": {"id": "d1"},
            }]}))
            .unwrap(),
            serde_json::from_value(json!({"num_retries": 2})).unwrap(),
            litellm_router_core::HostFacts::everything(),
        )
        .build()
        .unwrap();
        let route = Overloaded::default();
        let host = RouteHost {
            route: &route,
            cache: litellm_router_core::InMemoryRouterCache::default(),
        };
        let request = RouterRequest {
            model: "primary".to_owned(),
            ..RouterRequest::default()
        };
        let op = Op::ANTHROPIC_MESSAGES;
        let first = router
            .call(&host, op.clone(), request.clone())
            .await
            .unwrap();

        let last = taken_over_while_held::<Overloaded, _>(&router, &host, &op, &request, first)
            .await
            .unwrap();

        assert_eq!(
            (
                last.attempted_retries,
                route.open_at_each_run.lock().as_slice()
            ),
            (2, [0, 0, 0].as_slice())
        );
    }
}
