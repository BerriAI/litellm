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
    RouterHost, RouterRequest, RouterResponse, Served, call::StreamOrigin,
    health::should_retry_status, resolve,
};
use litellm_router_types::Spelled;
use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::{
    Error, Gateway,
    error::{Ended, NoDeployments},
    headers::Routing,
    stream_hold::{self, Cause, Chunks, Held, Peek},
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

    /// The stream of a served outcome and how its events read, for a route whose stream another
    /// attempt may answer in place of while it has sent no content. `None` for a complete
    /// answer, and for a route whose streams are never taken over.
    fn held(outcome: &mut Self::Outcome) -> Option<(&mut RouteChunks, Peek)> {
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

/// `route`, remembering the attempt the router ran last, which a stream's recovery starts from.
struct Remembering<'a, A> {
    route: &'a A,
    last: Mutex<Option<Attempt>>,
}

impl<A> Remembering<'_, A> {
    fn last(&self) -> Option<Attempt> {
        self.last.lock().clone()
    }
}

impl<A: RouteAttempt> RouteAttempt for Remembering<'_, A> {
    type Outcome = A::Outcome;

    async fn run(&self, attempt: &Attempt) -> Result<Served<A::Outcome>, Error> {
        *self.last.lock() = Some(attempt.clone());
        self.route.run(attempt).await
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
    let route = Remembering {
        route,
        last: Mutex::new(None),
    };
    let host = gateway.host.router_host(gateway, identity, &route);
    let ended = gateway
        .router
        .call(&host, op.clone(), request.clone())
        .await;
    let routed = answered(gateway, path, ended).await?;
    taken_over_while_held(&gateway.router, &host, &route, &op, &request, routed).await
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

/// `routed`, or the answer of the attempt that took its stream over. A stream is held while it
/// has sent no content and a retry or a fallback could still answer in its place. One the
/// provider fails there is retried in its group, then falls back, and the stream that answers
/// is held the same way (`wrap_anthropic_stream`). A stream no attempt may answer for goes out
/// as the provider sent it, and one whose take-over served nothing ends the call on its failure.
async fn taken_over_while_held<A, RH>(
    router: &Router,
    host: &RH,
    route: &Remembering<'_, A>,
    op: &Op,
    request: &RouterRequest,
    routed: RouterResponse<A::Outcome>,
) -> Result<RouterResponse<A::Outcome>, Ended>
where
    A: RouteAttempt,
    RH: RouterHost<Response = A::Outcome, Failure = Error>,
{
    let mut routed = routed;
    let mut committed_budget = None;
    loop {
        let Some(last) = route.last() else {
            return Ok(routed);
        };
        let Some((chunks, peek)) = A::held(&mut routed.output.response) else {
            return Ok(routed);
        };
        let origin = StreamOrigin {
            group: last.name.clone(),
            order: last.target_order,
            depth: last.fallback_depth,
            level: last.fallback_level,
            deployment_id: Some(last.deployment.id.clone()),
            attempted_retries: last.retry,
            committed_budget,
            routed: last.routed.clone(),
        };
        let policy = router.stream_policy(request, &origin, None);
        if !policy.can_retry && !policy.can_fall_back {
            return Ok(routed);
        }
        let unsent = std::mem::replace(chunks, stream::empty().boxed());
        let (failed, sent) = match stream_hold::hold(unsent, peek, stream_outcome::broke).await {
            Held::Committed(sent) => {
                *chunks = sent;
                return Ok(routed);
            }
            Held::Failed { failed, sent } => (failed, sent),
        };
        let unanswered = match take_over(router, host, op, request, &origin, failed.failure).await {
            TakenOver::By { answer, budget } => {
                routed = answer;
                committed_budget = budget;
                continue;
            }
            TakenOver::ByNone { raised } => Some(raised),
            TakenOver::Declined => None,
        };
        let Some(raised) = unanswered else {
            *chunks = sent;
            return Ok(routed);
        };
        // `_raise_stream_failure`: a take-over that served nothing ends the call on the error a
        // later attempt raised, else on the stream's own, with what the router did for the
        // stream.
        let error = match (raised, failed.cause) {
            (Some(raised), _) => raised,
            (None, Cause::ErrorEvent { status, body }) => {
                Error::Route(litellm_http::Error::upstream(status, body).into())
            }
            (None, Cause::Broke) => sent
                .filter_map(|chunk| future::ready(chunk.err()))
                .next()
                .await
                .map_or(Error::Internal, Error::from),
        };
        return Err(Ended {
            error: Box::new(error),
            routing: Some(Routing::served(&routed)),
        });
    }
}

/// How the take-over of a failed stream went.
enum TakenOver<R> {
    /// Another attempt answered. `budget` is what a retry committed the request's later stream
    /// retries to.
    By {
        answer: RouterResponse<R>,
        budget: Option<u32>,
    },
    /// Attempts ran and none answered. `raised` is the error of the newest one that failed,
    /// `None` when the take-over ends on the stream's own failure.
    ByNone { raised: Option<Error> },
    /// No attempt may answer in the stream's place.
    Declined,
}

/// The take-over of a stream from `origin` that failed with `failure` before any content
/// (`_anthropic_recover_stream_error`): a retry in the stream's group while the failure's retry
/// budget lasts, then the fallbacks. Where the stream is read, its failure was already counted
/// against its deployment.
async fn take_over<RH: RouterHost<Failure = Error>>(
    router: &Router,
    host: &RH,
    op: &Op,
    request: &RouterRequest,
    origin: &StreamOrigin,
    failure: AttemptError,
) -> TakenOver<RH::Response> {
    if !failure.status.is_none_or(should_retry_status) {
        return TakenOver::Declined;
    }
    let policy = router.stream_policy(request, origin, Some(failure.kind));
    let retries = policy
        .retry_budget
        .is_some_and(|(budget, _)| budget > origin.attempted_retries);
    if !retries && !policy.can_fall_back {
        return TakenOver::Declined;
    }
    let retried = router
        .retry_stream(
            host,
            op.clone(),
            request.clone(),
            origin.clone(),
            failure.clone(),
        )
        .await;
    let (newest, raised) = match retried {
        Ok(answer) => {
            let budget = answer.max_retries;
            return TakenOver::By { answer, budget };
        }
        Err(ended) => (
            ended.error.attempt().cloned().unwrap_or(failure),
            ended.raised.map(|raised| raised.failure),
        ),
    };
    let fallen_back = router
        .fallback_stream(host, op.clone(), request.clone(), origin.clone(), newest)
        .await;
    match fallen_back {
        Ok(answer) => TakenOver::By {
            answer,
            budget: None,
        },
        Err(ended) => TakenOver::ByNone {
            raised: ended.raised.map(|raised| raised.failure).or(raised),
        },
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
}
