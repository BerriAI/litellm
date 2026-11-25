//! How a route is served through router-core. [`call`] checks the name the request asks for is
//! one the router knows, has the auth core decide whether the caller's model lists reach it, then
//! runs it through [`litellm_router_core::Router::call`], where the core picks a deployment, retries and falls back, and each attempt runs the request's own
//! [`RouteAttempt`]. Every other hostcall the core makes is answered by the [`Host`] the gateway
//! was built with. The route's typed outcome comes back with the routing facts, and the route
//! renders it once. A served stream is held until it sends content, so one the provider fails
//! before that is answered by a retry or a fallback instead.

use std::{collections::BTreeSet, future::Future, time::Duration};

use futures_util::{StreamExt, stream};
use litellm_config::LitellmRole;
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Rows, StoredKey,
    authz::{self, Denied, Layer, ModelAsk},
    body::Credentials,
};
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
    hooks::CallHooks,
    record,
    stream_event::{Counts, Read},
    stream_hold::{self, Cause, Chunks, Held},
    stream_outcome,
};

/// The chunks of a route's stream.
pub type RouteChunks = Chunks<litellm_host_http::Error<InferenceError>>;

/// The host the routes are served through. It answers router-core's hostcalls and the callback
/// hooks for one call, and the gateway crate attaches the native one.
pub trait Host: Send + Sync + Sized + 'static {
    type RouterHost<'a, A: RouteAttempt + 'a>: RouterHost<Response = A::Outcome, Failure = Error>
    where
        Self: 'a;

    type Hooks: CallHooks;

    /// The rows of the callers this host authenticates.
    type Rows: Rows<Key: StoredKey>;

    /// The model lists this host read for `identity`, which the auth core decides access on. A
    /// list the host did not read belongs to a key it declined when it authenticated it.
    fn layers<'a>(
        &self,
        identity: &'a Authenticated<Self::Rows>,
    ) -> impl Iterator<Item = Layer<'a>>;

    /// The stored credentials a request body names, read through the host's auth store, which
    /// the request body check reads before it decides.
    fn credentials(
        &self,
        names: &BTreeSet<String>,
    ) -> impl Future<Output = Result<Credentials, litellm_gateway_auth_core::Error>> + Send;

    /// The callback hooks of one call `identity` makes, which read and write the call's record.
    fn hooks(&self, identity: &Authenticated<Self::Rows>) -> Self::Hooks;

    /// router-core's host for one call that runs `route` on each attempt for `identity`. The host
    /// counts `route`'s [`RouteAttempt::prompt`] for the pre-call checks.
    fn router_host<'a, A: RouteAttempt>(
        &'a self,
        gateway: &'a Gateway<Self>,
        identity: &'a Authenticated<Self::Rows>,
        route: &'a A,
    ) -> Self::RouterHost<'a, A>;
}

/// What one route does for one attempt: build its call for the picked deployment, run it, and
/// return its typed outcome. [`attempt`] tells the router what that outcome used.
pub trait RouteAttempt: Send + Sync {
    type Outcome: Send;

    fn run(&self, attempt: &Attempt) -> impl Future<Output = Result<Self::Outcome, Error>> + Send;

    /// What a complete outcome counted, which the router counts against the deployment. A
    /// stream's counts are read from its events.
    /// Required, so a new route says what it counts rather than counting nothing by default.
    fn counts(outcome: &Self::Outcome) -> Option<Counts>;

    /// The request's prompt, which the host counts for the pre-call checks. `None` holds the
    /// route's calls to no context window, so each route says so itself.
    fn prompt(&self) -> Option<Prompt<'_>>;

    /// The stream of an outcome and the reader of its format: its outcome is reported to the
    /// router as it is read, and another attempt may answer in its place while it has sent no
    /// content. `None` for a complete answer.
    fn held(outcome: &mut Self::Outcome) -> Option<(&mut RouteChunks, Read)> {
        let _ = outcome;
        None
    }
}

/// The prompt a request carries, in the format its route reads it in. A route only hands it over,
/// and the host decides how it is counted.
#[derive(Clone, Copy)]
pub enum Prompt<'a> {
    /// A Messages body: its `system`, its `messages` and its `tools`.
    Messages(&'a Map<String, Value>),
}

/// One attempt of `route` as router-core hears it. A complete outcome is served with the tokens
/// it used. A stream is served deferred, and reports how it went once that is known.
#[expect(
    clippy::result_large_err,
    reason = "the type router-core's RouterHost::attempt returns"
)]
pub async fn attempt<H: Host, A: RouteAttempt>(
    gateway: &Gateway<H>,
    route: &A,
    attempt: &Attempt,
) -> Result<Served<A::Outcome>, AttemptFailure<Error>> {
    let mut outcome = route
        .run(attempt)
        .await
        .map_err(|error| attempt_failure(attempt, error))?;
    let Some((chunks, read)) = A::held(&mut outcome) else {
        let total_tokens = A::counts(&outcome).map_or(0, Counts::total);
        return Ok(Served::new(outcome, total_tokens));
    };
    let unreported = std::mem::replace(chunks, stream::empty().boxed());
    *chunks = stream_outcome::reporting_outcome(gateway, &attempt.deployment, unreported, read);
    Ok(Served {
        deferred: true,
        ..Served::new(outcome, 0)
    })
}

/// Whether the request's name resolves to a deployment for its caller, default targets included.
fn known<H: Host>(gateway: &Gateway<H>, request: &RouterRequest) -> bool {
    let settings = gateway.router.settings();
    let routable = resolve::routable(
        &gateway.router.registry(),
        &settings.dynamic,
        request,
        settings.enable_cross_model_group_collision_check,
    );
    !routable.is_empty()
}

/// Whether the model lists the host read for `identity` reach `name`, which the auth core
/// decides. The route only says what the router resolves the name to: its model group and the
/// access groups of that group's deployments, as `can_key_call_model` reads them.
fn authorize<H: Host>(
    gateway: &Gateway<H>,
    identity: &Authenticated<H::Rows>,
    name: &str,
) -> Result<(), Denied> {
    let registry = gateway.router.registry();
    let access_groups: Vec<String> = resolve::model_access_groups(&registry, name, None)
        .into_iter()
        .collect();
    let ask = ModelAsk {
        name,
        group: registry.alias_of(name).unwrap_or(name),
        access_groups: &access_groups,
    };
    authz::authorize(gateway.host.layers(identity), &ask)
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

/// Whether `identity` may be served `target`, which a fallback is only taken to when it may. A
/// target the router does not know is not refused here.
pub fn allows<H: Host>(
    gateway: &Gateway<H>,
    identity: &Authenticated<H::Rows>,
    target: &str,
) -> bool {
    let request = caller_request(identity, gateway.host.layers(identity), target);
    !known(gateway, &request) || authorize(gateway, identity, target).is_ok()
}

/// A request for `model` by `identity`: an admin by the role the auth core gives it, never by
/// the kind of credential it presented, with the names its `layers` hold, which the router
/// narrows access groups by as Python's reads the key's `models`.
fn caller_request<'a, R: Rows<Key: StoredKey>>(
    identity: &Authenticated<R>,
    layers: impl Iterator<Item = Layer<'a>>,
    model: &str,
) -> RouterRequest {
    RouterRequest {
        model: model.to_owned(),
        proxy_admin: identity.role() == LitellmRole::ProxyAdmin,
        allowed_models: layers
            .flat_map(|layer| match layer {
                Layer::Key(models) => models.iter().cloned(),
            })
            .collect(),
        ..RouterRequest::default()
    }
}

/// Serve `model` through the router: pick, run `route` on each attempt, retry and fall back. A
/// call that ends on an attempt's error names the deployment the router says raised it.
pub(crate) async fn call<H: Host, A: RouteAttempt>(
    gateway: &Gateway<H>,
    identity: &Authenticated<H::Rows>,
    op: Op,
    path: &'static str,
    model: &str,
    route: &A,
) -> Result<RouterResponse<A::Outcome>, Ended> {
    let request = caller_request(identity, gateway.host.layers(identity), model);
    if !known(gateway, &request) {
        return Err(Error::UnknownModel {
            route: path,
            model: model.to_owned(),
        }
        .into());
    }
    authorize(gateway, identity, model).map_err(Error::from)?;
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
        CallError { error, .. } => Ended {
            routing: error
                .refused()
                .map(|refused| Routing::failed_on(&refused.deployment_id)),
            error: Box::new(Error::Routing(error)),
        },
    })
}

/// `routed`, or the answer of the attempt that took its stream over (`wrap_anthropic_stream`).
/// A stream is held while it has sent no content and the core says a retry or a fallback could
/// still answer in its place, and the stream that answers is held the same way. A stream the
/// provider fails there ends one of three ways:
///
/// - the core lets an attempt take it over: the stream is closed and the take-over runs, and
///   when that serves nothing the call ends on an error
/// - no attempt may answer and the provider sent its error as an event: the stream goes out as
///   the provider sent it
/// - no attempt may answer and the stream broke: the call ends on that failure, since nothing
///   reached the client yet
///
/// A call that ends on an error carries the routing headers of the stream that failed.
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
        if !router.may_take_over(request, &routed.origin) {
            return Ok(routed);
        }
        let unsent = std::mem::replace(chunks, stream::empty().boxed());
        let failed = match stream_hold::hold(unsent, read, stream_hold::HELD).await {
            Held::Committed(sent, first_event) => {
                if let Some(at) = first_event {
                    record::write(|entries| entries.see_first_event(at));
                }
                *chunks = sent;
                return Ok(routed);
            }
            Held::Failed(failed) => failed,
        };
        let take_over = router.take_over_stream(request, &routed.origin, &failure(&failed.cause));
        if take_over.is_none() && matches!(failed.cause, Cause::ErrorEvent { .. }) {
            *chunks = failed.replayed();
            return Ok(routed);
        }
        // Python closes the attempt a take-over supersedes before the take-over runs.
        let own = ended_on(failed.closed());
        let routing = Some(Routing::served(&routed));
        let answered = match take_over {
            Some(take_over) => take_over.run(host, op.clone()).await,
            None => Err(Unanswered { raised: None }),
        };
        match answered {
            Ok(answer) => routed = answer,
            Err(Unanswered { raised }) => {
                return Err(Ended {
                    error: Box::new(raised.unwrap_or(own)),
                    routing,
                });
            }
        }
    }
}

/// How a held stream failed, as router-core reads it to decide what may take the stream over.
fn failure(cause: &Cause<litellm_host_http::Error<InferenceError>>) -> AttemptError {
    match cause {
        Cause::ErrorEvent { status, answer, .. } => {
            AttemptError::from_provider(*status, answer, None)
        }
        Cause::Broke(error) => stream_outcome::broke(error),
    }
}

/// How a held stream failed, as the error its call ends on when no attempt answers in its place
/// and none raised an error of its own (`_raise_stream_failure`).
fn ended_on(cause: Cause<litellm_host_http::Error<InferenceError>>) -> Error {
    match cause {
        Cause::ErrorEvent { status, answer, .. } => {
            Error::Route(litellm_http::Error::upstream(status, answer).into())
        }
        Cause::Broke(error) => error.into(),
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
/// `stream_timeout` on the deployment, then the deployment's `timeout` and `request_timeout`, then
/// the router's, then `configured`, the process `request_timeout`. As the Python router does, a
/// `configured` timeout stands in for the router's own, and a stream reads it right after the
/// deployment's `stream_timeout`.
pub(crate) fn timeout(
    attempt: &Attempt,
    streamed: bool,
    configured: Option<Duration>,
) -> Option<Duration> {
    let seconds =
        |written: &Option<Spelled<f64>>| Duration::try_from_secs_f64(written.as_ref()?.lax()?).ok();
    let params = &attempt.deployment.params;
    let request_timeout = || {
        let seconds = params.extra.get("request_timeout")?.as_f64()?;
        Duration::try_from_secs_f64(seconds).ok()
    };
    streamed
        .then(|| seconds(&params.stream_timeout).or(attempt.timeout.and(configured)))
        .flatten()
        .or_else(|| seconds(&params.timeout))
        .or_else(request_timeout)
        .or(configured.or(attempt.timeout))
}

pub(crate) fn api_key(deployment: &Deployment) -> Option<String> {
    deployment
        .params
        .api_key
        .as_ref()
        .map(|key| key.expose().to_owned())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_gateway_auth_core::Caller;
    use litellm_router_core::DeploymentSpec;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::test_support::{Held, key};

    #[rstest]
    #[case::the_master_key(Caller::MasterKey, true)]
    #[case::a_key_its_row_calls_an_admin(key("stored", true), true)]
    #[case::any_other_key(key("stored", false), false)]
    fn only_a_caller_with_the_admin_role_is_routed_as_a_proxy_admin(
        #[case] caller: Caller<Held>,
        #[case] proxy_admin: bool,
    ) {
        let request = caller_request(&Authenticated::from(caller), [].into_iter(), "public/model");

        assert_eq!(request.proxy_admin, proxy_admin);
    }

    #[rstest]
    fn the_router_narrows_access_groups_by_the_names_the_key_layer_holds() {
        let models = ["claude".to_owned(), "fast".to_owned()];
        let caller = Authenticated::from(key("stored", false));

        let request = caller_request(&caller, [Layer::Key(&models)].into_iter(), "claude");

        assert_eq!(request.allowed_models, models);
    }

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
    #[case::the_deployments_request_timeout_after_its_timeout(
        json!({"model": "a/b", "timeout": 30, "request_timeout": 20}), Some(600), false, Some(30)
    )]
    #[case::the_deployments_request_timeout_before_the_routers(
        json!({"model": "a/b", "request_timeout": 20}), Some(600), true, Some(20)
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
            timeout(&attempt(params, router_timeout), streamed, None),
            expected.map(Duration::from_secs)
        );
    }

    #[rstest]
    #[case::the_configured_timeout_without_a_routers(json!({"model": "a/b"}), None, false, Some(2))]
    #[case::the_configured_timeout_in_place_of_the_routers(
        json!({"model": "a/b"}), Some(600), false, Some(2)
    )]
    #[case::the_deployments_timeout_before_the_configured(
        json!({"model": "a/b", "timeout": 30}), Some(600), false, Some(30)
    )]
    #[case::a_stream_reads_the_configured_timeout_before_the_deployments_when_the_router_has_one(
        json!({"model": "a/b", "timeout": 30}), Some(600), true, Some(2)
    )]
    #[case::a_stream_reads_the_deployments_timeout_first_when_the_router_has_none(
        json!({"model": "a/b", "timeout": 30}), None, true, Some(30)
    )]
    #[case::a_stream_reads_its_own_stream_timeout_first(
        json!({"model": "a/b", "stream_timeout": 5}), Some(600), true, Some(5)
    )]
    fn a_configured_request_timeout_is_read_where_python_reads_it(
        #[case] params: Value,
        #[case] router_timeout: Option<u64>,
        #[case] streamed: bool,
        #[case] expected: Option<u64>,
    ) {
        assert_eq!(
            timeout(
                &attempt(params, router_timeout),
                streamed,
                Some(Duration::from_secs(2))
            ),
            expected.map(Duration::from_secs)
        );
    }

    const START: &str =
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n";
    const OVERLOADED: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n";

    /// A route whose every stream fails before content, with an overloaded error or by breaking.
    /// Each run notes how many of the streams served before it were still open.
    #[derive(Default)]
    struct Failing {
        breaks: bool,
        open: Arc<()>,
        open_at_each_run: parking_lot::Mutex<Vec<usize>>,
    }

    impl RouteAttempt for Failing {
        type Outcome = RouteChunks;

        async fn run(&self, _attempt: &Attempt) -> Result<RouteChunks, Error> {
            self.open_at_each_run
                .lock()
                .push(Arc::strong_count(&self.open) - 1);
            let open = self.open.clone();
            let failure = if self.breaks {
                Err(litellm_host_http::Error::Protocol)
            } else {
                Ok(OVERLOADED)
            };
            let sent = stream::iter([Ok(START), failure]).map(move |event| {
                let _open = &open;
                event.map(|event| bytes::Bytes::from_static(event.as_bytes()))
            });
            Ok(litellm_framer::sse::blocks(sent).boxed())
        }

        /// Only ever a stream, whose counts are read from its events.
        fn counts(_outcome: &RouteChunks) -> Option<Counts> {
            None
        }

        fn prompt(&self) -> Option<Prompt<'_>> {
            None
        }

        fn held(outcome: &mut RouteChunks) -> Option<(&mut RouteChunks, Read)> {
            Some((outcome, crate::messages::read_event))
        }
    }

    struct RouteHost<'a> {
        route: &'a Failing,
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
            let sent = self
                .route
                .run(attempt)
                .await
                .map_err(|error| attempt_failure(attempt, error))?;
            Ok(Served {
                deferred: true,
                ..Served::new(sent, 0)
            })
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

    /// A router of one deployment of `primary`, under `retry_policy` and `num_retries`.
    fn lone_primary(retry_policy: Value, num_retries: u32) -> Router {
        let config = json!({
            "model_list": [{
                "model_name": "primary",
                "litellm_params": {"model": "anthropic/test-model"},
                "model_info": {"id": "d1"},
            }],
            "retry_policy": retry_policy,
        });
        Router::builder(
            serde_json::from_value(config).unwrap(),
            serde_json::from_value(json!({"num_retries": num_retries})).unwrap(),
        )
        .build()
        .unwrap()
    }

    /// How a stream of `route` that `router` served for `primary` was answered once it failed
    /// before content.
    async fn answered_after_failing(
        router: &Router,
        route: &Failing,
    ) -> Result<RouterResponse<RouteChunks>, Ended> {
        let host = RouteHost {
            route,
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
        let held = taken_over_while_held::<Failing, _>(router, &host, &op, &request, first);
        record::scope(record::CallRecord::new(), held).await
    }

    /// Python closes the attempt a take-over supersedes before the take-over runs, so a retry
    /// never runs while the stream it replaces still holds its connection.
    #[tokio::test]
    async fn a_failed_stream_is_closed_before_the_attempt_that_takes_it_over_runs() {
        let router = lone_primary(json!(null), 2);
        let route = Failing::default();

        let last = answered_after_failing(&router, &route).await.unwrap();

        assert_eq!(
            (
                last.attempted_retries,
                route.open_at_each_run.lock().as_slice()
            ),
            (2, [0, 0, 0].as_slice())
        );
    }

    /// The stream is held because a rate limit could still be retried, and a break is not one.
    /// Nothing reached the client yet, so it hears the failure and not a stream that ends early.
    #[tokio::test]
    async fn a_stream_that_breaks_before_content_ends_the_call_when_no_attempt_may_answer() {
        let router = lone_primary(json!({"RateLimitErrorRetries": 2}), 0);
        let route = Failing {
            breaks: true,
            ..Failing::default()
        };

        let ended = answered_after_failing(&router, &route).await;

        assert_eq!(
            (
                ended.err().map(|ended| ended.error.status()),
                route.open_at_each_run.lock().as_slice()
            ),
            (
                Some(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
                [0].as_slice()
            )
        );
    }
}
