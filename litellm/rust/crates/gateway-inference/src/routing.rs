//! How a route is served through router-core. [`call`] authorizes the name the request asks for,
//! then runs it through [`litellm_router_core::Router::call`], where the core picks a deployment,
//! retries and falls back, and each attempt runs the request's own [`RouteAttempt`]. Every other
//! hostcall the core makes is answered by the [`Host`] the gateway was built with. The route's
//! typed outcome comes back with the routing facts, and the route renders it once.

use std::{future::Future, sync::Arc, time::Duration};

use axum::{
    http::{HeaderMap, HeaderName, HeaderValue},
    response::Response,
};
use futures_util::{StreamExt, stream::BoxStream};
use litellm_gateway_auth::{AccessRequest, AuthenticatedRequest, Permissions};
use litellm_router_core::{
    Affinity, Attempt, AttemptError, AttemptFailure, CallError, Deployment, FailureKind, Op,
    RouterHost, RouterRequest, RouterResponse, Served, SuccessReport, resolve,
};
use litellm_router_types::Spelled;
use serde_json::{Map, Value};

use crate::{Error, Gateway};

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

    /// The deployments the request's own ids name.
    fn affinity(&self) -> Affinity {
        Affinity::default()
    }

    /// The request params the pre-call checks read.
    fn body(&self) -> Value {
        Value::Null
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

/// A route's failed attempt as router-core reads it to decide retries and cooldowns.
pub fn attempt_failure(error: Error) -> AttemptFailure<Error> {
    let facts = match &error {
        Error::Route(route) => match route.upstream_response() {
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
            None if route.is_request() => {
                Some(AttemptError::new(FailureKind::BadRequest, Some(400)))
            }
            None => Some(AttemptError::new(FailureKind::ApiConnection, None)),
        },
        Error::Ocr(ocr) => ocr
            .upstream_response()
            .map(|(status, body, ..)| AttemptError::from_provider(status, body, None)),
        _ => None,
    };
    match facts {
        Some(facts) => AttemptFailure::Failed {
            facts,
            raised: error,
        },
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
/// retry and fall back.
pub(crate) async fn call<H: Host, A: RouteAttempt>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    op: Op,
    path: &'static str,
    model: &str,
    route: &A,
) -> Result<RouterResponse<A::Outcome>, Error> {
    let request = RouterRequest {
        affinity: route.affinity(),
        body: route.body(),
        ..caller_request(identity, model)
    };
    let first = first_model(gateway, &request).ok_or_else(|| Error::UnknownModel {
        route: path,
        model: model.to_owned(),
    })?;
    authorize(identity, model, first).await?;
    let host = gateway.host.router_host(gateway, identity, route);
    gateway
        .router
        .call(&host, op, request)
        .await
        .map_err(|ended| match ended {
            CallError {
                raised: Some(error),
                ..
            } => error,
            CallError {
                error: litellm_router_core::Error::UnknownModel { model },
                ..
            } => Error::UnknownModel { route: path, model },
            CallError { error, .. } => Error::Routing(error),
        })
}

/// The request as one attempt sends it: the client's params, then the params the fallback chain's
/// dict targets merged in.
pub(crate) fn attempt_body(body: &Map<String, Value>, attempt: &Attempt) -> Map<String, Value> {
    body.iter()
        .chain(attempt.params.iter().flat_map(|params| params.iter()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// The deployment's own `timeout`, else the router's.
pub(crate) fn timeout(attempt: &Attempt) -> Option<Duration> {
    match attempt.deployment.params.timeout {
        Some(Spelled::Value(seconds)) => Duration::try_from_secs_f64(seconds).ok(),
        Some(Spelled::Text(_)) | None => attempt.timeout,
    }
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

/// A stream whose success is reported to the router once it ends without an error: its attempt
/// was served deferred, so it counted its request and no success.
pub(crate) fn reporting_success<H: Host, C: Send + 'static, E: Send + 'static>(
    gateway: &Gateway<H>,
    deployment_id: String,
    chunks: BoxStream<'static, Result<C, E>>,
) -> BoxStream<'static, Result<C, E>> {
    let router = gateway.router.clone();
    let routing = gateway.routing.clone();
    let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let seen = failed.clone();
    let ended = futures_util::stream::once(async move {
        if seen.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        if let Err(error) = router
            .report_success(
                &*routing,
                &deployment_id,
                0.0,
                0.0,
                SuccessReport::default(),
            )
            .await
        {
            tracing::warn!(%deployment_id, %error, "stream success was not recorded");
        }
    })
    .filter_map(|()| async { None::<Result<C, E>> });
    chunks
        .inspect(move |chunk| {
            if chunk.is_err() {
                failed.store(true, std::sync::atomic::Ordering::Release);
            }
        })
        .chain(ended)
        .boxed()
}

/// What the router did to serve a response, as the proxy's `x-litellm-*` headers.
pub(crate) fn headers<R>(routed: &RouterResponse<R>) -> HeaderMap {
    [
        ("x-litellm-model-id", Some(routed.deployment_id.clone())),
        ("x-litellm-model-group", Some(routed.model_group.clone())),
        (
            "x-litellm-attempted-retries",
            Some(routed.attempted_retries.to_string()),
        ),
        (
            "x-litellm-attempted-fallbacks",
            Some(routed.attempted_fallbacks.to_string()),
        ),
        (
            "x-litellm-max-retries",
            routed.max_retries.map(|max| max.to_string()),
        ),
    ]
    .into_iter()
    .filter_map(|(name, value)| {
        Some((
            HeaderName::from_static(name),
            HeaderValue::from_str(&value?).ok()?,
        ))
    })
    .collect()
}

/// `response` with the routing headers of the call that served it.
pub(crate) fn respond(routing: HeaderMap, response: Response) -> Response {
    let (mut parts, body) = response.into_parts();
    parts.headers.extend(routing);
    Response::from_parts(parts, body)
}
