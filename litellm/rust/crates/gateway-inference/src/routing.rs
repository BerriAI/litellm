//! How a route is served through router-core. [`call`] hands the router the request's
//! [`RouteCall`], the router picks a deployment, retries and falls back, and [`GatewayHost`] runs
//! the route on each attempt it makes. The response or error the router ended on is rendered once.

use std::{future::Future, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    http::{HeaderName, HeaderValue},
    response::Response,
};
use futures_util::{FutureExt, future::BoxFuture};
use litellm_gateway_auth::{AccessRequest, AuthenticatedRequest};
use litellm_inference_messages::MessagesShaping;
use litellm_router_core::{
    Attempt, AttemptError, ConfigError, Deployment, FailureKind, Host, InMemoryRouterCache, Op,
    Payload, RouterCache, RouterResponse, Unsupported,
    host::{Output, RouterEvent, RouterRequest, RoutingContext, SelectStage},
    resolve,
};
use litellm_router_types::Spelled;
use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::{Error, Gateway};

/// The host the gateway's router runs on. Only simple-shuffle routing is native, so it selects
/// nothing itself, and routing state lives in this process.
#[derive(Default)]
pub(crate) struct GatewayHost {
    cache: Arc<InMemoryRouterCache>,
}

#[async_trait]
impl Host for GatewayHost {
    async fn invoke(&self, attempt: &Attempt) -> Result<Output, AttemptError> {
        let Some(call) = RouteCall::of(&attempt.request) else {
            return Err(AttemptError::new(FailureKind::Other, None));
        };
        call.run(attempt).await
    }

    async fn select(
        &self,
        _stage: &SelectStage,
        _ctx: &RoutingContext<'_>,
        _candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, litellm_router_core::Error> {
        Err(ConfigError::Unsupported(Unsupported::Setting("a host selection stage")).into())
    }

    async fn allow_fallback(
        &self,
        request: &RouterRequest,
        target: &str,
    ) -> Result<bool, AttemptError> {
        Ok(match RouteCall::of(request) {
            Some(call) => call.allows(target).await,
            None => false,
        })
    }

    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }

    fn observe(&self, event: &RouterEvent) {
        tracing::debug!(target: "litellm::router", ?event, "routing event");
    }

    fn cache(&self) -> Arc<dyn RouterCache> {
        self.cache.clone()
    }

    fn can_select(&self) -> bool {
        false
    }
}

/// The deployment one attempt runs on, with the request as that attempt sends it.
pub(crate) struct Target {
    pub deployment: Arc<Deployment>,
    pub body: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

impl Target {
    fn of(gateway: &Gateway, attempt: &Attempt, body: &Map<String, Value>) -> Self {
        let params = attempt.params.iter().flat_map(|params| params.iter());
        Self {
            deployment: attempt.deployment.clone(),
            body: body
                .iter()
                .chain(params)
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            timeout: match attempt.deployment.params.timeout {
                Some(Spelled::Value(seconds)) => Duration::try_from_secs_f64(seconds).ok(),
                Some(Spelled::Text(_)) | None => attempt.timeout,
            },
            shaping: gateway.shaping(&attempt.deployment.id),
        }
    }

    pub fn api_key(&self) -> Option<String> {
        self.deployment
            .params
            .api_key
            .as_ref()
            .map(|key| key.expose().to_owned())
    }
}

type Run = Box<dyn Fn(Target) -> BoxFuture<'static, Result<Response, Error>> + Send + Sync>;

/// One request served through the router: the route each attempt runs, and what each attempt
/// returned, which the router names by index once it ends.
struct RouteCall {
    gateway: Arc<Gateway>,
    identity: AuthenticatedRequest,
    body: Map<String, Value>,
    route: Run,
    served: Mutex<Option<Response>>,
    failures: Mutex<Vec<Error>>,
}

impl RouteCall {
    fn of(request: &RouterRequest) -> Option<&Self> {
        request.payload.as_ref()?.0.downcast_ref()
    }

    async fn run(&self, attempt: &Attempt) -> Result<Output, AttemptError> {
        match (self.route)(Target::of(&self.gateway, attempt, &self.body)).await {
            Ok(response) => {
                *self.served.lock() = Some(response);
                Ok(Output {
                    value: Payload(Arc::new(())),
                    total_tokens: 0,
                    blocked: None,
                    deferred: false,
                    tracks_usage: None,
                })
            }
            Err(error) => {
                let mut failures = self.failures.lock();
                let failure = AttemptError {
                    payload: Some(Payload(Arc::new(failures.len()))),
                    ..attempt_error(&error)
                };
                failures.push(error);
                Err(failure)
            }
        }
    }

    /// Whether the caller may be served `target`, which a fallback is only taken to when it may.
    async fn allows(&self, target: &str) -> bool {
        match first_model(&self.gateway, target) {
            Some(model) => authorize(&self.identity, target, model).await.is_ok(),
            None => true,
        }
    }

    /// The error a failed attempt returned, by the index its [`AttemptError`] carries.
    fn failure(&self, attempt: &AttemptError) -> Option<Error> {
        let index = *attempt.payload.as_ref()?.0.downcast_ref::<usize>()?;
        std::mem::take(&mut *self.failures.lock())
            .into_iter()
            .nth(index)
    }
}

/// The litellm model of the first deployment `name` resolves to, which the caller is authorized
/// against.
fn first_model(gateway: &Gateway, name: &str) -> Option<String> {
    let request = RouterRequest {
        model: name.to_owned(),
        ..RouterRequest::default()
    };
    let routable = resolve::routable(
        &gateway.router.registry(),
        &gateway.router.settings().dynamic,
        &request,
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

/// A route's failure as the facts router-core decides retries, fallbacks and cooldowns by. A
/// failure the gateway raised before reaching the provider is a bad request: never retried, never
/// a fallback, never a cooldown.
fn attempt_error(error: &Error) -> AttemptError {
    let declined = || AttemptError {
        classes: vec!["BadRequestError".into()],
        ..AttemptError::new(FailureKind::Declined, Some(400))
    };
    match error {
        Error::Route(route) => match route.upstream_response() {
            Some((status, body)) => {
                AttemptError::from_provider(status, body, retry_after(route.upstream_headers()))
            }
            None if route.timeout().is_some() => AttemptError::new(FailureKind::Timeout, Some(408)),
            None if route.is_missing_credential() => {
                AttemptError::new(FailureKind::Authentication, Some(401))
            }
            None if route.declined().is_some() || route.is_request() => declined(),
            None => AttemptError::new(FailureKind::ApiConnection, None),
        },
        Error::Ocr(ocr) => match ocr.upstream_response() {
            Some((status, body, ..)) => AttemptError::from_provider(status, body, None),
            None => declined(),
        },
        _ => declined(),
    }
}

/// Serve `model` through the router: each attempt runs `route` on the deployment the router
/// picked, and the router retries and falls back as `router_settings` say.
pub(crate) async fn call<F, R>(
    gateway: Arc<Gateway>,
    identity: &AuthenticatedRequest,
    op: &'static str,
    path: &'static str,
    body: Map<String, Value>,
    route: F,
) -> Result<Response, Error>
where
    F: Fn(Target) -> R + Send + Sync + 'static,
    R: Future<Output = Result<Response, Error>> + Send + 'static,
{
    let model = crate::request::model(&body)?.to_owned();
    let first = first_model(&gateway, &model).ok_or_else(|| Error::UnknownModel {
        route: path,
        model: model.clone(),
    })?;
    authorize(identity, &model, first).await?;
    let call = Arc::new(RouteCall {
        gateway: gateway.clone(),
        identity: identity.clone(),
        body,
        route: Box::new(move |target| route(target).boxed()),
        served: Mutex::new(None),
        failures: Mutex::new(Vec::new()),
    });
    let request = RouterRequest {
        model,
        payload: Some(Payload(call.clone())),
        ..RouterRequest::default()
    };
    match gateway.router.call(Op::new(op), request).await {
        Ok(routed) => {
            let response = call.served.lock().take().ok_or(Error::Internal)?;
            Ok(with_routing_headers(&routed, response))
        }
        Err(litellm_router_core::Error::Attempt(failure)) => Err(call
            .failure(&failure)
            .unwrap_or(Error::Routing(failure.into()))),
        Err(litellm_router_core::Error::UnknownModel { model }) => {
            Err(Error::UnknownModel { route: path, model })
        }
        Err(error) => Err(Error::Routing(error)),
    }
}

/// `response` with the proxy's `x-litellm-*` headers saying what the router did to serve it.
fn with_routing_headers(routed: &RouterResponse, response: Response) -> Response {
    let headers = [
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
    });
    let (mut parts, body) = response.into_parts();
    parts.headers.extend(headers);
    Response::from_parts(parts, body)
}
