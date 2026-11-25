//! One inference call from its body to its response. A route declares what it is as an
//! [`Endpoint`], and [`run`] serves it with the host's callback hooks around it (`hooks.rs`): the
//! model its body names goes through router-core, the hooks hear the outcome that served it, the
//! route renders it, and a call that ends on an error is answered in the route's error format.

use std::{sync::Arc, time::Duration};

use axum::{
    extract::FromRequestParts,
    http::{HeaderMap, request::Parts},
    response::Response,
};
use litellm_gateway_auth::Authenticated;
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use litellm_router_core::Op;
use serde_json::{Map, Value};

use crate::{
    Ended, Error, Gateway, RequestId,
    error::ErrorFormat,
    headers::Routing,
    hooks::{self, CallType, ServedBy},
    model_restamp::ReportedModel,
    record::{self, CallFailure, ErrorResponse},
    request,
    routing::{self, Host, RouteAttempt},
    stream_event::Counts,
};

/// One request to an inference route, apart from its body.
pub struct Call<H: Host> {
    pub(crate) gateway: Arc<Gateway<H>>,
    pub(crate) identity: Authenticated<H::Rows>,
    pub(crate) headers: HeaderMap,
    pub(crate) request_id: Option<String>,
    /// The `x-litellm-call-id` the gateway's call id layer resolved, `None` on a router that does
    /// not wear it.
    pub(crate) call_id: Option<String>,
}

impl<H: Host> FromRequestParts<Arc<Gateway<H>>> for Call<H> {
    type Rejection = litellm_gateway_auth::Error;

    async fn from_request_parts(
        parts: &mut Parts,
        gateway: &Arc<Gateway<H>>,
    ) -> Result<Self, Self::Rejection> {
        let identity = Authenticated::from_request_parts(parts, gateway).await?;
        let Ok(RequestId(request_id)) = RequestId::from_request_parts(parts, gateway).await;
        let call_id = parts
            .extensions
            .get::<tower_http::request_id::RequestId>()
            .and_then(|id| id.header_value().to_str().ok())
            .map(str::to_owned);
        Ok(Self {
            gateway: gateway.clone(),
            identity,
            headers: parts.headers.clone(),
            request_id,
            call_id,
        })
    }
}

/// What a route says about itself, next to the attempt it runs for the router.
pub(crate) trait Endpoint: RouteAttempt {
    const CALL_TYPE: CallType;
    const OP: Op;
    /// The path an unknown model is reported under.
    const PATH: &'static str;
    const ERRORS: ErrorFormat;

    /// Those counts as Python's litellm `Usage`, `None` for a route that counts no tokens.
    fn usage(counts: Counts) -> Option<ChatCompletionsUsage>;

    /// The provider's id for a complete output's response.
    fn response_id(output: &Self::Output) -> Option<String>;

    /// The provider's id for a stream's answer, when the event whose `data` this is names it.
    fn stream_response_id(data: &str) -> Option<String>;

    /// Whether a call of this route with `body` runs under a deployment's stream timeout.
    fn streamed(body: &Map<String, Value>) -> bool {
        routing::streamed(body)
    }

    /// The process `request_timeout` a call of this route to deployment `deployment_id` reads
    /// after the router's timeout, `None` for a route that reads none.
    fn configured_timeout<H: Host>(gateway: &Gateway<H>, deployment_id: &str) -> Option<Duration> {
        let _ = (gateway, deployment_id);
        None
    }

    /// The output that served the call as the response the client reads.
    fn render(output: Self::Output, reported: ReportedModel<'_>) -> Result<Response, Error>;
}

/// Serves `body` as one call of the route `build` makes of it. A body that could not be read is
/// answered here too, so it is told in the route's error format.
pub(crate) async fn run<H, E, B>(
    call: &Call<H>,
    body: Result<Map<String, Value>, Error>,
    build: B,
) -> Response
where
    H: Host,
    E: Endpoint,
    B: FnOnce(Map<String, Value>) -> Result<E, Error>,
{
    let (gateway, identity) = (&*call.gateway, &call.identity);
    let serve = |hooks: Arc<H::Hooks>, body: Map<String, Value>| async move {
        let model = request::model(&body)?.to_owned();
        let streamed = E::streamed(&body);
        record::write(|entries| entries.stream_timeout = streamed);
        let endpoint = build(body)?;
        let routed = routing::call(gateway, identity, E::OP, E::PATH, &model, &endpoint).await?;
        let routing = Routing::served(&routed);
        let deployment = routing::served_deployment(
            gateway,
            &routed.deployment_id,
            streamed,
            E::configured_timeout(gateway, &routed.deployment_id),
        );
        let reported = ReportedModel::of(&model, &routed);
        let served = ServedBy {
            deployment: gateway.router.registry().deployment(&routed.deployment_id),
            model_group: routed.model_group,
        };
        let (mut output, recorded) = routed.output.response;
        let stream = E::held(&mut output).is_some();
        record::write(|entries| {
            entries.routed(routing, deployment, recorded);
            if stream {
                entries.stream_response = true;
            }
        });
        let post_call = hooks::post_call::<E, _>(&hooks, &mut output, served).await?;
        Ok(post_call.answer(E::render(output, reported)?))
    };
    record::write(|entries| {
        entries.call_type = Some(E::CALL_TYPE);
        if let Some(call_id) = &call.call_id {
            entries.call_id = Some(call_id.clone());
        }
    });
    let served = match body {
        Ok(body) => hooks::around(gateway, identity, body, serve).await,
        Err(error) => Err(Ended::from(error)),
    };
    served.unwrap_or_else(|mut ended| {
        let routing = ended.routing.take();
        let raised_on = ended.raised_on.as_deref();
        let error = error_response::<H, E>(gateway, raised_on, &ended.error);
        let provider = failed_provider(gateway, raised_on);
        let failure = CallFailure::raised(&ended.error, provider.as_deref(), error.timeout);
        record::write(|entries| {
            entries.respond_with_error(routing, error);
            entries.failure = Some(failure);
        });
        ended.response(E::ERRORS, call.request_id.as_deref())
    })
}

/// How a call of the route `E` that ended on `error` is responded to, `raised_on` naming the
/// deployment whose attempt raised it.
fn error_response<H: Host, E: Endpoint>(
    gateway: &Gateway<H>,
    raised_on: Option<&str>,
    error: &Error,
) -> ErrorResponse {
    let streamed = record::read(|entries| entries.stream_timeout).unwrap_or(false);
    let timeout = raised_on
        .filter(|_| error.raised_by_provider())
        .and_then(|deployment_id| {
            routing::served_deployment(
                gateway,
                deployment_id,
                streamed,
                E::configured_timeout(gateway, deployment_id),
            )
        })
        .and_then(|deployment| deployment.timeout);
    ErrorResponse {
        in_handler: error.in_handler(),
        timeout,
        provider_headers: error.provider_headers(),
    }
}

/// The provider the exception a failed call raised names: the provider of the deployment whose
/// attempt raised it, else the one the call's model resolves to as
/// `resolve_llm_provider_for_rate_limit` resolves it for a limiter's refusal, by the model's
/// provider prefix and then by the first deployment of the model group it names.
fn failed_provider<H: Host>(gateway: &Gateway<H>, raised_on: Option<&str>) -> Option<String> {
    let registry = gateway.router.registry();
    if let Some(deployment) = raised_on.and_then(|id| registry.deployment(id)) {
        return Some(deployment.provider().to_owned());
    }
    let model = record::read(|entries| entries.model().map(str::to_owned)).flatten()?;
    if let Some((provider, _)) = model.split_once('/') {
        return Some(provider.to_owned());
    }
    let group = registry.group(&model)?;
    group
        .first()
        .map(|deployment| deployment.provider().to_owned())
}
