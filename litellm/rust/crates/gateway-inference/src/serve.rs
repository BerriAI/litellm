//! One inference call from its body to its response. A route declares what it is as an
//! [`Endpoint`], and [`run`] serves it with the host's callback hooks around it (`hooks.rs`): the
//! model its body names goes through router-core, the hooks hear the outcome that served it, the
//! route renders it, and a call that ends on an error is answered in the route's error format.

use std::sync::Arc;

use axum::{
    Extension,
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
    record, request,
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
    pub(crate) admitted: Option<H::Admitted>,
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
            admitted: parts.extensions.get::<H::Admitted>().cloned(),
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

    /// The provider's id for a complete outcome's answer.
    fn response_id(outcome: &Self::Outcome) -> Option<String>;

    /// The provider's id for a stream's answer, when the event whose `data` this is names it.
    fn stream_response_id(data: &str) -> Option<String>;

    /// The outcome that served the call as the response the client reads.
    fn render(outcome: Self::Outcome, reported: ReportedModel<'_>) -> Result<Response, Error>;
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
        let endpoint = build(body)?;
        let routed = routing::call(gateway, identity, E::OP, E::PATH, &model, &endpoint).await?;
        let routing = Routing::served(&routed);
        let reported = ReportedModel::of(&model, &routed);
        let served = ServedBy {
            deployment: gateway.router.registry().deployment(&routed.deployment_id),
            model_group: routed.model_group,
        };
        let mut outcome = routed.output.response;
        let post_call = hooks::post_call::<E, _>(&hooks, &mut outcome, served).await?;
        Ok(post_call.answer((Extension(routing), E::render(outcome, reported)?)))
    };
    record::write(|entries| {
        entries.serve_as(E::CALL_TYPE);
        if let Some(call_id) = &call.call_id {
            entries.name_call(call_id.clone());
        }
    });
    let served = match body {
        Ok(body) => hooks::around(gateway, identity, call.admitted.as_ref(), body, serve).await,
        Err(error) => Err(Ended::from(error)),
    };
    served.unwrap_or_else(|ended| ended.response(E::ERRORS, call.request_id.as_deref()))
}
