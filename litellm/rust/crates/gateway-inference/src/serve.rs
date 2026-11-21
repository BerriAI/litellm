//! One inference call from its body to its response. A route declares what it is as an
//! [`Endpoint`], and [`run`] serves it: the model its body names goes through router-core, the
//! route renders the outcome that served it, and a call that ends on an error is answered in the
//! route's error format.

use std::sync::Arc;

use axum::{
    Extension,
    extract::FromRequestParts,
    http::{HeaderMap, request::Parts},
    response::{IntoResponse, Response},
};
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_router_core::Op;
use serde_json::{Map, Value};

use crate::{
    Ended, Error, Gateway, RequestId,
    error::ErrorFormat,
    headers::Routing,
    model_restamp::ReportedModel,
    request,
    routing::{self, Host, RouteAttempt},
};

/// One request to an inference route, apart from its body.
pub struct Call<H> {
    pub(crate) gateway: Arc<Gateway<H>>,
    pub(crate) identity: AuthenticatedRequest,
    pub(crate) headers: HeaderMap,
    pub(crate) request_id: Option<String>,
}

impl<H: Host> FromRequestParts<Arc<Gateway<H>>> for Call<H> {
    type Rejection = litellm_gateway_auth::Error;

    async fn from_request_parts(
        parts: &mut Parts,
        gateway: &Arc<Gateway<H>>,
    ) -> Result<Self, Self::Rejection> {
        let identity = AuthenticatedRequest::from_request_parts(parts, gateway).await?;
        let Ok(RequestId(request_id)) = RequestId::from_request_parts(parts, gateway).await;
        Ok(Self {
            gateway: gateway.clone(),
            identity,
            headers: parts.headers.clone(),
            request_id,
        })
    }
}

/// What a route says about itself, next to the attempt it runs for the router.
pub(crate) trait Endpoint: RouteAttempt {
    const OP: Op;
    /// The path an unknown model is reported under.
    const PATH: &'static str;
    const ERRORS: ErrorFormat;

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
    let served = async {
        let body = body?;
        let model = request::model(&body)?.to_owned();
        let endpoint = build(body)?;
        let routed = routing::call(
            &call.gateway,
            &call.identity,
            E::OP,
            E::PATH,
            &model,
            &endpoint,
        )
        .await?;
        let routing = Routing::served(&routed);
        let reported = ReportedModel::of(&model, &routed);
        let response = E::render(routed.output.response, reported)?;
        Ok::<_, Ended>((Extension(routing), response).into_response())
    };
    served
        .await
        .unwrap_or_else(|ended| ended.response(E::ERRORS, call.request_id.as_deref()))
}
