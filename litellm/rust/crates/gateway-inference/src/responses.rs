use std::sync::Arc;

use axum::{Json, body::Bytes, extract::State, response::Response};
use litellm_cache_response::CacheOptions;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host_http::Sse;
use litellm_inference_responses::{
    ResponsesRoute,
    route::Responses,
    types::{ResponsesCall, ResponsesPlan},
};

use crate::{
    Error, Gateway,
    admit::{Planned, Refusal},
    request,
};

pub(crate) struct Call {
    plan: ResponsesPlan,
    cache: CacheOptions,
}

pub(crate) async fn plan(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    body: &Bytes,
) -> Result<Call, Refusal> {
    let body = request::object(body)?;
    let deployment = request::resolve_deployment(gateway, "/responses", &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    let (body, cache) = crate::caching::prepare(identity, body)?;
    let plan = ResponsesRoute::plan(ResponsesCall {
        model: deployment.model.clone(),
        input: body.get("input").cloned().unwrap_or_default(),
        optional_params: body
            .into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "model" | "input"))
            .collect(),
        api_key: deployment.api_key.clone(),
        api_base: deployment.api_base.clone(),
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        extra_headers: None,
        timeout: deployment.timeout,
    })?;
    Ok(Call { plan, cache })
}

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    Planned(Call { plan, cache }): Planned<Call>,
) -> Result<Response, Error> {
    let route = gateway.responses.clone();
    let route = match &gateway.cache {
        Some(shared) => route.with_cache(litellm_cache_response::ScopedCache::new(
            shared.clone(),
            cache.scope.clone(),
        )),
        None => route,
    };
    let machine = route.machine(plan, cache.policy);
    let stream = Sse::<Responses, _, _>::new(Json, |error| {
        Bytes::from(Error::from(error).responses_sse_frame())
    });
    let headers = crate::caching::CacheHeaders::default();
    let response = litellm_host_http::serve(machine, (), headers.clone(), stream, None).await?;
    Ok(headers.apply(response))
}
