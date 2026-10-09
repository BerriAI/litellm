use std::sync::Arc;

use axum::{Json, body::Bytes, extract::State, response::Response};
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host_http::Sse;
use litellm_inference_responses::{route::Responses, types::ResponsesCall};
use serde_json::json;

use crate::{Error, Gateway, JsonObject, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(&gateway, &body)?;
    request::authorize_model(&identity, deployment, &body).await?;
    let (body, cache_options) = crate::caching::prepare(&identity, body)?;
    let route = gateway.responses.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };

    let call = ResponsesCall {
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
    };
    let machine = route.machine(call, Some(cache_options.policy));
    let stream = Sse::<Responses, _, _>::new(Json, |error| {
        let error = Error::from(error);
        Bytes::from(format!(
            "event: error\ndata: {}\n\n",
            json!({"type": "error", "code": error.status().as_u16().to_string(), "message": error.to_string(), "param": null})
        ))
    });
    let headers = crate::caching::CacheHeaders::default();
    let response = litellm_host_http::serve(machine, (), headers.clone(), stream).await?;
    Ok(headers.apply(response))
}
