use std::sync::Arc;

#[cfg(feature = "switchyard")]
use axum::http::HeaderMap;
use axum::{Json, body::Bytes, extract::State, response::Response};
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host_http::Sse;
use litellm_inference_responses::{route::Responses, types::ResponsesCall};
use serde_json::{Map, Value, json};
#[cfg(feature = "switchyard")]
use switchyard_protocol::WireFormat;

use crate::{Deployment, Error, Gateway, JsonObject, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    #[cfg(feature = "switchyard")] headers: HeaderMap,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(&gateway, &body)?;
    request::authorize_model(&identity, deployment, &body).await?;
    #[cfg(feature = "switchyard")]
    let switchyard_route = body
        .get("model")
        .and_then(Value::as_str)
        .and_then(|model| gateway.switchyard_routes.get(model));
    let (body, cache_options) = crate::caching::prepare(&identity, body)?;
    #[cfg(feature = "switchyard")]
    if let Some(switchyard_route) = switchyard_route {
        return switchyard_route
            .execute(
                &gateway,
                cache_options,
                WireFormat::OpenAiResponses,
                body,
                headers,
            )
            .await;
    }
    let route = gateway.responses.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };

    let call = provider_request(deployment, body);
    let machine = route.machine(call, cache_options.policy);
    let stream = Sse::<Responses, _, _>::new(Json, |error| {
        let error = Error::from(error);
        Bytes::from(format!(
            "event: error\ndata: {}\n\n",
            json!({"type": "error", "code": error.status().as_u16().to_string(), "message": error.to_string(), "param": null})
        ))
    });
    let headers = crate::caching::CacheHeaders::default();
    let response = litellm_host_http::serve(machine, (), headers.clone(), stream, None).await?;
    Ok(headers.apply(response))
}

pub(crate) fn provider_request(
    deployment: &Deployment,
    mut body: Map<String, Value>,
) -> ResponsesCall {
    let input = body.remove("input").unwrap_or_default();
    body.remove("model");
    ResponsesCall {
        model: deployment.model.clone(),
        input,
        optional_params: body,
        api_key: deployment.api_key.clone(),
        api_base: deployment.api_base.clone(),
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        extra_headers: None,
        timeout: deployment.timeout,
    }
}
