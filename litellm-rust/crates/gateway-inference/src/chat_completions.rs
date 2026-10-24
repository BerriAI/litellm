use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

#[cfg(feature = "switchyard")]
use axum::http::HeaderMap;
use axum::{
    Json,
    extract::{Path, State},
    response::{IntoResponse, Response},
};
use litellm_inference_chat::types::{ChatCompletionsCall, ChatCompletionsRequest};
use serde_json::{Map, Value};
#[cfg(feature = "switchyard")]
use switchyard_protocol::WireFormat;

use crate::{Deployment, Error, Gateway, JsonObject, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    #[cfg(feature = "switchyard")] headers: HeaderMap,
    JsonObject(body): JsonObject,
) -> Result<impl IntoResponse, Error> {
    handle(
        &gateway,
        &identity,
        #[cfg(feature = "switchyard")]
        headers,
        body,
    )
    .await
}

pub(crate) async fn create_from_model_path(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    #[cfg(feature = "switchyard")] headers: HeaderMap,
    Path(model): Path<String>,
    JsonObject(body): JsonObject,
) -> Result<impl IntoResponse, Error> {
    let body = match body.get("model") {
        None | Some(Value::Null) => body
            .into_iter()
            .chain([("model".into(), Value::String(model))])
            .collect(),
        Some(_) => body,
    };
    handle(
        &gateway,
        &identity,
        #[cfg(feature = "switchyard")]
        headers,
        body,
    )
    .await
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    #[cfg(feature = "switchyard")] inbound_headers: HeaderMap,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(gateway, &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    #[cfg(feature = "switchyard")]
    let switchyard_route = body
        .get("model")
        .and_then(Value::as_str)
        .and_then(|model| gateway.switchyard_routes.get(model));
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    #[cfg(feature = "switchyard")]
    if let Some(switchyard_route) = switchyard_route {
        return switchyard_route
            .execute(
                gateway,
                cache_options,
                WireFormat::OpenAiChat,
                body,
                inbound_headers,
            )
            .await;
    }
    let route = gateway.chat_completions.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };
    let headers = crate::caching::CacheHeaders::default();
    let response = litellm_host_http::serve_unary(
        route.machine(
            ChatCompletionsCall::from(provider_request(deployment, body)),
            cache_options.policy,
        ),
        (),
        headers.clone(),
        litellm_host_http::Unary::new(Json),
        None,
    )
    .await?;
    Ok(headers.apply(response))
}

pub(crate) fn provider_request(
    deployment: &Deployment,
    mut body: Map<String, Value>,
) -> ChatCompletionsRequest<'_> {
    let messages = body.remove("messages").unwrap_or_default();
    body.remove("model");
    ChatCompletionsRequest {
        model: &deployment.model,
        messages,
        optional_params: body,
        api_key: deployment.api_key.as_deref(),
        api_base: deployment.api_base.as_deref(),
        custom_llm_provider: deployment.custom_llm_provider.as_deref(),
        extra_headers: None,
        timeout: deployment.timeout,
    }
}
