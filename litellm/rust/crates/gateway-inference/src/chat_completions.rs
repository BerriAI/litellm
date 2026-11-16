use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{Json, body::Bytes, extract::State, response::Response};
use litellm_cache_response::CacheOptions;
use litellm_inference_chat::{
    ChatCompletionsRoute,
    types::{ChatCompletionsCall, ChatCompletionsPlan},
};
use serde_json::Value;

use crate::{
    Error, Gateway,
    admit::{Planned, Refusal},
    request,
};

pub(crate) struct Call {
    plan: ChatCompletionsPlan,
    cache: CacheOptions,
}

/// `path_model` is the model a route names in its path, which the body's own model replaces. The
/// route core's `plan` is the allowlist here: it declines every parameter and message shape its
/// provider config does not translate.
pub(crate) async fn plan(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    path_model: Option<String>,
    body: &Bytes,
) -> Result<Call, Refusal> {
    let body = request::object(body)?;
    let body = match (body.get("model"), path_model) {
        (None | Some(Value::Null), Some(model)) => body
            .into_iter()
            .chain([("model".into(), Value::String(model))])
            .collect(),
        _ => body,
    };
    let deployment = request::resolve_deployment(gateway, "/chat/completions", &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    let (body, cache) = crate::caching::prepare(identity, body)?;
    let messages = body.get("messages").cloned().unwrap_or_default();
    let plan = ChatCompletionsRoute::plan(ChatCompletionsCall {
        model: deployment.model.clone(),
        messages,
        optional_params: body
            .into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages"))
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
    let route = gateway.chat_completions.clone();
    let route = match &gateway.cache {
        Some(shared) => route.with_cache(litellm_cache_response::ScopedCache::new(
            shared.clone(),
            cache.scope.clone(),
        )),
        None => route,
    };
    let headers = crate::caching::CacheHeaders::default();
    let response = litellm_host_http::serve_unary(
        route.machine(plan, cache.policy),
        (),
        headers.clone(),
        litellm_host_http::Unary::new(Json),
        None,
    )
    .await?;
    Ok(headers.apply(response))
}
