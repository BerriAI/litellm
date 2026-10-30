use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use litellm_inference_chat::types::ChatCompletionsCall;
use serde_json::{Map, Value};

use crate::{
    Error, Gateway, JsonObject,
    request::{self, Path},
};

const DEPLOYMENT_DEFAULT_PARAMS: &[&str] = &[
    "functions",
    "function_call",
    "temperature",
    "top_p",
    "n",
    "stream_options",
    "stop",
    "max_completion_tokens",
    "modalities",
    "prediction",
    "audio",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "response_format",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "thinking",
    "web_search_options",
    "include_server_side_tool_invocations",
    "service_tier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "verbosity",
    "store",
];

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<impl IntoResponse, Error> {
    handle(&gateway, &identity, body).await
}

pub(crate) async fn create_from_model_path(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
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
    handle(&gateway, &identity, body).await
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(gateway, "/chat/completions", &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    let body: Map<String, Value> =
        request::deployment_defaults(deployment, DEPLOYMENT_DEFAULT_PARAMS)
            .chain(body)
            .collect();
    let route = gateway.chat_completions.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };

    let messages = body.get("messages").cloned().unwrap_or_default();
    let headers = crate::caching::CacheHeaders::default();
    let response = litellm_host_http::serve_unary(
        route.machine(
            ChatCompletionsCall {
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
            },
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
