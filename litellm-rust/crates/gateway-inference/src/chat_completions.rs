use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::{Path, State},
    http::header::CONTENT_TYPE,
    response::{IntoResponse, Response},
};
use litellm_host_http::Sse;
use litellm_inference_chat::{
    ChatCompletionsRoute,
    openai::{OpenAiChatCall, OpenAiChatCompletions},
    types::ChatCompletionsCall,
};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway, JsonObject, request};

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
    let deployment = request::resolve_deployment(gateway, &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    if ChatCompletionsRoute::serves_openai(
        &deployment.model,
        deployment.custom_llm_provider.as_deref(),
    ) {
        return openai(gateway, deployment, body).await;
    }
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

async fn openai(
    gateway: &Gateway,
    deployment: &Deployment,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let machine = gateway.chat_completions.clone().openai_machine(
        OpenAiChatCall {
            model: deployment.model.clone(),
            body,
            api_key: deployment.api_key.clone(),
            api_base: deployment.api_base.clone(),
            custom_llm_provider: deployment.custom_llm_provider.clone(),
            extra_headers: None,
            timeout: deployment.timeout,
        },
        None,
    );
    let encoder = Sse::<OpenAiChatCompletions, _, _>::new(
        |body: Bytes| ([(CONTENT_TYPE, "application/json")], body),
        |error| Bytes::from(Error::from(error).openai_sse_frame()),
    );
    Ok(litellm_host_http::serve(machine, (), (), encoder, None).await?)
}
