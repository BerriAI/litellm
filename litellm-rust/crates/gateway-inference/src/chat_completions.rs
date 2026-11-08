use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use litellm_inference::provider::resolve_llm_provider;
use litellm_inference_chat::types::ChatCompletionsCall;
use litellm_model_catalog::Catalog;
use serde_json::{Map, Value};

use crate::{
    Deployment, Error, Gateway, JsonObject,
    request::{self, Path},
};

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
    let route = gateway.chat_completions.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };

    let max_output_tokens = deployment
        .max_output_tokens
        .or_else(|| catalog_max_output_tokens(&(gateway.catalog)(), deployment));
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
                max_output_tokens,
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

fn catalog_max_output_tokens(catalog: &Catalog, deployment: &Deployment) -> Option<i64> {
    let provider = resolve_llm_provider(
        &deployment.model,
        deployment.custom_llm_provider.as_deref(),
        "chat completions",
    )
    .ok()?;
    let found = catalog
        .lookup(provider.model)
        .filter(|found| found.matched_key == provider.model)?;
    i64::try_from(found.entry.model()?.attributes.max_output_tokens?).ok()
}
