use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_core::chat_completions::{
    Error as RouteError, route::chat_completions_machine, types::ChatCompletionsCall,
};
use serde_json::{Map, Value};

use crate::{Error, Gateway, JsonObject, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    handle(&gateway, body).await
}

pub(crate) async fn deployment(
    State(gateway): State<Arc<Gateway>>,
    Path(path): Path<String>,
    body: Result<JsonObject, Error>,
) -> Result<Response, Error> {
    if let Some(model) = path
        .strip_suffix("/chat/completions")
        .filter(|model| !model.is_empty())
    {
        let JsonObject(body) = body?;
        let body = if body.get("model").is_some_and(|model| !model.is_null()) {
            body
        } else {
            body.into_iter()
                .chain([("model".into(), Value::String(model.into()))])
                .collect()
        };
        return handle(&gateway, body).await;
    }
    if path.ends_with("/embeddings") || path.ends_with("/completions") {
        return Err(Error::Unsupported(path));
    }
    Ok(StatusCode::NOT_FOUND.into_response())
}

async fn handle(gateway: &Gateway, body: Map<String, Value>) -> Result<Response, Error> {
    let deployment = request::deployment(gateway, &body)?;
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(Error::Unsupported("streaming chat completions".into()));
    }
    let messages = body
        .get("messages")
        .cloned()
        .ok_or_else(|| Error::InvalidBody("messages is required".into()))?;
    let machine =
        chat_completions_machine(&gateway.resources, &gateway.http).map_err(RouteError::from)?;
    let response = litellm_host_http::serve_unary(
        machine,
        ChatCompletionsCall {
            model: deployment.model.clone(),
            messages,
            optional_params: body
                .into_iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages" | "stream"))
                .collect(),
            api_key: deployment.api_key.clone(),
            api_base: deployment.api_base.clone(),
            custom_llm_provider: deployment.custom_llm_provider.clone(),
            extra_headers: None,
            timeout: deployment.timeout,
        },
        (),
        Json,
    )
    .await?;
    Ok(response)
}
