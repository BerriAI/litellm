use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_core::chat_completions::{chat_completions, types::ChatCompletionsRequest};
use serde_json::{Map, Value};

use crate::{Error, Gateway, request};

pub(crate) async fn create(State(gateway): State<Arc<Gateway>>, body: Bytes) -> Response {
    respond(&gateway, request::object(&body)).await
}

pub(crate) async fn deployment(
    State(gateway): State<Arc<Gateway>>,
    Path(path): Path<String>,
    body: Bytes,
) -> Response {
    if let Some(model) = path
        .strip_suffix("/chat/completions")
        .filter(|model| !model.is_empty())
    {
        let body = request::object(&body).map(|body| {
            if body.get("model").is_some_and(|model| !model.is_null()) {
                return body;
            }
            body.into_iter()
                .chain([("model".into(), Value::String(model.into()))])
                .collect()
        });
        return respond(&gateway, body).await;
    }
    if path.ends_with("/embeddings") || path.ends_with("/completions") {
        return Error::Unsupported(path).openai_response();
    }
    StatusCode::NOT_FOUND.into_response()
}

async fn respond(gateway: &Gateway, body: Result<Map<String, Value>, Error>) -> Response {
    let result = match body {
        Ok(body) => handle(gateway, body).await,
        Err(error) => Err(error),
    };
    match result {
        Ok(response) => response,
        Err(error) => error.openai_response(),
    }
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
    let response = chat_completions(
        &gateway.resources,
        &gateway.http,
        ChatCompletionsRequest {
            model: &deployment.model,
            messages,
            optional_params: body
                .into_iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages" | "stream"))
                .collect(),
            api_key: deployment.api_key.as_deref(),
            api_base: deployment.api_base.as_deref(),
            custom_llm_provider: deployment.custom_llm_provider.as_deref(),
            extra_headers: None,
            timeout: deployment.timeout,
        },
    )
    .await?;
    Ok(Json(response).into_response())
}
