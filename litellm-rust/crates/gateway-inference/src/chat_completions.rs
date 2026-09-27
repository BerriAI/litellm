use std::{convert::Infallible, sync::Arc};

use axum::{
    Json,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_core::chat_completions::{
    Error as RouteError,
    route::{ChatCompletions, chat_completions_machine},
    types::ChatCompletionsCall,
};
use litellm_host_http::HttpAdapter;
use litellm_types::utils::ChatCompletionsResponse;
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
    let machine =
        chat_completions_machine(&gateway.resources, &gateway.http).map_err(RouteError::from)?;
    let response = litellm_host_http::serve(
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
        ChatHttp,
        (),
    )
    .await?;
    Ok(response)
}

struct ChatHttp;

impl HttpAdapter for ChatHttp {
    type Protocol = ChatCompletions;

    async fn custom_op(&self, op: Infallible) -> Result<(), RouteError> {
        match op {}
    }

    fn complete(&self, response: ChatCompletionsResponse) -> Result<Response, RouteError> {
        Ok(Json(response).into_response())
    }

    fn head(&self, head: Infallible) -> Result<axum::http::Response<()>, RouteError> {
        match head {}
    }

    fn chunk(&self, chunk: Infallible) -> Result<Bytes, RouteError> {
        match chunk {}
    }

    fn stream_error(&self, error: litellm_host_http::Error<RouteError>) -> Bytes {
        Bytes::from(Error::from(error).sse_frame())
    }
}
