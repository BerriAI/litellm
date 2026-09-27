use std::sync::Arc;

use axum::{Json, extract::State, response::Response};
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

pub(crate) async fn create_from_model_path(
    gateway: &Gateway,
    model: &str,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    let body = if body.get("model").is_some_and(|model| !model.is_null()) {
        body
    } else {
        body.into_iter()
            .chain([("model".into(), Value::String(model.into()))])
            .collect()
    };
    handle(gateway, body).await
}

async fn handle(gateway: &Gateway, body: Map<String, Value>) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(gateway, &body)?;
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
