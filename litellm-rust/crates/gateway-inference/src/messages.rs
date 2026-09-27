//! `POST /v1/messages`, as the Python proxy's `anthropic_response` serves it.

use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_core::messages::{
    Error as RouteError, MessagesCall, messages_body,
    route::{Messages, messages_machine},
};
use litellm_host_http::Sse;
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway, JsonObject, RequestId, request};

const ANTHROPIC_API_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];

pub async fn create(
    State(gateway): State<Arc<Gateway>>,
    RequestId(request_id): RequestId,
    headers: HeaderMap,
    body: Result<JsonObject, Error>,
) -> impl IntoResponse {
    let result = match body {
        Ok(JsonObject(body)) => handle(&gateway, &headers, body).await,
        Err(error) => Err(error),
    };
    result.map_err(|error| (error.status(), Json(error.body(request_id.as_deref()))))
}

async fn handle(
    gateway: &Gateway,
    headers: &HeaderMap,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(gateway, &body)?;
    let call = project(deployment, body, headers)?;
    let machine = messages_machine(&gateway.resources, &gateway.http, gateway.secrets.clone())
        .map_err(RouteError::from)?;
    let stream =
        Sse::<Messages, _, _>::new(Json, |error| Bytes::from(Error::from(error).sse_frame()));
    Ok(litellm_host_http::serve(machine(call), (), (), stream).await?)
}

fn project(
    deployment: &Deployment,
    body: Map<String, Value>,
    headers: &HeaderMap,
) -> Result<MessagesCall, Error> {
    let body = body
        .into_iter()
        .map(|(name, value)| match name.as_str() {
            "model" => (name, Value::from(deployment.model.as_str())),
            _ => (name, value),
        })
        .collect();
    Ok(MessagesCall {
        body: messages_body(body)?,
        api_key: deployment.api_key.clone(),
        api_base: deployment.api_base.clone(),
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        extra_headers: Some(anthropic_api_headers(headers)),
        provider_specific_header: None,
        timeout: deployment.timeout,
        shaping: deployment.shaping.clone(),
    })
}

fn anthropic_api_headers(headers: &HeaderMap) -> Map<String, Value> {
    ANTHROPIC_API_HEADERS
        .into_iter()
        .filter_map(|name| {
            let value = headers.get(name)?.to_str().ok()?;
            Some((name.to_owned(), Value::from(value)))
        })
        .collect()
}
