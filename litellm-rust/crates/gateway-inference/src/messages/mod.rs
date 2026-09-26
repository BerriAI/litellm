//! `POST /v1/messages`, as the Python proxy's `anthropic_response` serves it.

use std::{convert::Infallible, sync::Arc};

use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::{StreamExt, stream::BoxStream};
use litellm_core::messages::{
    Error as RouteError, MessagesCall, MessagesResponse, messages, messages_body,
};
use litellm_types::utils::{ProviderSpecificHeader, ProviderSpecificHeaders};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway};

/// Client headers Python forwards to Anthropic-speaking providers on every call.
const ANTHROPIC_API_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];
const ANTHROPIC_API_HEADER_PROVIDERS: &str = "anthropic,bedrock,bedrock_mantle,vertex_ai";

pub async fn create(
    State(gateway): State<Arc<Gateway>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request_id = headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    match handle(&gateway, &headers, &body).await {
        Ok(response) => response,
        Err(error) => (error.status(), Json(error.body(request_id.as_deref()))).into_response(),
    }
}

async fn handle(gateway: &Gateway, headers: &HeaderMap, body: &[u8]) -> Result<Response, Error> {
    let body = match serde_json::from_slice(body) {
        Ok(Value::Object(body)) => body,
        Ok(_) => return Err(Error::InvalidBody("expected a JSON object".into())),
        Err(error) => return Err(Error::InvalidBody(error.to_string())),
    };
    let model_name = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::InvalidBody("model is required".into()))?;
    let deployment = gateway
        .models
        .get(model_name)
        .ok_or_else(|| Error::UnknownModel(model_name.to_owned()))?;
    let call = project(deployment, body, headers)?;
    match messages(
        &gateway.resources,
        &gateway.http,
        gateway.secrets.as_ref(),
        call,
    )
    .await?
    {
        MessagesResponse::Message(message) => Ok(Json(message).into_response()),
        MessagesResponse::Stream { chunks, .. } => Ok(stream(chunks)),
    }
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
        extra_headers: None,
        provider_specific_header: anthropic_api_headers(headers),
        timeout: deployment.timeout,
        shaping: deployment.shaping.clone(),
    })
}

fn anthropic_api_headers(headers: &HeaderMap) -> Option<ProviderSpecificHeaders> {
    let extra_headers: Map<String, Value> = ANTHROPIC_API_HEADERS
        .into_iter()
        .filter_map(|name| {
            let value = headers.get(name)?.to_str().ok()?;
            Some((name.to_owned(), Value::from(value)))
        })
        .collect();
    (!extra_headers.is_empty()).then(|| {
        ProviderSpecificHeaders::One(ProviderSpecificHeader {
            custom_llm_provider: ANTHROPIC_API_HEADER_PROVIDERS.into(),
            extra_headers,
        })
    })
}

/// A chunk that fails after the stream opened is delivered as an SSE error frame, since
/// the status line already went out; the stream ends on it.
fn stream(chunks: BoxStream<'static, Result<Bytes, RouteError>>) -> Response {
    let body = chunks.map(|chunk| {
        Ok::<_, Infallible>(
            chunk.unwrap_or_else(|error| Bytes::from(Error::Route(error).sse_frame())),
        )
    });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(body),
    )
        .into_response()
}
