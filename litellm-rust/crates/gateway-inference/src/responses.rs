use std::sync::Arc;

use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, header},
    response::{IntoResponse, Response},
};
use litellm_core::responses::{
    responses,
    types::{ResponsesBody, ResponsesRequest},
};
use serde_json::Value;

use crate::{Error, Gateway, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle(&gateway, headers, &body).await {
        Ok(response) => response,
        Err(error) => error.openai_response(),
    }
}

async fn handle(gateway: &Gateway, headers: HeaderMap, body: &[u8]) -> Result<Response, Error> {
    let body = request::object(body)?;
    let deployment = request::deployment(gateway, &body)?;
    let response = responses(
        &gateway.resources,
        &gateway.http,
        gateway.secrets.as_ref(),
        ResponsesRequest {
            model: &deployment.model,
            body,
            api_key: deployment.api_key.as_deref(),
            api_base: deployment.api_base.as_deref(),
            custom_llm_provider: deployment.custom_llm_provider.as_deref(),
            extra_headers: Some(
                [
                    "openai-organization",
                    "openai-project",
                    "openai-beta",
                    "x-client-request-id",
                ]
                .into_iter()
                .filter_map(|name| {
                    Some((name.into(), Value::from(headers.get(name)?.to_str().ok()?)))
                })
                .collect(),
            ),
            timeout: deployment.timeout,
        },
    )
    .await?;
    match response.body {
        ResponsesBody::Response(body) => Ok((response.headers, Json(body)).into_response()),
        ResponsesBody::Stream(stream) => Ok((
            response.headers,
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            Body::from_stream(stream),
        )
            .into_response()),
    }
}
