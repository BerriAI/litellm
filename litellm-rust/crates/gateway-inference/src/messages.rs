//! `POST /v1/messages`, as the Python proxy's `anthropic_response` serves it.

use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_core::messages::{MessagesCall, messages_body, route::Messages};
use litellm_host_http::Sse;
use litellm_types::utils::{ProviderSpecificHeader, ProviderSpecificHeaders};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway, JsonObject, RequestId, request};

/// Client headers Python forwards to Anthropic-speaking providers on every call.
const ANTHROPIC_API_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];
const ANTHROPIC_API_HEADER_PROVIDERS: &str = "anthropic,bedrock,bedrock_mantle,vertex_ai";

pub async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    RequestId(request_id): RequestId,
    headers: HeaderMap,
    body: Result<JsonObject, Error>,
) -> impl IntoResponse {
    let result = match body {
        Ok(JsonObject(body)) => handle(&gateway, &identity, &headers, body).await,
        Err(error) => Err(error),
    };
    result.map_err(|error| (error.status(), Json(error.body(request_id.as_deref()))))
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(gateway, &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    let call = project(deployment, body, headers)?;
    let machine = gateway.messages.clone().machine(call);
    let stream =
        Sse::<Messages, _, _>::new(Json, |error| Bytes::from(Error::from(error).sse_frame()));
    Ok(litellm_host_http::serve(machine, (), (), stream, None).await?)
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
