use serde_json::Value;

use crate::error::Error;
use crate::http_utils::{HttpClientProfile, http_client, http_request, truncate_error_body};
use crate::runtime::{HttpRequest, HttpTransport};

use super::request::{
    build_resolved_pre_call_request_with_services, settle_pre_call_request_with_services,
    unchanged_pre_call_readback,
};
use super::types::{
    ChatCompletionsResponse, ProviderChatCompletionsRequest, ProviderChatResponseData,
    ResolvedChatCompletionsRequest, SettledChatRequest,
};

pub(crate) async fn execute_chat_completions_provider_call_with_transport<S, T>(
    services: &S,
    transport: &T,
    request: ResolvedChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error>
where
    S: crate::providers::auth::ChatAuthorizationServices,
    T: HttpTransport,
{
    let prepared = build_resolved_pre_call_request_with_services(services, request).await?;
    let readback = unchanged_pre_call_readback(&prepared)?;
    let settled = settle_pre_call_request_with_services(services, prepared, readback).await?;
    execute_settled_request_with_transport(transport, settled).await
}

pub(crate) async fn execute_settled_request_with_transport<T>(
    transport: &T,
    request: SettledChatRequest,
) -> Result<ChatCompletionsResponse, Error>
where
    T: HttpTransport,
{
    let SettledChatRequest { endpoint, http } = request;
    let (body, headers) = http.into_parts();
    let response = transport
        .execute(HttpRequest {
            method: reqwest::Method::POST,
            url: endpoint.url.clone(),
            headers,
            body,
            timeout: endpoint.timeout,
        })
        .await?;
    let text = String::from_utf8(response.body).map_err(|error| {
        Error::InvalidResponse(format!("invalid chat completions response body: {error}"))
    })?;

    if !(200..300).contains(&response.status) {
        return Err(Error::Http {
            status: response.status,
            body: truncate_error_body(&text),
        });
    }

    let body: Value = serde_json::from_str(&text).map_err(|err| {
        Error::InvalidResponse(format!("invalid chat completions response JSON: {err}"))
    })?;
    endpoint
        .config
        .transform_response(&endpoint.model, ProviderChatResponseData { body })
        .map_err(as_response_error)
}

pub(super) async fn execute_settled_request(
    request: SettledChatRequest,
) -> Result<ChatCompletionsResponse, Error> {
    let SettledChatRequest { endpoint, http } = request;
    let (body, headers) = http.into_parts();
    let client = http_client(HttpClientProfile::Standard)
        .map_err(|error| Error::Connect(error.to_string()))?;
    let request_builder = headers.into_iter().fold(
        client.post(&endpoint.url).body(body),
        |builder, (key, value)| builder.header(key, value),
    );
    let request_builder = match endpoint.timeout {
        Some(timeout) => request_builder.timeout(timeout),
        None => request_builder,
    };

    let response = http_request(request_builder).await.map_err(|err| {
        // Failing to establish the connection means the request never went out,
        // so the host can still serve it. Everything else here, a timeout
        // above all, may have reached the provider and been answered.
        if err.is_connect() || err.is_builder() {
            Error::Connect(err.to_string())
        } else {
            Error::Network(err.to_string())
        }
    })?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let body: Value = serde_json::from_str(&text).map_err(|err| {
        Error::InvalidResponse(format!("invalid chat completions response JSON: {err}"))
    })?;
    endpoint
        .config
        .transform_response(&endpoint.model, ProviderChatResponseData { body })
        .map_err(as_response_error)
}

/// Re-tag an error raised while normalizing a response the provider already
/// returned.
///
/// A config reports the same variants on either side of the call: a missing
/// field or an unsupported block can mean "this request cannot be translated"
/// while building and "this response cannot be normalized" here. Only the
/// second kind has already been billed, and a host that keeps a reference
/// implementation must not retry those, so collapse them to one variant that
/// can only mean the provider was already called.
pub fn as_response_error(err: Error) -> Error {
    match err {
        already @ (Error::InvalidResponse(_) | Error::Http { .. }) => already,
        other => Error::InvalidResponse(other.to_string()),
    }
}

pub async fn signed_headers(
    request: &ProviderChatCompletionsRequest,
    body: &[u8],
) -> Result<Vec<(String, String)>, Error> {
    signed_headers_with_services(
        crate::providers::auth::native_authorization_services(),
        request,
        body,
    )
    .await
}

pub async fn signed_headers_with_services<S>(
    services: &S,
    request: &ProviderChatCompletionsRequest,
    body: &[u8],
) -> Result<Vec<(String, String)>, Error>
where
    S: crate::providers::auth::ChatAuthorizationServices,
{
    request
        .config
        .authorize(
            services,
            request.authorization_context(),
            crate::lifecycle::WireBody::from_bytes(body.to_vec()),
        )
        .await
        .map(|authorized| authorized.headers().to_vec())
}
