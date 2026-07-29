use std::collections::BTreeMap;
use std::time::SystemTime;

use litellm_core::CoreResult;
use litellm_core::error::CoreError;
use litellm_core::messages::transformation::MessagesAuthStrategy;
use litellm_core::providers::bedrock::aws_base::{
    AwsAuthConfig, resolve_credentials, sign_bedrock_post,
};
use serde_json::Value;

use super::client::http_client;
use super::common_utils::truncate_error_body;
use super::types::ProviderMessagesRequest;
use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;

fn environment_lookup(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

async fn signed_request(
    request: &ProviderMessagesRequest,
    body: &[u8],
) -> CoreResult<Vec<(String, String)>> {
    if !matches!(
        request.config.auth_strategy(),
        MessagesAuthStrategy::AwsSigV4
    ) {
        return Ok(request.upstream_headers.clone());
    }
    if let Some(token) = &request.bearer_token {
        return Ok(request
            .upstream_headers
            .iter()
            .filter(|(name, _)| {
                !matches!(
                    name.to_ascii_lowercase().as_str(),
                    "authorization" | "x-api-key" | "anthropic-version"
                )
            })
            .cloned()
            .chain([
                ("Authorization".to_string(), format!("Bearer {token}")),
                ("content-type".to_string(), "application/json".to_string()),
            ])
            .collect());
    }
    let headers = request
        .upstream_headers
        .iter()
        .filter(|(name, _)| {
            !matches!(
                name.to_ascii_lowercase().as_str(),
                "authorization" | "x-api-key" | "anthropic-version" | "host" | "content-length"
            )
        })
        .cloned()
        .chain(std::iter::once((
            "content-type".to_string(),
            "application/json".to_string(),
        )))
        .collect::<BTreeMap<_, _>>();
    let region = request.signing_region.as_deref().ok_or_else(|| {
        CoreError::InvalidRequest("Bedrock signing region was not resolved".to_string())
    })?;
    let credentials = resolve_credentials(AwsAuthConfig::default(), &environment_lookup).await?;
    let signed = sign_bedrock_post(
        &request.url,
        body,
        &headers,
        region,
        &credentials,
        SystemTime::now(),
    )?;
    Ok(signed.into_iter().collect())
}

pub(super) async fn execute_messages_provider_call(
    request: ProviderMessagesRequest,
) -> CoreResult<Value> {
    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid messages request body: {error}"))
    })?;
    let headers = signed_request(&request, &body).await?;
    let mut request_builder = http_client().post(&request.url).body(body);
    for (key, value) in &headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let response = serde_json::from_str(&text).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid messages response JSON: {err}"))
    })?;
    let transformed = request
        .config
        .transform_response(&request.model, response)?;
    serde_json::to_value(transformed).map_err(|err| {
        CoreError::InvalidResponse(format!("failed to serialize messages response: {err}"))
    })
}

pub(super) async fn execute_messages_provider_stream(
    request: ProviderMessagesRequest,
) -> CoreResult<reqwest::Response> {
    if request.provider != ANTHROPIC_MESSAGES_PROVIDER && request.signing_region.is_none() {
        return Err(CoreError::InvalidRequest(
            "streaming messages is not supported for this provider".to_string(),
        ));
    }

    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid messages request body: {error}"))
    })?;
    let headers = signed_request(&request, &body).await?;
    let mut request_builder = http_client().post(&request.url).body(body);
    for (key, value) in &headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        let text = response
            .text()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }
    Ok(response)
}
