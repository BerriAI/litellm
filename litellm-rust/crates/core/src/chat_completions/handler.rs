use serde_json::Value;

use crate::error::{CoreError, CoreResult};
use crate::http_utils::truncate_error_body;

use super::client::http_client;
use super::transformation::ChatCompletionsAuth;
use super::types::{
    ChatCompletionsResponse, ProviderChatCompletionsRequest, ProviderChatResponseData,
};

pub(super) async fn execute_chat_completions_provider_call(
    request: ProviderChatCompletionsRequest,
) -> CoreResult<ChatCompletionsResponse> {
    let body = serde_json::to_vec(&request.body).map_err(|err| {
        CoreError::InvalidRequest(format!(
            "failed to serialize chat completions request: {err}"
        ))
    })?;
    let headers = signed_headers(&request, &body).await?;

    let mut request_builder = http_client().post(&request.url).body(body);
    for (key, value) in &headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder.send().await.map_err(|err| {
        // Failing to establish the connection means the request never went out,
        // so the host can still serve it. Everything else here, a timeout
        // above all, may have reached the provider and been answered.
        if err.is_connect() || err.is_builder() {
            CoreError::Connect(err.to_string())
        } else {
            CoreError::Network(err.to_string())
        }
    })?;

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

    let body: Value = serde_json::from_str(&text).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid chat completions response JSON: {err}"))
    })?;
    request
        .config
        .transform_response(&request.model, ProviderChatResponseData { body })
        .map_err(as_response_error)
}

/// Re-tag an error raised while normalizing a response the provider already
/// returned.
///
/// A config reports the same variants on either side of the call: a missing
/// field or an unsupported block can mean "this request cannot be translated"
/// during prepare and "this response cannot be normalized" here. Only the
/// second kind has already been billed, and a host that keeps a reference
/// implementation must not retry those, so collapse them to one variant that
/// can only mean the provider was already called.
pub(super) fn as_response_error(err: CoreError) -> CoreError {
    match err {
        already @ (CoreError::InvalidResponse(_) | CoreError::Http { .. }) => already,
        other => CoreError::InvalidResponse(other.to_string()),
    }
}

#[cfg(feature = "bedrock-auth")]
pub(super) async fn signed_headers(
    request: &ProviderChatCompletionsRequest,
    body: &[u8],
) -> CoreResult<Vec<(String, String)>> {
    use std::collections::BTreeMap;
    use std::time::SystemTime;

    use crate::providers::bedrock::aws_base::{
        aws_auth_config, aws_signature_headers, host_supplied_credentials,
        is_sigv4_computed_header, resolve_credentials, sign_bedrock_post,
    };

    let ChatCompletionsAuth::AwsSigV4 { region } = &request.auth else {
        return Ok(request.upstream_headers.clone());
    };
    // Reattaching a header the signer also emits would put both copies on the
    // wire, and Bedrock rejects that pair. Python instead drops the caller's
    // copy and prefers a forwarded Authorization over the signature, so leave
    // the request to Python rather than serving it a different way here.
    if request
        .upstream_headers
        .iter()
        .any(|(name, _)| is_sigv4_computed_header(name))
    {
        return Err(CoreError::Unsupported(
            "request forwards a header AWS SigV4 computes",
        ));
    }
    let env_lookup = |key: &str| std::env::var(key).ok();
    let unsigned: BTreeMap<String, String> = request.upstream_headers.iter().cloned().collect();
    // A host with its own resolution chain hands the result down; only fall
    // back to deriving credentials here when it supplied none.
    let credentials = match host_supplied_credentials(&request.optional_params) {
        Some(credentials) => credentials,
        None => {
            resolve_credentials(
                aws_auth_config(&request.optional_params, &env_lookup),
                &env_lookup,
            )
            .await?
        }
    };
    let signature = sign_bedrock_post(
        &request.url,
        body,
        &aws_signature_headers(&unsigned),
        region,
        &credentials,
        SystemTime::now(),
    )?;
    // Every original header goes back on the wire alongside the computed ones,
    // as Python reattaches them. The guard above already rejected the names
    // that would collide, so no name appears twice.
    Ok(unsigned.into_iter().chain(signature).collect())
}

#[cfg(not(feature = "bedrock-auth"))]
pub(super) async fn signed_headers(
    request: &ProviderChatCompletionsRequest,
    _body: &[u8],
) -> CoreResult<Vec<(String, String)>> {
    match &request.auth {
        ChatCompletionsAuth::AwsSigV4 { .. } => Err(CoreError::Unsupported(
            "AWS SigV4 requires the bedrock-auth feature",
        )),
        _ => Ok(request.upstream_headers.clone()),
    }
}
