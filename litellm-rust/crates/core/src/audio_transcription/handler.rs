use crate::CoreResult;
use crate::error::CoreError;

#[cfg(feature = "bedrock-auth")]
use std::time::SystemTime;

#[cfg(feature = "bedrock-auth")]
use crate::providers::bedrock::audio_transcription::aws_auth_config;

#[cfg(feature = "bedrock-auth")]
use crate::providers::bedrock::aws_base::{resolve_credentials, sign_bedrock_post};

use super::client::http_client;
use super::common_utils::truncate_error_body;
use super::prepare::environment_lookup;
use super::transformation::AudioTranscriptionAuth;
use super::types::{AudioTranscriptionResponseData, PreparedAudioTranscriptionRequest};

pub(super) async fn execute_audio_transcription_provider_call(
    mut request: PreparedAudioTranscriptionRequest,
) -> CoreResult<AudioTranscriptionResponseData> {
    sign_request(&mut request).await?;

    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid audio request body: {error}"))
    })?;

    let mut request_builder = http_client().post(&request.url).body(body);

    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }

    if let Some(timeout) = request.timeout {
        request_builder = request_builder.timeout(timeout);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|error| CoreError::Network(error.to_string()))?;

    let status = response.status();

    let response_text = response
        .text()
        .await
        .map_err(|error| CoreError::Network(error.to_string()))?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&response_text),
        });
    }

    let response_json = serde_json::from_str(&response_text).map_err(|error| {
        CoreError::InvalidResponse(format!(
            "invalid audio transcription response JSON: {error}"
        ))
    })?;

    request
        .config
        .transform_transcription_response(&request.model, response_json)
}

async fn sign_request(request: &mut PreparedAudioTranscriptionRequest) -> CoreResult<()> {
    let env_lookup = environment_lookup;

    let auth =
        request
            .config
            .auth_strategy(&request.model, &request.optional_params, &env_lookup)?;

    match auth {
        AudioTranscriptionAuth::Bearer => {
            ensure_content_type(request);
            Ok(())
        }

        AudioTranscriptionAuth::AwsSigV4 { region, .. } => {
            sign_aws_sigv4_request(request, &region).await
        }
    }
}

fn ensure_content_type(request: &mut PreparedAudioTranscriptionRequest) {
    if !request
        .upstream_headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case("content-type"))
    {
        request
            .upstream_headers
            .push(("Content-Type".to_string(), "application/json".to_string()));
    }
}

#[cfg(feature = "bedrock-auth")]
async fn sign_aws_sigv4_request(
    request: &mut PreparedAudioTranscriptionRequest,
    region: &str,
) -> CoreResult<()> {
    let env_lookup = environment_lookup;

    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid audio request body: {error}"))
    })?;

    let mut headers = std::collections::BTreeMap::new();

    headers.insert("Content-Type".to_string(), "application/json".to_string());

    headers.extend(request.upstream_headers.iter().cloned());

    let credentials = resolve_credentials(
        aws_auth_config(&request.optional_params, &env_lookup),
        &env_lookup,
    )
    .await?;

    headers.extend(sign_bedrock_post(
        &request.url,
        &body,
        &headers,
        region,
        &credentials,
        SystemTime::now(),
    )?);

    request.upstream_headers = headers.into_iter().collect();

    Ok(())
}

#[cfg(not(feature = "bedrock-auth"))]
async fn sign_aws_sigv4_request(
    _request: &mut PreparedAudioTranscriptionRequest,
    _region: &str,
) -> CoreResult<()> {
    Err(CoreError::InvalidProvider(
        "AWS SigV4 authentication requires the `bedrock-auth` feature".to_string(),
    ))
}
