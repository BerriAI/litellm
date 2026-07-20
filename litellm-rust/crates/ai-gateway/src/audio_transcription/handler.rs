use std::time::SystemTime;

use litellm_core::CoreResult;
use litellm_core::audio_transcription::transformation::AudioTranscriptionAuth;
use litellm_core::error::CoreError;
use litellm_core::providers::bedrock::audio_transcription::aws_auth_config;
use litellm_core::providers::bedrock::aws_base::{resolve_credentials, sign_bedrock_post};
use serde_json::Value;

use super::common_utils::truncate_error_body;
use super::types::ProviderAudioTranscriptionRequest;
use crate::client::http_client;

pub(crate) async fn execute_audio_transcription_provider_call(
    request: ProviderAudioTranscriptionRequest,
) -> CoreResult<Value> {
    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid audio request body: {error}"))
    })?;
    let mut request_builder = http_client().post(&request.url).body(body.clone());
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }
    let response = request_builder
        .send()
        .await
        .map_err(|error| CoreError::Network(error.to_string()))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|error| CoreError::Network(error.to_string()))?;
    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }
    let response_json: Value = serde_json::from_str(&text).map_err(|error| {
        CoreError::InvalidResponse(format!("invalid audio response JSON: {error}"))
    })?;
    Ok(request
        .config
        .transform_transcription_response(&request.model, response_json)?
        .into_json())
}

pub(crate) async fn sign_request(
    request: &ProviderAudioTranscriptionRequest,
    optional_params: &serde_json::Map<String, Value>,
) -> CoreResult<ProviderAudioTranscriptionRequest> {
    let env_lookup = environment_lookup;
    let auth = request
        .config
        .auth_strategy(&request.model, optional_params, &env_lookup)?;
    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid audio request body: {error}"))
    })?;
    let mut headers = super::common_utils::string_headers(None)?;
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    headers.extend(request.upstream_headers.iter().cloned());
    match auth {
        AudioTranscriptionAuth::Bearer => {}
        AudioTranscriptionAuth::AwsSigV4 { region, .. } => {
            let credentials =
                resolve_credentials(aws_auth_config(optional_params, &env_lookup), &env_lookup)
                    .await?;
            headers.extend(sign_bedrock_post(
                &request.url,
                &body,
                &headers,
                &region,
                &credentials,
                SystemTime::now(),
            )?);
        }
    }
    Ok(ProviderAudioTranscriptionRequest {
        upstream_headers: headers.into_iter().collect(),
        ..request.clone()
    })
}

pub(super) fn environment_lookup(key: &str) -> Option<String> {
    std::env::var(key).ok()
}
