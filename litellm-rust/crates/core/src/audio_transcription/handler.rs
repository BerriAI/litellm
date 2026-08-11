use crate::CoreResult;
use crate::error::CoreError;

#[cfg(feature = "bedrock-auth")]
use std::collections::BTreeMap;
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
    request: PreparedAudioTranscriptionRequest,
) -> CoreResult<AudioTranscriptionResponseData> {
    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid audio request body: {error}"))
    })?;

    let headers = authenticated_headers(&request, &body).await?;

    let request_builder = headers.into_iter().fold(
        http_client().post(&request.url).body(body),
        |builder, (key, value)| builder.header(key, value),
    );

    let request_builder = match request.timeout {
        Some(timeout) => request_builder.timeout(timeout),
        None => request_builder,
    };

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

async fn authenticated_headers(
    request: &PreparedAudioTranscriptionRequest,
    body: &[u8],
) -> CoreResult<Vec<(String, String)>> {
    let env_lookup = environment_lookup;

    let auth =
        request
            .config
            .auth_strategy(&request.model, &request.optional_params, &env_lookup)?;

    match auth {
        AudioTranscriptionAuth::Bearer => Ok(with_content_type(&request.upstream_headers)),
        AudioTranscriptionAuth::AwsSigV4 { region, .. } => {
            aws_sigv4_headers(request, body, &region).await
        }
    }
}

fn with_content_type(headers: &[(String, String)]) -> Vec<(String, String)> {
    if headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case("content-type"))
    {
        headers.to_vec()
    } else {
        headers
            .iter()
            .cloned()
            .chain(std::iter::once((
                "Content-Type".to_string(),
                "application/json".to_string(),
            )))
            .collect()
    }
}

#[cfg(feature = "bedrock-auth")]
async fn aws_sigv4_headers(
    request: &PreparedAudioTranscriptionRequest,
    body: &[u8],
    region: &str,
) -> CoreResult<Vec<(String, String)>> {
    let env_lookup = environment_lookup;

    let headers = std::iter::once(("Content-Type".to_string(), "application/json".to_string()))
        .chain(request.upstream_headers.iter().cloned())
        .collect::<BTreeMap<_, _>>();

    let credentials = resolve_credentials(
        aws_auth_config(&request.optional_params, &env_lookup),
        &env_lookup,
    )
    .await?;

    let signed_headers = sign_bedrock_post(
        &request.url,
        body,
        &headers,
        region,
        &credentials,
        SystemTime::now(),
    )?;

    Ok(headers
        .into_iter()
        .chain(signed_headers)
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .collect())
}

#[cfg(not(feature = "bedrock-auth"))]
async fn aws_sigv4_headers(
    _request: &PreparedAudioTranscriptionRequest,
    _body: &[u8],
    _region: &str,
) -> CoreResult<Vec<(String, String)>> {
    Err(CoreError::InvalidProvider(
        "AWS SigV4 authentication requires the `bedrock-auth` feature".to_string(),
    ))
}
