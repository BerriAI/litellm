use serde_json::Value;

use crate::error::Error;
use crate::http_utils::truncate_error_body;

use super::client::http_client;
use super::types::ProviderAudioTranscriptionRequest;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn execute_audio_transcription_provider_call(
    request: ProviderAudioTranscriptionRequest,
) -> Result<Value, Error> {
    let body = serde_json::to_vec(&request.body)
        .map_err(|error| Error::InvalidRequest(format!("invalid audio request body: {error}")))?;
    let headers = signed_headers(&request, &body).await?;
    let mut request_builder = http_client().post(&request.url).body(body);
    for (key, value) in headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }
    let response = request_builder
        .send()
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }
    let response_json = serde_json::from_str(&text)
        .map_err(|error| Error::InvalidResponse(format!("invalid audio response JSON: {error}")))?;
    Ok(request
        .config
        .transform_transcription_response(&request.model, response_json)?
        .into_json())
}

#[cfg(feature = "bedrock-auth")]
async fn signed_headers(
    request: &ProviderAudioTranscriptionRequest,
    body: &[u8],
) -> Result<Vec<(String, String)>, Error> {
    use std::collections::BTreeMap;
    use std::time::SystemTime;

    use crate::audio_transcription::transformation::AudioTranscriptionAuth;
    use crate::providers::bedrock::audio_transcription::aws_auth_config;
    use crate::providers::bedrock::aws_base::{resolve_credentials, sign_bedrock_post};

    let AudioTranscriptionAuth::AwsSigV4 { region, .. } = &request.auth else {
        return Ok(request.upstream_headers.clone());
    };
    let env_lookup = |key: &str| std::env::var(key).ok();
    let credentials = resolve_credentials(
        aws_auth_config(&request.optional_params, &env_lookup),
        &env_lookup,
    )
    .await?;
    let unsigned: BTreeMap<String, String> = request.upstream_headers.iter().cloned().collect();
    let signature = sign_bedrock_post(
        &request.url,
        body,
        &unsigned,
        region,
        &credentials,
        SystemTime::now(),
    )?;
    Ok(unsigned.into_iter().chain(signature).collect())
}

#[cfg(not(feature = "bedrock-auth"))]
async fn signed_headers(
    request: &ProviderAudioTranscriptionRequest,
    _body: &[u8],
) -> Result<Vec<(String, String)>, Error> {
    use crate::audio_transcription::transformation::AudioTranscriptionAuth;

    match request.auth {
        AudioTranscriptionAuth::AwsSigV4 { .. } => Err(Error::Unsupported(
            "AWS SigV4 requires the bedrock-auth feature",
        )),
        AudioTranscriptionAuth::Bearer => Ok(request.upstream_headers.clone()),
    }
}
