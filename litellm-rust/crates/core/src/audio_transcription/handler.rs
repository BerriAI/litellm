use serde_json::Value;

use crate::constants::AUDIO_TRANSCRIPTION_TIMEOUT_SECS;
use crate::error::Error;
use crate::http_utils::body::PreparedJsonBody;
use crate::http_utils::replay::{BodySigner, replay_client, send_json};
use crate::http_utils::truncate_error_body;
use std::time::Duration;

use super::client::http_client;
use super::types::ProviderAudioTranscriptionRequest;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn execute_audio_transcription_provider_call(
    request: ProviderAudioTranscriptionRequest,
) -> Result<Value, Error> {
    let signer = request_signer(&request).await?;
    let body = PreparedJsonBody::new(request.body.clone())?;
    let response = send_json(
        if body.is_streamed() {
            replay_client()?
        } else {
            http_client()
        },
        &request.url,
        &body,
        &request.upstream_headers,
        request
            .timeout
            .unwrap_or(Duration::from_secs(AUDIO_TRANSCRIPTION_TIMEOUT_SECS)),
        signer.as_deref(),
    )
    .await?;
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
async fn request_signer(
    request: &ProviderAudioTranscriptionRequest,
) -> Result<Option<Box<BodySigner<'static>>>, Error> {
    use crate::audio_transcription::transformation::AudioTranscriptionAuth;
    use crate::providers::bedrock::audio_transcription::aws_auth_config;
    use crate::providers::bedrock::aws_base::{resolve_credentials, sign_bedrock_digest};
    use std::collections::BTreeMap;
    use std::time::SystemTime;

    let AudioTranscriptionAuth::AwsSigV4 { region, .. } = &request.auth else {
        return Ok(None);
    };
    let env_lookup = |key: &str| std::env::var(key).ok();
    let credentials = resolve_credentials(
        aws_auth_config(&request.optional_params, &env_lookup),
        &env_lookup,
    )
    .await?;
    let region = region.clone();
    Ok(Some(Box::new(move |url, digest, headers| {
        let unsigned = headers
            .iter()
            .filter(|(name, _)| {
                !matches!(
                    name.as_str(),
                    "authorization" | "x-amz-date" | "x-amz-security-token" | "host"
                )
            })
            .map(|(name, value)| {
                Ok((
                    name.to_string(),
                    value
                        .to_str()
                        .map_err(|_| Error::InvalidRequest("invalid signing header".into()))?
                        .to_owned(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        let signed = sign_bedrock_digest(
            url.as_str(),
            digest,
            &unsigned,
            &region,
            &credentials,
            SystemTime::now(),
        )?;
        let mut result = headers.clone();
        for (name, value) in signed {
            result.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| Error::Auth("invalid signing header".into()))?,
                reqwest::header::HeaderValue::from_str(&value)
                    .map_err(|_| Error::Auth("invalid signing value".into()))?,
            );
        }
        Ok(result)
    })))
}

#[cfg(not(feature = "bedrock-auth"))]
async fn request_signer(
    request: &ProviderAudioTranscriptionRequest,
) -> Result<Option<Box<BodySigner<'static>>>, Error> {
    use crate::audio_transcription::transformation::AudioTranscriptionAuth;
    match request.auth {
        AudioTranscriptionAuth::AwsSigV4 { .. } => Err(Error::Unsupported(
            "AWS SigV4 requires the bedrock-auth feature",
        )),
        AudioTranscriptionAuth::Bearer => Ok(None),
    }
}
