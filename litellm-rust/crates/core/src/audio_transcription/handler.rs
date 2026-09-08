use serde_json::Value;

use crate::error::Error;
use crate::http_utils::{http_request, truncate_error_body};

use super::client::http_client;
use super::types::ProviderAudioTranscriptionRequest;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) async fn execute_audio_transcription_provider_call(
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
    let response = http_request(request_builder)
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

async fn signed_headers(
    request: &ProviderAudioTranscriptionRequest,
    body: &[u8],
) -> Result<Vec<(String, String)>, Error> {
    request.config.authorize(request, body).await
}
