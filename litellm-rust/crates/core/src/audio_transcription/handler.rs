use serde_json::Value;

use crate::error::Error;
use crate::http_utils::{HttpClientProfile, http_client, http_request, truncate_error_body};

use super::types::ProviderAudioTranscriptionRequest;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) async fn execute_audio_transcription_provider_call<S>(
    services: &S,
    request: ProviderAudioTranscriptionRequest,
) -> Result<Value, Error>
where
    S: crate::providers::auth::AuthorizationServices,
{
    if request.config.request_body_policy()
        != crate::lifecycle::RequestBodyPolicy::StructuredAtBuild
    {
        return Err(Error::Unsupported(
            "audio transcription request body policy",
        ));
    }
    let body = crate::lifecycle::WireBody::encode(&request.body, "audio request body")?;
    let authorized = signed_body(services, &request, body).await?;
    let pre_call = crate::lifecycle::PreCallBody::StructuredAtBuild {
        callback: request.body,
        authorized,
    };
    let crate::lifecycle::PreCallBody::StructuredAtBuild { authorized, .. } = pre_call else {
        unreachable!("audio body policy was checked before capture")
    };
    let settled = authorized.settle();
    let (body, headers) = settled.into_parts();
    let client = http_client(HttpClientProfile::NoConnectTimeout)
        .map_err(|error| Error::Network(error.to_string()))?;
    let request_builder = headers.into_iter().fold(
        client.post(&request.url).body(body),
        |builder, (key, value)| builder.header(key, value),
    );
    let request_builder = match request.timeout {
        Some(timeout) => request_builder.timeout(timeout),
        None => request_builder,
    };
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

async fn signed_body<S>(
    services: &S,
    request: &ProviderAudioTranscriptionRequest,
    body: crate::lifecycle::WireBody,
) -> Result<crate::lifecycle::AuthorizedBody, Error>
where
    S: crate::providers::auth::AuthorizationServices,
{
    request
        .config
        .authorize(services, request.authorization_context(), body)
        .await
}
