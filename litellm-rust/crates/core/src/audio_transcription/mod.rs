pub mod types;
pub use crate::error::RouteError as Error;
mod handler;
mod prepare;
pub use handler::execute_audio_transcription_provider_call;
use litellm_http::{ClientVariant, HttpClientConfig};
pub use prepare::prepare_audio_transcription_provider_call;
use serde_json::Value;

use crate::audio_transcription::types::AudioTranscriptionRequest;

pub async fn audio_transcription(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    request: AudioTranscriptionRequest<'_>,
) -> Result<Value, Error> {
    let request = prepare_audio_transcription_provider_call(request)?;
    let http = resources.pool.client(config, ClientVariant::Provider)?;
    execute_audio_transcription_provider_call(&http, &resources.auth, request).await
}
