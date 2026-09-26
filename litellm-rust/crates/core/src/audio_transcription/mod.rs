mod error;
pub mod types;
pub use error::Error;
mod handler;
mod prepare;
pub use handler::execute_audio_transcription_provider_call;
use litellm_http::{ClientVariant, HttpClientConfig, HttpClientPool};
pub use prepare::prepare_audio_transcription_provider_call;
use serde_json::Value;

use crate::audio_transcription::types::AudioTranscriptionRequest;

pub async fn audio_transcription(
    pool: &HttpClientPool,
    config: &HttpClientConfig,
    request: AudioTranscriptionRequest<'_>,
) -> Result<Value, Error> {
    let request = prepare_audio_transcription_provider_call(request)?;
    let http = pool.client(config, ClientVariant::Provider)?;
    execute_audio_transcription_provider_call(&http, request).await
}
