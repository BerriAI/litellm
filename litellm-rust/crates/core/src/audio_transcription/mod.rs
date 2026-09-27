pub mod types;
pub use crate::error::RouteError as Error;
mod handler;
mod prepare;
pub use handler::execute_audio_transcription_provider_call;
pub use prepare::prepare_audio_transcription_provider_call;
use serde_json::Value;

use crate::audio_transcription::types::AudioTranscriptionRequest;

impl crate::CoreClient {
    pub async fn audio_transcription(
        &self,
        request: AudioTranscriptionRequest<'_>,
    ) -> Result<Value, Error> {
        let request = prepare_audio_transcription_provider_call(request)?;
        let http = self.provider_http()?;
        execute_audio_transcription_provider_call(&http, &self.resources().auth, request).await
    }
}
