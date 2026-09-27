pub mod types;
pub use crate::error::RouteError as Error;
mod handler;
mod prepare;
pub use handler::execute_audio_transcription_provider_call;
use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
pub use prepare::prepare_audio_transcription_provider_call;
use serde_json::Value;
use std::sync::Arc;

use crate::audio_transcription::types::AudioTranscriptionRequest;

#[derive(Clone)]
pub struct AudioTranscriptionRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

impl AudioTranscriptionRoute {
    pub fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            auth,
            secrets,
        }
    }

    pub async fn execute(&self, request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
        let request =
            prepare_audio_transcription_provider_call(request, self.secrets.as_ref()).await?;
        execute_audio_transcription_provider_call(&self.http, &self.auth, request).await
    }
}
