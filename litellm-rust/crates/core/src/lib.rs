mod diagnostic;

pub mod audio_transcription;
pub mod caching;
pub mod chat_completions;
pub mod constants;
pub mod error;
pub mod messages;
pub mod ocr;
mod outbound;
mod provider;
pub mod resources;
pub mod responses;

pub use error::RouteError;

#[derive(Clone, Default)]
pub struct CallOptions {
    pub cache: Option<litellm_cache_response::CacheOptions>,
    pub observers: Option<litellm_host::observation::ObservationSender>,
}

impl From<Option<litellm_host::observation::ObservationSender>> for CallOptions {
    fn from(observers: Option<litellm_host::observation::ObservationSender>) -> Self {
        Self {
            cache: None,
            observers,
        }
    }
}

impl From<litellm_cache_response::CacheOptions> for CallOptions {
    fn from(cache: litellm_cache_response::CacheOptions) -> Self {
        Self {
            cache: Some(cache),
            observers: None,
        }
    }
}
