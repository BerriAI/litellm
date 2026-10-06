pub mod context;
pub mod diagnostic;

pub mod caching;
pub mod error;
pub mod outbound;
pub mod provider;
pub mod resources;

pub use error::RouteError;

#[derive(Clone, Default)]
pub struct CallOptions {
    pub cache: Option<litellm_cache_response::CacheOptions>,
    pub observers: Option<litellm_host::observation::ObservationSender>,
}

impl From<Option<litellm_host::observation::ObservationSender>> for CallOptions {
    fn from(observers: Option<litellm_host::observation::ObservationSender>) -> Self {
        Self {
            observers,
            ..Self::default()
        }
    }
}

impl From<litellm_cache_response::CacheOptions> for CallOptions {
    fn from(cache: litellm_cache_response::CacheOptions) -> Self {
        Self {
            cache: Some(cache),
            ..Self::default()
        }
    }
}
