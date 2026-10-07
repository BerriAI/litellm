pub mod context;
pub mod diagnostic;

pub mod caching;
pub mod error;
pub mod outbound;
pub mod provider;
pub mod resources;

pub use error::RouteError;
/// The staged call steps every route composes; see `litellm_llms::base_llm::call`.
pub use litellm_llms::base_llm::call;

#[derive(Clone, Default)]
pub struct CallOptions {
    pub cache: Option<litellm_cache_response::CachePolicy>,
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

impl From<litellm_cache_response::CachePolicy> for CallOptions {
    fn from(cache: litellm_cache_response::CachePolicy) -> Self {
        Self {
            cache: Some(cache),
            observers: None,
        }
    }
}
