use std::sync::Arc;

use litellm_cache_response::{CacheKeyInput, CacheOptions, ResponseCacheService};

use super::CacheKeyProjection;
use crate::RouteError;

pub struct CachePlan {
    pub(super) service: Arc<dyn ResponseCacheService>,
    pub(super) options: CacheOptions,
    pub(super) input: CacheKeyInput,
}

impl CachePlan {
    pub fn new(
        service: Arc<dyn ResponseCacheService>,
        options: CacheOptions,
        input: CacheKeyInput,
    ) -> Option<Self> {
        options.policy.enabled().then_some(Self {
            service,
            options,
            input,
        })
    }

    pub fn for_request(
        service: Option<&Arc<dyn ResponseCacheService>>,
        options: Option<CacheOptions>,
        request: &impl CacheKeyProjection,
    ) -> Result<Option<Self>, RouteError> {
        let (Some(service), Some(options)) = (service, options) else {
            return Ok(None);
        };
        if !options.policy.enabled() {
            return Ok(None);
        }
        Ok(Some(Self {
            service: service.clone(),
            options,
            input: request.cache_key_input()?,
        }))
    }
}
