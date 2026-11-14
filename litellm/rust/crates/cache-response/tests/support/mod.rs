use std::{sync::Arc, time::Duration};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{CacheEntry, CacheKeyInput, ResponseCache, ResponseCacheRequest};
use rstest::fixture;

#[fixture]
pub fn memory() -> Arc<ResponseCache<InMemoryCache<CacheEntry>>> {
    Arc::new(ResponseCache::new(Arc::new(InMemoryCache::new(
        Some(8),
        Some(Duration::from_secs(600)),
    ))))
}

#[fixture]
pub fn request() -> ResponseCacheRequest {
    keyed("tenant:key")
}

pub fn keyed(key: &str) -> ResponseCacheRequest {
    ResponseCacheRequest::new(CacheKeyInput {
        preset: Some(key.into()),
        ..Default::default()
    })
}
