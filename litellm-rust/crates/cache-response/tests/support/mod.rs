use std::{sync::Arc, time::Duration};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::RedisCache;
use litellm_cache_response::{
    CacheEntry, CacheKeyInput, ResponseCache, ResponseCacheCodec, ResponseCacheRequest,
};
use redis_test::{MockCmd, MockRedisConnection};
use rstest::fixture;

pub type MockedRedis = RedisCache<ResponseCacheCodec, MockRedisConnection>;

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

/// A Redis response cache that must receive exactly `commands`, in order.
pub fn redis(commands: Vec<MockCmd>, namespace: Option<&str>) -> ResponseCache<MockedRedis> {
    let connection = MockRedisConnection::new(commands).assert_all_commands_consumed();
    ResponseCache::new(Arc::new(
        RedisCache::with_connection(connection, None, ResponseCacheCodec)
            .with_namespace(namespace.map(str::to_owned)),
    ))
}
