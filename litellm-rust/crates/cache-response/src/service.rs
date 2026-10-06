use std::time::Duration;

use futures_util::{StreamExt, TryStreamExt, future::BoxFuture};
use litellm_cache::{BaseCache, BatchCache, Error, ExactCacheContext};
use serde_json::Value;

use crate::{BatchLookup, CacheEntry, CacheKey, ResponseCache, ResponseCacheRequest};

type CacheFuture<'a, T> = BoxFuture<'a, Result<T, Error>>;

#[derive(Clone)]
pub struct ResponseCacheConfig {
    pub namespace: String,
    pub max_entry_bytes: usize,
    pub supports_isolated_scope: bool,
}

impl Default for ResponseCacheConfig {
    fn default() -> Self {
        Self {
            namespace: String::new(),
            max_entry_bytes: usize::MAX,
            supports_isolated_scope: true,
        }
    }
}

pub trait ResponseCacheService: Send + Sync {
    fn config(&self) -> &ResponseCacheConfig;

    fn key<'a>(&'a self, request: &'a ResponseCacheRequest) -> CacheFuture<'a, CacheKey>;

    fn lookup<'a>(
        &'a self,
        key: &'a CacheKey,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> CacheFuture<'a, Option<Value>>;

    fn lookup_batch<'a>(
        &'a self,
        requests: &'a [(CacheKey, ResponseCacheRequest)],
        now: Duration,
    ) -> CacheFuture<'a, BatchLookup<Value>> {
        Box::pin(async move {
            let values = futures_util::stream::iter(requests)
                .then(|(key, request)| self.lookup(key, request, now))
                .try_collect()
                .await?;
            Ok(BatchLookup { values })
        })
    }

    fn store<'a>(
        &'a self,
        key: &'a CacheKey,
        request: &'a ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> CacheFuture<'a, ()>;
}

impl<B> ResponseCacheService for ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry, Context = ExactCacheContext> + BatchCache,
{
    fn config(&self) -> &ResponseCacheConfig {
        self.config()
    }

    fn key<'a>(&'a self, request: &'a ResponseCacheRequest) -> CacheFuture<'a, CacheKey> {
        Box::pin(async move { Ok(ResponseCache::key(self, request)) })
    }

    fn lookup<'a>(
        &'a self,
        key: &'a CacheKey,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> CacheFuture<'a, Option<Value>> {
        Box::pin(self.async_lookup_keyed(key, request, now))
    }

    fn lookup_batch<'a>(
        &'a self,
        requests: &'a [(CacheKey, ResponseCacheRequest)],
        now: Duration,
    ) -> CacheFuture<'a, BatchLookup<Value>> {
        Box::pin(
            self.async_lookup_keyed_batch(
                requests.iter().map(|(key, request)| (key, request)),
                now,
            ),
        )
    }

    fn store<'a>(
        &'a self,
        key: &'a CacheKey,
        request: &'a ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> CacheFuture<'a, ()> {
        Box::pin(self.async_store_keyed(key, request, response, now))
    }
}
