use std::time::Duration;

use futures_util::{StreamExt, TryStreamExt, future::BoxFuture};
use litellm_cache::{BaseCache, BatchCache, Error, ExactCacheContext};
use serde_json::Value;

use crate::{BatchLookup, CacheEntry, CacheKey, CacheKeyInput, CacheScope, ResponseCache};

type CacheFuture<'a, T> = BoxFuture<'a, Result<T, Error>>;

#[derive(Clone)]
pub struct ResponseCacheConfig {
    pub namespace: String,
    pub max_entry_bytes: usize,
}

impl Default for ResponseCacheConfig {
    fn default() -> Self {
        Self {
            namespace: String::new(),
            max_entry_bytes: usize::MAX,
        }
    }
}

pub trait ResponseCacheService: Send + Sync {
    fn config(&self) -> &ResponseCacheConfig;

    fn key<'a>(
        &'a self,
        input: &'a CacheKeyInput,
        scope: &'a CacheScope,
    ) -> CacheFuture<'a, CacheKey>;

    fn lookup<'a>(
        &'a self,
        key: &'a CacheKey,
        max_age: Option<Duration>,
        now: Duration,
    ) -> CacheFuture<'a, Option<Value>>;

    fn lookup_batch<'a>(
        &'a self,
        keys: &'a [CacheKey],
        max_age: Option<Duration>,
        now: Duration,
    ) -> CacheFuture<'a, BatchLookup<Value>> {
        Box::pin(async move {
            let values = futures_util::stream::iter(keys)
                .then(|key| self.lookup(key, max_age, now))
                .try_collect()
                .await?;
            Ok(BatchLookup { values })
        })
    }

    fn store<'a>(
        &'a self,
        key: &'a CacheKey,
        ttl: Option<Duration>,
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

    fn key<'a>(
        &'a self,
        input: &'a CacheKeyInput,
        scope: &'a CacheScope,
    ) -> CacheFuture<'a, CacheKey> {
        Box::pin(async move { Ok(ResponseCache::key(self, input, scope)) })
    }

    fn lookup<'a>(
        &'a self,
        key: &'a CacheKey,
        max_age: Option<Duration>,
        now: Duration,
    ) -> CacheFuture<'a, Option<Value>> {
        Box::pin(async move {
            self.async_lookup(key, &ExactCacheContext::default(), max_age, now)
                .await
        })
    }

    fn lookup_batch<'a>(
        &'a self,
        keys: &'a [CacheKey],
        max_age: Option<Duration>,
        now: Duration,
    ) -> CacheFuture<'a, BatchLookup<Value>> {
        Box::pin(async move {
            self.async_lookup_batch(keys, &ExactCacheContext::default(), max_age, now)
                .await
        })
    }

    fn store<'a>(
        &'a self,
        key: &'a CacheKey,
        ttl: Option<Duration>,
        response: Value,
        now: Duration,
    ) -> CacheFuture<'a, ()> {
        Box::pin(self.async_store(key, ExactCacheContext { ttl }, response, now))
    }
}
