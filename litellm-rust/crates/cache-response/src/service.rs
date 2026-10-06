use std::time::Duration;

use futures_util::{StreamExt, TryStreamExt, future::BoxFuture};
use litellm_cache::{BaseCache, BatchCache, Error, ExactCacheContext};
use serde_json::Value;

use crate::{BatchLookup, CacheEntry, ResponseCache, ResponseCacheRequest, get_cache_key};

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

    fn get_cache_key<'a>(&'a self, request: &'a ResponseCacheRequest) -> CacheFuture<'a, String> {
        Box::pin(async move { Ok(get_cache_key(&request.key)) })
    }

    fn lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> CacheFuture<'a, Option<Value>>;

    fn lookup_batch<'a>(
        &'a self,
        requests: &'a [ResponseCacheRequest],
        now: Duration,
    ) -> CacheFuture<'a, BatchLookup<Value>> {
        Box::pin(async move {
            let values = futures_util::stream::iter(requests)
                .then(|request| self.lookup(request, now))
                .try_collect()
                .await?;
            Ok(BatchLookup { values })
        })
    }

    fn store<'a>(
        &'a self,
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

    fn lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> CacheFuture<'a, Option<Value>> {
        Box::pin(self.async_lookup(request, now))
    }

    fn lookup_batch<'a>(
        &'a self,
        requests: &'a [ResponseCacheRequest],
        now: Duration,
    ) -> CacheFuture<'a, BatchLookup<Value>> {
        Box::pin(self.async_lookup_batch(requests, now))
    }

    fn store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> CacheFuture<'a, ()> {
        Box::pin(self.async_store(request, response, now))
    }
}
