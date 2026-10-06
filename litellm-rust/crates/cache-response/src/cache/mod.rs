mod batch;
mod semantic;

use std::{sync::Arc, time::Duration};

pub use batch::{BatchLookup, PendingWrite};
use litellm_cache::{BaseCache, CacheConnectionResult, ConnectionCache, Error, FlushCache};
use serde_json::Value;

use crate::{CacheEntry, CacheKey, ResponseCacheConfig, ResponseCacheRequest};

pub struct ResponseCache<B: BaseCache<Value = CacheEntry>>
where
    B::Context: Default + PartialEq,
{
    backend: Arc<B>,
    config: ResponseCacheConfig,
}

impl<B> ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry>,
    B::Context: Default + PartialEq,
{
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            backend,
            config: ResponseCacheConfig::default(),
        }
    }

    pub fn with_config(self, config: ResponseCacheConfig) -> Self {
        Self { config, ..self }
    }

    pub fn config(&self) -> &ResponseCacheConfig {
        &self.config
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn default_ttl(&self) -> Option<Duration> {
        self.backend.get_ttl(&B::Context::default())
    }

    pub async fn async_flush(&self) -> Result<(), Error>
    where
        B: FlushCache,
    {
        self.backend.async_flush_cache().await
    }

    pub async fn test_connection(&self) -> Result<CacheConnectionResult, Error>
    where
        B: ConnectionCache,
    {
        self.backend.test_connection().await
    }

    pub fn key(&self, request: &ResponseCacheRequest<B::Context>) -> CacheKey {
        CacheKey::derive(&request.key)
    }

    pub fn lookup(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.access.reads {
            return Ok(None);
        }
        let entry = self
            .backend
            .get_cache(self.key(request).as_str(), &request.context);
        fresh_hit(entry, now, request.max_age)
    }

    pub async fn async_lookup(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        self.async_lookup_keyed(&self.key(request), request, now)
            .await
    }

    pub(crate) async fn async_lookup_keyed(
        &self,
        key: &CacheKey,
        request: &ResponseCacheRequest<B::Context>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.access.reads {
            return Ok(None);
        }
        let entry = self
            .backend
            .async_get_cache(key.as_str(), &request.context)
            .await;
        fresh_hit(entry, now, request.max_age)
    }

    pub fn store(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        let Some(entry) = self.writable(request, response, now) else {
            return Ok(());
        };
        self.backend
            .set_cache(self.key(request).as_str(), entry, &request.context)
    }

    pub async fn async_store(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        self.async_store_keyed(&self.key(request), request, response, now)
            .await
    }

    pub(crate) async fn async_store_keyed(
        &self,
        key: &CacheKey,
        request: &ResponseCacheRequest<B::Context>,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        let Some(entry) = self.writable(request, response, now) else {
            return Ok(());
        };
        self.backend
            .async_set_cache(key.as_str(), entry, request.context.clone())
            .await
    }

    fn writable(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        response: Value,
        produced_at: Duration,
    ) -> Option<CacheEntry> {
        (request.access.writes && self.fits(&response))
            .then(|| CacheEntry::produced_at(response, produced_at))
    }

    fn fits(&self, response: &Value) -> bool {
        self.config.max_entry_bytes == usize::MAX
            || response.to_string().len() <= self.config.max_entry_bytes
    }
}

fn fresh_hit(
    entry: Result<Option<CacheEntry>, Error>,
    now: Duration,
    max_age: Option<Duration>,
) -> Result<Option<Value>, Error> {
    match entry {
        Ok(entry) => Ok(entry.and_then(|entry| entry.into_fresh(now, max_age))),
        Err(Error::InvalidEntry) => Ok(None),
        Err(error) => Err(error),
    }
}
