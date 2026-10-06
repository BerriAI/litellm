mod batch;
mod semantic;

use std::{sync::Arc, time::Duration};

pub use batch::{PartialHits, PendingWrite};
use litellm_cache::{BaseCache, CacheConnectionResult, ConnectionCache, Error, FlushCache};
use serde_json::Value;

use crate::{CacheEntry, ResponseCacheConfig, ResponseCacheRequest, get_cache_key};

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

    pub fn lookup(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.controls.reads() {
            return Ok(None);
        }
        let entry = self
            .backend
            .get_cache(&get_cache_key(&request.key), &request.context);
        fresh_hit(entry, now, request.max_age)
    }

    pub async fn async_lookup(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.controls.reads() {
            return Ok(None);
        }
        let entry = self
            .backend
            .async_get_cache(&get_cache_key(&request.key), &request.context)
            .await;
        fresh_hit(entry, now, request.max_age)
    }

    pub fn store(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        let Some((key, entry)) = self.writable(request, response, now) else {
            return Ok(());
        };
        self.backend.set_cache(&key, entry, &request.context)
    }

    pub async fn async_store(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        let Some((key, entry)) = self.writable(request, response, now) else {
            return Ok(());
        };
        self.backend
            .async_set_cache(&key, entry, request.context.clone())
            .await
    }

    fn writable(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        response: Value,
        produced_at: Duration,
    ) -> Option<(String, CacheEntry)> {
        (request.controls.writes() && self.fits(&response)).then(|| {
            (
                get_cache_key(&request.key),
                CacheEntry::produced_at(response, produced_at),
            )
        })
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
