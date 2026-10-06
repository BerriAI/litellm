mod batch;
mod semantic;

use std::{sync::Arc, time::Duration};

pub use batch::{BatchLookup, PendingWrite};
use litellm_cache::{BaseCache, CacheConnectionResult, ConnectionCache, Error, FlushCache};
use serde_json::Value;

use crate::{
    CacheEntry, CacheKey, CacheKeyInput, CacheScope, ResponseCacheConfig, key::KeyContext,
};

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

    pub fn key(&self, input: &CacheKeyInput, scope: &CacheScope) -> CacheKey {
        CacheKey::derive(
            input,
            &KeyContext {
                namespace: &self.config.namespace,
                scope,
            },
        )
    }

    pub fn lookup(
        &self,
        key: &CacheKey,
        context: &B::Context,
        max_age: Option<Duration>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        fresh_hit(self.backend.get_cache(key.as_str(), context), now, max_age)
    }

    pub async fn async_lookup(
        &self,
        key: &CacheKey,
        context: &B::Context,
        max_age: Option<Duration>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        let entry = self.backend.async_get_cache(key.as_str(), context).await;
        fresh_hit(entry, now, max_age)
    }

    pub fn store(
        &self,
        key: &CacheKey,
        context: &B::Context,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        let Some(entry) = self.writable(response, now) else {
            return Ok(());
        };
        self.backend.set_cache(key.as_str(), entry, context)
    }

    pub async fn async_store(
        &self,
        key: &CacheKey,
        context: B::Context,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        let Some(entry) = self.writable(response, now) else {
            return Ok(());
        };
        self.backend
            .async_set_cache(key.as_str(), entry, context)
            .await
    }

    fn writable(&self, response: Value, produced_at: Duration) -> Option<CacheEntry> {
        self.fits(&response)
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
