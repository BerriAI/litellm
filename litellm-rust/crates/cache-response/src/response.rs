use std::{sync::Arc, time::Duration};

use litellm_cache::{BaseCache, CacheKwargs, Error};

use crate::{CacheControls, CacheEntry, CacheKeyInput, cache_key};
use serde_json::Value;

#[derive(Clone)]
pub struct ResponseCacheRequest {
    pub key: CacheKeyInput,
    pub controls: CacheControls,
    pub kwargs: CacheKwargs,
    pub max_age: Option<Duration>,
}

impl ResponseCacheRequest {
    pub fn new(key: CacheKeyInput) -> Self {
        Self {
            key,
            controls: CacheControls {
                configured: true,
                supported_call_type: true,
                native_backend: true,
                default_on: true,
                ..Default::default()
            },
            kwargs: CacheKwargs::default(),
            max_age: None,
        }
    }
}

pub struct ResponseCache<B: BaseCache<Value = CacheEntry>> {
    backend: Arc<B>,
}

impl<B: BaseCache<Value = CacheEntry>> ResponseCache<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    pub fn lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.controls.reads() {
            return Ok(None);
        }
        let entry = self
            .backend
            .get_cache(&cache_key(&request.key), &request.kwargs)?;
        Self::fresh_response(entry, now, request.max_age)
    }

    pub async fn async_lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.controls.reads() {
            return Ok(None);
        }
        let entry = self
            .backend
            .async_get_cache(&cache_key(&request.key), &request.kwargs)
            .await?;
        Self::fresh_response(entry, now, request.max_age)
    }

    pub fn store(
        &self,
        request: &ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        if !request.controls.writes() {
            return Ok(());
        }
        self.backend.set_cache(
            &cache_key(&request.key),
            CacheEntry {
                timestamp: now.as_secs_f64(),
                response,
            },
            request.kwargs.clone(),
        )
    }

    pub async fn async_store(
        &self,
        request: &ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        if !request.controls.writes() {
            return Ok(());
        }
        self.backend
            .async_set_cache(
                &cache_key(&request.key),
                CacheEntry {
                    timestamp: now.as_secs_f64(),
                    response,
                },
                request.kwargs.clone(),
            )
            .await
    }

    fn fresh_response(
        entry: Option<CacheEntry>,
        now: Duration,
        max_age: Option<Duration>,
    ) -> Result<Option<Value>, Error> {
        entry
            .filter(|entry| entry.fresh(now, max_age))
            .map(|entry| match entry.response {
                Value::String(text) => crate::codec::decode_value(&text),
                value => Ok(value),
            })
            .transpose()
    }
}
