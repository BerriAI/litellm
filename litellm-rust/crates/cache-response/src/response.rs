use std::{sync::Arc, time::Duration};

use litellm_cache::{BaseCache, BatchEntry, CacheConnectionResult, CacheKwargs, Error};

use crate::{CacheControls, CacheEntry, CacheKeyInput, PartialHits, cache_key};
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

    pub fn default_ttl(&self) -> Duration {
        self.backend.default_ttl()
    }

    pub async fn async_flush(&self) -> Result<(), Error> {
        self.backend.async_flush_cache().await
    }

    pub async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        self.backend.test_connection().await
    }

    pub fn lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.controls.reads() {
            return Ok(None);
        }
        let entry = match self
            .backend
            .get_cache(&cache_key(&request.key), &request.kwargs)
        {
            Ok(entry) => entry,
            Err(Error::InvalidEntry) => None,
            Err(error) => return Err(error),
        };
        Self::fresh_or_miss(entry, now, request.max_age)
    }

    pub async fn async_lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.controls.reads() {
            return Ok(None);
        }
        let entry = match self
            .backend
            .async_get_cache(&cache_key(&request.key), &request.kwargs)
            .await
        {
            Ok(entry) => entry,
            Err(Error::InvalidEntry) => None,
            Err(error) => return Err(error),
        };
        Self::fresh_or_miss(entry, now, request.max_age)
    }

    pub fn lookup_batch(
        &self,
        requests: &[ResponseCacheRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        let readable = requests
            .iter()
            .enumerate()
            .filter(|(_, request)| request.controls.reads())
            .collect::<Vec<_>>();
        let keys = readable
            .iter()
            .map(|(_, request)| cache_key(&request.key))
            .collect::<Vec<_>>();
        let entries = if let Some((_, request)) = readable.first() {
            self.backend.get_cache_batch(&keys, &request.kwargs)?
        } else {
            Vec::new()
        };
        Self::partial_hits(requests, readable, entries, now)
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[ResponseCacheRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        let readable = requests
            .iter()
            .enumerate()
            .filter(|(_, request)| request.controls.reads())
            .collect::<Vec<_>>();
        let keys = readable
            .iter()
            .map(|(_, request)| cache_key(&request.key))
            .collect::<Vec<_>>();
        let entries = if let Some((_, request)) = readable.first() {
            self.backend
                .async_get_cache_batch(keys, request.kwargs.clone())
                .await?
        } else {
            Vec::new()
        };
        Self::partial_hits(requests, readable, entries, now)
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
                timestamp: Some(now.as_secs_f64()),
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
                    timestamp: Some(now.as_secs_f64()),
                    response,
                },
                request.kwargs.clone(),
            )
            .await
    }

    pub async fn async_store_batch(
        &self,
        entries: Vec<(ResponseCacheRequest, Value)>,
        now: Duration,
    ) -> Result<(), Error> {
        let writable = entries
            .into_iter()
            .filter(|(request, _)| request.controls.writes())
            .map(|(request, response)| {
                (
                    cache_key(&request.key),
                    CacheEntry {
                        timestamp: Some(now.as_secs_f64()),
                        response,
                    },
                    request.kwargs,
                )
            })
            .collect::<Vec<_>>();
        let Some((_, _, first_kwargs)) = writable.first() else {
            return Ok(());
        };
        if writable.iter().all(|(_, _, kwargs)| kwargs == first_kwargs) {
            let kwargs = first_kwargs.clone();
            let cache_list = writable
                .into_iter()
                .map(|(key, entry, _)| (key, entry))
                .collect();
            return self
                .backend
                .async_set_cache_pipeline(cache_list, kwargs)
                .await;
        }
        for (key, entry, kwargs) in writable {
            self.backend.async_set_cache(&key, entry, kwargs).await?;
        }
        Ok(())
    }

    fn partial_hits(
        requests: &[ResponseCacheRequest],
        readable: Vec<(usize, &ResponseCacheRequest)>,
        entries: Vec<BatchEntry<CacheEntry>>,
        now: Duration,
    ) -> Result<PartialHits, Error> {
        if readable.len() != entries.len() {
            return Err(Error::Unavailable);
        }
        let mut values = vec![None; requests.len()];
        for ((index, request), entry) in readable.into_iter().zip(entries) {
            let response = match entry {
                BatchEntry::Hit(entry) => Self::fresh_or_miss(Some(entry), now, request.max_age)?,
                BatchEntry::Miss | BatchEntry::Invalid => None,
            };
            values[index] = response;
        }
        Ok(PartialHits::new(values))
    }

    fn fresh_or_miss(
        entry: Option<CacheEntry>,
        now: Duration,
        max_age: Option<Duration>,
    ) -> Result<Option<Value>, Error> {
        match Self::fresh_response(entry, now, max_age) {
            Err(Error::InvalidEntry) => Ok(None),
            result => result,
        }
    }

    fn fresh_response(
        entry: Option<CacheEntry>,
        now: Duration,
        max_age: Option<Duration>,
    ) -> Result<Option<Value>, Error> {
        entry
            .filter(|entry| entry.fresh(now, max_age))
            .map(|entry| match (entry.timestamp, entry.response) {
                (Some(_), Value::String(text)) => crate::codec::decode_value(&text),
                (_, value) => Ok(value),
            })
            .transpose()
    }
}
