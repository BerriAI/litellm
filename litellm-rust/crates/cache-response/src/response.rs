use std::{sync::Arc, time::Duration};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheConnectionResult, CacheContext, ConnectionCache, Error,
    FlushCache,
    semantic::{SemanticCache, SemanticLookup},
};
use serde_json::Value;

use crate::{CacheControls, CacheEntry, CacheKeyInput, PartialHits, cache_key};

#[derive(Clone)]
pub struct ResponseCacheRequest<C: CacheContext = litellm_cache::ExactCacheContext> {
    pub key: CacheKeyInput,
    pub controls: CacheControls,
    pub context: C,
    pub max_age: Option<Duration>,
}

impl<C: CacheContext + Default> ResponseCacheRequest<C> {
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
            context: C::default(),
            max_age: None,
        }
    }
}

impl<C: CacheContext> ResponseCacheRequest<C> {
    pub fn with_context<D: CacheContext>(self, context: D) -> ResponseCacheRequest<D> {
        ResponseCacheRequest {
            key: self.key,
            controls: self.controls,
            context,
            max_age: self.max_age,
        }
    }
}

pub struct ResponseCache<B: BaseCache<Value = CacheEntry>>
where
    B::Context: Default + PartialEq,
{
    backend: Arc<B>,
}

impl<B> ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry>,
    B::Context: Default + PartialEq,
{
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_arc(&self) -> &Arc<B> {
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
        let entry = match self
            .backend
            .get_cache(&cache_key(&request.key), &request.context)
        {
            Ok(entry) => entry,
            Err(Error::InvalidEntry) => None,
            Err(error) => return Err(error),
        };
        Ok(Self::fresh_or_miss(entry, now, request.max_age))
    }

    pub async fn async_lookup(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        if !request.controls.reads() {
            return Ok(None);
        }
        let entry = match self
            .backend
            .async_get_cache(&cache_key(&request.key), &request.context)
            .await
        {
            Ok(entry) => entry,
            Err(Error::InvalidEntry) => None,
            Err(error) => return Err(error),
        };
        Ok(Self::fresh_or_miss(entry, now, request.max_age))
    }

    /// `lookup` plus the similarity the semantic backend reports. Freshness applies to the
    /// value only: Python stamps the similarity before its max-age check.
    pub fn lookup_semantic(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        now: Duration,
    ) -> Result<SemanticLookup<Value>, Error>
    where
        B: SemanticCache,
    {
        if !request.controls.reads() {
            return Ok(SemanticLookup::miss(None));
        }
        let lookup = self
            .backend
            .get_cache_with_similarity(&cache_key(&request.key), &request.context);
        Self::fresh_semantic(lookup, now, request.max_age)
    }

    pub async fn async_lookup_semantic(
        &self,
        request: &ResponseCacheRequest<B::Context>,
        now: Duration,
    ) -> Result<SemanticLookup<Value>, Error>
    where
        B: SemanticCache,
    {
        if !request.controls.reads() {
            return Ok(SemanticLookup::miss(None));
        }
        let lookup = self
            .backend
            .async_get_cache_with_similarity(&cache_key(&request.key), &request.context)
            .await;
        Self::fresh_semantic(lookup, now, request.max_age)
    }

    pub fn lookup_batch(
        &self,
        requests: &[ResponseCacheRequest<B::Context>],
        now: Duration,
    ) -> Result<PartialHits, Error>
    where
        B: BatchCache,
    {
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
            self.backend.batch_get_cache(&keys, &request.context)?
        } else {
            Vec::new()
        };
        Self::partial_hits(requests, readable, entries, now)
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[ResponseCacheRequest<B::Context>],
        now: Duration,
    ) -> Result<PartialHits, Error>
    where
        B: BatchCache,
    {
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
                .async_batch_get_cache(keys, request.context.clone())
                .await?
        } else {
            Vec::new()
        };
        Self::partial_hits(requests, readable, entries, now)
    }

    pub fn store(
        &self,
        request: &ResponseCacheRequest<B::Context>,
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
            &request.context,
        )
    }

    pub async fn async_store(
        &self,
        request: &ResponseCacheRequest<B::Context>,
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
                request.context.clone(),
            )
            .await
    }

    pub async fn async_store_batch(
        &self,
        entries: Vec<(ResponseCacheRequest<B::Context>, Value)>,
        now: Duration,
    ) -> Result<(), Error> {
        self.async_store_entries(
            entries
                .into_iter()
                .map(|(request, response)| (request, response, now))
                .collect(),
        )
        .await
    }

    /// Stores entries that each carry the time they were produced, so a deferred write keeps
    /// the freshness of its original response.
    pub async fn async_store_entries(
        &self,
        entries: Vec<(ResponseCacheRequest<B::Context>, Value, Duration)>,
    ) -> Result<(), Error> {
        let writable = entries
            .into_iter()
            .filter(|(request, _, _)| request.controls.writes())
            .map(|(request, response, now)| {
                (
                    cache_key(&request.key),
                    CacheEntry {
                        timestamp: Some(now.as_secs_f64()),
                        response,
                    },
                    request.context,
                )
            })
            .collect::<Vec<_>>();
        let Some((_, _, first_kwargs)) = writable.first() else {
            return Ok(());
        };
        if writable
            .iter()
            .all(|(_, _, context)| context == first_kwargs)
        {
            let context = first_kwargs.clone();
            let cache_list = writable
                .into_iter()
                .map(|(key, entry, _)| (key, entry))
                .collect();
            return self
                .backend
                .async_set_cache_pipeline(cache_list, context)
                .await;
        }
        for (key, entry, context) in writable {
            self.backend.async_set_cache(&key, entry, context).await?;
        }
        Ok(())
    }

    fn partial_hits(
        requests: &[ResponseCacheRequest<B::Context>],
        readable: Vec<(usize, &ResponseCacheRequest<B::Context>)>,
        entries: Vec<BatchEntry<CacheEntry>>,
        now: Duration,
    ) -> Result<PartialHits, Error> {
        if readable.len() != entries.len() {
            return Err(Error::Unavailable);
        }
        let mut values = vec![None; requests.len()];
        for ((index, request), entry) in readable.into_iter().zip(entries) {
            let response = match entry {
                BatchEntry::Hit(entry) => Self::fresh_or_miss(Some(entry), now, request.max_age),
                BatchEntry::Miss | BatchEntry::Invalid => None,
            };
            values[index] = response;
        }
        Ok(PartialHits::new(values))
    }

    fn fresh_semantic(
        lookup: Result<SemanticLookup<CacheEntry>, Error>,
        now: Duration,
        max_age: Option<Duration>,
    ) -> Result<SemanticLookup<Value>, Error> {
        match lookup {
            Ok(lookup) => Ok(SemanticLookup {
                value: Self::fresh_or_miss(lookup.value, now, max_age),
                similarity: lookup.similarity,
            }),
            Err(Error::InvalidEntry) => Ok(SemanticLookup::miss(None)),
            Err(error) => Err(error),
        }
    }

    fn fresh_or_miss(
        entry: Option<CacheEntry>,
        now: Duration,
        max_age: Option<Duration>,
    ) -> Option<Value> {
        entry
            .filter(|entry| entry.fresh(now, max_age))
            .map(|entry| entry.response)
    }
}
