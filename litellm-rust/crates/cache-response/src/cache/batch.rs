use std::time::Duration;

use litellm_cache::{BaseCache, BatchCache, BatchEntry, CacheContext, Error, ExactCacheContext};
use serde::{Serialize, Serializer, ser::SerializeStruct};
use serde_json::Value;

use crate::{CacheEntry, ResponseCache, ResponseCacheRequest, get_cache_key};

#[derive(Clone, Debug, PartialEq)]
pub struct PartialHits {
    pub values: Vec<Option<Value>>,
}

impl PartialHits {
    pub fn misses(len: usize) -> Self {
        Self {
            values: vec![None; len],
        }
    }

    pub fn missing_indices(&self) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.is_none().then_some(index))
            .collect()
    }
}

impl Serialize for PartialHits {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut hits = serializer.serialize_struct("PartialHits", 2)?;
        hits.serialize_field("values", &self.values)?;
        hits.serialize_field("missing_indices", &self.missing_indices())?;
        hits.end()
    }
}

pub struct PendingWrite<C: CacheContext = ExactCacheContext> {
    pub request: ResponseCacheRequest<C>,
    pub response: Value,
    pub produced_at: Duration,
}

impl<B> ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry> + BatchCache,
    B::Context: Default + PartialEq,
{
    pub fn lookup_batch(
        &self,
        requests: &[ResponseCacheRequest<B::Context>],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        let readable = readable(requests);
        let entries = match readable.first() {
            Some((_, request)) => self
                .backend
                .batch_get_cache(&keys(&readable), &request.context)?,
            None => Vec::new(),
        };
        partial_hits(requests.len(), readable, entries, now)
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[ResponseCacheRequest<B::Context>],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        let readable = readable(requests);
        let entries = match readable.first() {
            Some((_, request)) => {
                self.backend
                    .async_batch_get_cache(keys(&readable), request.context.clone())
                    .await?
            }
            None => Vec::new(),
        };
        partial_hits(requests.len(), readable, entries, now)
    }
}

impl<B> ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry>,
    B::Context: Default + PartialEq,
{
    pub async fn async_store_batch(
        &self,
        entries: Vec<(ResponseCacheRequest<B::Context>, Value)>,
        now: Duration,
    ) -> Result<(), Error> {
        self.async_store_entries(
            entries
                .into_iter()
                .map(|(request, response)| PendingWrite {
                    request,
                    response,
                    produced_at: now,
                })
                .collect(),
        )
        .await
    }

    pub async fn async_store_entries(
        &self,
        writes: Vec<PendingWrite<B::Context>>,
    ) -> Result<(), Error> {
        let writable = writes
            .into_iter()
            .filter_map(|write| {
                let (key, entry) =
                    self.writable(&write.request, write.response, write.produced_at)?;
                Some((key, entry, write.request.context))
            })
            .collect::<Vec<_>>();
        let Some((_, _, first_context)) = writable.first() else {
            return Ok(());
        };
        if writable
            .iter()
            .all(|(_, _, context)| context == first_context)
        {
            let context = first_context.clone();
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
}

fn readable<C: CacheContext>(
    requests: &[ResponseCacheRequest<C>],
) -> Vec<(usize, &ResponseCacheRequest<C>)> {
    requests
        .iter()
        .enumerate()
        .filter(|(_, request)| request.controls.reads())
        .collect()
}

fn keys<C: CacheContext>(readable: &[(usize, &ResponseCacheRequest<C>)]) -> Vec<String> {
    readable
        .iter()
        .map(|(_, request)| get_cache_key(&request.key))
        .collect()
}

fn partial_hits<C: CacheContext>(
    len: usize,
    readable: Vec<(usize, &ResponseCacheRequest<C>)>,
    entries: Vec<BatchEntry<CacheEntry>>,
    now: Duration,
) -> Result<PartialHits, Error> {
    if readable.len() != entries.len() {
        return Err(Error::Unavailable);
    }
    let mut hits = PartialHits::misses(len);
    for ((index, request), entry) in readable.into_iter().zip(entries) {
        if let BatchEntry::Hit(entry) = entry {
            hits.values[index] = entry.into_fresh(now, request.max_age);
        }
    }
    Ok(hits)
}
