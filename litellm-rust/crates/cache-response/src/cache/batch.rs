use std::time::Duration;

use litellm_cache::{BaseCache, BatchCache, BatchEntry, CacheContext, Error, ExactCacheContext};
use serde::{Serialize, Serializer, ser::SerializeStruct};
use serde_json::Value;

use crate::{CacheEntry, CacheKey, ResponseCache, ResponseCacheRequest};

#[derive(Clone, Debug, PartialEq)]
pub struct BatchLookup<T> {
    pub values: Vec<Option<T>>,
}

impl<T> BatchLookup<T> {
    pub fn misses(len: usize) -> Self {
        Self {
            values: std::iter::repeat_with(|| None).take(len).collect(),
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

impl<T: Serialize> Serialize for BatchLookup<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut hits = serializer.serialize_struct("BatchLookup", 2)?;
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
    ) -> Result<BatchLookup<Value>, Error> {
        let keys = self.keys(requests);
        let readable = readable(keys.iter().zip(requests));
        let entries = match readable.first() {
            Some((_, _, request)) => self
                .backend
                .batch_get_cache(&key_strings(&readable), &request.context)?,
            None => Vec::new(),
        };
        batch_lookup(requests.len(), readable, entries, now)
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[ResponseCacheRequest<B::Context>],
        now: Duration,
    ) -> Result<BatchLookup<Value>, Error> {
        let keys = self.keys(requests);
        self.async_lookup_keyed_batch(keys.iter().zip(requests), now)
            .await
    }

    pub(crate) async fn async_lookup_keyed_batch<'a>(
        &self,
        requests: impl ExactSizeIterator<Item = (&'a CacheKey, &'a ResponseCacheRequest<B::Context>)>,
        now: Duration,
    ) -> Result<BatchLookup<Value>, Error>
    where
        B::Context: 'a,
    {
        let len = requests.len();
        let readable = readable(requests);
        let entries = match readable.first() {
            Some((_, _, request)) => {
                self.backend
                    .async_batch_get_cache(key_strings(&readable), request.context.clone())
                    .await?
            }
            None => Vec::new(),
        };
        batch_lookup(len, readable, entries, now)
    }

    fn keys(&self, requests: &[ResponseCacheRequest<B::Context>]) -> Vec<CacheKey> {
        requests.iter().map(|request| self.key(request)).collect()
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
                let entry = self.writable(&write.request, write.response, write.produced_at)?;
                Some((
                    String::from(self.key(&write.request)),
                    entry,
                    write.request.context,
                ))
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

type Readable<'a, C> = Vec<(usize, &'a CacheKey, &'a ResponseCacheRequest<C>)>;

fn readable<'a, C: CacheContext + 'a>(
    requests: impl Iterator<Item = (&'a CacheKey, &'a ResponseCacheRequest<C>)>,
) -> Readable<'a, C> {
    requests
        .enumerate()
        .filter(|(_, (_, request))| request.access.reads)
        .map(|(index, (key, request))| (index, key, request))
        .collect()
}

fn key_strings<C: CacheContext>(readable: &Readable<'_, C>) -> Vec<String> {
    readable
        .iter()
        .map(|(_, key, _)| key.as_str().to_owned())
        .collect()
}

fn batch_lookup<C: CacheContext>(
    len: usize,
    readable: Readable<'_, C>,
    entries: Vec<BatchEntry<CacheEntry>>,
    now: Duration,
) -> Result<BatchLookup<Value>, Error> {
    if readable.len() != entries.len() {
        return Err(Error::Unavailable);
    }
    let mut hits = BatchLookup::misses(len);
    for ((index, _, request), entry) in readable.into_iter().zip(entries) {
        if let BatchEntry::Hit(entry) = entry {
            hits.values[index] = entry.into_fresh(now, request.max_age);
        }
    }
    Ok(hits)
}
