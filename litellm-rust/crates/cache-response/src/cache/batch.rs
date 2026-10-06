use std::time::Duration;

use litellm_cache::{BaseCache, BatchCache, BatchEntry, CacheContext, Error, ExactCacheContext};
use serde::{Serialize, Serializer, ser::SerializeStruct};
use serde_json::Value;

use crate::{CacheEntry, CacheKey, ResponseCache};

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
    pub key: CacheKey,
    pub context: C,
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
        keys: &[CacheKey],
        context: &B::Context,
        max_age: Option<Duration>,
        now: Duration,
    ) -> Result<BatchLookup<Value>, Error> {
        if keys.is_empty() {
            return Ok(BatchLookup::misses(0));
        }
        let entries = self.backend.batch_get_cache(&key_strings(keys), context)?;
        batch_lookup(keys.len(), entries, max_age, now)
    }

    pub async fn async_lookup_batch(
        &self,
        keys: &[CacheKey],
        context: &B::Context,
        max_age: Option<Duration>,
        now: Duration,
    ) -> Result<BatchLookup<Value>, Error> {
        if keys.is_empty() {
            return Ok(BatchLookup::misses(0));
        }
        let entries = self
            .backend
            .async_batch_get_cache(key_strings(keys), context.clone())
            .await?;
        batch_lookup(keys.len(), entries, max_age, now)
    }
}

impl<B> ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry>,
    B::Context: Default + PartialEq,
{
    pub async fn async_store_batch(
        &self,
        entries: Vec<(CacheKey, Value)>,
        context: B::Context,
        now: Duration,
    ) -> Result<(), Error> {
        self.async_store_entries(
            entries
                .into_iter()
                .map(|(key, response)| PendingWrite {
                    key,
                    context: context.clone(),
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
                let entry = self.writable(write.response, write.produced_at)?;
                Some((String::from(write.key), entry, write.context))
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

fn key_strings(keys: &[CacheKey]) -> Vec<String> {
    keys.iter().map(|key| key.as_str().to_owned()).collect()
}

fn batch_lookup(
    len: usize,
    entries: Vec<BatchEntry<CacheEntry>>,
    max_age: Option<Duration>,
    now: Duration,
) -> Result<BatchLookup<Value>, Error> {
    if entries.len() != len {
        return Err(Error::Unavailable);
    }
    Ok(BatchLookup {
        values: entries
            .into_iter()
            .map(|entry| match entry {
                BatchEntry::Hit(entry) => entry.into_fresh(now, max_age),
                BatchEntry::Miss | BatchEntry::Invalid => None,
            })
            .collect(),
    })
}
