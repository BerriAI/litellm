use serde_json::Value;
use std::{sync::Mutex, time::Duration};

use litellm_cache::Error;

use crate::{ExactResponseCache, PendingWrite, ResponseCacheRequest};

pub struct WriteBuffer {
    flush_size: usize,
    entries: Mutex<Vec<PendingWrite>>,
}

impl WriteBuffer {
    pub fn new(flush_size: usize) -> Self {
        Self {
            flush_size: flush_size.max(1),
            entries: Mutex::new(Vec::new()),
        }
    }

    pub async fn async_store(
        &self,
        cache: &dyn ExactResponseCache,
        request: &ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        let pending = {
            let mut entries = self.entries.lock().map_err(|_| Error::Unavailable)?;
            entries.push(PendingWrite {
                request: request.clone(),
                response,
                produced_at: now,
            });
            (entries.len() >= self.flush_size).then(|| std::mem::take(&mut *entries))
        };
        match pending {
            Some(pending) => cache.async_store_entries(pending).await,
            None => Ok(()),
        }
    }

    pub fn clear(&self) -> Result<(), Error> {
        self.entries.lock().map_err(|_| Error::Unavailable)?.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_cache::{BaseCache, BatchCache, ExactCacheContext, FlushCache};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{CacheEntry, CacheKeyInput, ResponseCache};

    struct FailingBackend {
        failed_flushes: usize,
        batches: Mutex<Vec<Vec<Value>>>,
    }

    impl BaseCache for FailingBackend {
        type Value = CacheEntry;
        type Context = ExactCacheContext;

        fn get_ttl(&self, _: &Self::Context) -> Option<Duration> {
            None
        }

        fn set_cache(&self, _: &str, _: Self::Value, _: &Self::Context) -> Result<(), Error> {
            unreachable!()
        }

        fn get_cache(&self, _: &str, _: &Self::Context) -> Result<Option<Self::Value>, Error> {
            unreachable!()
        }

        async fn async_set_cache_pipeline(
            &self,
            entries: Vec<(String, Self::Value)>,
            _: Self::Context,
        ) -> Result<(), Error> {
            let mut batches = self.batches.lock().unwrap();
            batches.push(
                entries
                    .into_iter()
                    .map(|(_, entry)| entry.response)
                    .collect(),
            );
            if batches.len() <= self.failed_flushes {
                Err(Error::Unavailable)
            } else {
                Ok(())
            }
        }
    }

    impl BatchCache for FailingBackend {}

    impl FlushCache for FailingBackend {
        fn flush_cache(&self) -> Result<(), Error> {
            unreachable!()
        }
    }

    #[rstest]
    #[case::one_failed_flush(1)]
    #[case::repeated_failed_flushes(3)]
    #[tokio::test]
    async fn failed_flush_drops_its_batch_without_growing_or_resending_it(
        #[case] failed_flushes: usize,
    ) {
        let backend = Arc::new(FailingBackend {
            failed_flushes,
            batches: Mutex::new(Vec::new()),
        });
        let cache = ResponseCache::new(backend.clone());
        let buffer = WriteBuffer::new(2);
        let request = ResponseCacheRequest::new(CacheKeyInput::default());

        for batch in 0..=failed_flushes {
            let first = json!(batch * 2);
            let second = json!(batch * 2 + 1);
            assert_eq!(
                buffer
                    .async_store(&cache, &request, first, Duration::ZERO)
                    .await,
                Ok(())
            );
            assert_eq!(backend.batches.lock().unwrap().len(), batch);
            assert_eq!(
                buffer
                    .async_store(&cache, &request, second, Duration::ZERO)
                    .await,
                if batch < failed_flushes {
                    Err(Error::Unavailable)
                } else {
                    Ok(())
                }
            );
            assert_eq!(
                backend.batches.lock().unwrap().as_slice(),
                (0..=batch)
                    .map(|index| vec![json!(index * 2), json!(index * 2 + 1)])
                    .collect::<Vec<_>>()
                    .as_slice()
            );
            assert!(buffer.entries.lock().unwrap().is_empty());
        }
    }
}
