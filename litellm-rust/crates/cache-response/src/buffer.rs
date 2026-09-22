use std::{sync::Mutex, time::Duration};

use litellm_cache::Error;
use serde_json::Value;

use crate::{ExactResponseCache, ResponseCacheRequest};

pub struct WriteBuffer {
    flush_size: usize,
    entries: Mutex<Vec<(ResponseCacheRequest, Value, Duration)>>,
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
            entries.push((request.clone(), response, now));
            (entries.len() >= self.flush_size).then(|| std::mem::take(&mut *entries))
        };
        // A failed flush drops its batch, as Python does. Requeueing would grow the
        // buffer and re-send an ever larger pipeline on every write during an outage.
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
