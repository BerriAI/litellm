use std::time::Duration;

use litellm_redis_client::commands;

use super::DualCache;
use crate::{Error, FromValue, RequestRedis, codec, failure_log::redis_failure};

impl<V: Clone + FromValue, R> DualCache<V, R> {
    /// `DualCache.async_get_cache`: memory, else `None` for a key `request` already read as missing
    /// from Redis, else a `GET` written back to memory for `ttl` or `default_in_memory_ttl`. A failed
    /// or undecodable Redis read is a miss, as there.
    pub(crate) async fn get(
        &self,
        request: Option<&RequestRedis>,
        key: &str,
        ttl: Option<Duration>,
    ) -> Result<Option<V>, Error> {
        if let Some(value) = self.memory.get(key)? {
            return Ok(Some(value));
        }
        self.backfill(key, self.read_remote(request, key).await, ttl)
    }

    pub(crate) async fn read_remote(&self, request: Option<&RequestRedis>, key: &str) -> Option<V> {
        let client = self.redis.client()?;
        if request.is_some_and(|request| request.misses(client).contains(key)) {
            return None;
        }
        client
            .send(commands::get::<Vec<u8>>(key))
            .await
            .map_err(Error::from)
            .and_then(|raw| Ok(raw.as_deref().map(codec::decode).transpose()?))
            .unwrap_or_else(|error| {
                redis_failure!(&error, "LiteLLM Cache: exception in async_get_cache");
                None
            })
    }
}

impl<V: Clone, R> DualCache<V, R> {
    /// `_backfill_kwargs`: a value read from Redis stays in memory for `ttl` or
    /// `default_in_memory_ttl`.
    pub(super) fn backfill(
        &self,
        key: &str,
        value: Option<V>,
        ttl: Option<Duration>,
    ) -> Result<Option<V>, Error> {
        if let Some(value) = &value {
            self.memory
                .set(key, value.clone(), ttl.or(self.default_in_memory_ttl()))?;
        }
        Ok(value)
    }
}
