use std::time::Duration;

use litellm_redis_client::commands;

use super::DualCache;
use crate::{
    Error, IntoValue, Pending, Send, codec,
    failure_log::redis_failure,
    request::{pending, ready},
};

/// `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`, the chunk `async_delete_cache_keys` deletes at once.
const DELETE_CHUNK: usize = 1000;

impl<V: Clone + IntoValue, R> DualCache<V, R> {
    /// `DualCache.async_set_cache`: memory, then a `SET` of `json.dumps(value)`. A Redis failure, or
    /// a value `json.dumps` refuses, only logs.
    pub(crate) async fn set(
        &self,
        key: &str,
        value: V,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        self.set_now(key, value, ttl, codec::dumps).await
    }

    /// `DualCache.set_cache`, whose sync `RedisCache.set_cache` writes `str(value)`.
    pub(crate) async fn set_str(
        &self,
        key: &str,
        value: V,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        self.set_now(key, value, ttl, |value| Ok(codec::str(value)))
            .await
    }

    async fn set_now(
        &self,
        key: &str,
        value: V,
        ttl: Option<Duration>,
        encode: impl FnOnce(V) -> Result<String, Error>,
    ) -> Result<(), Error> {
        if let Err(error) = self
            .declare_encoded(Send::Now, key, value, ttl, encode)?
            .await
        {
            redis_failure!(&error, "LiteLLM Cache: exception in async add_cache");
        }
        Ok(())
    }

    fn declare_encoded(
        &self,
        send: Send<'_>,
        key: &str,
        value: V,
        ttl: Option<Duration>,
        encode: impl FnOnce(V) -> Result<String, Error>,
    ) -> Result<Pending<()>, Error> {
        let ttl = ttl.or(self.default_in_memory_ttl());
        let encoded = self.redis.get().map(|_| encode(value.clone()));
        self.memory.set(key, value, ttl)?;
        let (Some(remote), Some(encoded)) = (self.redis.get(), encoded) else {
            return Ok(ready(()));
        };
        let encoded = match encoded {
            Ok(encoded) => encoded,
            Err(error) => {
                redis_failure!(&error, "a cache value did not encode, kept in memory only");
                return Ok(ready(()));
            }
        };
        if let Some(misses) = send.misses(&remote.client) {
            misses.forget(key);
        }
        Ok(pending(send.batch(&remote.client).add(commands::set(
            key,
            encoded,
            ttl.unwrap_or(remote.default_ttl),
        ))))
    }
}

impl<V: Clone, R> DualCache<V, R> {
    /// `DualCache.async_delete_cache_keys`: memory, then one multi-key `DEL` per chunk, in order.
    pub(crate) async fn delete_keys(&self, keys: &[String]) -> Result<(), Error> {
        keys.iter().try_for_each(|key| self.memory.delete(key))?;
        let Some(client) = self.redis.client() else {
            return Ok(());
        };
        for chunk in keys.chunks(DELETE_CHUNK) {
            let chunk: Vec<&str> = chunk.iter().map(String::as_str).collect();
            client.send(commands::del(&chunk)).await?;
        }
        Ok(())
    }
}
