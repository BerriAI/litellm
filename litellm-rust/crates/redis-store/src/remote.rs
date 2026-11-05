use std::{sync::OnceLock, time::Duration};

use litellm_redis_client::{RedisBatch, RedisClient, commands};

use crate::{Error, Send, memory::Memory};

/// `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`, the chunk `DualCache.async_delete_cache_keys` deletes at once.
const DELETE_CHUNK: usize = 1000;

/// The Redis a store shares with every pod, with the `RedisCache.default_ttl` a write takes when it
/// names none.
#[derive(Clone)]
pub struct Remote {
    pub client: RedisClient,
    pub default_ttl: Duration,
}

/// A store's Redis, absent until the host attaches one, as `DualCache.attach_redis_cache` leaves
/// memory in place and only adds the second tier. It is attached at most once.
pub(crate) struct RemoteSlot(OnceLock<Remote>);

impl RemoteSlot {
    pub(crate) fn new(remote: Option<Remote>) -> Self {
        Self(remote.map(OnceLock::from).unwrap_or_default())
    }

    /// `false` when a Redis was already attached, which keeps it.
    pub(crate) fn attach(&self, remote: Remote) -> bool {
        self.0.set(remote).is_ok()
    }

    pub(crate) fn get(&self) -> Option<&Remote> {
        self.0.get()
    }

    pub(crate) fn client(&self) -> Option<&RedisClient> {
        self.get().map(|remote| &remote.client)
    }

    pub(crate) fn batch(&self, send: Send<'_>) -> Option<RedisBatch> {
        Some(send.batch(self.client()?))
    }

    /// `DualCache.async_get_cache`: memory, else Redis written back to memory for `backfill_ttl`. A
    /// failed or undecodable Redis read is a miss, as `DualCache` logs and swallows it.
    pub(crate) async fn read_through<V: Clone, E: Into<Error>>(
        &self,
        memory: &Memory<V>,
        key: &str,
        decode: impl FnOnce(&[u8]) -> Result<V, E>,
        backfill_ttl: Option<Duration>,
    ) -> Result<Option<V>, Error> {
        if let Some(value) = memory.get(key)? {
            return Ok(Some(value));
        }
        let Some(client) = self.client() else {
            return Ok(None);
        };
        let read = client.send(commands::get::<Vec<u8>>(key)).await;
        let value = read
            .map_err(Error::from)
            .and_then(|raw| raw.as_deref().map(decode).transpose().map_err(Into::into))
            .unwrap_or_else(|error| {
                tracing::error!(%error, "LiteLLM Cache: exception in async_get_cache");
                None
            });
        let Some(value) = value else {
            return Ok(None);
        };
        memory.set(key, value.clone(), backfill_ttl)?;
        Ok(Some(value))
    }
}

/// `DualCache.async_delete_cache_keys`: one multi-key `DEL` per chunk, in order.
pub(crate) async fn delete(client: &RedisClient, keys: &[String]) -> Result<(), Error> {
    for chunk in keys.chunks(DELETE_CHUNK) {
        let chunk: Vec<&str> = chunk.iter().map(String::as_str).collect();
        client.send(commands::del(&chunk)).await?;
    }
    Ok(())
}
