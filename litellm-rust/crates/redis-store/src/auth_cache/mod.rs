mod keys;
mod types;

use std::time::Duration;

use litellm_python_compat::{Value, json, repr::to_str};
use litellm_redis_client::{Declared, RedisBatch, commands};

pub use self::keys::is_key_object;
pub use self::types::Settings;
use crate::{
    Error, Number, Pending, Send,
    codec::python_value,
    memory::{Clock, Memory, system_clock},
    remote::{self, Remote, RemoteSlot},
    request::{pending, ready},
};

/// `RedisCache.async_increment_if_present`, byte for byte, sent as `EVAL` like it.
const INCREMENT_IF_PRESENT: &str = "if redis.call('EXISTS', KEYS[1]) == 0 then return false end return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])";

/// The proxy's auth object cache with `enable_redis_auth_cache` on: two memory partitions, so churn
/// in management objects cannot evict user-key objects, in front of one Redis shared by every pod.
/// Keys are the logical cache keys `UserApiKeyCache` callers already use; values are what
/// `json.loads` returns, written as `json.dumps`, like `RedisCache.async_set_cache`.
pub struct AuthCache {
    key_objects: Memory<Value>,
    objects: Memory<Value>,
    remote: RemoteSlot,
    settings: Settings,
}

impl AuthCache {
    pub fn new(remote: Option<Remote>, settings: Settings) -> Self {
        Self::with_clock(remote, settings, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, settings: Settings, now: Clock) -> Self {
        Self {
            key_objects: Memory::new(Some(settings.memory_entries), &now),
            objects: Memory::new(Some(settings.memory_entries), &now),
            remote: RemoteSlot::new(remote),
            settings,
        }
    }

    pub fn has_redis(&self) -> bool {
        self.remote.get().is_some()
    }

    pub fn settings(&self) -> Settings {
        self.settings
    }

    pub fn get_local(&self, key: &str) -> Result<Option<Value>, Error> {
        Ok(self.memory(key).get(key)?)
    }

    /// A memory write with `ttl`, or `default_in_memory_ttl` when it names none.
    pub fn set_local(&self, key: &str, value: Value, ttl: Option<Duration>) -> Result<(), Error> {
        Ok(self.memory(key).set(key, value, self.memory_ttl(ttl))?)
    }

    pub fn delete_local(&self, key: &str) -> Result<(), Error> {
        Ok(self.memory(key).delete(key)?)
    }

    pub fn flush_local(&self) -> Result<(), Error> {
        self.key_objects.flush()?;
        Ok(self.objects.flush()?)
    }

    /// Entries held by the management-object and the user-key partitions.
    pub fn len_local(&self) -> Result<(usize, usize), Error> {
        Ok((self.objects.len()?, self.key_objects.len()?))
    }

    /// `DualCache.async_get_cache`, backfilling memory for `default_in_memory_ttl`.
    pub async fn get(&self, key: &str) -> Result<Option<Value>, Error> {
        self.remote
            .read_through(self.memory(key), key, python_value, self.settings.memory_ttl)
            .await
    }

    /// `DualCache.async_set_cache`: memory, then `SET` of `json.dumps(value)`. Redis failures and
    /// values `json.dumps` refuses only log, as there.
    pub async fn set(&self, key: &str, value: Value, ttl: Option<Duration>) -> Result<(), Error> {
        let encoded = json::dumps(&value);
        self.set_local(key, value, ttl)?;
        let Some(remote) = self.remote.get() else {
            return Ok(());
        };
        let written = match encoded {
            Ok(bytes) => remote
                .client
                .send(commands::set(key, bytes, self.redis_ttl(remote, ttl)))
                .await
                .map_err(Error::from),
            Err(error) => Err(error.into()),
        };
        if let Err(error) = written {
            tracing::warn!(%error, "writing an auth cache entry to Redis failed");
        }
        Ok(())
    }

    /// `DualCache.set_cache`, whose `RedisCache.set_cache` writes `str(value)`.
    pub async fn set_str(&self, key: &str, value: Value, ttl: Option<Duration>) -> Result<(), Error> {
        let encoded = to_str(&value);
        self.set_local(key, value, ttl)?;
        let Some(remote) = self.remote.get() else {
            return Ok(());
        };
        let write = commands::set(key, encoded, self.redis_ttl(remote, ttl));
        if let Err(error) = remote.client.send(write).await {
            tracing::warn!(%error, "writing an auth cache entry to Redis failed");
        }
        Ok(())
    }

    /// `DualCache.async_set_cache_pre_call`: memory now, the `SET` on the batch `send` picks.
    pub fn declare_set(
        &self,
        send: Send<'_>,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<Pending<()>, Error> {
        let encoded = json::dumps(&value);
        self.set_local(key, value, ttl)?;
        let Some(remote) = self.remote.get() else {
            return Ok(ready(()));
        };
        match encoded {
            Ok(bytes) => Ok(pending(
                send.batch(&remote.client)
                    .add(commands::set(key, bytes, self.redis_ttl(remote, ttl))),
            )),
            Err(error) => {
                tracing::warn!(%error, "an auth cache value is not JSON serializable, kept in memory only");
                Ok(ready(()))
            }
        }
    }

    /// `DualCache.async_delete_cache_pre_call`, memory now and the `DEL` on the batch `send` picks.
    pub fn declare_delete(&self, send: Send<'_>, key: &str) -> Result<Pending<()>, Error> {
        self.delete_local(key)?;
        let Some(batch) = self.remote.batch(send) else {
            return Ok(ready(()));
        };
        let deleted = batch.add(commands::del(&[key]));
        Ok(Box::pin(async move { Ok(deleted.resolve().await.map(drop)?) }))
    }

    /// `UserApiKeyCache.async_set_cache_pipeline`: the key-object entries in one pipeline, then
    /// the rest in another, each with one TTL.
    pub async fn set_many(&self, entries: Vec<(String, Value)>, ttl: Option<Duration>) -> Result<(), Error> {
        let (key_objects, others): (Vec<_>, Vec<_>) =
            entries.into_iter().partition(|(key, _)| is_key_object(key));
        for partition in [key_objects, others] {
            self.set_partition(partition, ttl).await?;
        }
        Ok(())
    }

    /// The auth prefetch write-back, `async_set_cache_pipeline_with_ttls`: memory now, and an entry
    /// without a TTL takes `default_in_memory_ttl` in memory but the connection's default in Redis.
    /// On a request batch the `SET`s are declared and nothing is left to wait for. [`Send::Now`] is
    /// one pipeline of its own, as `async_set_cache_pipeline_with_ttls` sends it, rather than one
    /// round trip per entry.
    pub fn write_back(
        &self,
        send: Send<'_>,
        entries: Vec<(String, Value, Option<Duration>)>,
    ) -> Result<Pending<()>, Error> {
        let Some(remote) = self.remote.get() else {
            for (key, value, ttl) in entries {
                self.set_local(&key, value, ttl)?;
            }
            return Ok(ready(()));
        };
        let sets = entries
            .into_iter()
            .map(|(key, value, ttl)| {
                let redis_ttl = ttl.unwrap_or(remote.default_ttl);
                let set = commands::set(&key, json::dumps(&value)?, redis_ttl);
                self.set_local(&key, value, ttl)?;
                Ok(set)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let Send::Now = send else {
            let batch = send.batch(&remote.client);
            sets.into_iter().for_each(|set| drop(batch.add(set)));
            return Ok(ready(()));
        };
        let pipeline = RedisBatch::pipelined(remote.client.clone());
        let written: Vec<_> = sets.into_iter().map(|set| pipeline.add(set)).collect();
        Ok(Box::pin(async move {
            Declared::all(written).await.into_iter().try_for_each(|reply| Ok(reply?))
        }))
    }

    /// `UserApiKeyCache.async_delete_cache_keys`: memory, then the key-object partition and the
    /// rest as two concurrent `DualCache.async_delete_cache_keys`, each a multi-key `DEL` per
    /// chunk of 1000. Both run even when one fails.
    pub async fn delete(&self, keys: &[String]) -> Result<(), Error> {
        keys.iter().try_for_each(|key| self.delete_local(key))?;
        let Some(client) = self.remote.client() else {
            return Ok(());
        };
        let (key_objects, others): (Vec<String>, Vec<String>) =
            keys.iter().cloned().partition(|key| is_key_object(key));
        let (key_objects, others) = tokio::join!(
            remote::delete(client, &key_objects),
            remote::delete(client, &others)
        );
        key_objects.and(others)
    }

    /// `RedisCache.async_batch_get_cache` for keys the caller found missing in memory, one `MGET` on
    /// the batch `send` picks. `None` without Redis. Any failed read or undecodable value fails the
    /// whole read, as a failed `MGET` does.
    pub fn get_many(&self, send: Send<'_>, keys: &[String]) -> Option<Pending<Vec<Option<Value>>>> {
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let read = self.remote.batch(send)?.add(commands::mget::<Vec<u8>>(&keys));
        Some(Box::pin(async move {
            read.resolve()
                .await?
                .into_iter()
                .map(|raw| Ok(types::decode(raw)?))
                .collect()
        }))
    }

    /// `RedisCache.async_increment_if_present`: `INCRBYFLOAT` only an entry Redis still holds.
    /// `None` without Redis or for an absent entry.
    pub async fn increment_if_present(&self, key: &str, amount: Number) -> Result<Option<f64>, Error> {
        let Some(client) = self.remote.client() else {
            return Ok(None);
        };
        let amount = litellm_redis_client::Amount::from(amount);
        Ok(client.send(commands::eval_source(INCREMENT_IF_PRESENT, &[key], amount)).await?)
    }

    fn memory(&self, key: &str) -> &Memory<Value> {
        if is_key_object(key) {
            &self.key_objects
        } else {
            &self.objects
        }
    }

    /// `DualCache`'s rule: a write without a TTL takes `default_in_memory_ttl` in memory and Redis.
    fn memory_ttl(&self, ttl: Option<Duration>) -> Option<Duration> {
        ttl.or(self.settings.memory_ttl)
    }

    fn redis_ttl(&self, remote: &Remote, ttl: Option<Duration>) -> Duration {
        self.memory_ttl(ttl).unwrap_or(remote.default_ttl)
    }

    async fn set_partition(&self, entries: Vec<(String, Value)>, ttl: Option<Duration>) -> Result<(), Error> {
        let target = self
            .remote
            .get()
            .map(|remote| (RedisBatch::pipelined(remote.client.clone()), self.redis_ttl(remote, ttl)));
        let written = entries
            .into_iter()
            .map(|(key, value)| {
                let set = match (&target, json::dumps(&value)) {
                    (Some((pipeline, redis_ttl)), Ok(bytes)) => {
                        Some(pipeline.add(commands::set(&key, bytes, *redis_ttl)))
                    }
                    (Some(_), Err(error)) => {
                        tracing::warn!(%error, "an auth cache value is not JSON serializable, kept in memory only");
                        None
                    }
                    (None, _) => None,
                };
                self.set_local(&key, value, ttl)?;
                Ok(set)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        for reply in Declared::all(written.into_iter().flatten()).await {
            if let Err(error) = reply {
                tracing::warn!(%error, "writing an auth cache entry to Redis failed");
            }
        }
        Ok(())
    }
}
