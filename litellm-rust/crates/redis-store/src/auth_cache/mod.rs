mod keys;
mod types;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_python_compat::{Value, json};
use litellm_redis_client::{Declared, RedisBatch, commands};

pub use self::keys::is_key_object;
pub use self::types::Settings;
use crate::{
    Error, Number, Pending, RequestRedis, Send,
    dual_cache::DualCache,
    memory::{Clock, lock, system_clock},
    remote::{Remote, RemoteSlot},
    request::ready,
};

/// `RedisCache.async_increment_if_present`, byte for byte, sent as `EVAL` like it.
const INCREMENT_IF_PRESENT: &str = "if redis.call('EXISTS', KEYS[1]) == 0 then return false end return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])";

/// The proxy's auth object cache with `enable_redis_auth_cache` on: `UserApiKeyCache`'s two
/// `DualCache` partitions, so churn in management objects cannot evict user-key objects, in front of
/// one Redis shared by every pod. Keys are the logical cache keys `UserApiKeyCache` callers already
/// use; values are what `json.loads` returns, written as `json.dumps`, like `RedisCache.async_set_cache`.
pub struct AuthCache {
    key_objects: DualCache<Value>,
    objects: DualCache<Value>,
    redis: Arc<RemoteSlot>,
    settings: Mutex<Settings>,
}

impl AuthCache {
    pub fn new(remote: Option<Remote>, settings: Settings) -> Self {
        Self::with_clock(remote, settings, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, settings: Settings, now: Clock) -> Self {
        let redis = Arc::new(RemoteSlot::new(remote));
        let partition = || DualCache::new(redis.clone(), settings.memory(), now.clone());
        Self {
            key_objects: partition(),
            objects: partition(),
            redis,
            settings: Mutex::new(settings),
        }
    }

    pub fn has_redis(&self) -> bool {
        self.redis.get().is_some()
    }

    pub fn settings(&self) -> Settings {
        *lock(&self.settings)
    }

    /// `UserApiKeyCache.update_cache_ttl` and `update_in_memory_max_size` on both partitions, which
    /// keep what they hold.
    pub fn reconfigure(&self, settings: Settings) {
        *lock(&self.settings) = settings;
        self.key_objects.reconfigure(settings.memory());
        self.objects.reconfigure(settings.memory());
    }

    pub fn get_local(&self, key: &str) -> Result<Option<Value>, Error> {
        Ok(self.partition(key).memory().get(key)?)
    }

    /// A memory write with `ttl`, or `default_in_memory_ttl` when it names none.
    pub fn set_local(&self, key: &str, value: Value, ttl: Option<Duration>) -> Result<(), Error> {
        let partition = self.partition(key);
        Ok(partition
            .memory()
            .set(key, value, ttl.or(partition.default_in_memory_ttl()))?)
    }

    pub fn delete_local(&self, key: &str) -> Result<(), Error> {
        Ok(self.partition(key).memory().delete(key)?)
    }

    pub fn flush_local(&self) -> Result<(), Error> {
        self.key_objects.memory().flush()?;
        Ok(self.objects.memory().flush()?)
    }

    /// Entries held by the management-object and the user-key partitions.
    pub fn len_local(&self) -> Result<(usize, usize), Error> {
        Ok((
            self.objects.memory().len()?,
            self.key_objects.memory().len()?,
        ))
    }

    /// `DualCache.async_get_cache`. A key `request` already read as missing from Redis is not read
    /// again.
    pub async fn get(
        &self,
        request: Option<&RequestRedis>,
        key: &str,
    ) -> Result<Option<Value>, Error> {
        self.partition(key).get(request, key, None).await
    }

    /// `DualCache.async_set_cache`, writing `json.dumps(value)`.
    pub async fn set(&self, key: &str, value: Value, ttl: Option<Duration>) -> Result<(), Error> {
        self.partition(key).set(key, value, ttl).await
    }

    /// `DualCache.set_cache`, whose `RedisCache.set_cache` writes `str(value)`.
    pub async fn set_str(
        &self,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        self.partition(key).set_str(key, value, ttl).await
    }

    /// `DualCache.async_set_cache_pre_call`: memory now, the `SET` on the batch `send` picks.
    pub fn declare_set(
        &self,
        send: Send<'_>,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<Pending<()>, Error> {
        self.partition(key).declare_set(send, key, value, ttl)
    }

    /// `DualCache.async_delete_cache_pre_call`, memory now and the `DEL` on the batch `send` picks.
    pub fn declare_delete(&self, send: Send<'_>, key: &str) -> Result<Pending<()>, Error> {
        self.partition(key).declare_delete(send, key)
    }

    /// `UserApiKeyCache.async_set_cache_pipeline`: the key-object entries in one pipeline, then
    /// the rest in another, each with one TTL.
    pub async fn set_many(
        &self,
        entries: Vec<(String, Value)>,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let (key_objects, others): (Vec<_>, Vec<_>) =
            entries.into_iter().partition(|(key, _)| is_key_object(key));
        self.key_objects.set_pipeline(key_objects, ttl).await?;
        self.objects.set_pipeline(others, ttl).await
    }

    /// The auth prefetch write-back: memory first, an entry without a TTL taking
    /// `default_in_memory_ttl`, then `RedisCache.async_set_cache_pipeline_with_ttls`, which encodes
    /// every value before it sends anything and gives an entry without a TTL the connection's
    /// default. On a request batch the `SET`s are declared and nothing is left to wait for.
    /// [`Send::Now`] is one pipeline of its own rather than one round trip per entry.
    pub fn write_back(
        &self,
        send: Send<'_>,
        entries: Vec<(String, Value, Option<Duration>)>,
    ) -> Result<Pending<()>, Error> {
        let encoded = self
            .redis
            .get()
            .map(|_| {
                entries
                    .iter()
                    .map(|(_, value, _)| json::dumps(value))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose();
        let writes = entries
            .into_iter()
            .map(|(key, value, ttl)| {
                self.set_local(&key, value, ttl)?;
                Ok((key, ttl))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let (Some(remote), Some(encoded)) = (self.redis.get(), encoded?) else {
            return Ok(ready(()));
        };
        let sets = writes.iter().zip(encoded).map(|((key, ttl), value)| {
            commands::set(key, value, ttl.unwrap_or(remote.default_ttl))
        });
        let Send::Now = send else {
            let batch = send.batch(&remote.client);
            sets.for_each(|set| drop(batch.add(set)));
            return Ok(ready(()));
        };
        let pipeline = RedisBatch::pipelined(remote.client.clone());
        let written: Vec<_> = sets.map(|set| pipeline.add(set)).collect();
        Ok(Box::pin(async move {
            Declared::all(written)
                .await
                .into_iter()
                .try_for_each(|reply| Ok(reply?))
        }))
    }

    /// `UserApiKeyCache.async_delete_cache_keys`: the key-object partition and the rest as two
    /// concurrent `DualCache.async_delete_cache_keys`. Both run even when one fails.
    pub async fn delete(&self, keys: &[String]) -> Result<(), Error> {
        let (key_objects, others): (Vec<String>, Vec<String>) =
            keys.iter().cloned().partition(|key| is_key_object(key));
        let (key_objects, others) = tokio::join!(
            self.key_objects.delete_keys(&key_objects),
            self.objects.delete_keys(&others)
        );
        key_objects.and(others)
    }

    /// `RedisCache.async_batch_get_cache` for keys the caller found missing in memory, one `MGET` on
    /// the batch `send` picks, noting the keys Redis lacks as the request's misses. `None` without
    /// Redis. Any failed read or undecodable value fails the whole read, as a failed `MGET` does.
    pub fn get_many(&self, send: Send<'_>, keys: &[String]) -> Option<Pending<Vec<Option<Value>>>> {
        let client = self.redis.client()?;
        let misses = send.misses(client);
        let read = send.batch(client).add(commands::mget::<Vec<u8>>(
            &keys.iter().map(String::as_str).collect::<Vec<_>>(),
        ));
        let keys = keys.to_vec();
        Some(Box::pin(async move {
            let raw = read.resolve().await?;
            if let Some(misses) = misses {
                misses.note(
                    keys.into_iter()
                        .zip(&raw)
                        .filter(|(_, raw)| raw.is_none())
                        .map(|(key, _)| key),
                );
            }
            raw.into_iter().map(|raw| Ok(types::decode(raw)?)).collect()
        }))
    }

    /// `RedisCache.async_increment_if_present`: `INCRBYFLOAT` only an entry Redis still holds.
    /// `None` without Redis or for an absent entry.
    pub async fn increment_if_present(
        &self,
        key: &str,
        amount: Number,
    ) -> Result<Option<f64>, Error> {
        let Some(client) = self.redis.client() else {
            return Ok(None);
        };
        let amount = litellm_redis_client::Amount::from(amount);
        Ok(client
            .send(commands::eval_source(INCREMENT_IF_PRESENT, &[key], amount))
            .await?)
    }

    fn partition(&self, key: &str) -> &DualCache<Value> {
        if is_key_object(key) {
            &self.key_objects
        } else {
            &self.objects
        }
    }
}
