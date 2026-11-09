mod keys;
mod local;
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
    Error, Number, Pending, RequestRedis, Send, codec,
    failure_log::redis_failure,
    memory::{Clock, lock, system_clock},
    remote::{Remote, RemoteSlot},
    request::{pending, ready},
};

/// `RedisCache.async_increment_if_present`, byte for byte, sent as `EVAL` like it.
const INCREMENT_IF_PRESENT: &str = "if redis.call('EXISTS', KEYS[1]) == 0 then return false end return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])";

pub struct AuthCache {
    local: Mutex<local::Local>,
    now: Clock,
    redis: Arc<RemoteSlot>,
    settings: Mutex<Settings>,
}

impl AuthCache {
    pub fn new(remote: Option<Remote>, settings: Settings) -> Self {
        Self::with_clock(remote, settings, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, settings: Settings, now: Clock) -> Self {
        let redis = Arc::new(RemoteSlot::new(remote));
        Self {
            local: Mutex::new(local::Local::new(settings)),
            now,
            redis,
            settings: Mutex::new(settings),
        }
    }

    pub fn has_redis(&self) -> bool {
        self.redis.get().is_some()
    }

    pub fn attach(&self, remote: Remote) -> bool {
        self.redis.attach(remote)
    }

    pub fn current_generation(&self) -> u64 {
        lock(&self.local).generation
    }

    pub fn local_ready(&self) -> bool {
        lock(&self.local).healthy
    }

    pub fn subscriber_connected(&self, connected: bool) {
        let mut local = lock(&self.local);
        local.flush();
        local.healthy = connected;
    }

    pub fn set_if_generation(
        &self,
        key: &str,
        value: Value,
        generation: u64,
        ttl: Option<Duration>,
    ) -> bool {
        let ttl = ttl.or(self.settings().memory_ttl);
        let local = lock(&self.local);
        if local.generation != generation || !local.healthy || key.starts_with("sk-") {
            return false;
        }
        local.set(key, value, ttl, (self.now)());
        true
    }

    pub fn settings(&self) -> Settings {
        *lock(&self.settings)
    }

    /// `UserApiKeyCache.update_cache_ttl` and `update_in_memory_max_size` on both partitions, which
    /// keep what they hold.
    pub fn reconfigure(&self, settings: Settings) {
        *lock(&self.settings) = settings;
        lock(&self.local).reconfigure(settings);
    }

    pub fn get_local(&self, key: &str) -> Result<Option<Value>, Error> {
        Ok(lock(&self.local).get(key, (self.now)()))
    }

    /// A memory write with `ttl`, or `default_in_memory_ttl` when it names none.
    pub fn set_local(&self, key: &str, value: Value, ttl: Option<Duration>) -> Result<(), Error> {
        let ttl = ttl.or(self.settings().memory_ttl);
        lock(&self.local).set(key, value, ttl, (self.now)());
        Ok(())
    }

    pub fn delete_local(&self, key: &str) -> Result<(), Error> {
        lock(&self.local).delete(key);
        Ok(())
    }

    pub fn flush_local(&self) -> Result<(), Error> {
        lock(&self.local).flush();
        Ok(())
    }

    /// Entries held by the management-object and the user-key partitions.
    pub fn len_local(&self) -> Result<(usize, usize), Error> {
        Ok(lock(&self.local).len())
    }

    /// `DualCache.async_get_cache`. A key `request` already read as missing from Redis is not read
    /// again.
    pub async fn get(
        &self,
        request: Option<&RequestRedis>,
        key: &str,
    ) -> Result<Option<Value>, Error> {
        let generation = self.current_generation();
        if let Some(value) = self.get_local(key)? {
            return Ok(Some(value));
        }
        if !lock(&self.local).healthy || key.starts_with("sk-") {
            return Ok(None);
        }
        let Some(client) = self.redis.client() else {
            return Ok(None);
        };
        if request.is_some_and(|request| request.misses(client).contains(key)) {
            return Ok(None);
        }
        let value = client
            .send(commands::get::<Vec<u8>>(key))
            .await
            .map_err(Error::from)
            .and_then(|raw| Ok(types::decode(raw)?))
            .unwrap_or_else(|error| {
                redis_failure!(&error, "auth cache read failed");
                None
            });
        if let Some(value) = &value
            && !self.set_if_generation(key, value.clone(), generation, None)
        {
            return Ok(None);
        }
        Ok(value)
    }

    /// `DualCache.async_set_cache`, writing `json.dumps(value)`.
    pub async fn set(&self, key: &str, value: Value, ttl: Option<Duration>) -> Result<(), Error> {
        if let Err(error) = self.declare_set(Send::Now, key, value, ttl)?.await {
            redis_failure!(&error, "auth cache write failed");
        }
        Ok(())
    }

    /// `DualCache.set_cache`, whose `RedisCache.set_cache` writes `str(value)`.
    pub async fn set_str(
        &self,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let encoded = codec::str(value.clone());
        if let Err(error) = self
            .declare_encoded(Send::Now, key, value, ttl, Ok(encoded))?
            .await
        {
            redis_failure!(&error, "auth cache write failed");
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
        let encoded = codec::dumps(value.clone());
        self.declare_encoded(send, key, value, ttl, encoded)
    }

    fn declare_encoded(
        &self,
        send: Send<'_>,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
        encoded: Result<String, Error>,
    ) -> Result<Pending<()>, Error> {
        let ttl = ttl.or(self.settings().memory_ttl);
        self.set_local(key, value, ttl)?;
        if key.starts_with("sk-") {
            return Ok(ready(()));
        }
        let Some(remote) = self.redis.get() else {
            return Ok(ready(()));
        };
        let encoded = match encoded {
            Ok(encoded) => encoded,
            Err(error) => {
                redis_failure!(&error, "auth cache value did not encode");
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

    /// `DualCache.async_delete_cache_pre_call`, memory now and the `DEL` on the batch `send` picks.
    pub fn declare_delete(&self, send: Send<'_>, key: &str) -> Result<Pending<()>, Error> {
        self.delete_local(key)?;
        let Some(client) = self.redis.client() else {
            return Ok(ready(()));
        };
        if let Some(misses) = send.misses(client) {
            misses.note([key.to_owned()]);
        }
        let deleted = send.batch(client).add(commands::del(&[key]));
        Ok(Box::pin(
            async move { Ok(deleted.resolve().await.map(drop)?) },
        ))
    }

    /// `UserApiKeyCache.async_set_cache_pipeline`: the key-object entries in one pipeline, then
    /// the rest in another, each with one TTL.
    pub async fn set_many(
        &self,
        entries: Vec<(String, Value)>,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let (keys, objects): (Vec<_>, Vec<_>) =
            entries.into_iter().partition(|(key, _)| is_key_object(key));
        self.set_partition(keys, ttl).await?;
        self.set_partition(objects, ttl).await
    }

    async fn set_partition(
        &self,
        entries: Vec<(String, Value)>,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let ttl = ttl.or(self.settings().memory_ttl);
        let entries = entries
            .into_iter()
            .map(|(key, value)| (key, value, ttl))
            .collect();
        let result = match self.write_back(Send::Now, entries) {
            Ok(written) => written.await,
            Err(error) => Err(error),
        };
        if let Err(error) = result {
            redis_failure!(&error, "auth cache pipeline write failed");
        }
        Ok(())
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
        let entries: Vec<_> = entries
            .into_iter()
            .filter(|(key, _, _)| !key.starts_with("sk-"))
            .collect();
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
        for key in keys {
            self.delete_local(key)?;
        }
        let (keys, objects): (Vec<_>, Vec<_>) =
            keys.iter().cloned().partition(|key| is_key_object(key));
        let (keys, objects) = tokio::join!(
            self.delete_partition(&keys),
            self.delete_partition(&objects)
        );
        keys.and(objects)
    }

    async fn delete_partition(&self, keys: &[String]) -> Result<(), Error> {
        let Some(client) = self.redis.client() else {
            return Ok(());
        };
        for chunk in keys.chunks(1000) {
            client
                .send(commands::del(
                    &chunk.iter().map(String::as_str).collect::<Vec<_>>(),
                ))
                .await?;
        }
        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    const KEY: &str = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b";

    #[rstest]
    fn invalidation_fences_an_in_flight_fill_and_allows_a_new_one() {
        let cache = AuthCache::new(None, Settings::default());
        let before = cache.current_generation();
        cache.delete_local(KEY).unwrap();
        assert!(!cache.set_if_generation(KEY, Value::from("revoked"), before, None));
        assert_eq!(cache.get_local(KEY).unwrap(), None);
        let after = cache.current_generation();
        assert!(cache.set_if_generation(KEY, Value::from("fresh"), after, None));
        assert_eq!(cache.get_local(KEY).unwrap(), Some(Value::from("fresh")));
    }

    #[rstest]
    fn disconnect_bypasses_writes_and_reconnect_fences_old_fills() {
        let cache = AuthCache::new(None, Settings::default());
        cache.set_local(KEY, Value::Bool(true), None).unwrap();
        cache.subscriber_connected(false);
        assert_eq!(cache.get_local(KEY).unwrap(), None);
        cache.set_local(KEY, Value::Bool(true), None).unwrap();
        let disconnected = cache.current_generation();
        assert!(!cache.set_if_generation(KEY, Value::Bool(true), disconnected, None));
        cache.subscriber_connected(true);
        assert_eq!(cache.get_local(KEY).unwrap(), None);
        assert!(!cache.set_if_generation(KEY, Value::Bool(true), disconnected, None));
    }

    #[rstest]
    fn auth_entries_expire_even_without_a_configured_ttl() {
        let seconds = Arc::new(AtomicU64::new(100));
        let clock = seconds.clone();
        let cache = AuthCache::with_clock(
            None,
            Settings::default(),
            Arc::new(move || Duration::from_secs(clock.load(Ordering::SeqCst))),
        );
        cache.set_local(KEY, Value::Bool(true), None).unwrap();
        assert_eq!(cache.get_local(KEY).unwrap(), Some(Value::Bool(true)));
        seconds.store(160, Ordering::SeqCst);
        assert_eq!(cache.get_local(KEY).unwrap(), None);
    }

    #[rstest]
    fn raw_credentials_are_not_cached() {
        let cache = AuthCache::new(None, Settings::default());
        cache
            .set_local("sk-do-not-cache", Value::Bool(true), None)
            .unwrap();
        assert_eq!(cache.get_local("sk-do-not-cache").unwrap(), None);
        assert_eq!(cache.len_local().unwrap(), (0, 0));
    }
}
