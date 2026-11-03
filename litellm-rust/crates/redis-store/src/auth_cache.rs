use std::time::Duration;

use litellm_cache::Error as MemoryError;
use litellm_cache_memory::InMemoryCache;
use litellm_python_compat::{Value, json, repr::to_str};
use litellm_redis_client::{Declared, RedisBatch, RedisClient};

use crate::tier::{Clock, system_clock};
use crate::{DecodeError, Number, RequestRedis, codec::python_value};

/// `InMemoryCache`'s TTL for a write that names none and has no `default_in_memory_ttl`.
pub const MEMORY_TTL: Duration = Duration::from_secs(600);
/// `BaseCache.default_ttl`, which `RedisCache` keeps when `litellm.default_redis_ttl` is unset.
pub const REDIS_TTL: Duration = Duration::from_secs(60);
pub const MEMORY_ENTRIES: usize = 200;

/// `RedisCache.async_increment_if_present`, byte for byte, sent as `EVAL` like it.
const INCREMENT_IF_PRESENT: &str = "if redis.call('EXISTS', KEYS[1]) == 0 then return false end return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])";
/// `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`, the chunk `DualCache.async_delete_cache_keys` deletes at once.
const DELETE_CHUNK: usize = 1000;

/// `UserApiKeyCache`'s configuration: each memory partition's size, `default_in_memory_ttl`, and the
/// `RedisCache.default_ttl` of the Redis it was attached to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    pub memory_entries: usize,
    pub memory_ttl: Option<Duration>,
    pub redis_ttl: Duration,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            memory_entries: MEMORY_ENTRIES,
            memory_ttl: None,
            redis_ttl: REDIS_TTL,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Redis(#[from] litellm_redis_client::Error),
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error(transparent)]
    Decode(#[from] DecodeError),
    #[error("the value is not JSON serializable: {0}")]
    Encode(#[from] litellm_python_compat::Error),
}

/// `is_user_key_cache_key`: user-key objects are cached under a bare `hash_token` digest, every
/// other object under a prefixed key.
pub fn is_key_object(key: &str) -> bool {
    key.len() == 64
        && key
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// `BaseCache.get_ttl` sends `int(ttl)`, so Redis sees whole seconds.
fn whole_seconds(ttl: Duration) -> Duration {
    Duration::from_secs(ttl.as_secs())
}

fn decode(raw: Option<Vec<u8>>) -> Result<Option<Value>, DecodeError> {
    raw.as_deref().map(python_value).transpose()
}

/// The proxy's auth object cache with `enable_redis_auth_cache` on: two memory partitions, so churn
/// in management objects cannot evict user-key objects, in front of one Redis shared by every pod.
/// Keys are the logical cache keys `UserApiKeyCache` callers already use; values are what
/// `json.loads` returns, written as `json.dumps`, like `RedisCache.async_set_cache`.
pub struct AuthCache {
    key_objects: InMemoryCache<Value>,
    objects: InMemoryCache<Value>,
    remote: Option<RedisClient>,
    settings: Settings,
}

impl AuthCache {
    pub fn new(client: Option<RedisClient>, settings: Settings) -> Self {
        Self::with_clock(client, settings, system_clock())
    }

    pub fn with_clock(client: Option<RedisClient>, settings: Settings, now: Clock) -> Self {
        let memory = || {
            let now = now.clone();
            InMemoryCache::with_clock(Some(settings.memory_entries), Some(MEMORY_TTL), move || {
                now()
            })
        };
        Self {
            key_objects: memory(),
            objects: memory(),
            remote: client,
            settings,
        }
    }

    pub fn has_redis(&self) -> bool {
        self.remote.is_some()
    }

    pub fn settings(&self) -> Settings {
        self.settings
    }

    fn memory(&self, key: &str) -> &InMemoryCache<Value> {
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

    fn redis_ttl(&self, ttl: Option<Duration>) -> Duration {
        whole_seconds(self.memory_ttl(ttl).unwrap_or(self.settings.redis_ttl))
    }

    pub fn get_local(&self, key: &str) -> Result<Option<Value>, MemoryError> {
        self.memory(key).get_cache(key)
    }

    /// A memory write with `ttl`, or `default_in_memory_ttl` when it names none.
    pub fn set_local(
        &self,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), MemoryError> {
        self.memory(key)
            .set_cache(key, value, self.memory_ttl(ttl))
            .map(drop)
    }

    pub fn delete_local(&self, key: &str) -> Result<(), MemoryError> {
        self.memory(key).delete_cache(key)
    }

    pub fn flush_local(&self) -> Result<(), MemoryError> {
        self.key_objects.flush_cache()?;
        self.objects.flush_cache()
    }

    /// Entries held by the management-object and the user-key partitions.
    pub fn len_local(&self) -> Result<(usize, usize), MemoryError> {
        Ok((self.objects.len()?, self.key_objects.len()?))
    }

    /// `DualCache.async_get_cache`: memory, then Redis, backfilling memory. A failed Redis read is a
    /// miss, as `DualCache` logs and swallows it.
    pub async fn get(&self, key: &str) -> Result<Option<Value>, MemoryError> {
        if let Some(value) = self.get_local(key)? {
            return Ok(Some(value));
        }
        let Some(remote) = &self.remote else {
            return Ok(None);
        };
        let read = remote
            .get::<Vec<u8>>(key)
            .await
            .map_err(Error::from)
            .and_then(|raw| Ok(decode(raw)?));
        match read {
            Ok(Some(value)) => {
                self.set_local(key, value.clone(), None)?;
                Ok(Some(value))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                tracing::warn!(%error, "auth cache Redis read failed, treating it as a miss");
                Ok(None)
            }
        }
    }

    /// `DualCache.async_set_cache`: memory, then `SET` of `json.dumps(value)`. Redis failures and
    /// values `json.dumps` refuses only log, as there.
    pub async fn set(
        &self,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), MemoryError> {
        let encoded = json::dumps(&value);
        self.set_local(key, value, ttl)?;
        let Some(remote) = &self.remote else {
            return Ok(());
        };
        let written = match encoded {
            Ok(bytes) => remote
                .set(key, bytes, self.redis_ttl(ttl))
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
    pub async fn set_str(
        &self,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), MemoryError> {
        let encoded = to_str(&value);
        self.set_local(key, value, ttl)?;
        let Some(remote) = &self.remote else {
            return Ok(());
        };
        if let Err(error) = remote.set(key, encoded, self.redis_ttl(ttl)).await {
            tracing::warn!(%error, "writing an auth cache entry to Redis failed");
        }
        Ok(())
    }

    /// `DualCache.async_set_cache_pre_call`: memory now, the `SET` on the request's pre-call batch,
    /// sent with whatever read goes out next. `None` without Redis.
    pub fn set_pre_call(
        &self,
        request: &RequestRedis,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<Option<DeclaredWrite>, MemoryError> {
        let encoded = json::dumps(&value);
        let Some(remote) = &self.remote else {
            return Ok(None);
        };
        self.set_local(key, value, ttl)?;
        Ok(Some(match encoded {
            Ok(bytes) => {
                DeclaredWrite::Set(request.pre(remote).set(key, bytes, self.redis_ttl(ttl)))
            }
            Err(error) => DeclaredWrite::Unencodable(error),
        }))
    }

    /// `DualCache.async_delete_cache_pre_call`. `None` without Redis.
    pub fn delete_pre_call(
        &self,
        request: &RequestRedis,
        key: &str,
    ) -> Result<Option<DeclaredWrite>, MemoryError> {
        let Some(remote) = &self.remote else {
            return Ok(None);
        };
        self.delete_local(key)?;
        Ok(Some(DeclaredWrite::Delete(request.pre(remote).del(key))))
    }

    /// `UserApiKeyCache.async_set_cache_pipeline`: the key-object entries in one pipeline, then
    /// the rest in another, each with one TTL.
    pub async fn set_many(
        &self,
        entries: Vec<(String, Value)>,
        ttl: Option<Duration>,
    ) -> Result<(), MemoryError> {
        let (key_objects, others): (Vec<_>, Vec<_>) =
            entries.into_iter().partition(|(key, _)| is_key_object(key));
        for partition in [key_objects, others] {
            if !partition.is_empty() {
                self.set_partition(partition, ttl).await?;
            }
        }
        Ok(())
    }

    async fn set_partition(
        &self,
        entries: Vec<(String, Value)>,
        ttl: Option<Duration>,
    ) -> Result<(), MemoryError> {
        let mut pipeline = self.remote.as_ref().map(RedisClient::pipeline);
        for (key, value) in entries {
            if let Some(writes) = pipeline.take() {
                pipeline = Some(match json::dumps(&value) {
                    Ok(bytes) => writes.set(&key, bytes, self.redis_ttl(ttl)),
                    Err(error) => {
                        tracing::warn!(%error, "an auth cache value is not JSON serializable, kept in memory only");
                        writes
                    }
                });
            }
            self.set_local(&key, value, ttl)?;
        }
        if let (Some(remote), Some(pipeline)) = (&self.remote, pipeline) {
            for reply in remote.run(&pipeline).await {
                if let Err(error) = reply {
                    tracing::warn!(%error, "writing an auth cache entry to Redis failed");
                }
            }
        }
        Ok(())
    }

    /// The auth prefetch write-back, `async_set_cache_pipeline_with_ttls`: an entry without a TTL
    /// takes `default_in_memory_ttl` in memory but the connection's default in Redis. On the
    /// request's pre-call batch when there is one, else in a round trip of its own.
    pub async fn write_back(
        &self,
        request: Option<&RequestRedis>,
        entries: Vec<(String, Value, Option<Duration>)>,
    ) -> Result<(), Error> {
        let Some(remote) = &self.remote else {
            for (key, value, ttl) in entries {
                self.set_local(&key, value, ttl)?;
            }
            return Ok(());
        };
        if let Some(request) = request {
            let batch = request.pre(remote);
            for (key, value, ttl) in entries {
                let redis_ttl = whole_seconds(ttl.unwrap_or(self.settings.redis_ttl));
                drop(batch.set(&key, json::dumps(&value)?, redis_ttl));
                self.set_local(&key, value, ttl)?;
            }
            return Ok(());
        }
        let mut pipeline = remote.pipeline();
        for (key, value, ttl) in entries {
            let redis_ttl = whole_seconds(ttl.unwrap_or(self.settings.redis_ttl));
            pipeline = pipeline.set(&key, json::dumps(&value)?, redis_ttl);
            self.set_local(&key, value, ttl)?;
        }
        for reply in remote.run(&pipeline).await {
            reply?;
        }
        Ok(())
    }

    /// `UserApiKeyCache.async_delete_cache_keys`: memory, then the key-object partition and the
    /// rest as two concurrent `DualCache.async_delete_cache_keys`, each a multi-key `DEL` per
    /// chunk of 1000. Both run even when one fails.
    pub async fn delete(&self, keys: &[String]) -> Result<(), Error> {
        for key in keys {
            self.delete_local(key)?;
        }
        let Some(remote) = &self.remote else {
            return Ok(());
        };
        let (key_objects, others): (Vec<&str>, Vec<&str>) = keys
            .iter()
            .map(String::as_str)
            .partition(|key| is_key_object(key));
        let (key_objects, others) = tokio::join!(
            delete_chunks(remote, &key_objects),
            delete_chunks(remote, &others)
        );
        key_objects.and(others)
    }

    /// Reads for keys the caller found missing in memory, on the request's pre-call batch. `None`
    /// without Redis.
    pub fn declare_get_many(
        &self,
        request: &RequestRedis,
        keys: &[String],
    ) -> Option<DeclaredGets> {
        Some(self.declare_gets(&request.pre(self.remote.as_ref()?), keys))
    }

    /// `RedisCache.async_batch_get_cache` for keys missing in memory, in one round trip. `None`
    /// without Redis.
    pub async fn get_many(&self, keys: &[String]) -> Option<Result<Vec<Option<Value>>, Error>> {
        Some(
            self.declare_gets(&self.remote.as_ref()?.batch(), keys)
                .resolve()
                .await,
        )
    }

    fn declare_gets(&self, batch: &RedisBatch, keys: &[String]) -> DeclaredGets {
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        DeclaredGets(batch.mget::<Vec<u8>>(&keys))
    }

    /// `RedisCache.async_increment_if_present`: `INCRBYFLOAT` only an entry Redis still holds.
    /// `None` without Redis or for an absent entry.
    pub async fn increment_if_present(
        &self,
        key: &str,
        amount: Number,
    ) -> Result<Option<f64>, Error> {
        let Some(remote) = &self.remote else {
            return Ok(None);
        };
        let total: Option<Vec<u8>> = remote
            .eval_source(
                INCREMENT_IF_PRESENT,
                &[key],
                litellm_redis_client::Amount::from(amount),
            )
            .await?;
        total
            .as_deref()
            .map(|raw| {
                std::str::from_utf8(raw)
                    .ok()
                    .and_then(|text| text.trim().parse::<f64>().ok())
                    .ok_or(Error::Decode(DecodeError::Shape(
                        "INCRBYFLOAT returned a non-number",
                    )))
            })
            .transpose()
    }
}

/// A write declared on a request's pre-call batch, settled once the batch goes out.
pub enum DeclaredWrite {
    Set(Declared<()>),
    Delete(Declared<usize>),
    /// Kept in memory only, as `json.dumps` refused the value.
    Unencodable(litellm_python_compat::Error),
}

impl DeclaredWrite {
    /// Flushes the batch if nobody has yet.
    pub async fn resolve(self) -> Result<(), Error> {
        match self {
            Self::Set(written) => written.resolve().await?,
            Self::Delete(deleted) => drop(deleted.resolve().await?),
            Self::Unencodable(error) => {
                tracing::warn!(%error, "an auth cache value is not JSON serializable, kept in memory only");
            }
        }
        Ok(())
    }
}

/// Reads declared on a batch, answered once it goes out.
pub struct DeclaredGets(Declared<Vec<Option<Vec<u8>>>>);

impl DeclaredGets {
    /// Flushes the batch if nobody has yet. Any failed read or undecodable value fails the whole
    /// read, as a failed `MGET` does.
    pub async fn resolve(self) -> Result<Vec<Option<Value>>, Error> {
        self.0
            .resolve()
            .await?
            .into_iter()
            .map(|reply| Ok(decode(reply)?))
            .collect()
    }
}

async fn delete_chunks(remote: &RedisClient, keys: &[&str]) -> Result<(), Error> {
    for chunk in keys.chunks(DELETE_CHUNK) {
        remote.del(chunk).await?;
    }
    Ok(())
}
