mod keys;
mod local;
mod types;

use std::{sync::Arc, time::Duration};

use self::local::LocalValue;
use litellm_python_compat::{Value, json};
use litellm_redis_client::{Declared, RedisBatch, commands};
use parking_lot::Mutex;

pub use self::keys::AuthObject;
pub use self::local::TeamRoster;
pub use self::types::{
    Bag, BudgetEntry, BudgetFullEntry, BudgetLimitEntry, Dict, EndUserEntry, ExcludedIfNone, Ints,
    KeyEntry, Lists, MemberEntry, MemberRole, ModelAliases, ModelRegion, ModelTableEntry,
    ObjectPermissionEntry, ProjectEntry, Settings, TeamEntry, TeamMembershipEntry,
};
use crate::{
    Error, IntoValue, Number, Pending, RequestRedis, Send, codec,
    dual_cache::DualCache,
    failure_log::redis_failure,
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
    request::ready,
};

/// `RedisCache.async_increment_if_present`, byte for byte, sent as `EVAL` like it.
const INCREMENT_IF_PRESENT: &str = "if redis.call('EXISTS', KEYS[1]) == 0 then return false end return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])";

/// `SET` the entry only while its generation is the one the lease read.
const SET_LEASED: &str = "if (redis.call('GET', KEYS[2]) or '') ~= ARGV[1] then return 0 end redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3]) return 1";

/// The `DEL` and a new generation as one step, whatever order a batch sends its ops in. `DEL`
/// leads so a Redis at its memory limit still deletes the entry.
const DELETE_VOIDING_LEASES: &str = "local deleted = redis.call('DEL', KEYS[1]) redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[2]) return deleted";

const GENERATION_LABEL: &str = "auth_cache_generation";

/// Longer than any load a lease covers. A generation that expired reads like one never written.
const GENERATION_TTL: Duration = Duration::from_secs(600);

/// The proxy's auth object cache with `enable_redis_auth_cache` on: `UserApiKeyCache`'s two
/// `DualCache` partitions, so churn in management objects cannot evict user-key objects, in front of
/// one Redis shared by every pod. Writes and reads name an [`AuthObject`], which picks the partition;
/// an eviction carries a wire key and drops it from both. Values are what `json.loads` returns,
/// written as `json.dumps`, like `RedisCache.async_set_cache`.
pub struct AuthCache {
    key_objects: DualCache<LocalValue>,
    objects: DualCache<LocalValue>,
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

    /// `DualCache.attach_redis_cache`: memory stays, and `false` means a Redis was already attached.
    pub fn attach(&self, remote: Remote) -> bool {
        self.redis.attach(remote)
    }

    pub fn has_redis(&self) -> bool {
        self.redis.get().is_some()
    }

    pub fn settings(&self) -> Settings {
        *self.settings.lock()
    }

    /// `UserApiKeyCache.update_cache_ttl` and `update_in_memory_max_size` on both partitions, which
    /// keep what they hold.
    pub fn reconfigure(&self, settings: Settings) {
        *self.settings.lock() = settings;
        self.key_objects.reconfigure(settings.memory());
        self.objects.reconfigure(settings.memory());
    }

    pub fn get_local(&self, object: &AuthObject) -> Result<Option<Value>, Error> {
        Ok(self
            .partition(object)
            .memory()
            .get(&object.key())?
            .map(IntoValue::into_value))
    }

    pub fn local_team_fields(
        &self,
        object: &AuthObject,
        fields: Option<&[String]>,
        user_id: Option<&str>,
    ) -> Result<Option<Value>, Error> {
        if !matches!(object, AuthObject::Team(_)) {
            return Ok(None);
        }
        Ok(self
            .objects
            .memory()
            .get(&object.key())?
            .map(|row| row.team_fields(fields, user_id)))
    }

    /// Compact fields and the corresponding indexed roster from one cache entry.
    pub async fn team_snapshot(
        &self,
        request: Option<&RequestRedis>,
        object: &AuthObject,
    ) -> Result<Option<(Value, Option<TeamRoster>)>, Error> {
        if !matches!(object, AuthObject::Team(_) | AuthObject::TeamAlias(_)) {
            return Ok(None);
        }
        Ok(self
            .objects
            .get(request, &object.key(), None)
            .await?
            .map(|row| {
                let roster = row.team_roster();
                (row.team_fields(None, None), roster)
            }))
    }

    /// The compact team snapshot, optionally restricted to named scalar fields or one roster member.
    pub async fn team_fields(
        &self,
        request: Option<&RequestRedis>,
        object: &AuthObject,
        fields: Option<&[String]>,
        user_id: Option<&str>,
    ) -> Result<Option<Value>, Error> {
        if !matches!(object, AuthObject::Team(_)) {
            return Ok(None);
        }
        Ok(self
            .objects
            .get(request, &object.key(), None)
            .await?
            .map(|row| row.team_fields(fields, user_id)))
    }

    /// A local-only scalar update which neither inserts an entry nor resets its deadline.
    pub fn add_local_team_spend(&self, object: &AuthObject, delta: f64) -> Result<bool, Error> {
        if !matches!(object, AuthObject::Team(_)) {
            return Ok(false);
        }
        Ok(self
            .objects
            .memory()
            .get(&object.key())?
            .is_some_and(|row| row.add_team_spend(delta)))
    }

    /// This pod's copy under a wire key, whichever partition holds it. For a caller that knows only
    /// the key, as a Python pre-call hook does.
    pub fn get_local_by_key(&self, key: &str) -> Result<Option<Value>, Error> {
        match self.objects.memory().get(key)? {
            Some(value) => Ok(Some(value.into_value())),
            None => Ok(self
                .key_objects
                .memory()
                .get(key)?
                .map(IntoValue::into_value)),
        }
    }

    /// A memory write with `ttl`, or `default_in_memory_ttl` when it names none.
    pub fn set_local(
        &self,
        object: &AuthObject,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let partition = self.partition(object);
        Ok(partition.memory().set(
            &object.key(),
            value.into(),
            ttl.or(partition.default_in_memory_ttl()),
        )?)
    }

    /// Drops this pod's copy of a wire key from both partitions, as an eviction broadcast does.
    pub fn delete_local(&self, key: &str) -> Result<(), Error> {
        self.key_objects.memory().delete(key)?;
        Ok(self.objects.memory().delete(key)?)
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
        object: &AuthObject,
    ) -> Result<Option<Value>, Error> {
        Ok(self
            .partition(object)
            .get(request, &object.key(), None)
            .await?
            .map(IntoValue::into_value))
    }

    /// `DualCache.async_set_cache`, writing `json.dumps(value)`. `request` reads the key again
    /// from now on.
    pub async fn set(
        &self,
        request: Option<&RequestRedis>,
        object: &AuthObject,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        self.forget_misses(request, [object]);
        self.partition(object)
            .set(&object.key(), value.into(), ttl)
            .await
    }

    /// `DualCache.set_cache`, whose `RedisCache.set_cache` writes `str(value)`. `request` reads the
    /// key again from now on.
    pub async fn set_str(
        &self,
        request: Option<&RequestRedis>,
        object: &AuthObject,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        self.forget_misses(request, [object]);
        self.partition(object)
            .set_str(&object.key(), value.into(), ttl)
            .await
    }

    /// `DualCache.async_set_cache_pre_call`: memory now, the `SET` on the batch `send` picks.
    pub fn declare_set(
        &self,
        send: Send<'_>,
        object: &AuthObject,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<Pending<()>, Error> {
        self.partition(object)
            .declare_set(send, &object.key(), value.into(), ttl)
    }

    /// `DualCache.async_delete_cache_pre_call` of a wire key: both partitions now and the `DEL` on
    /// the batch `send` picks, in one script with the write that voids the key's leases.
    pub fn declare_delete(&self, send: Send<'_>, key: &str) -> Result<Pending<()>, Error> {
        let Some((client, generation)) = self
            .redis
            .client()
            .and_then(|client| Some((client, client.sibling_key(key, GENERATION_LABEL)?)))
        else {
            self.key_objects.memory().delete(key)?;
            return self.objects.declare_delete(send, key);
        };
        self.delete_local(key)?;
        if let Some(misses) = send.misses(client) {
            misses.note([key.to_owned()]);
        }
        let deleted = send.batch(client).add(commands::eval_source::<usize>(
            DELETE_VOIDING_LEASES,
            &[key, &generation],
            (new_generation(), GENERATION_TTL.as_secs()),
        ));
        Ok(Box::pin(
            async move { Ok(deleted.resolve().await.map(drop)?) },
        ))
    }

    /// The generation of `object` every pod shares, read on a cache miss before the database read
    /// that will fill it and handed to [`AuthCache::set_leased`]. Every delete of the key starts a
    /// new one. `None` without Redis, before the first delete, and when Redis did not answer.
    pub async fn lease(&self, object: &AuthObject) -> Option<String> {
        let client = self.redis.client()?;
        let generation = client.sibling_key(&object.key(), GENERATION_LABEL)?;
        client
            .send(commands::get::<String>(&generation))
            .await
            .unwrap_or_else(|error| {
                redis_failure!(&error, "an auth cache lease was not read from Redis");
                None
            })
    }

    /// The Redis `SET` of `object`, refused when its key was deleted on any pod since `lease` was
    /// read: the value may predate the write behind that delete. Memory is the caller's to write,
    /// and `request` reads the key again from now on.
    /// Also refused for a key no generation key can share a slot with, which no delete could void.
    /// `true` without Redis, and when Redis failed or `json.dumps` refuses the value, which only
    /// log as in [`AuthCache::set`].
    pub async fn set_leased(
        &self,
        request: Option<&RequestRedis>,
        object: &AuthObject,
        lease: Option<&str>,
        value: Value,
        ttl: Option<Duration>,
    ) -> bool {
        self.forget_misses(request, [object]);
        let Some(remote) = self.redis.get() else {
            return true;
        };
        let key = object.key();
        let key = key.as_str();
        let ttl = ttl
            .or(self.partition(object).default_in_memory_ttl())
            .unwrap_or(remote.default_ttl);
        let written = match codec::dumps(value) {
            Err(error) => Err(error),
            Ok(encoded) => match remote.client.sibling_key(key, GENERATION_LABEL) {
                None => Ok(false),
                Some(generation) => {
                    remote
                        .client
                        .send(commands::eval_source(
                            SET_LEASED,
                            &[key, &generation],
                            (lease.unwrap_or_default(), encoded, ttl.as_secs()),
                        ))
                        .await
                }
            }
            .map_err(Error::from),
        };
        written.unwrap_or_else(|error| {
            redis_failure!(&error, "LiteLLM Cache: exception in async add_cache");
            true
        })
    }

    /// `UserApiKeyCache.async_set_cache_pipeline`: the key-object entries in one pipeline, then
    /// the rest in another, each with one TTL.
    pub async fn set_many(
        &self,
        request: Option<&RequestRedis>,
        entries: Vec<(AuthObject, Value)>,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        self.forget_misses(request, entries.iter().map(|(object, _)| object));
        let (key_objects, others): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .partition(|(object, _)| object.is_key_object());
        let keyed = |entries: Vec<(AuthObject, Value)>| -> Vec<(String, LocalValue)> {
            entries
                .into_iter()
                .map(|(object, value)| (object.key(), value.into()))
                .collect()
        };
        self.key_objects
            .set_pipeline(keyed(key_objects), ttl)
            .await?;
        self.objects.set_pipeline(keyed(others), ttl).await
    }

    /// The auth prefetch write-back: memory first, an entry without a TTL taking
    /// `default_in_memory_ttl`, then `RedisCache.async_set_cache_pipeline_with_ttls`, which encodes
    /// every value before it sends anything and gives an entry without a TTL the connection's
    /// default. On a request batch the `SET`s are declared and nothing is left to wait for.
    /// [`Send::Now`] is one pipeline of its own rather than one round trip per entry.
    pub fn write_back(
        &self,
        send: Send<'_>,
        entries: Vec<(AuthObject, Value, Option<Duration>)>,
    ) -> Result<Pending<()>, Error> {
        self.forget_misses(send.request(), entries.iter().map(|(object, _, _)| object));
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
            .map(|(object, value, ttl)| {
                self.set_local(&object, value, ttl)?;
                Ok((object.key(), ttl))
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

    /// `UserApiKeyCache.async_delete_cache_keys` of wire keys. Both partitions drop every key
    /// before the first wait on Redis. A key deletes in one script with the write that voids its leases, all of them in one
    /// pipeline. Keys no generation can share a slot with go as `DualCache.async_delete_cache_keys`
    /// per partition, split by `main`'s `is_user_key_cache_key` rule. Every part runs even when another fails.
    pub async fn delete(&self, keys: &[String]) -> Result<(), Error> {
        keys.iter().try_for_each(|key| self.delete_local(key))?;
        let Some(client) = self.redis.client() else {
            return Ok(());
        };
        let pipeline = RedisBatch::pipelined(client.clone());
        let (leased, plain): (Vec<_>, Vec<_>) = keys
            .iter()
            .map(|key| (key, client.sibling_key(key, GENERATION_LABEL)))
            .partition(|(_, generation)| generation.is_some());
        let deleted: Vec<_> = leased
            .into_iter()
            .filter_map(|(key, generation)| Some((key, generation?)))
            .map(|(key, generation)| {
                pipeline.add(commands::eval_source::<usize>(
                    DELETE_VOIDING_LEASES,
                    &[key, &generation],
                    (new_generation(), GENERATION_TTL.as_secs()),
                ))
            })
            .collect();
        let (key_objects, others): (Vec<String>, Vec<String>) = plain
            .into_iter()
            .map(|(key, _)| key.clone())
            .partition(|key| is_user_key_cache_key(key));
        let (deleted, key_objects, others) = tokio::join!(
            Declared::all(deleted),
            self.key_objects.delete_keys(&key_objects),
            self.objects.delete_keys(&others)
        );
        deleted
            .into_iter()
            .try_for_each(|reply| Ok(reply.map(drop)?))
            .and(key_objects)
            .and(others)
    }

    /// `RedisCache.async_batch_get_cache` for keys the caller found missing in memory, one `MGET` on
    /// the batch `send` picks, noting the keys Redis lacks as the request's misses. `None` without
    /// Redis. Any failed read or undecodable value fails the whole read, as a failed `MGET` does.
    pub fn get_many(
        &self,
        send: Send<'_>,
        objects: &[AuthObject],
    ) -> Option<Pending<Vec<Option<Value>>>> {
        let client = self.redis.client()?;
        let keys: Vec<String> = objects.iter().map(AuthObject::key).collect();
        let misses = send.misses(client);
        let read = send.batch(client).add(commands::mget::<Vec<u8>>(
            &keys.iter().map(String::as_str).collect::<Vec<_>>(),
        ));
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
        object: &AuthObject,
        amount: Number,
    ) -> Result<Option<f64>, Error> {
        let Some(client) = self.redis.client() else {
            return Ok(None);
        };
        let amount = litellm_redis_client::Amount::from(amount);
        Ok(client
            .send(commands::eval_source(
                INCREMENT_IF_PRESENT,
                &[object.key().as_str()],
                amount,
            ))
            .await?)
    }

    fn forget_misses<'a>(
        &self,
        request: Option<&RequestRedis>,
        objects: impl IntoIterator<Item = &'a AuthObject>,
    ) {
        let (Some(request), Some(client)) = (request, self.redis.client()) else {
            return;
        };
        let misses = request.misses(client);
        objects
            .into_iter()
            .for_each(|object| misses.forget(&object.key()));
    }

    fn partition(&self, object: &AuthObject) -> &DualCache<LocalValue> {
        if object.is_key_object() {
            &self.key_objects
        } else {
            &self.objects
        }
    }
}

fn new_generation() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// `is_user_key_cache_key`, which `main` split a raw eviction's `DEL`s by.
fn is_user_key_cache_key(key: &str) -> bool {
    key.len() == 64
        && key
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
