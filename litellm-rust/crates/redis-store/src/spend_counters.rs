use std::{
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::Duration,
};

use litellm_cache::Error as MemoryError;
use litellm_cache_memory::InMemoryCache;
use litellm_python_compat::{json, repr::to_str};
use litellm_redis_client::{Declared, Expiry, RedisBatch, RedisClient};

use crate::tier::{Clock, system_clock};
use crate::{DecodeError, Number, RequestRedis, codec::python_value};

/// `InMemoryCache`'s defaults, which `spend_counter_cache` runs its memory tier with.
const MEMORY_ENTRIES: usize = 200;
pub const MEMORY_TTL: Duration = Duration::from_secs(600);
/// `spend_counter_cache`'s `default_in_memory_ttl`, which only its memory-only increment applies.
pub const LOCAL_INCREMENT_TTL: Duration = Duration::from_secs(60);
/// `RedisCache`'s TTL when `litellm.default_redis_ttl` is unset.
pub const REDIS_TTL: Duration = Duration::from_secs(60);

/// `RedisCache.async_set_max`, byte for byte, sent as `EVAL` like it: a counter only ever rises to
/// the authoritative spend.
const SET_MAX: &str = concat!(
    "local cur = redis.call('GET', KEYS[1]) ",
    "if cur == false or tonumber(cur) < tonumber(ARGV[1]) then ",
    "redis.call('SET', KEYS[1], ARGV[1]) ",
    "if tonumber(ARGV[2]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end ",
    "return ARGV[1] end ",
    "return cur"
);
/// `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`, the chunk `DualCache.async_delete_cache_keys` deletes at once.
const DELETE_CHUNK: usize = 1000;

/// One budget's spend counter. `parse` and `key` round-trip `main`'s `spend:*` keys exactly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpendCounter {
    Key {
        token: String,
        window: Option<String>,
    },
    Team {
        team_id: String,
        window: Option<String>,
    },
    TeamMember {
        user_id: String,
        team_id: String,
    },
    User(String),
    EndUser(String),
    Org(String),
    Tag(String),
    Project(String),
    ModelAccessGroup(String),
    ShadowEval(String),
}

#[derive(Debug, thiserror::Error)]
#[error("{0:?} is not a spend counter")]
pub struct NotASpendCounter(pub String);

impl SpendCounter {
    pub fn key(&self) -> String {
        let windowed = |scope: &str, id: &str, window: &Option<String>| match window {
            Some(window) => format!("spend:{scope}:{id}:window:{window}"),
            None => format!("spend:{scope}:{id}"),
        };
        match self {
            Self::Key { token, window } => windowed("key", token, window),
            Self::Team { team_id, window } => windowed("team", team_id, window),
            Self::TeamMember { user_id, team_id } => {
                format!("spend:team_member:{user_id}:{team_id}")
            }
            Self::User(id) => format!("spend:user:{id}"),
            Self::EndUser(id) => format!("spend:end_user:{id}"),
            Self::Org(id) => format!("spend:org:{id}"),
            Self::Tag(name) => format!("spend:tag:{name}"),
            Self::Project(id) => format!("spend:project:{id}"),
            Self::ModelAccessGroup(name) => format!("spend:model_access_group:{name}"),
            Self::ShadowEval(job_id) => format!("spend:shadow_eval:{job_id}"),
        }
    }

    pub fn parse(key: &str) -> Result<Self, NotASpendCounter> {
        let invalid = || NotASpendCounter(key.to_owned());
        let rest = key.strip_prefix("spend:").ok_or_else(invalid)?;
        let (scope, id) = rest.split_once(':').ok_or_else(invalid)?;
        let windowed = || match id.rsplit_once(":window:") {
            Some((id, window)) if !window.is_empty() => (id.to_owned(), Some(window.to_owned())),
            _ => (id.to_owned(), None),
        };
        Ok(match scope {
            "key" => {
                let (token, window) = windowed();
                Self::Key { token, window }
            }
            "team" => {
                let (team_id, window) = windowed();
                Self::Team { team_id, window }
            }
            "team_member" => {
                let (user_id, team_id) = id.rsplit_once(':').ok_or_else(invalid)?;
                Self::TeamMember {
                    user_id: user_id.to_owned(),
                    team_id: team_id.to_owned(),
                }
            }
            "user" => Self::User(id.to_owned()),
            "end_user" => Self::EndUser(id.to_owned()),
            "org" => Self::Org(id.to_owned()),
            "tag" => Self::Tag(id.to_owned()),
            "project" => Self::Project(id.to_owned()),
            "model_access_group" => Self::ModelAccessGroup(id.to_owned()),
            "shadow_eval" => Self::ShadowEval(id.to_owned()),
            _ => return Err(invalid()),
        })
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
}

/// Where a read was answered. Redis is cross-pod truth, including a clean miss; memory only holds
/// this pod's writes and answers only when there is no Redis or it failed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reading {
    Redis(Option<Number>),
    Memory(Option<Number>),
}

/// Who wins when a seed's `SET NX` lost and the winner's value is gone by the time it is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LostSeed {
    /// `SpendCounterReseed.coalesced`: answer with the seed without writing it.
    Answer,
    /// `SpendCounterReseed.coalesced_window`: `INCRBYFLOAT` the seed, keeping any TTL.
    Increment,
}

fn decode(raw: &[u8]) -> Result<Number, DecodeError> {
    Number::from_value(&python_value(raw)?).ok_or(DecodeError::Shape("a spend counter is a number"))
}

fn decode_reply(raw: Option<Vec<u8>>) -> Result<Option<Number>, DecodeError> {
    raw.as_deref().map(decode).transpose()
}

/// `json.dumps` of the value, as `RedisCache.async_set_cache` writes it.
fn json_bytes(value: Number) -> String {
    json::dumps(&value.to_value()).unwrap_or_else(|_| to_str(&value.to_value()))
}

fn whole_seconds(ttl: Duration) -> u64 {
    ttl.as_secs()
}

/// The proxy's spend counters: Redis shared by every pod, and this pod's memory copy that answers
/// only when Redis cannot.
pub struct SpendCounters {
    memory: InMemoryCache<Number>,
    remote: Option<RedisClient>,
    redis_ttl: Duration,
    updates: Mutex<()>,
}

impl SpendCounters {
    pub fn new(client: Option<RedisClient>, redis_ttl: Duration) -> Self {
        Self::with_clock(client, redis_ttl, system_clock())
    }

    pub fn with_clock(client: Option<RedisClient>, redis_ttl: Duration, now: Clock) -> Self {
        Self {
            memory: InMemoryCache::with_clock(Some(MEMORY_ENTRIES), Some(MEMORY_TTL), move || {
                now()
            }),
            remote: client,
            redis_ttl,
            updates: Mutex::new(()),
        }
    }

    pub fn has_redis(&self) -> bool {
        self.remote.is_some()
    }

    pub fn redis_ttl(&self) -> Duration {
        self.redis_ttl
    }

    /// `read_spend_counter_cache_value`: Redis first, memory when Redis is absent or failed.
    pub async fn read(&self, counter: &SpendCounter) -> Result<Reading, MemoryError> {
        if let Some(remote) = &self.remote {
            match remote
                .get::<Vec<u8>>(&counter.key())
                .await
                .map_err(Error::from)
                .and_then(|raw| decode_reply(raw).map_err(Error::from))
            {
                Ok(value) => return Ok(Reading::Redis(value)),
                Err(error) => {
                    tracing::debug!(%error, "spend counter Redis read failed, falling back to memory");
                }
            }
        }
        Ok(Reading::Memory(self.get_local(counter)?))
    }

    /// One round trip for many counters; `None` when there is no Redis or the read failed, so each
    /// caller reads on its own.
    pub async fn read_many(&self, counters: &[SpendCounter]) -> Option<Vec<Option<Number>>> {
        let declared = self.declare_read_many(RedisClient::batch, counters)?;
        declared.resolve().await
    }

    /// `read_many` declared on the request's pre-call batch, so it goes out with the request's other
    /// pre-call reads.
    pub fn declare_read_many_pre_call(
        &self,
        request: &RequestRedis,
        counters: &[SpendCounter],
    ) -> Option<DeclaredReads> {
        self.declare_read_many(|client| request.pre(client), counters)
    }

    fn declare_read_many(
        &self,
        batch_for: impl FnOnce(&RedisClient) -> RedisBatch,
        counters: &[SpendCounter],
    ) -> Option<DeclaredReads> {
        let batch = batch_for(self.remote.as_ref()?);
        let keys: Vec<String> = counters.iter().map(SpendCounter::key).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        Some(DeclaredReads(batch.mget::<Vec<u8>>(&keys)))
    }

    /// `increment_spend_counter_cache`: `INCRBYFLOAT` refreshing the TTL, mirrored into memory. A
    /// failed increment may or may not have landed, so the counter is dropped everywhere before the
    /// error returns. Without Redis it is a memory increment.
    pub async fn increment(&self, counter: &SpendCounter, amount: Number) -> Result<Number, Error> {
        let Some(remote) = &self.remote else {
            return Ok(self.increment_local(counter, amount)?);
        };
        let key = counter.key();
        match remote
            .incr_by_float(&key, amount.into(), Expiry::Refresh(self.redis_ttl))
            .await
        {
            Ok(total) => {
                let total = Number::Float(total);
                self.memory.set_cache(key, total, None)?;
                Ok(total)
            }
            Err(error) => {
                self.invalidate(std::slice::from_ref(counter)).await;
                Err(error.into())
            }
        }
    }

    /// `run_spend_counter_pipeline`: every increment in one round trip, totals in order. The caller
    /// decides what a failure means for counters whose increment may or may not have landed.
    pub async fn increment_many(
        &self,
        increments: &[(SpendCounter, Number)],
    ) -> Result<Vec<Number>, Error> {
        let Some(remote) = &self.remote else {
            return increments
                .iter()
                .map(|(counter, amount)| Ok(self.increment_local(counter, *amount)?))
                .collect();
        };
        let pipeline = increments
            .iter()
            .fold(remote.pipeline(), |pipeline, (counter, amount)| {
                pipeline.incr_by_float(&counter.key(), (*amount).into(), Some(self.redis_ttl))
            });
        let replies = remote.run(&pipeline).await;
        let mut totals = Vec::with_capacity(replies.len());
        for ((counter, _), reply) in increments.iter().zip(replies) {
            let total = Number::Float(float_reply(reply?)?);
            self.memory.set_cache(counter.key(), total, None)?;
            totals.push(total);
        }
        Ok(totals)
    }

    /// `increment_many` on the request's post-call batch. Each total reaches memory once the batch
    /// goes out, and a counter whose increment failed is dropped everywhere. `false` without Redis.
    pub fn increment_many_post_call(
        self: &Arc<Self>,
        request: &RequestRedis,
        increments: &[(SpendCounter, Number)],
    ) -> bool {
        let Some(remote) = &self.remote else {
            return false;
        };
        let declared = self.declare_increments(&request.post(remote), increments);
        let counters = increments
            .iter()
            .map(|(counter, _)| counter.clone())
            .collect::<Vec<_>>();
        let store = Arc::clone(self);
        request.spawn(async move {
            for ((key, total), counter) in declared.into_iter().zip(counters) {
                match total.settled().await {
                    Ok(total) => {
                        if let Err(error) = store.memory.set_cache(key, Number::Float(total), None)
                        {
                            tracing::warn!(%error, "mirroring a spend counter total into memory failed");
                        }
                    }
                    Err(error) => {
                        tracing::warn!(%error, %key, "spend counter increment did not land in the post-call pipeline; invalidating it");
                        store.invalidate(std::slice::from_ref(&counter)).await;
                    }
                }
            }
        });
        true
    }

    /// `_repair_stale_spend_counter`: raise memory and Redis to `spend`, never lowering either.
    /// `None` without Redis, else what Redis holds afterwards.
    pub async fn set_max(
        &self,
        counter: &SpendCounter,
        spend: Number,
    ) -> Result<Option<Number>, Error> {
        let key = counter.key();
        {
            let _guard = lock(&self.updates);
            let raise = self
                .memory
                .get_cache(&key)?
                .is_none_or(|cached| cached.as_f64() < spend.as_f64());
            if raise {
                self.memory.set_cache(key.clone(), spend, None)?;
            }
        }
        let Some(remote) = &self.remote else {
            return Ok(None);
        };
        let stored: Option<Vec<u8>> = remote
            .eval_source(
                SET_MAX,
                &[&key],
                (
                    to_str(&spend.to_value()),
                    whole_seconds(self.redis_ttl).to_string(),
                ),
            )
            .await?;
        Ok(stored
            .as_deref()
            .map(|raw| {
                std::str::from_utf8(raw)
                    .ok()
                    .and_then(|text| text.trim().parse::<f64>().ok())
                    .map(Number::Float)
                    .ok_or(DecodeError::Shape("set_max returned a non-number"))
            })
            .transpose()?)
    }

    /// `SpendCounterReseed.coalesced(_window)`: seed a cold counter with `SET NX` so one pod wins,
    /// answer with whatever the counter holds, and mirror it into memory. Without Redis, memory
    /// keeps the larger of the seed and what it held.
    pub async fn seed(
        &self,
        counter: &SpendCounter,
        spend: Number,
        lost: LostSeed,
    ) -> Result<Number, Error> {
        let key = counter.key();
        let Some(remote) = &self.remote else {
            let _guard = lock(&self.updates);
            let seeded = match self.memory.get_cache(&key)? {
                Some(cached) if cached.as_f64() > spend.as_f64() => Number::Float(cached.as_f64()),
                _ => spend,
            };
            self.memory.set_cache(key, seeded, None)?;
            return Ok(seeded);
        };
        let current = if remote
            .set_nx(&key, json_bytes(spend), self.redis_ttl)
            .await?
        {
            spend
        } else {
            match decode_reply(remote.get::<Vec<u8>>(&key).await?)? {
                Some(winner) => Number::Float(winner.as_f64()),
                None => match lost {
                    LostSeed::Answer => spend,
                    LostSeed::Increment => Number::Float(
                        remote
                            .incr_by_float(&key, spend.into(), Expiry::KeepExisting(self.redis_ttl))
                            .await?,
                    ),
                },
            }
        };
        self.memory.set_cache(key, current, None)?;
        Ok(current)
    }

    /// A spend reset: memory, then Redis, each for `ttl` or its own default.
    pub async fn set(
        &self,
        counter: &SpendCounter,
        spend: Number,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let key = counter.key();
        self.memory.set_cache(key.clone(), spend, ttl)?;
        let Some(remote) = &self.remote else {
            return Ok(());
        };
        remote
            .set(&key, json_bytes(spend), ttl.unwrap_or(self.redis_ttl))
            .await?;
        Ok(())
    }

    /// Drop counters from memory, then from Redis in one round trip.
    pub async fn delete(&self, counters: &[SpendCounter]) -> Result<(), Error> {
        for counter in counters {
            self.memory.delete_cache(&counter.key())?;
        }
        let Some(remote) = &self.remote else {
            return Ok(());
        };
        let keys: Vec<String> = counters.iter().map(SpendCounter::key).collect();
        for chunk in keys.chunks(DELETE_CHUNK) {
            let chunk: Vec<&str> = chunk.iter().map(String::as_str).collect();
            remote.del(&chunk).await?;
        }
        Ok(())
    }

    /// `refresh_spend_counter_ttl`: `false` without Redis or for a counter Redis no longer holds.
    pub async fn refresh_ttl(&self, counter: &SpendCounter) -> Result<bool, Error> {
        let Some(remote) = &self.remote else {
            return Ok(false);
        };
        Ok(remote.expire(&counter.key(), self.redis_ttl).await?)
    }

    pub fn get_local(&self, counter: &SpendCounter) -> Result<Option<Number>, MemoryError> {
        self.memory.get_cache(&counter.key())
    }

    pub fn set_local(
        &self,
        counter: &SpendCounter,
        spend: Number,
        ttl: Option<Duration>,
    ) -> Result<(), MemoryError> {
        self.memory.set_cache(counter.key(), spend, ttl).map(drop)
    }

    pub fn delete_local(&self, counter: &SpendCounter) -> Result<(), MemoryError> {
        self.memory.delete_cache(&counter.key())
    }

    /// `DualCache.async_increment_cache(local_only=True)`.
    pub fn increment_local(
        &self,
        counter: &SpendCounter,
        amount: Number,
    ) -> Result<Number, MemoryError> {
        let key = counter.key();
        let _guard = lock(&self.updates);
        let total = self.memory.get_cache(&key)?.unwrap_or(Number::Int(0)) + amount;
        self.memory
            .set_cache(key, total, Some(LOCAL_INCREMENT_TTL))?;
        Ok(total)
    }

    pub fn snapshot(&self) -> Result<Vec<(String, Number)>, MemoryError> {
        self.memory.entries()
    }

    pub fn flush_local(&self) -> Result<(), MemoryError> {
        self.memory.flush_cache()
    }

    /// `invalidate_spend_counter`: memory always, Redis best effort.
    pub async fn invalidate(&self, counters: &[SpendCounter]) {
        if let Err(error) = self.delete(counters).await {
            tracing::debug!(%error, "dropping spend counters from Redis failed");
        }
    }

    fn declare_increments(
        &self,
        batch: &RedisBatch,
        increments: &[(SpendCounter, Number)],
    ) -> Vec<(String, Declared<f64>)> {
        increments
            .iter()
            .map(|(counter, amount)| {
                let key = counter.key();
                let total = batch.incr_by_float(&key, (*amount).into(), Some(self.redis_ttl));
                (key, total)
            })
            .collect()
    }
}

/// One `MGET` declared on a batch, answered once it goes out.
pub struct DeclaredReads(Declared<Vec<Option<Vec<u8>>>>);

impl DeclaredReads {
    /// Flushes the batch if nobody has yet. `None` when the read or a decode failed.
    pub async fn resolve(self) -> Option<Vec<Option<Number>>> {
        self.0
            .resolve()
            .await
            .map_err(Error::from)
            .and_then(|replies| {
                replies
                    .into_iter()
                    .map(|raw| decode_reply(raw).map_err(Error::from))
                    .collect::<Result<Vec<_>, _>>()
            })
            .inspect_err(|error| {
                tracing::debug!(%error, "spend counter batch read failed, falling back to per-key reads");
            })
            .ok()
    }
}

fn lock(mutex: &Mutex<()>) -> MutexGuard<'_, ()> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn float_reply(value: litellm_redis_client::Value) -> Result<f64, Error> {
    let total = match &value {
        litellm_redis_client::Value::BulkString(bytes) => std::str::from_utf8(bytes)
            .ok()
            .and_then(|text| text.parse().ok()),
        litellm_redis_client::Value::Double(total) => Some(*total),
        _ => None,
    };
    Ok(total.ok_or(DecodeError::Shape("INCRBYFLOAT did not answer a number"))?)
}
