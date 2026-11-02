use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_redis_client::{Error as RedisError, RedisClient};

use crate::{BaseCache, DeleteCache, Error, ExactCacheContext, FlushCache};

pub type Clock = Arc<dyn Fn() -> Duration + Send + Sync>;

/// How a read-through family uses its memory tier, declared once by the family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierPolicy {
    /// Python's `redis_batch_cache_expiry`: a key memory misses is read from Redis at most this often.
    pub refresh_after: Duration,
    /// How long a value read from Redis stays in memory.
    pub backfill_ttl: Duration,
    pub on_breaker_open: OnBreakerOpen,
    /// Python's `default_max_redis_batch_cache_size`: how many keys the refresh throttle tracks.
    pub tracked_reads: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnBreakerOpen {
    /// Serve what memory holds and let the next read try Redis again.
    ServeMemory,
}

/// The memory tier in front of Redis that a read-through family builds for itself, as Python's
/// `DualCache` with a `RedisCache` does.
pub struct DualCache<L1, L2> {
    memory: L1,
    remote: Option<L2>,
    policy: TierPolicy,
    reads: Mutex<RecentReads>,
    updates: Mutex<()>,
    now: Clock,
}

pub fn system_clock() -> Clock {
    Arc::new(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    })
}

impl<V, L1> DualCache<L1, RedisClient>
where
    V: Clone + Send + Sync + 'static,
    L1: BaseCache<Value = V, Context = ExactCacheContext> + DeleteCache + FlushCache,
{
    pub fn read_through(
        memory: L1,
        remote: Option<RedisClient>,
        policy: TierPolicy,
        now: Clock,
    ) -> Self {
        Self {
            memory,
            remote,
            reads: Mutex::new(RecentReads::new(policy.tracked_reads)),
            policy,
            updates: Mutex::new(()),
            now,
        }
    }

    pub fn memory(&self) -> &L1 {
        &self.memory
    }

    pub fn remote(&self) -> Option<&RedisClient> {
        self.remote.as_ref()
    }

    pub fn now(&self) -> Duration {
        (self.now)()
    }

    pub fn set_memory(&self, key: &str, value: V, ttl: Duration) -> Result<(), Error> {
        self.memory
            .set_cache(key, value, &ExactCacheContext { ttl: Some(ttl) })
    }

    /// Read-modify-write of one memory entry, atomic against every other update of this tier.
    pub fn update_memory(
        &self,
        key: &str,
        ttl: Duration,
        update: impl FnOnce(Option<V>) -> V,
    ) -> Result<V, Error> {
        let _guard = lock(&self.updates)?;
        let value = update(self.memory.get_cache(key, &ExactCacheContext::default())?);
        self.set_memory(key, value.clone(), ttl)?;
        Ok(value)
    }

    /// Python's `DualCache.async_batch_get_cache`: memory first, then one MGET for the misses
    /// whose refresh is due, written back to memory. A failed Redis read serves memory, as
    /// `RedisCache` swallows it, and only a breaker refusal releases the refresh it reserved.
    pub async fn get_many<E: std::fmt::Display>(
        &self,
        keys: &[String],
        decode: impl Fn(&[u8]) -> Result<V, E>,
    ) -> Result<Vec<Option<V>>, Error> {
        let context = ExactCacheContext::default();
        let local = keys
            .iter()
            .map(|key| self.memory.get_cache(key, &context))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(remote) = &self.remote else {
            return Ok(local);
        };
        let reserved = self.reserve(keys, &local)?;
        if reserved.keys.is_empty() {
            return Ok(local);
        }
        let due = reserved.keys.iter().map(String::as_str).collect::<Vec<_>>();
        let fetched = match remote.mget::<Vec<u8>>(&due).await {
            Ok(fetched) => fetched,
            Err(RedisError::BreakerOpen) => {
                self.release(reserved)?;
                return Ok(local);
            }
            Err(error) => {
                tracing::warn!(%error, "Redis read failed, serving the memory tier");
                return Ok(local);
            }
        };
        let decoded = match fetched
            .iter()
            .map(|raw| raw.as_deref().map(&decode).transpose())
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(decoded) => decoded,
            Err(error) => {
                tracing::warn!(%error, "a Redis value did not decode, serving the memory tier");
                return Ok(local);
            }
        };
        let read = reserved
            .keys
            .into_iter()
            .zip(decoded)
            .collect::<HashMap<_, _>>();
        for (key, value) in &read {
            if let Some(value) = value {
                self.set_memory(key, value.clone(), self.policy.backfill_ttl)?;
            }
        }
        Ok(keys
            .iter()
            .zip(local)
            .map(|(key, local)| read.get(key).cloned().unwrap_or(local))
            .collect())
    }

    pub fn delete_memory(&self, key: &str) -> Result<(), Error> {
        self.memory.delete_cache(key)
    }

    pub fn flush_memory(&self) -> Result<(), Error> {
        self.memory.flush_cache()
    }

    fn reserve(&self, keys: &[String], local: &[Option<V>]) -> Result<Reservation, Error> {
        let now = self.now();
        let mut reads = lock(&self.reads)?;
        let due = keys
            .iter()
            .zip(local)
            .filter(|(_, value)| value.is_none())
            .filter_map(|(key, _)| {
                let previous = reads.get(key);
                let due = previous
                    .is_none_or(|read| now.saturating_sub(read) >= self.policy.refresh_after);
                due.then(|| {
                    reads.insert(key.clone(), now);
                    (key.clone(), previous)
                })
            })
            .collect::<Vec<_>>();
        let (keys, previous) = due.into_iter().unzip();
        Ok(Reservation { keys, previous })
    }

    fn release(&self, reservation: Reservation) -> Result<(), Error> {
        let mut reads = lock(&self.reads)?;
        for (key, previous) in reservation.keys.into_iter().zip(reservation.previous) {
            match previous {
                Some(read) => reads.insert(key, read),
                None => reads.remove(&key),
            }
        }
        Ok(())
    }
}

fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, Error> {
    Ok(mutex.lock()?)
}

struct Reservation {
    keys: Vec<String>,
    previous: Vec<Option<Duration>>,
}

/// Python's `LimitedSizeOrderedDict`: insertion-ordered, and any write at capacity drops the
/// oldest entry first, even when the written key is already present.
struct RecentReads {
    capacity: usize,
    order: VecDeque<String>,
    times: HashMap<String, Duration>,
}

impl RecentReads {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::new(),
            times: HashMap::new(),
        }
    }

    fn get(&self, key: &str) -> Option<Duration> {
        self.times.get(key).copied()
    }

    fn insert(&mut self, key: String, time: Duration) {
        if self.times.len() >= self.capacity
            && let Some(oldest) = self.order.pop_front()
        {
            self.times.remove(&oldest);
        }
        if self.times.insert(key.clone(), time).is_none() {
            self.order.push_back(key);
        }
    }

    fn remove(&mut self, key: &str) {
        if self.times.remove(key).is_some() {
            self.order.retain(|tracked| tracked != key);
        }
    }
}
