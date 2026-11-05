use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
    time::Duration,
};

use litellm_cache::Error;
use litellm_redis_client::{Declared, Error as RedisError, commands};

use crate::{
    memory::{Clock, Memory, lock},
    remote::RemoteSlot,
};

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
/// `DualCache` with a `RedisCache` does. The family owns the Redis and passes it to each read.
pub struct MemoryTier<V: Clone> {
    memory: Memory<V>,
    policy: TierPolicy,
    reads: Mutex<RecentReads>,
    now: Clock,
}

impl<V: Clone + Send + Sync + 'static> MemoryTier<V> {
    pub fn read_through(policy: TierPolicy, now: Clock) -> Self {
        Self {
            memory: Memory::new(None, &now),
            reads: Mutex::new(RecentReads::new(policy.tracked_reads)),
            policy,
            now,
        }
    }

    pub fn memory(&self) -> &Memory<V> {
        &self.memory
    }

    pub fn now(&self) -> Duration {
        (self.now)()
    }

    /// Python's `DualCache.async_batch_get_cache`: memory first, then one round trip for the misses
    /// whose refresh is due, written back to memory. A failed Redis read serves memory, as
    /// `RedisCache` swallows it, and only a breaker refusal releases the refresh it reserved.
    pub async fn get_many<E: std::fmt::Display>(
        &self,
        remote: &RemoteSlot,
        keys: &[String],
        decode: impl Fn(&[u8]) -> Result<V, E>,
    ) -> Result<Vec<Option<V>>, Error> {
        let declared = self.declare_many(remote, crate::Send::Now, keys)?;
        self.resolve_many(declared, decode).await
    }

    /// The first half of `get_many`: reserves the due refreshes and declares their `MGET` on the
    /// batch `send` picks for the family's client, without sending anything.
    pub fn declare_many(
        &self,
        remote: &RemoteSlot,
        send: crate::Send<'_>,
        keys: &[String],
    ) -> Result<DeclaredRead, Error> {
        let fetch = match remote.client() {
            None => None,
            Some(client) => {
                let reservation = self.reserve(keys, &self.read_memory(keys)?)?;
                (!reservation.keys.is_empty()).then(|| {
                    let keys: Vec<&str> = reservation.keys.iter().map(String::as_str).collect();
                    let replies = send.batch(client).add(commands::mget::<Vec<u8>>(&keys));
                    Fetch {
                        reservation,
                        replies,
                    }
                })
            }
        };
        Ok(DeclaredRead {
            keys: keys.to_vec(),
            fetch,
        })
    }

    /// The second half of `get_many`. Memory is read again here, so a value written to memory
    /// since the declaration wins over what Redis answered.
    pub async fn resolve_many<E: std::fmt::Display>(
        &self,
        declared: DeclaredRead,
        decode: impl Fn(&[u8]) -> Result<V, E>,
    ) -> Result<Vec<Option<V>>, Error> {
        let DeclaredRead { keys, fetch } = declared;
        let Some(Fetch {
            reservation,
            replies,
        }) = fetch
        else {
            return self.read_memory(&keys);
        };
        let fetched = replies.resolve().await;
        let local = self.read_memory(&keys)?;
        let fetched = match fetched {
            Ok(fetched) => fetched,
            Err(RedisError::BreakerOpen) => {
                self.release(reservation)?;
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
        let read = reservation
            .keys
            .into_iter()
            .zip(decoded)
            .collect::<HashMap<_, _>>();
        keys.iter()
            .zip(local)
            .map(|(key, local)| match local {
                Some(value) => Ok(Some(value)),
                None => self.backfill(key, read.get(key).cloned().flatten()),
            })
            .collect()
    }

    fn read_memory(&self, keys: &[String]) -> Result<Vec<Option<V>>, Error> {
        keys.iter().map(|key| self.memory.get(key)).collect()
    }

    fn backfill(&self, key: &str, value: Option<V>) -> Result<Option<V>, Error> {
        if let Some(value) = &value {
            self.memory
                .set(key, value.clone(), Some(self.policy.backfill_ttl))?;
        }
        Ok(value)
    }

    fn reserve(&self, keys: &[String], local: &[Option<V>]) -> Result<Reservation, Error> {
        let now = self.now();
        let mut reads = lock(&self.reads);
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
        let mut reads = lock(&self.reads);
        for (key, previous) in reservation.keys.into_iter().zip(reservation.previous) {
            match previous {
                Some(read) => reads.insert(key, read),
                None => reads.remove(&key),
            }
        }
        Ok(())
    }
}

/// The GETs `declare_many` put on a batch, with the refreshes they reserved.
pub(crate) struct DeclaredRead {
    keys: Vec<String>,
    fetch: Option<Fetch>,
}

struct Fetch {
    reservation: Reservation,
    replies: Declared<Vec<Option<Vec<u8>>>>,
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
