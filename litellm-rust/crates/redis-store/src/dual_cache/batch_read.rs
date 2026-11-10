use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
    time::Duration,
};

use litellm_redis_client::{Declared, Error as RedisError, RedisBatch, commands};

use super::{DualCache, MemorySettings};
use crate::{Error, FromValue, Send, codec, memory::lock, request::Misses};

/// A batch-reading family's `DualCache` settings, as the host reads them from `litellm` globals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierPolicy {
    /// Python's `redis_batch_cache_expiry`: a key memory misses is read from Redis at most this often.
    pub refresh_after: Duration,
    /// Python's `default_max_redis_batch_cache_size`: how many keys the refresh throttle tracks.
    pub tracked_reads: usize,
    /// `litellm.default_in_memory_ttl`, which a value read from Redis keeps in memory.
    pub default_in_memory_ttl: Option<Duration>,
}

impl TierPolicy {
    pub(crate) fn memory(&self) -> MemorySettings {
        MemorySettings {
            entries: None,
            ttl: self.default_in_memory_ttl,
        }
    }
}

/// `DualCache.last_redis_batch_access_time` with its expiry, which only `async_batch_get_cache` reads.
pub(crate) struct BatchReads {
    refresh_after: Duration,
    recent: Mutex<RecentReads>,
}

impl BatchReads {
    pub(super) fn new(policy: TierPolicy) -> Self {
        Self {
            refresh_after: policy.refresh_after,
            recent: Mutex::new(RecentReads::new(policy.tracked_reads)),
        }
    }
}

impl<V: Clone + FromValue + std::marker::Send + Sync + 'static> DualCache<V, BatchReads> {
    /// `DualCache.async_batch_get_cache`: memory first, then one `MGET` for the misses whose refresh
    /// is due, written back to memory. A failed Redis read serves memory, as `RedisCache` swallows
    /// it, and only a breaker refusal releases the refresh it reserved.
    pub(crate) async fn batch_get(&self, keys: &[String]) -> Result<Vec<Option<V>>, Error> {
        let declared = self.declare_batch_get(Send::Now, keys)?;
        self.resolve_batch_get(declared).await
    }

    /// The first half of `batch_get`: reserves the due refreshes and declares their `MGET` on the
    /// batch `send` picks, without sending anything.
    pub(crate) fn declare_batch_get(
        &self,
        send: Send<'_>,
        keys: &[String],
    ) -> Result<DeclaredRead, Error> {
        let fetch = match self.redis.client() {
            None => None,
            Some(client) => {
                let reservation = self.reserve(keys, &self.read_memory(keys)?);
                (!reservation.keys.is_empty()).then(|| {
                    let keys: Vec<&str> = reservation.keys.iter().map(String::as_str).collect();
                    let batch = send.batch(client);
                    let replies = batch.add(commands::mget::<Vec<u8>>(&keys));
                    Fetch {
                        reservation,
                        batch,
                        replies,
                        misses: send.misses(client),
                    }
                })
            }
        };
        Ok(DeclaredRead {
            keys: keys.to_vec(),
            fetch,
        })
    }

    /// The second half of `batch_get`, flushing the read's batch if nobody has yet.
    pub(crate) async fn resolve_batch_get(
        &self,
        declared: DeclaredRead,
    ) -> Result<Vec<Option<V>>, Error> {
        if let Some(batch) = declared.batch() {
            batch.flush().await;
        }
        self.settle_batch_get(declared).await
    }

    /// The second half of `batch_get` once the read's batch answers, without flushing it. Memory
    /// is read again here, so a value written to memory since the declaration wins over what Redis
    /// answered. The keys Redis lacks are noted as the request's misses. An undecodable value
    /// serves memory for the whole read, where `main`'s `RedisCache` returns the raw bytes and
    /// `DualCache` backfills memory with them.
    pub(crate) async fn settle_batch_get(
        &self,
        declared: DeclaredRead,
    ) -> Result<Vec<Option<V>>, Error> {
        let DeclaredRead { keys, fetch } = declared;
        let Some(Fetch {
            reservation,
            batch: _,
            replies,
            misses,
        }) = fetch
        else {
            return self.read_memory(&keys);
        };
        let fetched = replies.settled().await;
        let local = self.read_memory(&keys)?;
        let fetched = match fetched {
            Ok(fetched) => fetched,
            Err(RedisError::BreakerOpen) => {
                self.release(reservation);
                return Ok(local);
            }
            Err(error) => {
                crate::failure_log::redis_failure!(
                    &error,
                    "Redis read failed, serving the memory tier"
                );
                return Ok(local);
            }
        };
        if let Some(misses) = misses {
            misses.note(
                reservation
                    .keys
                    .iter()
                    .zip(&fetched)
                    .filter(|(_, raw)| raw.is_none())
                    .map(|(key, _)| key.clone()),
            );
        }
        let decoded = match fetched
            .iter()
            .map(|raw| raw.as_deref().map(codec::decode).transpose())
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
                None => self.backfill(key, read.get(key).cloned().flatten(), None),
            })
            .collect()
    }

    fn read_memory(&self, keys: &[String]) -> Result<Vec<Option<V>>, Error> {
        keys.iter().map(|key| Ok(self.memory.get(key)?)).collect()
    }

    fn reserve(&self, keys: &[String], local: &[Option<V>]) -> Reservation {
        let now = self.now();
        let refresh_after = self.reads.refresh_after;
        let mut recent = lock(&self.reads.recent);
        let due = keys
            .iter()
            .zip(local)
            .filter(|(_, value)| value.is_none())
            .filter_map(|(key, _)| {
                let previous = recent.get(key);
                let due = previous.is_none_or(|read| now.saturating_sub(read) >= refresh_after);
                due.then(|| {
                    recent.insert(key.clone(), now);
                    (key.clone(), previous)
                })
            })
            .collect::<Vec<_>>();
        let (keys, previous) = due.into_iter().unzip();
        Reservation { keys, previous }
    }

    fn release(&self, reservation: Reservation) {
        let mut recent = lock(&self.reads.recent);
        for (key, previous) in reservation.keys.into_iter().zip(reservation.previous) {
            match previous {
                Some(read) => recent.insert(key, read),
                None => recent.remove(&key),
            }
        }
    }
}

/// The `MGET` `declare_batch_get` put on a batch, with the refreshes it reserved.
pub(crate) struct DeclaredRead {
    keys: Vec<String>,
    fetch: Option<Fetch>,
}

impl DeclaredRead {
    /// The batch carrying the read, `None` when memory answers it alone.
    pub(crate) fn batch(&self) -> Option<RedisBatch> {
        self.fetch.as_ref().map(|fetch| fetch.batch.clone())
    }
}

struct Fetch {
    reservation: Reservation,
    batch: RedisBatch,
    replies: Declared<Vec<Option<Vec<u8>>>>,
    misses: Option<Misses>,
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
