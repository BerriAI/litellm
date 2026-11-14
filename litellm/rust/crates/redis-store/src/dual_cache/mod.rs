mod batch_read;
mod increment;
mod read;
mod write;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

pub use self::batch_read::TierPolicy;
pub(crate) use self::{
    batch_read::{BatchReads, Reservation},
    increment::PipelineIncrement,
};
use crate::{
    memory::{Clock, Memory, lock},
    remote::RemoteSlot,
};

/// The memory tier's `InMemoryCache` size and the `default_in_memory_ttl` a write or a Redis
/// backfill takes when it names no TTL. `None` keeps `InMemoryCache`'s own default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MemorySettings {
    pub(crate) entries: Option<usize>,
    pub(crate) ttl: Option<Duration>,
}

/// Python's `DualCache`: a per-worker `InMemoryCache` in front of the Redis every pod shares. Each
/// method is the `DualCache` method it names. A family whose Python reached past `DualCache` to its
/// `in_memory_cache` or `redis_cache` uses `memory` and `redis` the same way.
pub(crate) struct DualCache<V: Clone, R = ()> {
    memory: Memory<V>,
    redis: Arc<RemoteSlot>,
    default_in_memory_ttl: Mutex<Option<Duration>>,
    now: Clock,
    reads: R,
}

impl<V: Clone> DualCache<V> {
    /// `redis` may be shared with the family's other caches, as one `RedisCache` is attached to each.
    pub(crate) fn new(redis: Arc<RemoteSlot>, memory: MemorySettings, now: Clock) -> Self {
        Self {
            memory: Memory::new(memory.entries, &now),
            redis,
            default_in_memory_ttl: Mutex::new(memory.ttl),
            now,
            reads: (),
        }
    }

    /// A cache that batch-reads, with the `async_batch_get_cache` throttle `policy` configures.
    pub(crate) fn batch_reading(
        redis: Arc<RemoteSlot>,
        policy: TierPolicy,
        now: Clock,
    ) -> DualCache<V, BatchReads> {
        let Self {
            memory,
            redis,
            default_in_memory_ttl,
            now,
            reads: (),
        } = Self::new(redis, policy.memory(), now);
        DualCache {
            memory,
            redis,
            default_in_memory_ttl,
            now,
            reads: BatchReads::new(policy),
        }
    }
}

impl<V: Clone, R> DualCache<V, R> {
    pub(crate) fn memory(&self) -> &Memory<V> {
        &self.memory
    }

    pub(crate) fn redis(&self) -> &RemoteSlot {
        &self.redis
    }

    pub(crate) fn now(&self) -> Duration {
        (self.now)()
    }

    pub(crate) fn default_in_memory_ttl(&self) -> Option<Duration> {
        *lock(&self.default_in_memory_ttl)
    }

    /// `DualCache.update_cache_ttl` and `update_in_memory_max_size` together: both apply in place,
    /// and the entries memory already holds stay.
    pub(crate) fn reconfigure(&self, memory: MemorySettings) {
        *lock(&self.default_in_memory_ttl) = memory.ttl;
        self.memory.update_max_size(memory.entries);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
        time::Duration,
    };

    use rstest::rstest;
    use tokio::runtime::Handle;

    use super::{DualCache, MemorySettings, PipelineIncrement};
    use crate::{Clock, Number, RequestRedis, memory::Counter, remote::RemoteSlot};

    #[derive(Clone, Debug, PartialEq)]
    enum Entry {
        Number(Number),
        Text,
    }

    impl From<Number> for Entry {
        fn from(number: Number) -> Self {
            Self::Number(number)
        }
    }

    impl Counter for Entry {
        fn as_number(&self) -> Option<Number> {
            match self {
                Self::Number(number) => Some(*number),
                Self::Text => None,
            }
        }
    }

    fn clock() -> (Clock, Arc<AtomicU64>) {
        let seconds = Arc::new(AtomicU64::new(1_000));
        let read = seconds.clone();
        let clock: Clock = Arc::new(move || Duration::from_secs(read.load(Ordering::SeqCst)));
        (clock, seconds)
    }

    fn without_redis<V: Clone>(
        default_in_memory_ttl: Option<Duration>,
        now: Clock,
    ) -> DualCache<V> {
        DualCache::new(
            Arc::new(RemoteSlot::new(None)),
            MemorySettings {
                entries: None,
                ttl: default_in_memory_ttl,
            },
            now,
        )
    }

    fn increment(key: &str, amount: Number) -> PipelineIncrement {
        PipelineIncrement {
            key: key.to_owned(),
            amount,
            ttl: None,
        }
    }

    // InMemoryCache's own default TTL is 600 s, which an entry outlives here only when the cache
    // configured none
    #[rstest]
    #[case::default_in_memory_ttl(Some(Duration::from_secs(5)), None)]
    #[case::in_memory_cache_default(None, Some(Number::Int(1)))]
    #[tokio::test]
    async fn a_set_without_a_ttl_takes_default_in_memory_ttl(
        #[case] default_in_memory_ttl: Option<Duration>,
        #[case] after_six_seconds: Option<Number>,
    ) {
        let (now, seconds) = clock();
        let cache = without_redis(default_in_memory_ttl, now);

        cache.set("k", Number::Int(1), None).await.unwrap();
        seconds.fetch_add(6, Ordering::SeqCst);

        assert_eq!(cache.memory().get("k").unwrap(), after_six_seconds);
    }

    #[rstest]
    #[case::absent_starts_at_int_zero(None, Number::Int(2), Number::Int(2))]
    #[case::int_plus_int_stays_int(Some(Number::Int(1)), Number::Int(2), Number::Int(3))]
    #[case::a_float_makes_a_float(Some(Number::Int(1)), Number::Float(0.5), Number::Float(1.5))]
    fn a_local_increment_adds_as_python_adds(
        #[case] stored: Option<Number>,
        #[case] amount: Number,
        #[case] expected: Number,
    ) {
        let (now, _) = clock();
        let cache = without_redis::<Number>(None, now);
        if let Some(stored) = stored {
            cache.memory().set("k", stored, None).unwrap();
        }

        let total = cache.memory().increment("k", amount, None).unwrap();

        assert_eq!(total, expected);
        assert_eq!(cache.memory().get("k").unwrap(), Some(expected));
    }

    #[test]
    fn a_local_increment_of_a_non_number_fails_and_keeps_the_entry() {
        let (now, _) = clock();
        let cache = without_redis::<Entry>(None, now);
        cache.memory().set("k", Entry::Text, None).unwrap();

        let total = cache.memory().increment("k", Number::Int(1), None);

        assert!(total.is_err());
        assert_eq!(cache.memory().get("k").unwrap(), Some(Entry::Text));
    }

    // async_increment_cache and async_increment_cache_pipeline catch the in-memory failure with the
    // Redis one, so the caller gets None and nothing is sent
    #[tokio::test]
    async fn an_increment_memory_refuses_answers_none() {
        let (now, _) = clock();
        let cache = without_redis::<Entry>(None, now);
        cache.memory().set("text", Entry::Text, None).unwrap();

        let single = cache.increment("text", Number::Int(1), None).await;
        let pipelined = cache
            .increment_pipeline(&[
                increment("before", Number::Int(1)),
                increment("text", Number::Int(1)),
            ])
            .await;

        assert_eq!(single, None);
        assert_eq!(pipelined, None);
        assert_eq!(cache.memory().get("text").unwrap(), Some(Entry::Text));
    }

    // update_cache_ttl and update_in_memory_max_size change the settings in place, so what memory
    // already holds survives and the next write takes the new TTL
    #[tokio::test]
    async fn reconfigure_keeps_entries_and_applies_the_new_ttl() {
        let (now, seconds) = clock();
        let cache = without_redis(None, now);
        cache.set("kept", Number::Int(1), None).await.unwrap();

        cache.reconfigure(MemorySettings {
            entries: Some(10),
            ttl: Some(Duration::from_secs(5)),
        });
        cache.set("short", Number::Int(2), None).await.unwrap();
        seconds.fetch_add(6, Ordering::SeqCst);

        assert_eq!(cache.memory().get("kept").unwrap(), Some(Number::Int(1)));
        assert_eq!(cache.memory().get("short").unwrap(), None);
    }

    // async_increment_cache_pipeline_post_call: InMemoryCache.async_increment_pipeline raises at the
    // first entry it cannot add to, so later increments never reach memory, and DualCache only logs
    #[tokio::test]
    async fn a_post_call_increment_stops_at_the_first_entry_memory_refuses() {
        let (now, _) = clock();
        let cache = without_redis::<Entry>(None, now);
        let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));
        cache.memory().set("text", Entry::Text, None).unwrap();

        cache.increment_pipeline_post_call(
            &request,
            &[
                increment("before", Number::Int(1)),
                increment("text", Number::Int(1)),
                increment("after", Number::Int(1)),
            ],
        );

        assert_eq!(
            cache.memory().get("before").unwrap(),
            Some(Entry::Number(Number::Int(1)))
        );
        assert_eq!(cache.memory().get("after").unwrap(), None);
    }
}
