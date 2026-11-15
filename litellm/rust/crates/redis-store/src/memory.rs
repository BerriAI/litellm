use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::Error;
use litellm_cache_memory::{Expiry, InMemoryCache};

use crate::Number;

pub type Clock = Arc<dyn Fn() -> Duration + Send + Sync>;

pub fn system_clock() -> Clock {
    Arc::new(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    })
}

/// A store's memory tier: 200 entries and a 600 s TTL unless the store names others.
pub struct Memory<V: Clone> {
    cache: InMemoryCache<V>,
}

impl<V: Clone> Memory<V> {
    pub fn new(entries: Option<usize>, now: &Clock) -> Self {
        let now = now.clone();
        Self {
            cache: InMemoryCache::with_clock(entries, None, move || now()),
        }
    }

    pub fn get(&self, key: &str) -> Result<Option<V>, Error> {
        self.cache.get_cache(key)
    }

    /// A key memory already holds keeps its deadline, as a counter keeps its window.
    pub fn set(&self, key: &str, value: V, ttl: Option<Duration>) -> Result<(), Error> {
        self.cache
            .set_cache(key, value, Expiry::KeepExisting(ttl))
            .map(drop)
    }

    pub fn refresh(&self, key: &str, value: V, ttl: Duration) -> Result<(), Error> {
        self.cache
            .set_cache(key, value, Expiry::Refresh(Some(ttl)))
            .map(drop)
    }

    /// Writes `value` when memory holds nothing for `key` or `replaces` what it holds.
    pub fn set_if(
        &self,
        key: &str,
        value: V,
        ttl: Option<Duration>,
        replaces: impl FnOnce(&V) -> bool,
    ) -> Result<(), Error> {
        self.cache.update(key, Expiry::KeepExisting(ttl), |held| {
            Ok((held.is_none_or(replaces).then_some(value), ()))
        })
    }

    pub fn delete(&self, key: &str) -> Result<(), Error> {
        self.cache.delete_cache(key)
    }

    pub fn flush(&self) -> Result<(), Error> {
        self.cache.flush_cache()
    }

    pub fn len(&self) -> Result<usize, Error> {
        self.cache.len()
    }

    pub fn expires_at(&self, key: &str) -> Result<Option<Duration>, Error> {
        self.cache.expires_at(key)
    }

    pub fn entries(&self) -> Result<Vec<(String, V)>, Error> {
        self.cache.entries()
    }

    /// `DualCache.update_in_memory_max_size`: `None` restores the 200-entry default.
    pub fn update_max_size(&self, entries: Option<usize>) {
        self.cache.update_max_size_in_memory(entries);
    }

    /// Read-modify-write of one entry, atomic against every other read and write of this tier.
    pub fn update(
        &self,
        key: &str,
        ttl: Option<Duration>,
        update: impl FnOnce(Option<V>) -> Result<V, Error>,
    ) -> Result<V, Error> {
        self.cache.update(key, Expiry::KeepExisting(ttl), |held| {
            let value = update(held.cloned())?;
            Ok((Some(value.clone()), value))
        })
    }
}

/// A value `Memory::increment` can add to.
pub(crate) trait Counter: Clone + From<Number> {
    fn as_number(&self) -> Option<Number>;
}

impl Counter for Number {
    fn as_number(&self) -> Option<Number> {
        Some(*self)
    }
}

impl<V: Counter> Memory<V> {
    /// `(current or 0) + amount`, refusing an entry that is not a number and leaving it in place.
    pub(crate) fn increment(
        &self,
        key: &str,
        amount: Number,
        ttl: Option<Duration>,
    ) -> Result<Number, Error> {
        self.cache.update(key, Expiry::KeepExisting(ttl), |held| {
            let current = match held {
                None => Number::Int(0),
                Some(value) => value.as_number().ok_or(Error::InvalidEntry(None))?,
            };
            let total = current + amount;
            Ok((Some(V::from(total)), total))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::Ordering, time::Duration};

    use rstest::rstest;

    use super::Memory;
    use crate::{
        Number,
        test_support::{START, clock},
    };

    type Write = fn(&Memory<Number>);

    fn secs(seconds: u64) -> Duration {
        Duration::from_secs(seconds)
    }

    fn ttl(seconds: u64) -> Option<Duration> {
        Some(secs(seconds))
    }

    fn after(seconds: u64) -> Option<Duration> {
        ttl(START + seconds)
    }

    fn set(memory: &Memory<Number>) {
        memory.set("key", Number::Int(2), ttl(50)).unwrap();
    }

    fn set_if(memory: &Memory<Number>) {
        memory
            .set_if("key", Number::Int(2), ttl(50), |_| true)
            .unwrap();
    }

    fn update(memory: &Memory<Number>) {
        memory
            .update("key", ttl(50), |_| Ok(Number::Int(2)))
            .unwrap();
    }

    fn increment(memory: &Memory<Number>) {
        memory.increment("key", Number::Int(2), ttl(50)).unwrap();
    }

    fn refresh(memory: &Memory<Number>) {
        memory.refresh("key", Number::Int(2), secs(50)).unwrap();
    }

    #[rstest]
    #[case::set(set, 10)]
    #[case::set_if(set_if, 10)]
    #[case::update(update, 10)]
    #[case::increment(increment, 10)]
    #[case::refresh(refresh, 55)]
    fn a_write_to_a_held_key_keeps_its_deadline_unless_it_refreshes(
        #[case] write: Write,
        #[case] expires_after: u64,
    ) {
        let (now, seconds) = clock();
        let memory = Memory::new(None, &now);
        memory.set("key", Number::Int(0), ttl(10)).unwrap();
        seconds.fetch_add(5, Ordering::SeqCst);

        write(&memory);

        assert_eq!(memory.get("key").unwrap(), Some(Number::Int(2)));
        assert_eq!(memory.expires_at("key").unwrap(), after(expires_after));
    }

    #[rstest]
    #[case::set(set)]
    #[case::set_if(set_if)]
    #[case::update(update)]
    #[case::increment(increment)]
    #[case::refresh(refresh)]
    fn a_write_to_a_new_key_takes_its_ttl(#[case] write: Write) {
        let (now, _) = clock();
        let memory = Memory::new(None, &now);

        write(&memory);

        assert_eq!(memory.get("key").unwrap(), Some(Number::Int(2)));
        assert_eq!(memory.expires_at("key").unwrap(), after(50));
    }

    #[rstest]
    #[case::nothing_held(None, false, 2)]
    #[case::a_value_it_replaces(Some(Number::Int(1)), true, 2)]
    #[case::a_value_it_does_not_replace(Some(Number::Int(1)), false, 1)]
    fn set_if_writes_over(
        #[case] held: Option<Number>,
        #[case] replaces: bool,
        #[case] expected: i64,
    ) {
        let (now, _) = clock();
        let memory = Memory::new(None, &now);
        if let Some(held) = held {
            memory.set("key", held, None).unwrap();
        }

        memory
            .set_if("key", Number::Int(2), None, |_| replaces)
            .unwrap();

        assert_eq!(memory.get("key").unwrap(), Some(Number::Int(expected)));
    }
}
