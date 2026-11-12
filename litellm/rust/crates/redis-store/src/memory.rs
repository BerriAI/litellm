use std::{
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::Error;
use litellm_cache_memory::InMemoryCache;

use crate::Number;

pub type Clock = Arc<dyn Fn() -> Duration + Send + Sync>;

pub fn system_clock() -> Clock {
    Arc::new(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    })
}

/// A store's memory tier: Python's `InMemoryCache`, with its 200 entries and 600 s TTL unless the
/// store names others, plus one lock for every read-modify-write on it.
pub struct Memory<V: Clone> {
    cache: InMemoryCache<V>,
    updates: Mutex<()>,
}

impl<V: Clone> Memory<V> {
    pub fn new(entries: Option<usize>, now: &Clock) -> Self {
        let now = now.clone();
        Self {
            cache: InMemoryCache::with_clock(entries, None, move || now()),
            updates: Mutex::new(()),
        }
    }

    pub fn get(&self, key: &str) -> Result<Option<V>, Error> {
        self.cache.get_cache(key)
    }

    pub fn set(&self, key: &str, value: V, ttl: Option<Duration>) -> Result<(), Error> {
        self.cache.set_cache(key, value, ttl).map(drop)
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

    /// Read-modify-write of one entry, atomic against every other update of this tier.
    pub fn update(
        &self,
        key: &str,
        ttl: Option<Duration>,
        update: impl FnOnce(Option<V>) -> Result<V, Error>,
    ) -> Result<V, Error> {
        let _guard = self.guard();
        let value = update(self.get(key)?)?;
        self.set(key, value.clone(), ttl)?;
        Ok(value)
    }

    /// Held across a read and a conditional write that `update` cannot express.
    pub fn guard(&self) -> MutexGuard<'_, ()> {
        lock(&self.updates)
    }
}

/// A value `InMemoryCache.increment_cache` can add to.
pub(crate) trait Counter: Clone + From<Number> {
    fn as_number(&self) -> Option<Number>;
}

impl Counter for Number {
    fn as_number(&self) -> Option<Number> {
        Some(*self)
    }
}

impl<V: Counter> Memory<V> {
    /// `InMemoryCache.increment_cache`: `(current or 0) + amount`, refusing an entry that is not a
    /// number and leaving it in place.
    pub(crate) fn increment(
        &self,
        key: &str,
        amount: Number,
        ttl: Option<Duration>,
    ) -> Result<Number, Error> {
        let _guard = self.guard();
        let current = match self.get(key)? {
            None => Number::Int(0),
            Some(value) => value.as_number().ok_or(Error::InvalidEntry(None))?,
        };
        let total = current + amount;
        self.set(key, V::from(total), ttl)?;
        Ok(total)
    }
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
