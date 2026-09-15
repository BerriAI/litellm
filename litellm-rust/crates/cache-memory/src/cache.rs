use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::{BaseCache, CacheEntry, CacheKwargs, Error};

const DEFAULT_MAX_SIZE_IN_MEMORY: usize = 200;
const DEFAULT_TTL: Duration = Duration::from_secs(600);

type ValueMeasure<V> = Arc<dyn Fn(&V) -> Result<usize, Error> + Send + Sync>;
type ValueValidator<V> = Arc<dyn Fn(&V) -> Result<(), Error> + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheWrite {
    Stored,
    Disabled,
    TooLarge,
}

struct CacheState<V> {
    values: HashMap<String, V>,
    expirations: HashMap<String, Duration>,
    expiration_heap: BinaryHeap<Reverse<(Duration, String)>>,
}

pub struct InMemoryCache<V: Clone> {
    state: Mutex<CacheState<V>>,
    max_size_in_memory: usize,
    default_ttl: Duration,
    max_entry_bytes: Option<usize>,
    measure_value: Option<ValueMeasure<V>>,
    validate_value: Option<ValueValidator<V>>,
    now: Arc<dyn Fn() -> Duration + Send + Sync>,
}

impl<V: Clone> Default for InMemoryCache<V> {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl<V: Clone> InMemoryCache<V> {
    pub fn new(max_size_in_memory: Option<usize>, default_ttl: Option<Duration>) -> Self {
        Self::with_clock(max_size_in_memory, default_ttl, || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
        })
    }

    pub fn with_clock(
        max_size_in_memory: Option<usize>,
        default_ttl: Option<Duration>,
        now: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> Self {
        Self::with_clock_and_size_measurement(max_size_in_memory, default_ttl, None, None, now)
    }

    pub fn with_clock_and_size_measurement(
        max_size_in_memory: Option<usize>,
        default_ttl: Option<Duration>,
        max_entry_bytes: Option<usize>,
        measure_value: Option<ValueMeasure<V>>,
        now: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> Self {
        Self {
            state: Mutex::new(CacheState {
                values: HashMap::new(),
                expirations: HashMap::new(),
                expiration_heap: BinaryHeap::new(),
            }),
            max_size_in_memory: max_size_in_memory.unwrap_or(DEFAULT_MAX_SIZE_IN_MEMORY),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
            max_entry_bytes,
            measure_value,
            validate_value: None,
            now: Arc::new(now),
        }
    }

    pub fn set_cache(
        &self,
        key: impl Into<String>,
        value: V,
        ttl: Option<Duration>,
    ) -> Result<CacheWrite, Error> {
        if self.max_size_in_memory == 0 {
            return Ok(CacheWrite::Disabled);
        }
        if let Some(validate) = &self.validate_value {
            validate(&value)?;
        }
        if let (Some(limit), Some(measure)) = (self.max_entry_bytes, &self.measure_value)
            && measure(&value)? > limit
        {
            return Ok(CacheWrite::TooLarge);
        }
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::evict(&mut state, self.max_size_in_memory, now);
        let key = key.into();
        state.values.insert(key.clone(), value);
        let expiration = state.expirations.get(&key).copied();
        if expiration.is_none_or(|expiration| expiration < now) {
            let expiration = now + ttl.unwrap_or(self.default_ttl);
            state.expirations.insert(key.clone(), expiration);
            state.expiration_heap.push(Reverse((expiration, key)));
        }
        Ok(CacheWrite::Stored)
    }

    pub fn get_cache(&self, key: &str) -> Result<Option<V>, Error> {
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        if state
            .expirations
            .get(key)
            .is_some_and(|expiration| *expiration < now)
        {
            Self::remove(&mut state, key);
        }
        Ok(state.values.get(key).cloned())
    }

    pub fn get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        Ok(self
            .state
            .lock()
            .map_err(|_| Error::Unavailable)?
            .expirations
            .get(key)
            .copied())
    }
    pub fn delete_cache(&self, key: &str) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::remove(&mut state, key);
        Ok(())
    }
    pub fn flush_cache(&self) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        state.values.clear();
        state.expirations.clear();
        state.expiration_heap.clear();
        Ok(())
    }

    fn evict(state: &mut CacheState<V>, capacity: usize, now: Duration) {
        while let Some(Reverse((expiration, key))) = state.expiration_heap.peek().cloned() {
            if state.expirations.get(&key).copied() != Some(expiration) {
                state.expiration_heap.pop();
            } else if expiration < now {
                state.expiration_heap.pop();
                Self::remove(state, &key);
            } else {
                break;
            }
        }
        while state.values.len() >= capacity {
            let Some(Reverse((expiration, key))) = state.expiration_heap.pop() else {
                break;
            };
            if state.expirations.get(&key).copied() == Some(expiration) {
                Self::remove(state, &key);
            }
        }
    }

    fn remove(state: &mut CacheState<V>, key: &str) {
        state.values.remove(key);
        state.expirations.remove(key);
    }
}

impl InMemoryCache<CacheEntry> {
    pub fn response_cache(capacity: usize, ttl: Duration, max_entry_bytes: usize) -> Self {
        Self::response_cache_with_clock(capacity, ttl, max_entry_bytes, || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
        })
    }

    pub fn response_cache_with_clock(
        capacity: usize,
        ttl: Duration,
        max_entry_bytes: usize,
        now: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> Self {
        let mut cache = Self::with_clock_and_size_measurement(
            Some(capacity),
            Some(ttl),
            Some(max_entry_bytes),
            Some(Arc::new(|entry: &CacheEntry| {
                serde_json::to_vec(entry)
                    .map(|bytes| bytes.len())
                    .map_err(|_| Error::InvalidEntry)
            })),
            now,
        );
        cache.validate_value = Some(Arc::new(|entry: &CacheEntry| {
            entry
                .timestamp
                .is_finite()
                .then_some(())
                .ok_or(Error::InvalidEntry)
        }));
        cache
    }
}

impl BaseCache for InMemoryCache<CacheEntry> {
    type Value = CacheEntry;
    fn set_cache(&self, key: &str, value: Self::Value, kwargs: CacheKwargs) -> Result<(), Error> {
        self.set_cache(key, value, kwargs.ttl).map(|_| ())
    }
    fn get_cache(&self, key: &str, _: &CacheKwargs) -> Result<Option<Self::Value>, Error> {
        self.get_cache(key)
    }
    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.delete_cache(key)
    }
    fn flush_cache(&self) -> Result<(), Error> {
        self.flush_cache()
    }
}
