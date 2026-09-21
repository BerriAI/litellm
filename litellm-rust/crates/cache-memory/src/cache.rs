use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheConnectionStatus, CacheKwargs, ClaimCache, CounterCache,
    Error,
};

const DEFAULT_MAX_SIZE_IN_MEMORY: usize = 200;
const DEFAULT_TTL: Duration = Duration::from_secs(600);

type ValueMeasure<V> = Arc<dyn Fn(&V) -> Result<usize, Error> + Send + Sync>;

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

    pub fn expires_at(&self, key: &str) -> Result<Option<Duration>, Error> {
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
            } else if expiration <= now {
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

impl<V> ClaimCache for InMemoryCache<V>
where
    V: Clone + PartialEq + Send + Sync + 'static,
{
    fn claim_cache(
        &self,
        key: &str,
        candidate: V,
        eligible: &[V],
        kwargs: CacheKwargs,
    ) -> Result<V, Error> {
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::evict(&mut state, self.max_size_in_memory, now);
        let winner = match state.values.get(key) {
            Some(existing) if eligible.is_empty() => existing.clone(),
            Some(existing) if eligible.contains(existing) => existing.clone(),
            _ => candidate,
        };
        let expiration = now + self.get_ttl(&kwargs);
        state.values.insert(key.into(), winner.clone());
        state.expirations.insert(key.into(), expiration);
        state
            .expiration_heap
            .push(Reverse((expiration, key.into())));
        Ok(winner)
    }
}

impl CounterCache for InMemoryCache<f64> {
    fn increment_cache(&self, key: &str, amount: f64, kwargs: CacheKwargs) -> Result<f64, Error> {
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::evict(&mut state, self.max_size_in_memory, now);
        let value = state.values.get(key).copied().unwrap_or_default() + amount;
        let expiration = state
            .expirations
            .get(key)
            .copied()
            .unwrap_or_else(|| now + self.get_ttl(&kwargs));
        state.values.insert(key.into(), value);
        state.expirations.insert(key.into(), expiration);
        state
            .expiration_heap
            .push(Reverse((expiration, key.into())));
        Ok(value)
    }
}

impl<V: Clone + Send + Sync + 'static> BaseCache for InMemoryCache<V> {
    type Value = V;

    fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    fn set_cache(&self, key: &str, value: Self::Value, kwargs: CacheKwargs) -> Result<(), Error> {
        let ttl = self.get_ttl(&kwargs);
        self.set_cache(key, value, Some(ttl)).map(|_| ())
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

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        Ok(CacheConnectionResult {
            status: CacheConnectionStatus::Success,
            message: "In-memory cache connection test successful".into(),
            error: None,
        })
    }
}
