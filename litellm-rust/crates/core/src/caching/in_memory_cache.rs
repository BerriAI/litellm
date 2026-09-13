use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_SIZE_IN_MEMORY: usize = 200;
const DEFAULT_TTL: Duration = Duration::from_secs(600);

pub struct InMemoryCache<V: Clone> {
    pub cache_dict: HashMap<String, V>,
    pub ttl_dict: HashMap<String, Duration>,
    pub expiration_heap: BinaryHeap<Reverse<(Duration, String)>>,
    pub max_size_in_memory: usize,
    pub default_ttl: Duration,
    now: Box<dyn Fn() -> Duration + Send + Sync>,
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
        Self {
            cache_dict: HashMap::new(),
            ttl_dict: HashMap::new(),
            expiration_heap: BinaryHeap::new(),
            max_size_in_memory: max_size_in_memory.unwrap_or(DEFAULT_MAX_SIZE_IN_MEMORY),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
            now: Box::new(now),
        }
    }

    pub fn evict_cache(&mut self) {
        if self.max_size_in_memory == 0 {
            return;
        }

        let current_time = (self.now)();
        while let Some(Reverse((expiration_time, key))) = self.expiration_heap.peek().cloned() {
            if self.ttl_dict.get(&key).copied() != Some(expiration_time) {
                self.expiration_heap.pop();
            } else if expiration_time <= current_time {
                self.expiration_heap.pop();
                self.remove_key(&key);
            } else {
                break;
            }
        }

        while self.cache_dict.len() >= self.max_size_in_memory {
            let Some(Reverse((expiration_time, key))) = self.expiration_heap.pop() else {
                break;
            };
            if self.ttl_dict.get(&key).copied() == Some(expiration_time) {
                self.remove_key(&key);
            }
        }
    }

    pub fn allow_ttl_override(&self, key: &str) -> bool {
        match self.ttl_dict.get(key).copied() {
            None => true,
            Some(expiration_time) => expiration_time < (self.now)(),
        }
    }

    pub fn set_cache(&mut self, key: impl Into<String>, value: V, ttl: Option<Duration>) {
        if self.max_size_in_memory == 0 {
            return;
        }

        self.evict_cache();
        let key = key.into();
        self.cache_dict.insert(key.clone(), value);
        if self.allow_ttl_override(&key) {
            let expiration_time = (self.now)() + ttl.unwrap_or(self.default_ttl);
            self.ttl_dict.insert(key.clone(), expiration_time);
            self.expiration_heap.push(Reverse((expiration_time, key)));
        }
    }

    // Generic values intentionally omit Python's per-item size check.
    pub fn get_cache(&mut self, key: &str) -> Option<V> {
        if self.cache_dict.contains_key(key) {
            if self.is_key_expired(key) {
                self.remove_key(key);
                return None;
            }
            return self.cache_dict.get(key).cloned();
        }
        None
    }

    pub fn get_ttl(&self, key: &str) -> Option<Duration> {
        self.ttl_dict.get(key).copied()
    }

    pub fn delete_cache(&mut self, key: &str) {
        self.remove_key(key);
    }

    pub fn flush_cache(&mut self) {
        self.cache_dict.clear();
        self.ttl_dict.clear();
        self.expiration_heap.clear();
    }

    fn is_key_expired(&self, key: &str) -> bool {
        self.ttl_dict
            .get(key)
            .is_some_and(|expiration_time| *expiration_time < (self.now)())
    }

    fn remove_key(&mut self, key: &str) {
        self.cache_dict.remove(key);
        self.ttl_dict.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    use super::InMemoryCache;
    use std::time::Duration;

    fn cache(now: Arc<AtomicU64>, max_size: usize, default_ttl: Duration) -> InMemoryCache<String> {
        InMemoryCache::with_clock(Some(max_size), Some(default_ttl), move || {
            Duration::from_secs(now.load(Ordering::Relaxed))
        })
    }

    #[test]
    fn ttl_expiry_is_deterministic() {
        let now = Arc::new(AtomicU64::new(100));
        let mut cache = cache(now.clone(), 10, Duration::from_secs(60));
        cache.set_cache("key", "value".to_string(), None);
        assert_eq!(cache.get_cache("key"), Some("value".to_string()));
        now.store(161, Ordering::Relaxed);
        assert_eq!(cache.get_cache("key"), None);
        assert_eq!(cache.get_ttl("key"), None);
    }

    #[test]
    fn default_and_per_set_ttl_are_applied() {
        let now = Arc::new(AtomicU64::new(100));
        let mut cache = cache(now.clone(), 10, Duration::from_secs(60));
        cache.set_cache("default", "value".to_string(), None);
        cache.set_cache("custom", "value".to_string(), Some(Duration::from_secs(20)));
        assert_eq!(cache.get_ttl("default"), Some(Duration::from_secs(160)));
        assert_eq!(cache.get_ttl("custom"), Some(Duration::from_secs(120)));
    }

    #[test]
    fn unexpired_entries_do_not_allow_ttl_override() {
        let now = Arc::new(AtomicU64::new(100));
        let mut cache = cache(now.clone(), 10, Duration::from_secs(60));
        cache.set_cache("key", "first".to_string(), Some(Duration::from_secs(20)));
        cache.set_cache("key", "second".to_string(), Some(Duration::from_secs(80)));
        assert_eq!(cache.get_cache("key"), Some("second".to_string()));
        assert_eq!(cache.get_ttl("key"), Some(Duration::from_secs(120)));
        now.store(121, Ordering::Relaxed);
        cache.set_cache("key", "third".to_string(), Some(Duration::from_secs(80)));
        assert_eq!(cache.get_ttl("key"), Some(Duration::from_secs(201)));
    }

    #[test]
    fn max_size_evicts_earliest_expiration() {
        let now = Arc::new(AtomicU64::new(100));
        let mut cache = cache(now, 2, Duration::from_secs(60));
        cache.set_cache("early", "value".to_string(), Some(Duration::from_secs(10)));
        cache.set_cache("late", "value".to_string(), Some(Duration::from_secs(20)));
        cache.set_cache("new", "value".to_string(), Some(Duration::from_secs(30)));
        assert_eq!(cache.get_cache("early"), None);
        assert!(cache.get_cache("late").is_some());
        assert!(cache.get_cache("new").is_some());
    }

    #[test]
    fn expired_entries_are_evicted_before_live_entries() {
        let now = Arc::new(AtomicU64::new(100));
        let mut cache = cache(now.clone(), 3, Duration::from_secs(60));
        cache.set_cache(
            "expired-one",
            "value".to_string(),
            Some(Duration::from_secs(10)),
        );
        cache.set_cache(
            "expired-two",
            "value".to_string(),
            Some(Duration::from_secs(20)),
        );
        cache.set_cache("live", "value".to_string(), Some(Duration::from_secs(100)));
        now.store(121, Ordering::Relaxed);
        cache.set_cache("new", "value".to_string(), Some(Duration::from_secs(100)));
        assert_eq!(cache.get_cache("expired-one"), None);
        assert_eq!(cache.get_cache("expired-two"), None);
        assert!(cache.get_cache("live").is_some());
        assert!(cache.get_cache("new").is_some());
    }

    #[test]
    fn stale_heap_entries_are_skipped() {
        let now = Arc::new(AtomicU64::new(100));
        let mut cache = cache(now, 1, Duration::from_secs(60));
        cache.set_cache(
            "removed",
            "value".to_string(),
            Some(Duration::from_secs(10)),
        );
        cache.delete_cache("removed");
        cache.set_cache("kept", "value".to_string(), Some(Duration::from_secs(20)));
        cache.set_cache("new", "value".to_string(), Some(Duration::from_secs(30)));
        assert_eq!(cache.get_cache("removed"), None);
        assert_eq!(cache.get_cache("kept"), None);
        assert!(cache.get_cache("new").is_some());
    }

    #[test]
    fn delete_and_flush_remove_values_and_ttls() {
        let now = Arc::new(AtomicU64::new(100));
        let mut cache = cache(now, 10, Duration::from_secs(60));
        cache.set_cache("one", "value".to_string(), None);
        cache.set_cache("two", "value".to_string(), None);
        cache.delete_cache("one");
        assert_eq!(cache.get_cache("one"), None);
        cache.flush_cache();
        assert!(cache.cache_dict.is_empty());
        assert!(cache.ttl_dict.is_empty());
        assert!(cache.expiration_heap.is_empty());
    }

    #[test]
    fn zero_max_size_does_not_cache() {
        let now = Arc::new(AtomicU64::new(100));
        let mut cache = cache(now, 0, Duration::from_secs(60));
        cache.set_cache("key", "value".to_string(), None);
        assert_eq!(cache.get_cache("key"), None);
        assert!(cache.cache_dict.is_empty());
    }
}
