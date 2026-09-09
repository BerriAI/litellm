use std::hash::Hash;
use std::time::{Duration, Instant};

use moka::Expiry;
use moka::future::Cache;

#[derive(Clone)]
struct CacheEntry<V> {
    value: V,
    ttl: Duration,
}

struct EntryExpiry;

impl<K, V> Expiry<K, CacheEntry<V>> for EntryExpiry {
    fn expire_after_create(
        &self,
        _key: &K,
        value: &CacheEntry<V>,
        _created_at: Instant,
    ) -> Option<Duration> {
        Some(value.ttl)
    }

    fn expire_after_update(
        &self,
        _key: &K,
        value: &CacheEntry<V>,
        _updated_at: Instant,
        _duration_until_expiry: Option<Duration>,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
}

pub(super) struct CredentialCache<K, V> {
    entries: Cache<K, CacheEntry<V>>,
}

impl<K, V> CredentialCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub(super) fn new(max_capacity: u64) -> Self {
        Self {
            entries: Cache::builder()
                .max_capacity(max_capacity)
                .expire_after(EntryExpiry)
                .build(),
        }
    }

    pub(super) async fn get(&self, key: &K) -> Option<V> {
        self.entries.get(key).await.map(|entry| entry.value)
    }

    pub(super) async fn insert(&self, key: K, value: V, ttl: Duration) {
        self.entries.insert(key, CacheEntry { value, ttl }).await;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::CredentialCache;

    #[tokio::test]
    async fn round_trip_preserves_value() {
        let cache = CredentialCache::new(2);
        cache.insert("key", "value", Duration::from_secs(60)).await;

        assert_eq!(cache.get(&"key").await, Some("value"));
    }

    #[tokio::test]
    async fn zero_ttl_expires_immediately() {
        let cache = CredentialCache::new(2);
        cache.insert("key", "value", Duration::ZERO).await;

        assert_eq!(cache.get(&"key").await, None);
    }
}
