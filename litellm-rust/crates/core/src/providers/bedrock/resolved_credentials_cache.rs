use std::time::{Duration, Instant};

use aws_credential_types::Credentials;
use moka::Expiry;
use moka::future::Cache;

#[derive(Clone)]
struct CacheEntry {
    value: Credentials,
    ttl: Duration,
}

struct EntryExpiry;

impl Expiry<String, CacheEntry> for EntryExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &CacheEntry,
        _created_at: Instant,
    ) -> Option<Duration> {
        Some(value.ttl)
    }

    fn expire_after_update(
        &self,
        _key: &String,
        value: &CacheEntry,
        _updated_at: Instant,
        _duration_until_expiry: Option<Duration>,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
}

pub(super) struct AwsCredentialsCache {
    entries: Cache<String, CacheEntry>,
}

impl AwsCredentialsCache {
    pub(super) fn new(max_capacity: u64) -> Self {
        Self {
            entries: Cache::builder()
                .max_capacity(max_capacity)
                .expire_after(EntryExpiry)
                .build(),
        }
    }

    pub(super) async fn get(&self, key: &String) -> Option<Credentials> {
        self.entries.get(key).await.map(|entry| entry.value)
    }

    pub(super) async fn insert(&self, key: String, value: Credentials, ttl: Duration) {
        self.entries.insert(key, CacheEntry { value, ttl }).await;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use aws_credential_types::Credentials;

    use super::AwsCredentialsCache;

    #[tokio::test]
    async fn round_trip_preserves_credentials() {
        let cache = AwsCredentialsCache::new(2);
        let credentials = Credentials::new("access", "secret", None, None, "test");
        cache
            .insert("key".to_string(), credentials, Duration::from_secs(60))
            .await;

        assert_eq!(
            cache
                .get(&"key".to_string())
                .await
                .map(|value| value.access_key_id().to_string()),
            Some("access".to_string())
        );
    }

    #[tokio::test]
    async fn zero_ttl_expires_immediately() {
        let cache = AwsCredentialsCache::new(2);
        let credentials = Credentials::new("access", "secret", None, None, "test");
        cache
            .insert("key".to_string(), credentials, Duration::ZERO)
            .await;

        assert_eq!(cache.get(&"key".to_string()).await, None);
    }
}
