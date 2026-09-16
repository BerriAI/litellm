use std::sync::Mutex;
use std::time::Duration;

use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheConnectionStatus, CacheEntry, CacheFuture, CacheKwargs,
    Error,
};
use redis::Commands;

const DEFAULT_TTL: Duration = Duration::from_secs(600);

pub struct RedisCache {
    connection: Mutex<redis::Connection>,
    default_ttl: Duration,
}

impl RedisCache {
    pub fn new(url: &str, default_ttl: Option<Duration>) -> Result<Self, Error> {
        let client = redis::Client::open(url).map_err(|_| Error::Unavailable)?;
        let connection = client.get_connection().map_err(|_| Error::Unavailable)?;
        Ok(Self {
            connection: Mutex::new(connection),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
        })
    }

    fn connection(&self) -> Result<std::sync::MutexGuard<'_, redis::Connection>, Error> {
        self.connection.lock().map_err(|_| Error::Unavailable)
    }

    fn encode(value: &CacheEntry) -> Result<Vec<u8>, Error> {
        serde_json::to_vec(value).map_err(|_| Error::InvalidEntry)
    }

    fn decode(value: Vec<u8>) -> Result<CacheEntry, Error> {
        serde_json::from_slice(&value).map_err(|_| Error::InvalidEntry)
    }

    fn ttl_seconds(ttl: Duration) -> u64 {
        ttl.as_secs().max(1)
    }
}

impl BaseCache for RedisCache {
    type Value = CacheEntry;

    fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    fn set_cache(&self, key: &str, value: Self::Value, kwargs: CacheKwargs) -> Result<(), Error> {
        let payload = Self::encode(&value)?;
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        self.connection()?
            .set_ex::<_, _, ()>(key, payload, ttl)
            .map_err(|_| Error::Unavailable)
    }

    fn get_cache(&self, key: &str, _: &CacheKwargs) -> Result<Option<Self::Value>, Error> {
        self.connection()?
            .get::<_, Option<Vec<u8>>>(key)
            .map_err(|_| Error::Unavailable)?
            .map(Self::decode)
            .transpose()
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.connection()?
            .del::<_, ()>(key)
            .map_err(|_| Error::Unavailable)
    }

    fn flush_cache(&self) -> Result<(), Error> {
        self.connection()?
            .flushdb::<()>()
            .map_err(|_| Error::Unavailable)
    }

    fn disconnect(&self) -> CacheFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn test_connection(&self) -> CacheFuture<'_, CacheConnectionResult> {
        Box::pin(async {
            let mut connection = self.connection()?;
            redis::cmd("PING")
                .query::<String>(&mut *connection)
                .map_err(|_| Error::Unavailable)?;
            Ok(CacheConnectionResult {
                status: CacheConnectionStatus::Success,
                message: "Redis cache connection test successful".into(),
                error: None,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::RedisCache;
    use litellm_cache::CacheEntry;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn cache_entries_round_trip_through_json() {
        let entry = CacheEntry {
            timestamp: 123.0,
            response: json!({"choices": [{"text": "cached"}]}),
        };

        let encoded = RedisCache::encode(&entry).unwrap();
        assert_eq!(RedisCache::decode(encoded).unwrap(), entry);
    }

    #[test]
    fn invalid_json_is_rejected() {
        assert!(RedisCache::decode(b"not json".to_vec()).is_err());
    }

    #[test]
    fn ttl_seconds_keeps_redis_expiration_positive() {
        assert_eq!(RedisCache::ttl_seconds(Duration::ZERO), 1);
        assert_eq!(RedisCache::ttl_seconds(Duration::from_millis(1500)), 1);
        assert_eq!(RedisCache::ttl_seconds(Duration::from_secs(15)), 15);
    }
}
