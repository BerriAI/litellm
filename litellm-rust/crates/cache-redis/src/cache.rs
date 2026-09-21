use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheConnectionStatus, CacheEntry, CacheFuture, CacheKwargs,
    Error,
};
use redis::Commands;

const DEFAULT_TTL: Duration = Duration::from_secs(600);
const KEY_PREFIX: &str = "litellm-cache:";

pub struct RedisCache<C = redis::Connection> {
    connection: Arc<Mutex<C>>,
    default_ttl: Duration,
}

impl RedisCache<redis::Connection> {
    pub fn new(url: &str, default_ttl: Option<Duration>) -> Result<Self, Error> {
        let client = redis::Client::open(url).map_err(|_| Error::Unavailable)?;
        let connection = client.get_connection().map_err(|_| Error::Unavailable)?;
        Ok(Self::with_connection(connection, default_ttl))
    }
}

impl<C> RedisCache<C>
where
    C: redis::ConnectionLike + Send + 'static,
{
    fn with_connection(connection: C, default_ttl: Option<Duration>) -> Self {
        Self {
            connection: Arc::new(Mutex::new(connection)),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
        }
    }

    fn connection(&self) -> Result<MutexGuard<'_, C>, Error> {
        self.connection.lock().map_err(|_| Error::Unavailable)
    }

    fn namespaced_key(key: &str) -> String {
        format!("{KEY_PREFIX}{key}")
    }

    fn namespaced_pattern() -> &'static str {
        const PATTERN: &str = "litellm-cache:*";
        PATTERN
    }

    fn encode(value: &CacheEntry) -> Result<Vec<u8>, Error> {
        serde_json::to_vec(value).map_err(|_| Error::InvalidEntry)
    }

    fn decode(value: Vec<u8>) -> Result<CacheEntry, Error> {
        serde_json::from_slice(&value).map_err(|_| Error::InvalidEntry)
    }

    fn ttl_seconds(ttl: Duration) -> u64 {
        ttl.as_secs()
            .saturating_add(u64::from(ttl.subsec_nanos() > 0))
            .max(1)
    }

    fn run_blocking<T, F>(connection: Arc<Mutex<C>>, operation: F) -> CacheFuture<'static, T>
    where
        T: Send + 'static,
        F: FnOnce(&mut C) -> Result<T, Error> + Send + 'static,
    {
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let mut connection = connection.lock().map_err(|_| Error::Unavailable)?;
                operation(&mut connection)
            })
            .await
            .map_err(|_| Error::Unavailable)?
        })
    }
}

impl<C> BaseCache for RedisCache<C>
where
    C: redis::ConnectionLike + Send + 'static,
{
    type Value = CacheEntry;

    fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    fn set_cache(&self, key: &str, value: Self::Value, kwargs: CacheKwargs) -> Result<(), Error> {
        let payload = Self::encode(&value)?;
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        self.connection()?
            .set_ex::<_, _, ()>(Self::namespaced_key(key), payload, ttl)
            .map_err(|_| Error::Unavailable)
    }

    fn get_cache(&self, key: &str, _: &CacheKwargs) -> Result<Option<Self::Value>, Error> {
        self.connection()?
            .get::<_, Option<Vec<u8>>>(Self::namespaced_key(key))
            .map_err(|_| Error::Unavailable)?
            .map(Self::decode)
            .transpose()
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.connection()?
            .del::<_, ()>(Self::namespaced_key(key))
            .map_err(|_| Error::Unavailable)
    }

    fn flush_cache(&self) -> Result<(), Error> {
        let mut connection = self.connection()?;
        let keys = connection
            .scan_match(Self::namespaced_pattern())
            .map_err(|_| Error::Unavailable)?
            .collect::<redis::RedisResult<Vec<String>>>()
            .map_err(|_| Error::Unavailable)?;
        if keys.is_empty() {
            return Ok(());
        }
        connection
            .del::<_, usize>(keys)
            .map(|_| ())
            .map_err(|_| Error::Unavailable)
    }

    fn async_set_cache<'a>(
        &'a self,
        key: &'a str,
        value: Self::Value,
        kwargs: CacheKwargs,
    ) -> CacheFuture<'a, ()> {
        let payload = Self::encode(&value);
        let key = Self::namespaced_key(key);
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        Self::run_blocking(Arc::clone(&self.connection), move |connection| {
            connection
                .set_ex::<_, _, ()>(key, payload?, ttl)
                .map_err(|_| Error::Unavailable)
        })
    }

    fn async_get_cache<'a>(
        &'a self,
        key: &'a str,
        _: &'a CacheKwargs,
    ) -> CacheFuture<'a, Option<Self::Value>> {
        let key = Self::namespaced_key(key);
        Box::pin(async move {
            Self::run_blocking(Arc::clone(&self.connection), move |connection| {
                connection
                    .get::<_, Option<Vec<u8>>>(key)
                    .map_err(|_| Error::Unavailable)
            })
            .await?
            .map(Self::decode)
            .transpose()
        })
    }

    fn async_set_cache_pipeline<'a>(
        &'a self,
        cache_list: Vec<(String, Self::Value)>,
        kwargs: CacheKwargs,
    ) -> CacheFuture<'a, ()> {
        let entries = cache_list
            .into_iter()
            .map(|(key, value)| {
                Self::encode(&value).map(|payload| (Self::namespaced_key(&key), payload))
            })
            .collect::<Result<Vec<_>, _>>();
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        Self::run_blocking(Arc::clone(&self.connection), move |connection| {
            for (key, payload) in entries? {
                connection
                    .set_ex::<_, _, ()>(key, payload, ttl)
                    .map_err(|_| Error::Unavailable)?;
            }
            Ok(())
        })
    }

    fn async_delete_cache<'a>(&'a self, key: &'a str) -> CacheFuture<'a, ()> {
        let key = Self::namespaced_key(key);
        Self::run_blocking(Arc::clone(&self.connection), move |connection| {
            connection.del::<_, ()>(key).map_err(|_| Error::Unavailable)
        })
    }

    fn disconnect(&self) -> CacheFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn test_connection(&self) -> CacheFuture<'_, CacheConnectionResult> {
        Box::pin(async move {
            Self::run_blocking(Arc::clone(&self.connection), |connection| {
                redis::cmd("PING")
                    .query::<String>(connection)
                    .map_err(|_| Error::Unavailable)
            })
            .await?;
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
    use litellm_cache::{BaseCache, CacheEntry, CacheKwargs};
    use redis_test::{MockCmd, MockRedisConnection};
    use serde_json::json;
    use std::time::Duration;

    fn entry() -> CacheEntry {
        CacheEntry {
            timestamp: 123.0,
            response: json!({"choices": [{"text": "cached"}]}),
        }
    }

    #[test]
    fn cache_entries_round_trip_through_json() {
        let entry = entry();
        let encoded = RedisCache::<redis::Connection>::encode(&entry).unwrap();
        assert_eq!(
            RedisCache::<redis::Connection>::decode(encoded).unwrap(),
            entry
        );
    }

    #[test]
    fn invalid_json_is_rejected() {
        assert!(RedisCache::<redis::Connection>::decode(b"not json".to_vec()).is_err());
    }

    #[test]
    fn ttl_seconds_rounds_up_and_keeps_expiration_positive() {
        assert_eq!(
            RedisCache::<redis::Connection>::ttl_seconds(Duration::ZERO),
            1
        );
        assert_eq!(
            RedisCache::<redis::Connection>::ttl_seconds(Duration::from_millis(1500)),
            2
        );
        assert_eq!(
            RedisCache::<redis::Connection>::ttl_seconds(Duration::from_secs(15)),
            15
        );
    }

    #[test]
    fn redis_commands_round_trip_entries_and_delete_only_namespaced_keys() {
        let value = entry();
        let payload = RedisCache::<redis::Connection>::encode(&value).unwrap();
        let connection = MockRedisConnection::new([
            MockCmd::new(
                redis::cmd("SETEX")
                    .arg("litellm-cache:key")
                    .arg(600)
                    .arg(payload.clone()),
                Ok("OK"),
            ),
            MockCmd::new(redis::cmd("GET").arg("litellm-cache:key"), Ok(payload)),
            MockCmd::new(redis::cmd("DEL").arg("litellm-cache:key"), Ok(1u32)),
        ])
        .assert_all_commands_consumed();
        let cache = RedisCache::with_connection(connection, None);

        cache
            .set_cache("key", value.clone(), CacheKwargs::default())
            .unwrap();
        assert_eq!(
            cache.get_cache("key", &CacheKwargs::default()).unwrap(),
            Some(value)
        );
        cache.delete_cache("key").unwrap();
    }

    #[test]
    fn flush_scans_and_deletes_only_cache_keys() {
        let connection = MockRedisConnection::new([
            MockCmd::new(
                redis::cmd("SCAN")
                    .cursor_arg(0)
                    .arg("MATCH")
                    .arg("litellm-cache:*"),
                Ok(redis_test::redis_value!(["0", ["litellm-cache:key"]])),
            ),
            MockCmd::new(redis::cmd("DEL").arg("litellm-cache:key"), Ok(1u32)),
        ])
        .assert_all_commands_consumed();
        let cache = RedisCache::with_connection(connection, None);

        cache.flush_cache().unwrap();
    }

    #[tokio::test]
    async fn test_connection_runs_ping_off_executor() {
        let connection = MockRedisConnection::new([MockCmd::new(redis::cmd("PING"), Ok("PONG"))])
            .assert_all_commands_consumed();
        let cache = RedisCache::with_connection(connection, None);

        assert_eq!(
            cache.test_connection().await.unwrap().status,
            litellm_cache::CacheConnectionStatus::Success
        );
    }
}
