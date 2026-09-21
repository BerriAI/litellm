use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use litellm_cache::{
    BaseCache, CacheCodec, CacheConnectionResult, CacheConnectionStatus, CacheKwargs, Error,
};
use redis::Commands;

const DEFAULT_TTL: Duration = Duration::from_secs(600);

pub struct RedisCache<S, C = redis::Connection> {
    connection: Arc<Mutex<C>>,
    default_ttl: Duration,
    codec: S,
    namespace: Option<String>,
}

impl<S: CacheCodec> RedisCache<S> {
    pub fn new(url: &str, default_ttl: Option<Duration>, codec: S) -> Result<Self, Error> {
        let client = redis::Client::open(url).map_err(|_| Error::Unavailable)?;
        let connection = client.get_connection().map_err(|_| Error::Unavailable)?;
        Ok(Self::with_connection(connection, default_ttl, codec))
    }
}

impl<S, C> RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    pub fn with_connection(connection: C, default_ttl: Option<Duration>, codec: S) -> Self {
        Self {
            connection: Arc::new(Mutex::new(connection)),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
            codec,
            namespace: None,
        }
    }

    fn connection(&self) -> Result<MutexGuard<'_, C>, Error> {
        self.connection.lock().map_err(|_| Error::Unavailable)
    }

    pub fn with_namespace(self, namespace: Option<String>) -> Self {
        Self {
            namespace: namespace.filter(|value| !value.is_empty()),
            ..self
        }
    }

    fn namespaced_key(&self, key: &str) -> String {
        match &self.namespace {
            Some(namespace) if !key.starts_with(&format!("{namespace}:")) => {
                format!("{namespace}:{key}")
            }
            _ => key.into(),
        }
    }

    fn namespaced_pattern(&self) -> Result<String, Error> {
        let namespace = self.namespace.as_ref().ok_or(Error::UnscopedFlush)?;
        let escaped: String = namespace
            .chars()
            .flat_map(|ch| {
                if matches!(ch, '*' | '?' | '[' | ']' | '\\') {
                    vec!['\\', ch]
                } else {
                    vec![ch]
                }
            })
            .collect();
        Ok(format!("{escaped}:*"))
    }

    fn decode_response(&self, value: redis::Value) -> Result<Option<S::Value>, Error> {
        match value {
            redis::Value::Nil => Ok(None),
            redis::Value::BulkString(bytes) => self.codec.decode(&bytes).map(Some),
            redis::Value::SimpleString(text) => self.codec.decode(text.as_bytes()).map(Some),
            _ => Err(Error::InvalidEntry),
        }
    }

    fn ttl_seconds(ttl: Duration) -> u64 {
        ttl.as_secs()
            .saturating_add(u64::from(ttl.subsec_nanos() > 0))
            .max(1)
    }

    async fn run_blocking<T, F>(connection: Arc<Mutex<C>>, operation: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&mut C) -> Result<T, Error> + Send + 'static,
    {
        tokio::task::spawn_blocking(move || {
            let mut connection = connection.lock().map_err(|_| Error::Unavailable)?;
            operation(&mut connection)
        })
        .await
        .map_err(|_| Error::Unavailable)?
    }
}

impl<S, C> BaseCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type Value = S::Value;

    fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    fn set_cache(&self, key: &str, value: Self::Value, kwargs: CacheKwargs) -> Result<(), Error> {
        let payload = self.codec.encode(&value)?;
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        self.connection()?
            .set_ex::<_, _, ()>(self.namespaced_key(key), payload, ttl)
            .map_err(|_| Error::Unavailable)
    }

    fn get_cache(&self, key: &str, _: &CacheKwargs) -> Result<Option<Self::Value>, Error> {
        let value = self
            .connection()?
            .get::<_, redis::Value>(self.namespaced_key(key))
            .map_err(|_| Error::Unavailable)?;
        self.decode_response(value)
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.connection()?
            .del::<_, ()>(self.namespaced_key(key))
            .map_err(|_| Error::Unavailable)
    }

    fn flush_cache(&self) -> Result<(), Error> {
        let pattern = self.namespaced_pattern()?;
        let mut connection = self.connection()?;
        let keys = connection
            .scan_match(pattern)
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

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        kwargs: CacheKwargs,
    ) -> Result<(), Error> {
        let payload = self.codec.encode(&value)?;
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        Self::run_blocking(Arc::clone(&self.connection), move |connection| {
            connection
                .set_ex::<_, _, ()>(key, payload, ttl)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        _: &CacheKwargs,
    ) -> Result<Option<Self::Value>, Error> {
        let key = self.namespaced_key(key);
        let value = Self::run_blocking(Arc::clone(&self.connection), move |connection| {
            connection
                .get::<_, redis::Value>(key)
                .map_err(|_| Error::Unavailable)
        })
        .await?;
        self.decode_response(value)
    }

    async fn async_set_cache_pipeline(
        &self,
        cache_list: Vec<(String, Self::Value)>,
        kwargs: CacheKwargs,
    ) -> Result<(), Error> {
        let entries = cache_list
            .into_iter()
            .map(|(key, value)| {
                self.codec
                    .encode(&value)
                    .map(|payload| (self.namespaced_key(&key), payload))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        Self::run_blocking(Arc::clone(&self.connection), move |connection| {
            for (key, payload) in entries {
                connection
                    .set_ex::<_, _, ()>(key, payload, ttl)
                    .map_err(|_| Error::Unavailable)?;
            }
            Ok(())
        })
        .await
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        let key = self.namespaced_key(key);
        Self::run_blocking(Arc::clone(&self.connection), move |connection| {
            connection.del::<_, ()>(key).map_err(|_| Error::Unavailable)
        })
        .await
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
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
    }
}

#[cfg(test)]
mod tests {
    use super::RedisCache;
    use litellm_cache::{BaseCache, CacheCodec, CacheEntry, CacheKwargs, JsonCodec};
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
    fn ttl_seconds_rounds_up_and_keeps_expiration_positive() {
        assert_eq!(
            RedisCache::<JsonCodec<CacheEntry>>::ttl_seconds(Duration::ZERO),
            1
        );
        assert_eq!(
            RedisCache::<JsonCodec<CacheEntry>>::ttl_seconds(Duration::from_millis(1500)),
            2
        );
        assert_eq!(
            RedisCache::<JsonCodec<CacheEntry>>::ttl_seconds(Duration::from_secs(15)),
            15
        );
    }

    #[test]
    fn redis_commands_round_trip_entries_and_delete_only_namespaced_keys() {
        let value = entry();
        let payload = JsonCodec::<CacheEntry>::new().encode(&value).unwrap();
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
        let cache = RedisCache::with_connection(connection, None, JsonCodec::<CacheEntry>::new())
            .with_namespace(Some("litellm-cache".into()));

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
        let cache = RedisCache::with_connection(connection, None, JsonCodec::<CacheEntry>::new())
            .with_namespace(Some("litellm-cache".into()));

        cache.flush_cache().unwrap();
    }

    #[tokio::test]
    async fn test_connection_runs_ping_off_executor() {
        let connection = MockRedisConnection::new([MockCmd::new(redis::cmd("PING"), Ok("PONG"))])
            .assert_all_commands_consumed();
        let cache = RedisCache::with_connection(connection, None, JsonCodec::<CacheEntry>::new())
            .with_namespace(Some("litellm-cache".into()));

        assert_eq!(
            cache.test_connection().await.unwrap().status,
            litellm_cache::CacheConnectionStatus::Success
        );
    }
}
