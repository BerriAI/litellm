use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, BatchEntry, CacheCodec, CacheConnectionResult, CacheConnectionStatus, CacheKwargs,
    ClaimCache, CounterCache, Error,
};
use redis::Commands;

const DEFAULT_TTL: Duration = Duration::from_secs(600);
const REDIS_TIMEOUT: Duration = Duration::from_secs(5);
const REDIS_POOL_SIZE: u32 = 16;

struct PooledConnection {
    connection: redis::Connection,
    failed: bool,
}

/// Pools connections without a checkout PING, which would double every operation's round trips.
/// A timed-out command leaves its reply on the socket while redis still reports the connection
/// open, so any connection whose operation failed is discarded instead of being reused.
struct ConnectionManager(redis::Client);

impl r2d2::ManageConnection for ConnectionManager {
    type Connection = PooledConnection;
    type Error = redis::RedisError;

    fn connect(&self) -> Result<PooledConnection, redis::RedisError> {
        let connection = self.0.get_connection()?;
        connection.set_read_timeout(Some(REDIS_TIMEOUT))?;
        connection.set_write_timeout(Some(REDIS_TIMEOUT))?;
        Ok(PooledConnection {
            connection,
            failed: false,
        })
    }

    fn is_valid(&self, connection: &mut PooledConnection) -> Result<(), redis::RedisError> {
        redis::cmd("PING").query::<String>(&mut connection.connection)?;
        Ok(())
    }

    fn has_broken(&self, connection: &mut PooledConnection) -> bool {
        connection.failed || !redis::ConnectionLike::is_open(&connection.connection)
    }
}

const INCREMENT_SCRIPT: &str = concat!(
    "local value = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1]); ",
    "if redis.call('TTL', KEYS[1]) == -1 then ",
    "redis.call('EXPIRE', KEYS[1], ARGV[2]); end; return value"
);

// Compare-and-set against the exact bytes the claim decision was made on.
// ARGV: [1] expected payload or "" when absent, [2] ttl, [3] new payload, [4] refresh ttl.
const CLAIM_SCRIPT: &str = concat!(
    "local current = redis.call('GET', KEYS[1]); ",
    "if ARGV[1] == '' then if current ~= false and current ~= '' then return 0; end; ",
    "elseif current ~= ARGV[1] then return 0; end; ",
    "if ARGV[3] ~= '' then redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[2]); ",
    "elseif ARGV[4] == '1' then redis.call('EXPIRE', KEYS[1], ARGV[2]); end; return 1"
);
const CLAIM_ATTEMPTS: usize = 8;

enum Connections<C> {
    Pool(r2d2::Pool<ConnectionManager>),
    Fixed(Mutex<C>),
}

struct ConnectionRef<'a>(&'a mut dyn redis::ConnectionLike);

impl redis::ConnectionLike for ConnectionRef<'_> {
    fn req_packed_command(&mut self, cmd: &[u8]) -> redis::RedisResult<redis::Value> {
        self.0.req_packed_command(cmd)
    }

    fn req_packed_commands(
        &mut self,
        cmd: &[u8],
        offset: usize,
        count: usize,
    ) -> redis::RedisResult<Vec<redis::Value>> {
        self.0.req_packed_commands(cmd, offset, count)
    }

    fn get_db(&self) -> i64 {
        self.0.get_db()
    }

    fn supports_pipelining(&self) -> bool {
        self.0.supports_pipelining()
    }

    fn check_connection(&mut self) -> bool {
        self.0.check_connection()
    }

    fn is_open(&self) -> bool {
        self.0.is_open()
    }
}

impl<C> Connections<C>
where
    C: redis::ConnectionLike + Send + 'static,
{
    fn execute<T>(
        &self,
        operation: impl FnOnce(&mut ConnectionRef<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        match self {
            Self::Pool(pool) => {
                let mut pooled = pool.get().map_err(|_| Error::Unavailable)?;
                let result = operation(&mut ConnectionRef(&mut pooled.connection));
                pooled.failed = matches!(result, Err(Error::Unavailable));
                result
            }
            Self::Fixed(connection) => {
                let mut connection = connection.lock().map_err(|_| Error::Unavailable)?;
                operation(&mut ConnectionRef(&mut *connection))
            }
        }
    }
}

pub struct RedisCache<S, C = redis::Connection> {
    connections: Arc<Connections<C>>,
    default_ttl: Duration,
    codec: S,
    namespace: Option<String>,
}

impl<S: CacheCodec> RedisCache<S> {
    pub fn new(url: &str, default_ttl: Option<Duration>, codec: S) -> Result<Self, Error> {
        let client = redis::Client::open(url).map_err(|_| Error::Unavailable)?;
        let pool = r2d2::Pool::builder()
            .max_size(REDIS_POOL_SIZE)
            .min_idle(Some(0))
            .connection_timeout(REDIS_TIMEOUT)
            .test_on_check_out(false)
            .build(ConnectionManager(client))
            .map_err(|_| Error::Unavailable)?;
        Ok(Self {
            connections: Arc::new(Connections::Pool(pool)),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
            codec,
            namespace: None,
        })
    }
}

impl<S, C> RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    pub fn with_connection(connection: C, default_ttl: Option<Duration>, codec: S) -> Self {
        Self {
            connections: Arc::new(Connections::Fixed(Mutex::new(connection))),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
            codec,
            namespace: None,
        }
    }

    pub fn with_namespace(self, namespace: Option<String>) -> Self {
        Self {
            namespace: namespace.filter(|value| !value.is_empty()),
            ..self
        }
    }

    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
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

    fn flush_matching(connection: &mut ConnectionRef<'_>, pattern: &str) -> Result<(), Error> {
        let mut cursor = 0u64;
        loop {
            let (next_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .cursor_arg(cursor)
                .arg("MATCH")
                .arg(pattern)
                .arg("COUNT")
                .arg(1000)
                .query(connection)
                .map_err(|_| Error::Unavailable)?;
            if !keys.is_empty() {
                connection
                    .del::<_, usize>(keys)
                    .map_err(|_| Error::Unavailable)?;
            }
            if next_cursor == 0 {
                return Ok(());
            }
            cursor = next_cursor;
        }
    }

    fn decode_response(&self, value: redis::Value) -> Result<Option<S::Value>, Error> {
        match value {
            redis::Value::Nil => Ok(None),
            redis::Value::BulkString(bytes) => self.codec.decode(&bytes).map(Some),
            redis::Value::SimpleString(text) => self.codec.decode(text.as_bytes()).map(Some),
            _ => Err(Error::InvalidEntry),
        }
    }

    fn decode_batch_response(&self, value: redis::Value) -> Result<BatchEntry<S::Value>, Error> {
        match self.decode_response(value) {
            Ok(Some(value)) => Ok(BatchEntry::Hit(value)),
            Ok(None) => Ok(BatchEntry::Miss),
            Err(Error::InvalidEntry) => Ok(BatchEntry::Invalid),
            Err(error) => Err(error),
        }
    }

    fn ttl_seconds(ttl: Duration) -> u64 {
        ttl.as_secs()
            .saturating_add(u64::from(ttl.subsec_nanos() > 0))
            .max(1)
    }

    async fn run_blocking<T, F>(connections: Arc<Connections<C>>, operation: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&mut ConnectionRef<'_>) -> Result<T, Error> + Send + 'static,
    {
        tokio::task::spawn_blocking(move || connections.execute(operation))
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
        let key = self.namespaced_key(key);
        self.connections.execute(|connection| {
            connection
                .set_ex::<_, _, ()>(key, payload, ttl)
                .map_err(|_| Error::Unavailable)
        })
    }

    fn get_cache(&self, key: &str, _: &CacheKwargs) -> Result<Option<Self::Value>, Error> {
        let key = self.namespaced_key(key);
        let value = self.connections.execute(|connection| {
            connection
                .get::<_, redis::Value>(key)
                .map_err(|_| Error::Unavailable)
        })?;
        self.decode_response(value)
    }

    fn get_cache_batch(
        &self,
        keys: &[String],
        _: &CacheKwargs,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        let keys = keys
            .iter()
            .map(|key| self.namespaced_key(key))
            .collect::<Vec<_>>();
        let values = self.connections.execute(|connection| {
            redis::cmd("MGET")
                .arg(keys)
                .query::<Vec<redis::Value>>(connection)
                .map_err(|_| Error::Unavailable)
        })?;
        values
            .into_iter()
            .map(|value| self.decode_batch_response(value))
            .collect()
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        let key = self.namespaced_key(key);
        self.connections
            .execute(|connection| connection.del::<_, ()>(key).map_err(|_| Error::Unavailable))
    }

    fn flush_cache(&self) -> Result<(), Error> {
        let pattern = self.namespaced_pattern()?;
        self.connections
            .execute(|connection| Self::flush_matching(connection, &pattern))
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
        Self::run_blocking(Arc::clone(&self.connections), move |connection| {
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
        let value = Self::run_blocking(Arc::clone(&self.connections), move |connection| {
            connection
                .get::<_, redis::Value>(key)
                .map_err(|_| Error::Unavailable)
        })
        .await?;
        self.decode_response(value)
    }

    async fn async_get_cache_batch(
        &self,
        keys: Vec<String>,
        _: CacheKwargs,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        let keys = keys
            .iter()
            .map(|key| self.namespaced_key(key))
            .collect::<Vec<_>>();
        let values = Self::run_blocking(Arc::clone(&self.connections), move |connection| {
            redis::cmd("MGET")
                .arg(keys)
                .query::<Vec<redis::Value>>(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await?;
        values
            .into_iter()
            .map(|value| self.decode_batch_response(value))
            .collect()
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
        Self::run_blocking(Arc::clone(&self.connections), move |connection| {
            let mut pipeline = redis::pipe();
            for (key, payload) in entries {
                pipeline
                    .cmd("SETEX")
                    .arg(key)
                    .arg(ttl)
                    .arg(payload)
                    .ignore();
            }
            pipeline
                .query::<()>(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        let key = self.namespaced_key(key);
        Self::run_blocking(Arc::clone(&self.connections), move |connection| {
            connection.del::<_, ()>(key).map_err(|_| Error::Unavailable)
        })
        .await
    }

    async fn async_flush_cache(&self) -> Result<(), Error> {
        let pattern = self.namespaced_pattern()?;
        Self::run_blocking(Arc::clone(&self.connections), move |connection| {
            Self::flush_matching(connection, &pattern)
        })
        .await
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        match Self::run_blocking(Arc::clone(&self.connections), |connection| {
            Ok(match redis::cmd("PING").query::<String>(connection) {
                Ok(_) => CacheConnectionResult {
                    status: CacheConnectionStatus::Success,
                    message: "Redis cache connection test successful".into(),
                    error: None,
                },
                Err(error) => CacheConnectionResult {
                    status: CacheConnectionStatus::Failed,
                    message: format!("Redis connection failed: {error}"),
                    error: Some(error.to_string()),
                },
            })
        })
        .await
        {
            Ok(result) => Ok(result),
            Err(error) => Ok(CacheConnectionResult {
                status: CacheConnectionStatus::Failed,
                message: format!("Redis connection failed: {error}"),
                error: Some(error.to_string()),
            }),
        }
    }
}

impl<S, C> CounterCache for RedisCache<S, C>
where
    S: CacheCodec<Value = f64>,
    C: redis::ConnectionLike + Send + 'static,
{
    fn increment_cache(&self, key: &str, amount: f64, kwargs: CacheKwargs) -> Result<f64, Error> {
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        self.connections
            .execute(|connection| increment(connection, key, amount, ttl))
    }

    async fn async_increment_cache(
        &self,
        key: &str,
        amount: f64,
        kwargs: CacheKwargs,
    ) -> Result<f64, Error> {
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        Self::run_blocking(Arc::clone(&self.connections), move |connection| {
            increment(connection, key, amount, ttl)
        })
        .await
    }
}

fn increment(
    connection: &mut ConnectionRef<'_>,
    key: String,
    amount: f64,
    ttl: u64,
) -> Result<f64, Error> {
    redis::cmd("EVAL")
        .arg(INCREMENT_SCRIPT)
        .arg(1)
        .arg(key)
        .arg(amount)
        .arg(ttl)
        .query(connection)
        .map_err(|_| Error::Unavailable)
}

fn stored_bytes(value: redis::Value) -> Result<Option<Vec<u8>>, Error> {
    match value {
        redis::Value::Nil => Ok(None),
        redis::Value::BulkString(bytes) => Ok(Some(bytes)),
        redis::Value::SimpleString(text) => Ok(Some(text.into_bytes())),
        _ => Err(Error::InvalidEntry),
    }
}

/// Eligibility is decided on decoded values, so a pin written by another encoder (Python's
/// `json.dumps` spacing or key order) still matches. The write is a compare-and-set on the
/// bytes that decision was made on, retried when another claimant wins the race.
fn claim<S: CacheCodec>(
    connection: &mut ConnectionRef<'_>,
    codec: &S,
    key: &str,
    candidate: S::Value,
    eligible: &[S::Value],
    ttl: u64,
) -> Result<S::Value, Error>
where
    S::Value: PartialEq,
{
    let payload = codec.encode(&candidate)?;
    if payload.is_empty() {
        return Err(Error::InvalidEntry);
    }
    for _ in 0..CLAIM_ATTEMPTS {
        let current = stored_bytes(
            connection
                .get::<_, redis::Value>(key)
                .map_err(|_| Error::Unavailable)?,
        )?
        .filter(|bytes| !bytes.is_empty());
        let existing = current
            .as_deref()
            .and_then(|bytes| codec.decode(bytes).ok())
            .filter(|existing| eligible.is_empty() || eligible.contains(existing));
        let refresh = existing
            .as_ref()
            .is_some_and(|existing| !eligible.is_empty() || *existing == candidate);
        let write: &[u8] = if existing.is_some() { b"" } else { &payload };
        let applied = redis::cmd("EVAL")
            .arg(CLAIM_SCRIPT)
            .arg(1)
            .arg(key)
            .arg(current.as_deref().unwrap_or_default())
            .arg(ttl)
            .arg(write)
            .arg(u8::from(refresh))
            .query::<bool>(connection)
            .map_err(|_| Error::Unavailable)?;
        if applied {
            return Ok(existing.unwrap_or(candidate));
        }
    }
    Err(Error::Unavailable)
}

impl<S, C> ClaimCache for RedisCache<S, C>
where
    S: CacheCodec + Clone + 'static,
    S::Value: PartialEq,
    C: redis::ConnectionLike + Send + 'static,
{
    fn claim_cache(
        &self,
        key: &str,
        candidate: S::Value,
        eligible: &[S::Value],
        kwargs: CacheKwargs,
    ) -> Result<S::Value, Error> {
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        self.connections
            .execute(|connection| claim(connection, &self.codec, &key, candidate, eligible, ttl))
    }

    async fn async_claim_cache(
        &self,
        key: &str,
        candidate: S::Value,
        eligible: Vec<S::Value>,
        kwargs: CacheKwargs,
    ) -> Result<S::Value, Error> {
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(self.get_ttl(&kwargs));
        let codec = self.codec.clone();
        Self::run_blocking(Arc::clone(&self.connections), move |connection| {
            claim(connection, &codec, &key, candidate, &eligible, ttl)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use litellm_cache::{BaseCache, CacheCodec, CacheKwargs, JsonCodec};
    use redis_test::{MockCmd, MockRedisConnection};
    use serde_json::json;

    use super::RedisCache;

    fn entry() -> serde_json::Value {
        json!({"deployment": "model-a", "cooldown_seconds": 30})
    }

    #[test]
    fn ttl_seconds_rounds_up_and_keeps_expiration_positive() {
        assert_eq!(
            RedisCache::<JsonCodec<serde_json::Value>>::ttl_seconds(Duration::ZERO),
            1
        );
        assert_eq!(
            RedisCache::<JsonCodec<serde_json::Value>>::ttl_seconds(Duration::from_millis(1500)),
            2
        );
        assert_eq!(
            RedisCache::<JsonCodec<serde_json::Value>>::ttl_seconds(Duration::from_secs(15)),
            15
        );
    }

    #[test]
    fn redis_commands_round_trip_entries_and_delete_only_namespaced_keys() {
        let value = entry();
        let payload = JsonCodec::<serde_json::Value>::new()
            .encode(&value)
            .unwrap();
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
        let cache =
            RedisCache::with_connection(connection, None, JsonCodec::<serde_json::Value>::new())
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
                    .arg("litellm-cache:*")
                    .arg("COUNT")
                    .arg(1000),
                Ok(redis_test::redis_value!(["0", ["litellm-cache:key"]])),
            ),
            MockCmd::new(redis::cmd("DEL").arg("litellm-cache:key"), Ok(1u32)),
        ])
        .assert_all_commands_consumed();
        let cache =
            RedisCache::with_connection(connection, None, JsonCodec::<serde_json::Value>::new())
                .with_namespace(Some("litellm-cache".into()));

        cache.flush_cache().unwrap();
    }

    #[tokio::test]
    async fn test_connection_runs_ping_off_executor() {
        let connection = MockRedisConnection::new([MockCmd::new(redis::cmd("PING"), Ok("PONG"))])
            .assert_all_commands_consumed();
        let cache =
            RedisCache::with_connection(connection, None, JsonCodec::<serde_json::Value>::new())
                .with_namespace(Some("litellm-cache".into()));

        assert_eq!(
            cache.test_connection().await.unwrap().status,
            litellm_cache::CacheConnectionStatus::Success
        );
    }
}
