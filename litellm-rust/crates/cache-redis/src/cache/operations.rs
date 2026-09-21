use std::{sync::Arc, time::Duration};

use litellm_cache::{
    CacheCodec, CacheScript, ClientInfoCache, Error, IncrementOperation, QueueCache, ScanCache,
    ScriptCache, SetCache, TtlCache,
};
use redis::Commands;

use super::{ConnectionRef, Connections, RedisCache, namespaced_key};

const INCREMENT_WITH_FLOOR_SCRIPT: &str = concat!(
    "local count = redis.call('INCRBY', KEYS[1], ARGV[1]); ",
    "if count < 0 then count = redis.call('INCRBY', KEYS[1], -count); end; ",
    "if redis.call('TTL', KEYS[1]) < 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]); end; ",
    "return count"
);
const SET_MAX_SCRIPT: &str = concat!(
    "local current = redis.call('GET', KEYS[1]); ",
    "if current == false or tonumber(current) < tonumber(ARGV[1]) then ",
    "redis.call('SET', KEYS[1], ARGV[1]); ",
    "if tonumber(ARGV[2]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]); end; ",
    "return ARGV[1]; end; return current"
);

#[derive(Clone, Debug, PartialEq)]
pub enum RedisArg {
    Bytes(Vec<u8>),
    Integer(i64),
    Float(f64),
}

impl From<&str> for RedisArg {
    fn from(value: &str) -> Self {
        Self::Bytes(value.as_bytes().to_vec())
    }
}

impl From<String> for RedisArg {
    fn from(value: String) -> Self {
        Self::Bytes(value.into_bytes())
    }
}

impl From<Vec<u8>> for RedisArg {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<i64> for RedisArg {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for RedisArg {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl redis::ToRedisArgs for RedisArg {
    fn write_redis_args<W>(&self, out: &mut W)
    where
        W: ?Sized + redis::RedisWrite,
    {
        match self {
            Self::Bytes(value) => value.write_redis_args(out),
            Self::Integer(value) => value.write_redis_args(out),
            Self::Float(value) => value.write_redis_args(out),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RedisRpushOperation {
    pub key: String,
    pub values: Vec<RedisArg>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RedisLpopOperation {
    pub key: String,
    pub count: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RedisLpopResult {
    Missing,
    Value(Vec<u8>),
    Values(Vec<Vec<u8>>),
}

pub struct RedisScript<C> {
    connections: Arc<Connections<C>>,
    namespace: Option<String>,
    source: String,
}

impl<C> CacheScript for RedisScript<C>
where
    C: redis::ConnectionLike + Send + 'static,
{
    type Argument = RedisArg;
    type Output = redis::Value;

    async fn invoke(
        &self,
        keys: Vec<String>,
        arguments: Vec<Self::Argument>,
    ) -> Result<Self::Output, Error> {
        let keys = keys
            .into_iter()
            .map(|key| namespaced_key(self.namespace.as_deref(), &key))
            .collect::<Vec<_>>();
        let connections = Arc::clone(&self.connections);
        let source = self.source.clone();
        tokio::task::spawn_blocking(move || {
            connections.execute(|connection| {
                redis::cmd("EVAL")
                    .arg(source)
                    .arg(keys.len())
                    .arg(keys)
                    .arg(arguments)
                    .query(connection)
                    .map_err(|_| Error::Unavailable)
            })
        })
        .await
        .map_err(|_| Error::Unavailable)?
    }
}

impl<S, C> RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    pub async fn delete_cache_keys(&self, keys: Vec<String>) -> Result<usize, Error> {
        if keys.is_empty() {
            return Ok(0);
        }
        let keys = keys
            .into_iter()
            .map(|key| self.namespaced_key(&key))
            .collect::<Vec<_>>();
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            connection.del(keys).map_err(|_| Error::Unavailable)
        })
        .await
    }

    pub fn batch_get_counts(&self, keys: &[String]) -> Result<Vec<Option<i64>>, Error> {
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
        values.into_iter().map(count).collect()
    }

    pub async fn async_batch_get_counts(
        &self,
        keys: Vec<String>,
    ) -> Result<Vec<Option<i64>>, Error> {
        let keys = keys
            .iter()
            .map(|key| self.namespaced_key(key))
            .collect::<Vec<_>>();
        let values = Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            redis::cmd("MGET")
                .arg(keys)
                .query::<Vec<redis::Value>>(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await?;
        values.into_iter().map(count).collect()
    }

    pub fn sync_ping(&self) -> Result<bool, Error> {
        self.connections.execute(|connection| {
            redis::cmd("PING")
                .query::<String>(connection)
                .map(|response| response == "PONG")
                .map_err(|_| Error::Unavailable)
        })
    }

    pub async fn ping(&self) -> Result<bool, Error> {
        Connections::run_blocking(Arc::clone(&self.connections), |connection| {
            redis::cmd("PING")
                .query::<String>(connection)
                .map(|response| response == "PONG")
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    pub async fn async_get_ttl(&self, key: &str) -> Result<Option<i64>, Error> {
        let key = self.namespaced_key(key);
        let ttl = Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            redis::cmd("TTL")
                .arg(key)
                .query::<i64>(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await?;
        Ok((ttl >= 0).then_some(ttl))
    }

    pub async fn async_scan_iter(&self, pattern: &str, count: usize) -> Result<Vec<String>, Error> {
        let pattern = format!("{}*", self.namespaced_key(pattern));
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            let mut cursor = 0u64;
            let mut matches = Vec::new();
            loop {
                let (next_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                    .cursor_arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(count)
                    .query(connection)
                    .map_err(|_| Error::Unavailable)?;
                matches.extend(keys);
                if matches.len() >= count || next_cursor == 0 {
                    matches.truncate(count);
                    return Ok(matches);
                }
                cursor = next_cursor;
            }
        })
        .await
    }

    pub async fn async_set_cache_sadd(
        &self,
        key: &str,
        values: Vec<RedisArg>,
        ttl: Option<Duration>,
    ) -> Result<usize, Error> {
        if values.is_empty() {
            return Err(Error::InvalidEntry);
        }
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(ttl.unwrap_or(self.default_ttl));
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            let mut pipeline = redis::pipe();
            pipeline.cmd("SADD").arg(&key).arg(values);
            pipeline.cmd("EXPIRE").arg(&key).arg(ttl).ignore();
            pipeline
                .query::<(usize,)>(connection)
                .map(|(added,)| added)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    pub async fn async_rpush(&self, key: &str, values: Vec<RedisArg>) -> Result<usize, Error> {
        if values.is_empty() {
            return Err(Error::InvalidEntry);
        }
        let key = self.namespaced_key(key);
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            redis::cmd("RPUSH")
                .arg(key)
                .arg(values)
                .query(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    pub async fn async_rpush_pipeline(
        &self,
        operations: Vec<RedisRpushOperation>,
    ) -> Result<Vec<usize>, Error> {
        let operations = operations
            .into_iter()
            .map(|operation| {
                if operation.values.is_empty() {
                    return Err(Error::InvalidEntry);
                }
                Ok((self.namespaced_key(&operation.key), operation.values))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            let mut pipeline = redis::pipe();
            for (key, values) in operations {
                pipeline.cmd("RPUSH").arg(key).arg(values);
            }
            pipeline.query(connection).map_err(|_| Error::Unavailable)
        })
        .await
    }

    pub async fn async_lpop(
        &self,
        key: &str,
        count: Option<usize>,
    ) -> Result<RedisLpopResult, Error> {
        let key = self.namespaced_key(key);
        let multiple = count.is_some();
        let value = Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            let mut command = redis::cmd("LPOP");
            command.arg(key);
            if let Some(count) = count {
                command.arg(count);
            }
            command
                .query::<redis::Value>(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await?;
        lpop_result(value, multiple)
    }

    pub async fn async_lpop_pipeline(
        &self,
        operations: Vec<RedisLpopOperation>,
    ) -> Result<Vec<RedisLpopResult>, Error> {
        let operations = operations
            .into_iter()
            .map(|operation| (self.namespaced_key(&operation.key), operation.count))
            .collect::<Vec<_>>();
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        let multiple = operations
            .iter()
            .map(|(_, count)| count.is_some())
            .collect::<Vec<_>>();
        let values = Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            let mut pipeline = redis::pipe();
            for (key, count) in operations {
                let command = pipeline.cmd("LPOP").arg(key);
                if let Some(count) = count {
                    command.arg(count);
                }
            }
            pipeline
                .query::<Vec<redis::Value>>(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await?;
        values
            .into_iter()
            .zip(multiple)
            .map(|(value, multiple)| lpop_result(value, multiple))
            .collect()
    }

    pub async fn async_eval(
        &self,
        script: String,
        keys: Vec<String>,
        arguments: Vec<RedisArg>,
    ) -> Result<redis::Value, Error> {
        let keys = keys
            .into_iter()
            .map(|key| self.namespaced_key(&key))
            .collect::<Vec<_>>();
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            redis::cmd("EVAL")
                .arg(script)
                .arg(keys.len())
                .arg(keys)
                .arg(arguments)
                .query(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    pub fn client_list(&self) -> Result<String, Error> {
        self.connections.execute(|connection| {
            redis::cmd("CLIENT")
                .arg("LIST")
                .query(connection)
                .map_err(|_| Error::Unavailable)
        })
    }

    pub fn info(&self) -> Result<String, Error> {
        self.connections.execute(|connection| {
            redis::cmd("INFO")
                .query(connection)
                .map_err(|_| Error::Unavailable)
        })
    }

    pub fn flushall(&self) -> Result<(), Error> {
        self.connections.execute(|connection| {
            redis::cmd("FLUSHALL")
                .query(connection)
                .map_err(|_| Error::Unavailable)
        })
    }
}

impl<S, C> RedisCache<S, C>
where
    S: CacheCodec<Value = f64>,
    C: redis::ConnectionLike + Send + 'static,
{
    pub fn increment_with_floor(
        &self,
        key: &str,
        amount: i64,
        ttl: Duration,
    ) -> Result<i64, Error> {
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(ttl);
        self.connections
            .execute(|connection| increment_with_floor(connection, key, amount, ttl))
    }

    pub async fn async_increment_pipeline(
        &self,
        operations: Vec<IncrementOperation>,
    ) -> Result<Vec<f64>, Error> {
        let operations = operations
            .into_iter()
            .map(|operation| {
                (
                    self.namespaced_key(&operation.key),
                    operation.amount,
                    operation.ttl.map(Self::ttl_seconds),
                )
            })
            .collect::<Vec<_>>();
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            let mut pipeline = redis::pipe();
            for (key, amount, ttl) in operations {
                pipeline.cmd("INCRBYFLOAT").arg(&key).arg(amount);
                if let Some(ttl) = ttl {
                    pipeline.cmd("EXPIRE").arg(key).arg(ttl).ignore();
                }
            }
            pipeline.query(connection).map_err(|_| Error::Unavailable)
        })
        .await
    }

    pub async fn async_increment_with_floor(
        &self,
        key: &str,
        amount: i64,
        ttl: Duration,
    ) -> Result<i64, Error> {
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(ttl);
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            increment_with_floor(connection, key, amount, ttl)
        })
        .await
    }

    pub async fn async_set_max(
        &self,
        key: &str,
        value: f64,
        ttl: Option<Duration>,
    ) -> Result<f64, Error> {
        let key = self.namespaced_key(key);
        let ttl = Self::ttl_seconds(ttl.unwrap_or(self.default_ttl));
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            redis::cmd("EVAL")
                .arg(SET_MAX_SCRIPT)
                .arg(1)
                .arg(key)
                .arg(value)
                .arg(ttl)
                .query(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }
}

fn redis_bytes(value: redis::Value) -> Result<Vec<u8>, Error> {
    match value {
        redis::Value::BulkString(bytes) => Ok(bytes),
        redis::Value::SimpleString(text) => Ok(text.into_bytes()),
        _ => Err(Error::InvalidEntry),
    }
}

fn lpop_result(value: redis::Value, multiple: bool) -> Result<RedisLpopResult, Error> {
    match value {
        redis::Value::Nil => Ok(RedisLpopResult::Missing),
        redis::Value::Array(values) if multiple => values
            .into_iter()
            .map(redis_bytes)
            .collect::<Result<Vec<_>, _>>()
            .map(RedisLpopResult::Values),
        value if !multiple => redis_bytes(value).map(RedisLpopResult::Value),
        _ => Err(Error::InvalidEntry),
    }
}

fn count(value: redis::Value) -> Result<Option<i64>, Error> {
    match value {
        redis::Value::Nil => Ok(None),
        redis::Value::Int(value) => Ok(Some(value)),
        redis::Value::BulkString(value) => std::str::from_utf8(&value)
            .ok()
            .and_then(|value| value.parse().ok())
            .map(Some)
            .ok_or(Error::InvalidEntry),
        redis::Value::SimpleString(value) => {
            value.parse().map(Some).map_err(|_| Error::InvalidEntry)
        }
        _ => Err(Error::InvalidEntry),
    }
}

fn increment_with_floor(
    connection: &mut ConnectionRef<'_>,
    key: String,
    amount: i64,
    ttl: u64,
) -> Result<i64, Error> {
    redis::cmd("EVAL")
        .arg(INCREMENT_WITH_FLOOR_SCRIPT)
        .arg(1)
        .arg(key)
        .arg(amount)
        .arg(ttl)
        .query(connection)
        .map_err(|_| Error::Unavailable)
}

impl<S, C> TtlCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    async fn async_get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        RedisCache::async_get_ttl(self, key)
            .await
            .map(|ttl| ttl.map(|seconds| Duration::from_secs(seconds as u64)))
    }
}

impl<S, C> ScanCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    async fn async_scan_iter(&self, pattern: &str, count: usize) -> Result<Vec<String>, Error> {
        RedisCache::async_scan_iter(self, pattern, count).await
    }
}

impl<S, C> ClientInfoCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type ClientList = String;
    type Info = String;

    fn client_list(&self) -> Result<Self::ClientList, Error> {
        RedisCache::client_list(self)
    }

    fn info(&self) -> Result<Self::Info, Error> {
        RedisCache::info(self)
    }
}

impl<S, C> SetCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type SetValue = RedisArg;
    type SetResult = usize;

    async fn async_set_cache_sadd(
        &self,
        key: &str,
        values: Vec<Self::SetValue>,
        ttl: Option<Duration>,
    ) -> Result<Self::SetResult, Error> {
        RedisCache::async_set_cache_sadd(self, key, values, ttl).await
    }
}

impl<S, C> QueueCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type QueueValue = RedisArg;
    type PopResult = RedisLpopResult;

    async fn async_rpush(&self, key: &str, values: Vec<Self::QueueValue>) -> Result<usize, Error> {
        RedisCache::async_rpush(self, key, values).await
    }

    async fn async_lpop(&self, key: &str, count: Option<usize>) -> Result<Self::PopResult, Error> {
        RedisCache::async_lpop(self, key, count).await
    }
}

impl<S, C> ScriptCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type Script = RedisScript<C>;

    fn async_register_script(&self, source: String) -> Self::Script {
        RedisScript {
            connections: Arc::clone(&self.connections),
            namespace: self.namespace.clone(),
            source,
        }
    }
}
