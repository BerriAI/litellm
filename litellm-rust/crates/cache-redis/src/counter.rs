use std::time::Duration;

use litellm_cache::{
    BoundedCounterCache, CacheCodec, CountReadCache, CounterCache, Error, ExactCacheContext,
    IncrementOperation,
};

use crate::{
    cache::{RedisCache, ttl_seconds},
    connection::ConnectionRef,
    store::mget,
};

const INCREMENT_SCRIPT: &str = concat!(
    "local value = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1]); ",
    "if redis.call('TTL', KEYS[1]) == -1 then ",
    "redis.call('EXPIRE', KEYS[1], ARGV[2]); end; return value"
);
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

impl<S, C> CounterCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn increment_cache(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
    ) -> Result<f64, Error> {
        let key = self.namespaced_key(key);
        let ttl = self.ttl_or_default(context.ttl);
        self.execute(|connection| increment(connection, key, amount, ttl, false))
    }

    /// Python `_incrbyfloat_with_ttl`: without `refresh_ttl` the TTL is set only on a key that
    /// has none, in one atomic script; with it, every increment re-arms the TTL.
    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
        refresh_ttl: bool,
    ) -> Result<f64, Error> {
        let key = self.namespaced_key(key);
        let ttl = self.ttl_or_default(context.ttl);
        self.run(move |connection| increment(connection, key, amount, ttl, refresh_ttl))
            .await
    }

    async fn async_increment_pipeline(
        &self,
        operations: Vec<IncrementOperation>,
    ) -> Result<Vec<f64>, Error> {
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        let mut pipeline = redis::pipe();
        for operation in operations {
            let key = self.namespaced_key(&operation.key);
            pipeline.cmd("INCRBYFLOAT").arg(&key).arg(operation.amount);
            if let Some(ttl) = operation.ttl {
                pipeline
                    .cmd("EXPIRE")
                    .arg(key)
                    .arg(ttl_seconds(ttl))
                    .ignore();
            }
        }
        self.run(move |connection| connection.query_pipeline(&pipeline))
            .await
    }
}

fn increment(
    connection: &mut ConnectionRef<'_>,
    key: String,
    amount: f64,
    ttl: u64,
    refresh_ttl: bool,
) -> Result<f64, Error> {
    if !refresh_ttl {
        return redis::cmd("EVAL")
            .arg(INCREMENT_SCRIPT)
            .arg(1)
            .arg(key)
            .arg(amount)
            .arg(ttl)
            .query(connection)
            .map_err(|_| Error::Unavailable);
    }
    connection
        .query_pipeline(
            redis::pipe()
                .cmd("INCRBYFLOAT")
                .arg(&key)
                .arg(amount)
                .cmd("EXPIRE")
                .arg(&key)
                .arg(ttl)
                .ignore(),
        )
        .map(|(value,)| value)
}

impl<S, C> CountReadCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn batch_get_counts(&self, keys: &[String]) -> Result<Vec<Option<i64>>, Error> {
        let keys = self.namespaced_keys(keys);
        self.execute(|connection| mget(connection, keys))?
            .into_iter()
            .map(count)
            .collect()
    }

    async fn async_batch_get_counts(&self, keys: Vec<String>) -> Result<Vec<Option<i64>>, Error> {
        let keys = self.namespaced_keys(&keys);
        self.run(move |connection| mget(connection, keys))
            .await?
            .into_iter()
            .map(count)
            .collect()
    }
}

fn count(value: redis::Value) -> Result<Option<i64>, Error> {
    redis::from_redis_value(value).map_err(|_| Error::InvalidEntry)
}

impl<S, C> BoundedCounterCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn increment_with_floor(&self, key: &str, amount: i64, ttl: Duration) -> Result<i64, Error> {
        let key = self.namespaced_key(key);
        let ttl = ttl_seconds(ttl);
        self.execute(|connection| increment_with_floor(connection, key, amount, ttl))
    }

    async fn async_increment_with_floor(
        &self,
        key: &str,
        amount: i64,
        ttl: Duration,
    ) -> Result<i64, Error> {
        let key = self.namespaced_key(key);
        let ttl = ttl_seconds(ttl);
        self.run(move |connection| increment_with_floor(connection, key, amount, ttl))
            .await
    }

    async fn async_set_max(
        &self,
        key: &str,
        value: f64,
        ttl: Option<Duration>,
    ) -> Result<f64, Error> {
        let key = self.namespaced_key(key);
        let ttl = self.ttl_or_default(ttl);
        self.run(move |connection| {
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
