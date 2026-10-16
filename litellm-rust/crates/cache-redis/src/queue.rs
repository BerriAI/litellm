use std::time::Duration;

use litellm_cache::{CacheCodec, Error, PopOperation, PushOperation, QueueCache, SetCache};

use crate::{cache::RedisCache, script::RedisArg};

pub type RedisRpushOperation = PushOperation<RedisArg>;
pub type RedisLpopOperation = PopOperation;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RedisLpopResult {
    Missing,
    Value(Vec<u8>),
    Values(Vec<Vec<u8>>),
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
        if values.is_empty() {
            return Err(Error::InvalidEntry);
        }
        let key = self.namespaced_key(key);
        let ttl = self.ttl_or_default(ttl);
        let mut pipeline = redis::pipe();
        pipeline
            .cmd("SADD")
            .arg(&key)
            .arg(values)
            .cmd("EXPIRE")
            .arg(&key)
            .arg(ttl)
            .ignore();
        self.run(move |connection| connection.query_pipeline(&pipeline))
            .await
            .map(|(added,)| added)
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
        if values.is_empty() {
            return Err(Error::InvalidEntry);
        }
        let key = self.namespaced_key(key);
        self.run(move |connection| {
            redis::cmd("RPUSH")
                .arg(key)
                .arg(values)
                .query(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    async fn async_rpush_and_trim(
        &self,
        key: &str,
        values: Vec<Self::QueueValue>,
        max_len: usize,
    ) -> Result<usize, Error> {
        if values.is_empty() {
            return Err(Error::InvalidEntry);
        }
        let key = self.namespaced_key(key);
        let start = i64::try_from(max_len).map_or(i64::MIN, |max_len| -max_len);
        let mut pipeline = redis::pipe();
        pipeline
            .atomic()
            .cmd("RPUSH")
            .arg(&key)
            .arg(values)
            .cmd("LTRIM")
            .arg(&key)
            .arg(start)
            .arg(-1)
            .ignore();
        self.run(move |connection| connection.query_pipeline(&pipeline))
            .await
            .map(|(length,)| length)
    }

    async fn async_rpush_pipeline(
        &self,
        operations: Vec<RedisRpushOperation>,
    ) -> Result<Vec<usize>, Error> {
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        let mut pipeline = redis::pipe();
        for operation in operations {
            if operation.values.is_empty() {
                return Err(Error::InvalidEntry);
            }
            pipeline
                .cmd("RPUSH")
                .arg(self.namespaced_key(&operation.key))
                .arg(operation.values);
        }
        self.run(move |connection| connection.query_pipeline(&pipeline))
            .await
    }

    async fn async_lpop(&self, key: &str, count: Option<usize>) -> Result<Self::PopResult, Error> {
        if let Some(count) = count
            && self.major_version().await < 7
        {
            return self.lpop_one_at_a_time(key, count).await;
        }
        let command = lpop(self.namespaced_key(key), count);
        let value = self
            .run(move |connection| {
                command
                    .query::<redis::Value>(connection)
                    .map_err(|_| Error::Unavailable)
            })
            .await?;
        lpop_result(value, count.is_some())
    }

    async fn async_lpop_pipeline(
        &self,
        operations: Vec<RedisLpopOperation>,
    ) -> Result<Vec<Self::PopResult>, Error> {
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        if operations.iter().any(|operation| operation.count.is_some())
            && self.major_version().await < 7
        {
            let mut results = Vec::with_capacity(operations.len());
            for operation in &operations {
                results.push(self.async_lpop(&operation.key, operation.count).await?);
            }
            return Ok(results);
        }
        let multiple = operations
            .iter()
            .map(|operation| operation.count.is_some())
            .collect::<Vec<_>>();
        let mut pipeline = redis::pipe();
        for operation in operations {
            pipeline.add_command(lpop(self.namespaced_key(&operation.key), operation.count));
        }
        self.run(move |connection| connection.query_pipeline::<Vec<redis::Value>>(&pipeline))
            .await?
            .into_iter()
            .zip(multiple)
            .map(|(value, multiple)| lpop_result(value, multiple))
            .collect()
    }
}

fn lpop(key: String, count: Option<usize>) -> redis::Cmd {
    let mut command = redis::cmd("LPOP");
    command.arg(key);
    if let Some(count) = count {
        command.arg(count);
    }
    command
}

fn redis_bytes(value: redis::Value) -> Result<Vec<u8>, Error> {
    match value {
        redis::Value::BulkString(bytes) => Ok(bytes),
        redis::Value::SimpleString(text) => Ok(text.into_bytes()),
        _ => Err(Error::InvalidEntry),
    }
}

impl<S, C> RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    /// `handle_lpop_count_for_older_redis_versions`: `count` single-`LPOP` pipelines, keeping
    /// only the values actually popped.
    async fn lpop_one_at_a_time(&self, key: &str, count: usize) -> Result<RedisLpopResult, Error> {
        let key = self.namespaced_key(key);
        let mut values = Vec::new();
        for _ in 0..count {
            let mut pipeline = redis::pipe();
            pipeline.add_command(lpop(key.clone(), None));
            let replies = self
                .run(move |connection| connection.query_pipeline::<Vec<redis::Value>>(&pipeline))
                .await?;
            for reply in replies {
                if reply != redis::Value::Nil {
                    values.push(redis_bytes(reply)?);
                }
            }
        }
        Ok(RedisLpopResult::Values(values))
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
