use std::time::Duration;

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, BulkDeleteCache, CacheCodec, DeleteCache, Error,
    ExactCacheContext, FlushAllCache, FlushCache, TtlPipelineCache,
};
use redis::Commands;

use crate::{cache::RedisCache, connection::ConnectionRef};

impl<S, C> BaseCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type Value = S::Value;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl.or(Some(self.default_ttl))
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &ExactCacheContext,
    ) -> Result<(), Error> {
        let payload = self.codec.encode(&value)?;
        let ttl = self.ttl_or_default(context.ttl);
        let key = self.namespaced_key(key);
        self.execute(|connection| {
            connection
                .set_ex::<_, _, ()>(key, payload, ttl)
                .map_err(|_| Error::Unavailable)
        })
    }

    fn get_cache(&self, key: &str, _: &ExactCacheContext) -> Result<Option<Self::Value>, Error> {
        let key = self.namespaced_key(key);
        let value = self.execute(|connection| {
            connection
                .get::<_, redis::Value>(key)
                .map_err(|_| Error::Unavailable)
        })?;
        self.decode_response(value)
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        let payload = self.codec.encode(&value)?;
        let key = self.namespaced_key(key);
        let ttl = self.ttl_or_default(context.ttl);
        self.run(move |connection| {
            connection
                .set_ex::<_, _, ()>(key, payload, ttl)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        _: &ExactCacheContext,
    ) -> Result<Option<Self::Value>, Error> {
        let key = self.namespaced_key(key);
        let value = self
            .run(move |connection| {
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
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        self.async_set_cache_pipeline_with_ttls(
            cache_list
                .into_iter()
                .map(|(key, value)| (key, value, context.ttl))
                .collect(),
        )
        .await
    }
}

impl<S, C> TtlPipelineCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    async fn async_set_cache_pipeline_with_ttls(
        &self,
        entries: Vec<(String, Self::Value, Option<Duration>)>,
    ) -> Result<(), Error> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut pipeline = redis::pipe();
        for (key, value, ttl) in entries {
            pipeline
                .cmd("SETEX")
                .arg(self.namespaced_key(&key))
                .arg(self.ttl_or_default(ttl))
                .arg(self.codec.encode(&value)?)
                .ignore();
        }
        self.run(move |connection| connection.query_pipeline(&pipeline))
            .await
    }
}

impl<S, C> BatchCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn batch_get_cache(
        &self,
        keys: &[String],
        _: &ExactCacheContext,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        let keys = self.namespaced_keys(keys);
        self.execute(|connection| mget(connection, keys))?
            .into_iter()
            .map(|value| self.decode_batch_response(value))
            .collect()
    }

    async fn async_batch_get_cache(
        &self,
        keys: Vec<String>,
        _: ExactCacheContext,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        let keys = self.namespaced_keys(&keys);
        self.run(move |connection| mget(connection, keys))
            .await?
            .into_iter()
            .map(|value| self.decode_batch_response(value))
            .collect()
    }
}

pub(crate) fn mget(
    connection: &mut ConnectionRef<'_>,
    keys: Vec<String>,
) -> Result<Vec<redis::Value>, Error> {
    redis::cmd("MGET")
        .arg(keys)
        .query(connection)
        .map_err(|_| Error::Unavailable)
}

impl<S, C> DeleteCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        let key = self.namespaced_key(key);
        self.execute(|connection| connection.del::<_, ()>(key).map_err(|_| Error::Unavailable))
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        let key = self.namespaced_key(key);
        self.run(move |connection| connection.del::<_, ()>(key).map_err(|_| Error::Unavailable))
            .await
    }
}

impl<S, C> BulkDeleteCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    async fn delete_cache_keys(&self, keys: Vec<String>) -> Result<usize, Error> {
        if keys.is_empty() {
            return Ok(0);
        }
        let keys = self.namespaced_keys(&keys);
        self.run(move |connection| connection.del(keys).map_err(|_| Error::Unavailable))
            .await
    }
}

impl<S, C> FlushCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn flush_cache(&self) -> Result<(), Error> {
        let pattern = self.namespaced_pattern()?;
        self.execute(|connection| flush_matching(connection, &pattern))
    }

    async fn async_flush_cache(&self) -> Result<(), Error> {
        let pattern = self.namespaced_pattern()?;
        self.run(move |connection| flush_matching(connection, &pattern))
            .await
    }
}

impl<S, C> FlushAllCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn flushall(&self) -> Result<(), Error> {
        self.execute(|connection| connection.flushall())
    }
}

fn flush_matching(connection: &mut ConnectionRef<'_>, pattern: &str) -> Result<(), Error> {
    connection.scan(pattern, 1000, |connection, keys| {
        if !keys.is_empty() {
            connection
                .del::<_, usize>(keys)
                .map_err(|_| Error::Unavailable)?;
        }
        Ok(true)
    })
}
