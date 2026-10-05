use std::time::Duration;

use litellm_cache::{CacheCodec, Error, RefreshTtlCache, ScanCache, TtlCache};

use crate::cache::RedisCache;

impl<S, C> TtlCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    async fn async_get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        let key = self.namespaced_key(key);
        let ttl = self
            .run(move |connection| {
                redis::cmd("TTL")
                    .arg(key)
                    .query::<i64>(connection)
                    .map_err(|_| Error::Unavailable)
            })
            .await?;
        Ok(u64::try_from(ttl).ok().map(Duration::from_secs))
    }
}

impl<S, C> RefreshTtlCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    async fn async_refresh_ttl(&self, key: &str, ttl: Option<Duration>) -> Result<bool, Error> {
        let key = self.namespaced_key(key);
        let ttl = self.ttl_or_default(ttl);
        self.run(move |connection| {
            redis::cmd("EXPIRE")
                .arg(key)
                .arg(ttl)
                .query(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }
}

impl<S, C> ScanCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    async fn async_scan_iter(&self, pattern: &str, count: usize) -> Result<Vec<String>, Error> {
        let pattern = format!("{}*", self.namespaced_key(pattern));
        self.run(move |connection| {
            let mut matches = Vec::new();
            connection.scan(&pattern, count, |_, keys| {
                matches.extend(keys);
                Ok(matches.len() < count)
            })?;
            matches.truncate(count);
            Ok(matches)
        })
        .await
    }
}
