use std::{future::Future, time::Duration};

use crate::{BaseCache, BatchEntry, CacheConnectionResult, CacheContext, Error};

#[derive(Clone, Debug, PartialEq)]
pub struct IncrementOperation {
    pub key: String,
    pub amount: f64,
    pub ttl: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PushOperation<V> {
    pub key: String,
    pub values: Vec<V>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PopOperation {
    pub key: String,
    pub count: Option<usize>,
}

/// `disconnect`, for backends whose Python class releases connections or clients.
pub trait DisconnectCache: BaseCache {
    fn disconnect(&self) -> impl Future<Output = Result<(), Error>> + Send;
}

/// `test_connection`, for backends whose Python class overrides the base `NotImplementedError`.
pub trait ConnectionCache: BaseCache {
    fn test_connection(&self) -> impl Future<Output = Result<CacheConnectionResult, Error>> + Send;
}

/// `sync_ping` and `ping`.
pub trait PingCache: BaseCache {
    fn sync_ping(&self) -> Result<bool, Error>;

    fn ping(&self) -> impl Future<Output = Result<bool, Error>> + Send;
}

pub trait BatchCache: BaseCache {
    fn batch_get_cache(
        &self,
        keys: &[String],
        context: &Self::Context,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        keys.iter()
            .map(|key| match self.get_cache(key, context) {
                Ok(Some(value)) => Ok(BatchEntry::Hit(value)),
                Ok(None) => Ok(BatchEntry::Miss),
                Err(Error::InvalidEntry) => Ok(BatchEntry::Invalid),
                Err(error) => Err(error),
            })
            .collect()
    }

    fn async_batch_get_cache(
        &self,
        keys: Vec<String>,
        context: Self::Context,
    ) -> impl Future<Output = Result<Vec<BatchEntry<Self::Value>>, Error>> + Send {
        async move {
            let mut entries = Vec::with_capacity(keys.len());
            for key in keys {
                entries.push(match self.async_get_cache(&key, &context).await {
                    Ok(Some(value)) => BatchEntry::Hit(value),
                    Ok(None) => BatchEntry::Miss,
                    Err(Error::InvalidEntry) => BatchEntry::Invalid,
                    Err(error) => return Err(error),
                });
            }
            Ok(entries)
        }
    }
}

/// `async_set_cache_pipeline_with_ttls`: one pipeline where every entry carries its own TTL.
pub trait TtlPipelineCache: BaseCache {
    fn async_set_cache_pipeline_with_ttls(
        &self,
        entries: Vec<(String, Self::Value, Option<Duration>)>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
}

pub trait DeleteCache: BaseCache {
    fn delete_cache(&self, key: &str) -> Result<(), Error>;

    fn async_delete_cache(&self, key: &str) -> impl Future<Output = Result<(), Error>> + Send {
        async move { self.delete_cache(key) }
    }
}

/// `delete_cache_keys`: one round trip that reports how many keys existed.
pub trait BulkDeleteCache: DeleteCache {
    fn delete_cache_keys(
        &self,
        keys: Vec<String>,
    ) -> impl Future<Output = Result<usize, Error>> + Send;
}

pub trait FlushCache: BaseCache {
    fn flush_cache(&self) -> Result<(), Error>;

    fn async_flush_cache(&self) -> impl Future<Output = Result<(), Error>> + Send {
        async move { self.flush_cache() }
    }
}

/// `flushall`: drops every key on the server, ignoring any namespace.
pub trait FlushAllCache: FlushCache {
    fn flushall(&self) -> Result<(), Error>;
}

/// Numeric counters. Counters are independent of `Value`, so a response-valued backend can
/// expose them, the way one Python `RedisCache` serves both.
pub trait CounterCache: BaseCache {
    fn increment_cache(&self, key: &str, amount: f64, context: Self::Context)
    -> Result<f64, Error>;

    /// `refresh_ttl` re-arms the TTL on every write instead of only when the key is new;
    /// backends without expiring counters ignore it, as Python's `**kwargs` does.
    fn async_increment(
        &self,
        key: &str,
        amount: f64,
        context: Self::Context,
        _refresh_ttl: bool,
    ) -> impl Future<Output = Result<f64, Error>> + Send {
        async move { self.increment_cache(key, amount, context) }
    }

    /// `async_increment_pipeline`, one result per operation in order. The default increments
    /// one key at a time, as the in-memory cache does.
    fn async_increment_pipeline(
        &self,
        operations: Vec<IncrementOperation>,
    ) -> impl Future<Output = Result<Vec<f64>, Error>> + Send
    where
        Self::Context: Default,
    {
        async move {
            let mut results = Vec::with_capacity(operations.len());
            for operation in operations {
                let context = Self::Context::default().with_ttl(operation.ttl);
                results.push(
                    self.async_increment(&operation.key, operation.amount, context, false)
                        .await?,
                );
            }
            Ok(results)
        }
    }
}

/// `batch_get_counts` and `async_batch_get_counts`: counter values read in one round trip.
pub trait CountReadCache: CounterCache {
    fn batch_get_counts(&self, keys: &[String]) -> Result<Vec<Option<i64>>, Error>;

    fn async_batch_get_counts(
        &self,
        keys: Vec<String>,
    ) -> impl Future<Output = Result<Vec<Option<i64>>, Error>> + Send;
}

/// `increment_with_floor`, `async_increment_with_floor`, and `async_set_max`.
pub trait BoundedCounterCache: CounterCache {
    fn increment_with_floor(&self, key: &str, amount: i64, ttl: Duration) -> Result<i64, Error>;

    fn async_increment_with_floor(
        &self,
        key: &str,
        amount: i64,
        ttl: Duration,
    ) -> impl Future<Output = Result<i64, Error>> + Send;

    fn async_set_max(
        &self,
        key: &str,
        value: f64,
        ttl: Option<Duration>,
    ) -> impl Future<Output = Result<f64, Error>> + Send;
}

pub trait ClaimCache: BaseCache
where
    Self::Value: PartialEq,
{
    fn claim_cache(
        &self,
        key: &str,
        candidate: Self::Value,
        eligible: &[Self::Value],
        context: Self::Context,
    ) -> Result<Self::Value, Error>;

    fn async_claim_cache(
        &self,
        key: &str,
        candidate: Self::Value,
        eligible: Vec<Self::Value>,
        context: Self::Context,
    ) -> impl Future<Output = Result<Self::Value, Error>> + Send {
        async move { self.claim_cache(key, candidate, &eligible, context) }
    }
}

pub trait TtlCache: BaseCache {
    fn async_get_ttl(
        &self,
        key: &str,
    ) -> impl Future<Output = Result<Option<Duration>, Error>> + Send;
}

pub trait RefreshTtlCache: TtlCache {
    /// `async_refresh_ttl`: re-arms an existing key without touching its value. `ttl` falls
    /// back to the backend default, and the result is `false` when the key is absent or
    /// neither TTL is set.
    fn async_refresh_ttl(
        &self,
        key: &str,
        ttl: Option<Duration>,
    ) -> impl Future<Output = Result<bool, Error>> + Send;
}

pub trait SetCache: BaseCache {
    type SetValue: Clone + Send + Sync + 'static;
    type SetResult: Send + Sync + 'static;

    fn async_set_cache_sadd(
        &self,
        key: &str,
        values: Vec<Self::SetValue>,
        ttl: Option<Duration>,
    ) -> impl Future<Output = Result<Self::SetResult, Error>> + Send;
}

pub trait QueueCache: BaseCache {
    type QueueValue: Clone + Send + Sync + 'static;
    type PopResult: Send + Sync + 'static;

    fn async_rpush(
        &self,
        key: &str,
        values: Vec<Self::QueueValue>,
    ) -> impl Future<Output = Result<usize, Error>> + Send;

    /// `async_rpush_and_trim`: pushes, then keeps only the newest `max_len` entries, atomically.
    /// Returns the list length right after the push, before the trim.
    fn async_rpush_and_trim(
        &self,
        key: &str,
        values: Vec<Self::QueueValue>,
        max_len: usize,
    ) -> impl Future<Output = Result<usize, Error>> + Send;

    fn async_rpush_pipeline(
        &self,
        operations: Vec<PushOperation<Self::QueueValue>>,
    ) -> impl Future<Output = Result<Vec<usize>, Error>> + Send;

    fn async_lpop(
        &self,
        key: &str,
        count: Option<usize>,
    ) -> impl Future<Output = Result<Self::PopResult, Error>> + Send;

    fn async_lpop_pipeline(
        &self,
        operations: Vec<PopOperation>,
    ) -> impl Future<Output = Result<Vec<Self::PopResult>, Error>> + Send;
}

pub trait ScanCache: BaseCache {
    fn async_scan_iter(
        &self,
        pattern: &str,
        count: usize,
    ) -> impl Future<Output = Result<Vec<String>, Error>> + Send;
}

pub trait ClientInfoCache: BaseCache {
    type ClientList: Send + Sync + 'static;
    type Info: Send + Sync + 'static;

    fn client_list(&self) -> Result<Self::ClientList, Error>;

    fn info(&self) -> Result<Self::Info, Error>;
}

pub trait CacheScript: Send + Sync + 'static {
    type Argument: Clone + Send + Sync + 'static;
    type Output: Send + Sync + 'static;

    fn invoke(
        &self,
        keys: Vec<String>,
        arguments: Vec<Self::Argument>,
    ) -> impl Future<Output = Result<Self::Output, Error>> + Send;
}

pub trait ScriptCache: BaseCache {
    type Script: CacheScript;

    fn async_register_script(&self, source: String) -> Self::Script;
}
