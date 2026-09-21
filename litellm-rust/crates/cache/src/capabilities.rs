use std::{future::Future, time::Duration};

use crate::{BaseCache, BatchEntry, Error};

#[derive(Clone, Debug, PartialEq)]
pub struct IncrementOperation {
    pub key: String,
    pub amount: f64,
    pub ttl: Option<Duration>,
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

pub trait DeleteCache: BaseCache {
    fn delete_cache(&self, key: &str) -> Result<(), Error>;

    fn async_delete_cache(&self, key: &str) -> impl Future<Output = Result<(), Error>> + Send {
        async move { self.delete_cache(key) }
    }
}

pub trait FlushCache: BaseCache {
    fn flush_cache(&self) -> Result<(), Error>;

    fn async_flush_cache(&self) -> impl Future<Output = Result<(), Error>> + Send {
        async move { self.flush_cache() }
    }
}

pub trait CounterCache: BaseCache<Value = f64> {
    fn increment_cache(&self, key: &str, amount: f64, context: Self::Context)
    -> Result<f64, Error>;

    fn async_increment(
        &self,
        key: &str,
        amount: f64,
        context: Self::Context,
    ) -> impl Future<Output = Result<f64, Error>> + Send {
        async move { self.increment_cache(key, amount, context) }
    }
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

    fn async_lpop(
        &self,
        key: &str,
        count: Option<usize>,
    ) -> impl Future<Output = Result<Self::PopResult, Error>> + Send;
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
