use std::{future::Future, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum BatchEntry<V> {
    Hit(V),
    Miss,
    Invalid,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CacheKwargs {
    pub ttl: Option<Duration>,
    pub extras: Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CacheConnectionStatus {
    Success,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CacheConnectionResult {
    pub status: CacheConnectionStatus,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub trait BaseCache: Send + Sync {
    type Value: Clone + Send + Sync + 'static;

    fn default_ttl(&self) -> Duration {
        Duration::from_secs(60)
    }

    fn get_ttl(&self, kwargs: &CacheKwargs) -> Duration {
        kwargs.ttl.unwrap_or_else(|| self.default_ttl())
    }

    fn set_cache(&self, key: &str, value: Self::Value, kwargs: CacheKwargs) -> Result<(), Error>;

    fn get_cache(&self, key: &str, kwargs: &CacheKwargs) -> Result<Option<Self::Value>, Error>;

    fn get_cache_batch(
        &self,
        keys: &[String],
        kwargs: &CacheKwargs,
    ) -> Result<Vec<BatchEntry<Self::Value>>, Error> {
        keys.iter()
            .map(|key| match self.get_cache(key, kwargs) {
                Ok(Some(value)) => Ok(BatchEntry::Hit(value)),
                Ok(None) => Ok(BatchEntry::Miss),
                Err(Error::InvalidEntry) => Ok(BatchEntry::Invalid),
                Err(error) => Err(error),
            })
            .collect()
    }

    fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        kwargs: CacheKwargs,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        async move { self.set_cache(key, value, kwargs) }
    }

    fn async_get_cache(
        &self,
        key: &str,
        kwargs: &CacheKwargs,
    ) -> impl Future<Output = Result<Option<Self::Value>, Error>> + Send {
        async move { self.get_cache(key, kwargs) }
    }

    fn async_get_cache_batch(
        &self,
        keys: Vec<String>,
        kwargs: CacheKwargs,
    ) -> impl Future<Output = Result<Vec<BatchEntry<Self::Value>>, Error>> + Send {
        async move {
            let mut entries = Vec::with_capacity(keys.len());
            for key in keys {
                entries.push(match self.async_get_cache(&key, &kwargs).await {
                    Ok(Some(value)) => BatchEntry::Hit(value),
                    Ok(None) => BatchEntry::Miss,
                    Err(Error::InvalidEntry) => BatchEntry::Invalid,
                    Err(error) => return Err(error),
                });
            }
            Ok(entries)
        }
    }

    fn async_set_cache_pipeline(
        &self,
        cache_list: Vec<(String, Self::Value)>,
        kwargs: CacheKwargs,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        async move {
            for (key, value) in cache_list {
                self.async_set_cache(&key, value, kwargs.clone()).await?;
            }
            Ok(())
        }
    }

    fn batch_cache_write(
        &self,
        key: &str,
        value: Self::Value,
        kwargs: CacheKwargs,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        self.async_set_cache(key, value, kwargs)
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error>;

    fn async_delete_cache(&self, key: &str) -> impl Future<Output = Result<(), Error>> + Send {
        async move { self.delete_cache(key) }
    }

    fn flush_cache(&self) -> Result<(), Error>;

    fn async_flush_cache(&self) -> impl Future<Output = Result<(), Error>> + Send {
        async move { self.flush_cache() }
    }

    fn disconnect(&self) -> impl Future<Output = Result<(), Error>> + Send;

    fn test_connection(&self) -> impl Future<Output = Result<CacheConnectionResult, Error>> + Send;
}
