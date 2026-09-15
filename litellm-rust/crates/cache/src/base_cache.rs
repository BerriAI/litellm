use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde_json::{Map, Value};

use crate::Error;

pub type CacheFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CacheKwargs {
    pub ttl: Option<Duration>,
    pub extras: Map<String, Value>,
}

pub trait BaseCache: Send + Sync {
    type Value: Clone + Send + Sync + 'static;

    fn set_cache(&self, key: &str, value: Self::Value, kwargs: CacheKwargs) -> Result<(), Error>;

    fn get_cache(&self, key: &str, kwargs: &CacheKwargs) -> Result<Option<Self::Value>, Error>;

    fn async_set_cache<'a>(
        &'a self,
        key: &'a str,
        value: Self::Value,
        kwargs: CacheKwargs,
    ) -> CacheFuture<'a, ()> {
        Box::pin(async move { self.set_cache(key, value, kwargs) })
    }

    fn async_get_cache<'a>(
        &'a self,
        key: &'a str,
        kwargs: &'a CacheKwargs,
    ) -> CacheFuture<'a, Option<Self::Value>> {
        Box::pin(async move { self.get_cache(key, kwargs) })
    }

    fn async_set_cache_pipeline<'a>(
        &'a self,
        cache_list: Vec<(String, Self::Value)>,
        kwargs: CacheKwargs,
    ) -> CacheFuture<'a, ()> {
        Box::pin(async move {
            for (key, value) in cache_list {
                self.set_cache(&key, value, kwargs.clone())?;
            }
            Ok(())
        })
    }

    fn batch_cache_write<'a>(
        &'a self,
        key: &'a str,
        value: Self::Value,
        kwargs: CacheKwargs,
    ) -> CacheFuture<'a, ()> {
        self.async_set_cache(key, value, kwargs)
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error>;

    fn async_delete_cache<'a>(&'a self, key: &'a str) -> CacheFuture<'a, ()> {
        Box::pin(async move { self.delete_cache(key) })
    }

    fn flush_cache(&self) -> Result<(), Error>;

    fn disconnect(&self) -> CacheFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn test_connection(&self) -> CacheFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
