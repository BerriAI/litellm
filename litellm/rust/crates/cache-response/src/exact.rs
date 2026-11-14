use std::{future::Future, pin::Pin, time::Duration};

use litellm_cache::{
    BaseCache, BatchCache, CacheConnectionResult, ConnectionCache, Error, ExactCacheContext,
    FlushCache,
};
use serde_json::Value;

use crate::{CacheEntry, PartialHits, ResponseCache, ResponseCacheRequest};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Object-safe view of a `ResponseCache` over an exact-match backend, so hosts can hold every
/// exact backend behind one pointer without erasing which backend it is elsewhere.
pub trait ExactResponseCache: Send + Sync {
    fn default_ttl(&self) -> Option<Duration>;

    fn lookup(&self, request: &ResponseCacheRequest, now: Duration)
    -> Result<Option<Value>, Error>;

    fn store(
        &self,
        request: &ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error>;

    fn lookup_batch(
        &self,
        requests: &[ResponseCacheRequest],
        now: Duration,
    ) -> Result<PartialHits, Error>;

    fn async_lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> BoxFuture<'a, Result<Option<Value>, Error>>;

    fn async_store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> BoxFuture<'a, Result<(), Error>>;

    fn async_lookup_batch<'a>(
        &'a self,
        requests: &'a [ResponseCacheRequest],
        now: Duration,
    ) -> BoxFuture<'a, Result<PartialHits, Error>>;

    fn async_store_batch<'a>(
        &'a self,
        entries: Vec<(ResponseCacheRequest, Value)>,
        now: Duration,
    ) -> BoxFuture<'a, Result<(), Error>>;

    fn async_store_entries<'a>(
        &'a self,
        entries: Vec<(ResponseCacheRequest, Value, Duration)>,
    ) -> BoxFuture<'a, Result<(), Error>>;

    fn async_flush<'a>(&'a self) -> BoxFuture<'a, Result<(), Error>>;
}

/// Object-safe `test_connection` for the exact backends whose Python class defines it. Hosts hold
/// one next to their `ExactResponseCache` when the backend has it, and report the operation as
/// unsupported otherwise, as Python's `BaseCache.test_connection` does.
pub trait ConnectionProbe: Send + Sync {
    fn test_connection<'a>(&'a self) -> BoxFuture<'a, Result<CacheConnectionResult, Error>>;
}

impl<B> ConnectionProbe for ResponseCache<B>
where
    B: ConnectionCache<Value = CacheEntry>,
    B::Context: Default + PartialEq,
{
    fn test_connection<'a>(&'a self) -> BoxFuture<'a, Result<CacheConnectionResult, Error>> {
        Box::pin(ResponseCache::test_connection(self))
    }
}

impl<B> ExactResponseCache for ResponseCache<B>
where
    B: BaseCache<Value = CacheEntry, Context = ExactCacheContext> + BatchCache + FlushCache,
{
    fn default_ttl(&self) -> Option<Duration> {
        ResponseCache::default_ttl(self)
    }

    fn lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        ResponseCache::lookup(self, request, now)
    }

    fn store(
        &self,
        request: &ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        ResponseCache::store(self, request, response, now)
    }

    fn lookup_batch(
        &self,
        requests: &[ResponseCacheRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        ResponseCache::lookup_batch(self, requests, now)
    }

    fn async_lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> BoxFuture<'a, Result<Option<Value>, Error>> {
        Box::pin(ResponseCache::async_lookup(self, request, now))
    }

    fn async_store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(ResponseCache::async_store(self, request, response, now))
    }

    fn async_lookup_batch<'a>(
        &'a self,
        requests: &'a [ResponseCacheRequest],
        now: Duration,
    ) -> BoxFuture<'a, Result<PartialHits, Error>> {
        Box::pin(ResponseCache::async_lookup_batch(self, requests, now))
    }

    fn async_store_batch<'a>(
        &'a self,
        entries: Vec<(ResponseCacheRequest, Value)>,
        now: Duration,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(ResponseCache::async_store_batch(self, entries, now))
    }

    fn async_store_entries<'a>(
        &'a self,
        entries: Vec<(ResponseCacheRequest, Value, Duration)>,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(ResponseCache::async_store_entries(self, entries))
    }

    fn async_flush<'a>(&'a self) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(ResponseCache::async_flush(self))
    }
}
