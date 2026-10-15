use std::{future::Future, marker::PhantomData, pin::Pin, sync::Arc, time::Duration};

use litellm_cache::Error;
use litellm_cache_response::{
    CacheScope, ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService, ResponseEnvelope,
    ScopedCache,
};
use litellm_host::{machine::HostServices, protocol::Reply};
use litellm_inference::caching::{Cachable, CachedOutput};
use serde_json::Value;

use crate::cache::Cached;

const STREAM_EVENTS_KEY: &str = "litellm_cached_anthropic_sse_events";

pub(crate) enum CacheCall {
    Lookup {
        reply: Reply<Result<Option<Value>, Error>>,
    },
    Store {
        value: Value,
        reply: Reply<Result<(), Error>>,
    },
}

struct PythonCacheService<P: Cachable> {
    services: HostServices<Cached<P>>,
    config: ResponseCacheConfig,
    protocol: PhantomData<fn() -> P>,
}

pub(crate) fn scoped<P: Cachable>(services: HostServices<Cached<P>>) -> ScopedCache {
    ScopedCache::new(
        Arc::new(PythonCacheService {
            services,
            config: ResponseCacheConfig::default(),
            protocol: PhantomData,
        }),
        CacheScope::Shared,
    )
}

fn from_python<P: Cachable>(value: Value) -> Result<Value, Error> {
    let output = match value.get(STREAM_EVENTS_KEY) {
        Some(events) => {
            let events: Vec<String> =
                serde_json::from_value(events.clone()).map_err(|_| Error::InvalidEntry)?;
            CachedOutput::Stream(events.concat())
        }
        None => CachedOutput::Response(value),
    };
    serde_json::to_value(ResponseEnvelope::new(P::SURFACE, output)).map_err(|_| Error::InvalidEntry)
}

fn to_python<P: Cachable>(value: Value) -> Result<Value, Error> {
    let envelope: ResponseEnvelope<CachedOutput<Value>> =
        serde_json::from_value(value).map_err(|_| Error::InvalidEntry)?;
    match envelope.decode(P::SURFACE).ok_or(Error::InvalidEntry)? {
        CachedOutput::Response(response) => Ok(response),
        CachedOutput::Stream(text) => Ok(serde_json::json!({
            STREAM_EVENTS_KEY: text.split_inclusive("\n\n").collect::<Vec<_>>()
        })),
    }
}

impl<P: Cachable> ResponseCacheService for PythonCacheService<P> {
    fn config(&self) -> &ResponseCacheConfig {
        &self.config
    }

    fn lookup<'a>(
        &'a self,
        _: &'a ResponseCacheRequest,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, Error>> + Send + 'a>> {
        Box::pin(async move {
            self.services
                .call(|reply| CacheCall::Lookup { reply })
                .await
                .map_err(|_| Error::Unavailable)??
                .map(from_python::<P>)
                .transpose()
        })
    }

    fn store<'a>(
        &'a self,
        _: &'a ResponseCacheRequest,
        value: Value,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            let value = to_python::<P>(value)?;
            self.services
                .call(|reply| CacheCall::Store { value, reply })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }
}
