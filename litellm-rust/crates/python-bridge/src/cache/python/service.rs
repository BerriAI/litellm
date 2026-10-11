use std::{future::Future, pin::Pin, time::Duration};

use litellm_cache::Error;
use litellm_cache_response::{
    CacheKeyBuilder, CacheKeyInput, ResponseCacheConfig, ResponseCacheRequest,
    ResponseCacheService, ResponseEnvelope, cache_key,
};
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::{Protocol, Reply},
};
use litellm_inference::caching::CachedOutput;
use serde_json::Value;

const STREAM_EVENTS_KEY: &str = "litellm_cached_anthropic_sse_events";

fn from_python(surface: &str, value: Value) -> Result<Value, Error> {
    let output = match value.get(STREAM_EVENTS_KEY) {
        Some(events) => {
            let events: Vec<String> =
                serde_json::from_value(events.clone()).map_err(|_| Error::InvalidEntry)?;
            CachedOutput::Stream(events.concat())
        }
        None => CachedOutput::Response(value),
    };
    serde_json::to_value(ResponseEnvelope::new(surface, output)).map_err(|_| Error::InvalidEntry)
}

fn to_python(surface: &str, value: Value) -> Result<Value, Error> {
    let envelope: ResponseEnvelope<CachedOutput<Value>> =
        serde_json::from_value(value).map_err(|_| Error::InvalidEntry)?;
    match envelope.decode(surface).ok_or(Error::InvalidEntry)? {
        CachedOutput::Response(response) => Ok(response),
        CachedOutput::Stream(text) => Ok(serde_json::json!({
            STREAM_EVENTS_KEY: text.split_inclusive("\n\n").collect::<Vec<_>>()
        })),
    }
}

pub(crate) enum CacheCall {
    BuildKey {
        reply: Reply<Result<Option<String>, Error>>,
    },
    Lookup {
        key: String,
        reply: Reply<Result<Option<Value>, Error>>,
    },
    Store {
        key: String,
        value: Value,
        reply: Reply<Result<(), Error>>,
    },
}

struct PythonCacheService<P: Protocol> {
    services: HostServices<P>,
    config: ResponseCacheConfig,
    surface: &'static str,
}

pub(in crate::cache) fn service<P: Protocol<HostCall = CacheCall>>(
    services: HostServices<P>,
    namespace: String,
    surface: &'static str,
) -> std::sync::Arc<dyn ResponseCacheService>
where
    P::Error: From<MachineFault>,
{
    std::sync::Arc::new(PythonCacheService {
        services,
        config: ResponseCacheConfig {
            namespace,
            ..Default::default()
        },
        surface,
    })
}

impl<P: Protocol<HostCall = CacheCall>> CacheKeyBuilder for PythonCacheService<P>
where
    P::Error: From<MachineFault>,
{
    fn build_key<'a>(
        &'a self,
        _: &'a CacheKeyInput,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, Error>> + Send + 'a>> {
        Box::pin(async move {
            self.services
                .call(|reply| CacheCall::BuildKey { reply })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }
}

impl<P: Protocol<HostCall = CacheCall>> ResponseCacheService for PythonCacheService<P>
where
    P::Error: From<MachineFault>,
{
    fn config(&self) -> &ResponseCacheConfig {
        &self.config
    }

    fn lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, Error>> + Send + 'a>> {
        Box::pin(async move {
            self.services
                .call(|reply| CacheCall::Lookup {
                    key: cache_key(&request.key),
                    reply,
                })
                .await
                .map_err(|_| Error::Unavailable)??
                .map(|value| from_python(self.surface, value))
                .transpose()
        })
    }

    fn store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        value: Value,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            let value = to_python(self.surface, value)?;
            self.services
                .call(|reply| CacheCall::Store {
                    key: cache_key(&request.key),
                    value,
                    reply,
                })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }
}
