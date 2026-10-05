use std::{future::Future, pin::Pin, time::Duration};

use bytes::Bytes;
use litellm_cache::Error;
use litellm_cache_response::{
    DEFAULT_MAX_ENTRY_BYTES, ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService,
    ResponseEnvelope,
};
use litellm_core::caching::CachedOutput;
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::{Protocol, Reply},
};
use serde_json::Value;

const STREAM_EVENTS_KEY: &str = "litellm_cached_anthropic_sse_events";

fn from_python(value: Value) -> Result<Value, Error> {
    let output = match value.get(STREAM_EVENTS_KEY) {
        Some(events) => {
            let events: Vec<String> =
                serde_json::from_value(events.clone()).map_err(|_| Error::InvalidEntry)?;
            CachedOutput::Stream(events.concat())
        }
        None => CachedOutput::Response(value),
    };
    serde_json::to_value(ResponseEnvelope::new("messages", output)).map_err(|_| Error::InvalidEntry)
}

fn to_python(value: Value) -> Result<Value, Error> {
    let envelope: ResponseEnvelope<CachedOutput<Value>> =
        serde_json::from_value(value).map_err(|_| Error::InvalidEntry)?;
    match envelope.decode("messages").ok_or(Error::InvalidEntry)? {
        CachedOutput::Response(response) => Ok(response),
        CachedOutput::Stream(text) => {
            let events: Result<Vec<String>, Error> =
                litellm_framing::sse::split_raw_blocks(Bytes::from(text))
                    .map(|block| String::from_utf8(block.to_vec()).map_err(|_| Error::InvalidEntry))
                    .collect();
            Ok(serde_json::json!({ STREAM_EVENTS_KEY: events? }))
        }
    }
}

pub(crate) enum CacheCall {
    ResolveKey {
        reply: Reply<Result<String, Error>>,
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
}

pub(in crate::cache) fn service<P: Protocol<HostCall = CacheCall>>(
    services: HostServices<P>,
    namespace: String,
) -> std::sync::Arc<dyn ResponseCacheService>
where
    P::Error: From<MachineFault>,
{
    std::sync::Arc::new(PythonCacheService {
        services,
        config: ResponseCacheConfig {
            namespace,
            max_entry_bytes: DEFAULT_MAX_ENTRY_BYTES,
        },
    })
}

impl<P: Protocol<HostCall = CacheCall>> ResponseCacheService for PythonCacheService<P>
where
    P::Error: From<MachineFault>,
{
    fn config(&self) -> &ResponseCacheConfig {
        &self.config
    }

    fn resolve_key<'a>(
        &'a self,
        _: &'a ResponseCacheRequest,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + 'a>> {
        Box::pin(async move {
            self.services
                .call(|reply| CacheCall::ResolveKey { reply })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }

    fn lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, Error>> + Send + 'a>> {
        let Some(key) = request.key.preset.clone() else {
            return Box::pin(async { Err(Error::Unavailable) });
        };
        Box::pin(async move {
            self.services
                .call(|reply| CacheCall::Lookup { key, reply })
                .await
                .map_err(|_| Error::Unavailable)??
                .map(from_python)
                .transpose()
        })
    }

    fn store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        value: Value,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        let Some(key) = request.key.preset.clone() else {
            return Box::pin(async { Err(Error::Unavailable) });
        };
        Box::pin(async move {
            let value = to_python(value)?;
            self.services
                .call(|reply| CacheCall::Store { key, value, reply })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }
}
