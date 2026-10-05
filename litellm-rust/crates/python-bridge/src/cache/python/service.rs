use std::{future::Future, pin::Pin, time::Duration};

use litellm_cache::Error;
use litellm_cache_response::{
    ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService, ResponseEnvelope,
};
use litellm_core::{
    caching::{Cachable, CachedOutput},
    messages::route::Messages,
};
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::{Protocol, Reply},
};
use litellm_inference::caching::CachedOutput;
use serde_json::Value;

const STREAM_EVENTS_KEY: &str = "litellm_cached_anthropic_sse_events";

fn from_python(value: Value, surface: &'static str) -> Result<Value, Error> {
    let output = match value
        .get(STREAM_EVENTS_KEY)
        .filter(|_| surface == Messages::SURFACE)
    {
        Some(events) => {
            let events: Vec<String> =
                serde_json::from_value(events.clone()).map_err(|_| Error::InvalidEntry)?;
            CachedOutput::Stream(events.concat())
        }
        None => CachedOutput::Response(value),
    };
    serde_json::to_value(ResponseEnvelope::new(surface, output)).map_err(|_| Error::InvalidEntry)
}

fn to_python(value: Value, surface: &'static str) -> Result<Value, Error> {
    let envelope: ResponseEnvelope<CachedOutput<Value>> =
        serde_json::from_value(value).map_err(|_| Error::InvalidEntry)?;
    match envelope.decode(surface).ok_or(Error::InvalidEntry)? {
        CachedOutput::Response(response) => Ok(response),
        CachedOutput::Stream(text) if surface == Messages::SURFACE => {
            let response = serde_json::Map::from_iter([(
                STREAM_EVENTS_KEY.to_owned(),
                Value::Array(
                    litellm_framer::sse::text_blocks(&text)
                        .map(|block| Value::String(block.to_owned()))
                        .collect(),
                ),
            )]);
            Ok(Value::Object(response))
        }
        CachedOutput::Stream(_) => Err(Error::InvalidEntry),
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
    surface: &'static str,
}

pub(in crate::cache) fn service<P: Protocol<HostCall = CacheCall>>(
    services: HostServices<P>,
    surface: &'static str,
    config: ResponseCacheConfig,
) -> std::sync::Arc<dyn ResponseCacheService>
where
    P::Error: From<MachineFault>,
{
    std::sync::Arc::new(PythonCacheService {
        services,
        config,
        surface,
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
        request: &'a ResponseCacheRequest,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + 'a>> {
        if request.key.fields.iter().any(|field| {
            field.name == "wire_changes" && field.internal_parameter && field.value.is_some()
        }) {
            return Box::pin(async { Err(Error::UnsupportedOperation) });
        }
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
                .map(|value| from_python(value, self.surface))
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
            let value = to_python(value, self.surface)?;
            self.services
                .call(|reply| CacheCall::Store { key, value, reply })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }
}
