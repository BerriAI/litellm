use std::{future::Future, pin::Pin, time::Duration};

use litellm_cache::Error;
use litellm_cache_response::{
    ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService, ResponseEnvelope,
};
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::{Protocol, Reply},
};
use litellm_inference::caching::CachedOutput;
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
        CachedOutput::Stream(text) => Ok(serde_json::json!({
            STREAM_EVENTS_KEY: text.split_inclusive("\n\n").collect::<Vec<_>>()
        })),
    }
}

pub(crate) enum CacheCall {
    Lookup {
        reply: Reply<Result<Option<Value>, Error>>,
    },
    Store {
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
            ..Default::default()
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
                .map(from_python)
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
            let value = to_python(value)?;
            self.services
                .call(|reply| CacheCall::Store { value, reply })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }
}
