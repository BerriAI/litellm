use crate::cache::Selection;
use std::{future::Future, pin::Pin, time::Duration};

use litellm_cache::Error;
use litellm_cache_response::{
    ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService, cache_key,
};
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::{Protocol, Reply},
};
use serde_json::Value;

pub(crate) enum CacheCall {
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

pub(crate) struct Cached<P>(std::marker::PhantomData<P>);

impl<P: Protocol> Protocol for Cached<P> {
    type Request = (P::Request, Selection);
    type Response = P::Response;
    type Error = P::Error;
    type HostCall = CacheCall;
    type Chunk = P::Chunk;
    type StreamHead = P::StreamHead;
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
                .map_err(|_| Error::Unavailable)?
        })
    }

    fn store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        value: Value,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
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
