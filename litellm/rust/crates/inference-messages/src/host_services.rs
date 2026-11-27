//! The Messages route's host calls, and the route services a host answers through them.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use bytes::Bytes;
use futures_util::{StreamExt, future::BoxFuture};
use litellm_cache_response::{
    CacheScope, ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService, ScopedCache,
};
use litellm_host::{machine::HostServices, protocol::Reply};
use serde_json::Value;

use crate::{
    Delivered, Error, HostAnswer, MessagesBody, MessagesCall, MessagesRoute, PreRequestHooks,
    Safeguards, Transport, TransportRequest, TransportResponse, route::Messages,
};

/// The parts of a Messages call its host serves, each through a [`MessagesHostCall`].
#[derive(Clone, Debug, Default)]
pub struct MessagesHostServices {
    pub transport: bool,
    pub pre_request_hooks: bool,
    pub safeguards: bool,
    pub answer: bool,
    pub cache_namespace: Option<String>,
}

/// An operation the Messages route asks its host to perform.
pub enum MessagesHostCall {
    Answer { stream: bool, reply: Reply<()> },
    Intercept { stream: bool, reply: Reply<bool> },
    Cache(CacheCall),
    PreRequest(Reply<Result<MessagesCall, Error>>),
    Safeguards(SafeguardsCall),
    Transport(TransportCall),
}

pub enum CacheCall {
    Lookup {
        reply: Reply<Result<Option<Value>, litellm_cache::Error>>,
    },
    Store {
        value: Value,
        reply: Reply<Result<(), litellm_cache::Error>>,
    },
}

pub enum SafeguardsCall {
    Results {
        response: MessagesBody,
        reply: Reply<Option<Value>>,
    },
    OpenStream {
        reply: Reply<bool>,
    },
    Frames {
        chunk: Bytes,
        reply: Reply<Vec<Bytes>>,
    },
    Flush {
        reply: Reply<Vec<Bytes>>,
    },
}

pub enum TransportCall {
    Send {
        request: TransportRequest,
        reply: Reply<Result<Sent, Error>>,
    },
    Read {
        reply: Reply<Result<Option<Bytes>, Error>>,
    },
}

/// What the host's client received for a sent request. A streamed body is read with
/// [`TransportCall::Read`].
pub struct Sent {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: SentBody,
}

pub enum SentBody {
    Text(String),
    Stream,
    Rejected { text: String, error: Error },
}

impl MessagesHostServices {
    pub(crate) fn serve(
        self,
        route: MessagesRoute,
        services: HostServices<Messages>,
    ) -> MessagesRoute {
        let calls = Arc::new(HostCalls(services.clone()));
        let cache = self.cache_namespace.map(|namespace| {
            ScopedCache::new(
                Arc::new(HostCache {
                    services,
                    config: ResponseCacheConfig {
                        namespace,
                        ..Default::default()
                    },
                }),
                CacheScope::Shared,
            )
        });
        MessagesRoute {
            transport: self
                .transport
                .then(|| Arc::clone(&calls) as Arc<dyn Transport>)
                .or(route.transport),
            pre_request_hooks: self
                .pre_request_hooks
                .then(|| Arc::clone(&calls) as Arc<dyn PreRequestHooks>)
                .or(route.pre_request_hooks),
            safeguards: self
                .safeguards
                .then(|| Arc::clone(&calls) as Arc<dyn Safeguards>)
                .or(route.safeguards),
            host_answer: self
                .answer
                .then_some(calls as Arc<dyn HostAnswer>)
                .or(route.host_answer),
            cache: cache.or(route.cache),
            ..route
        }
    }
}

struct HostCalls(HostServices<Messages>);

impl Transport for HostCalls {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'static, Result<TransportResponse, Error>> {
        let services = self.0.clone();
        Box::pin(async move {
            let sent = services
                .call(|reply| MessagesHostCall::Transport(TransportCall::Send { request, reply }))
                .await??;
            let delivered = match sent.body {
                SentBody::Text(text) => Delivered::Complete(text),
                SentBody::Rejected { text, error } => Delivered::Rejected { text, error },
                SentBody::Stream => Delivered::Stream(
                    futures_util::stream::try_unfold(services, |services| async move {
                        let chunk = services
                            .call(|reply| {
                                MessagesHostCall::Transport(TransportCall::Read { reply })
                            })
                            .await??;
                        Ok(chunk.map(|chunk| (chunk, services)))
                    })
                    .boxed(),
                ),
            };
            Ok(TransportResponse {
                status: sent.status,
                headers: sent.headers,
                delivered,
            })
        })
    }
}

impl PreRequestHooks for HostCalls {
    fn run(&self, call: MessagesCall) -> BoxFuture<'_, Result<MessagesCall, Error>> {
        drop(call);
        Box::pin(async move {
            self.0
                .call(MessagesHostCall::PreRequest)
                .await
                .and_then(|hooked| hooked)
        })
    }
}

impl HostCalls {
    async fn safeguards<A: Send>(
        &self,
        call: impl FnOnce(Reply<A>) -> SafeguardsCall + Send,
    ) -> Result<A, Error> {
        self.0
            .call(|reply| MessagesHostCall::Safeguards(call(reply)))
            .await
    }
}

impl Safeguards for HostCalls {
    fn results<'a>(
        &'a self,
        response: &'a MessagesBody,
    ) -> BoxFuture<'a, Result<Option<Value>, Error>> {
        let response = response.clone();
        Box::pin(self.safeguards(|reply| SafeguardsCall::Results { response, reply }))
    }

    fn open_stream(&self) -> BoxFuture<'_, Result<bool, Error>> {
        Box::pin(self.safeguards(|reply| SafeguardsCall::OpenStream { reply }))
    }

    fn frames(&self, chunk: Bytes) -> BoxFuture<'_, Result<Vec<Bytes>, Error>> {
        Box::pin(self.safeguards(|reply| SafeguardsCall::Frames { chunk, reply }))
    }

    fn flush(&self) -> BoxFuture<'_, Result<Vec<Bytes>, Error>> {
        Box::pin(self.safeguards(|reply| SafeguardsCall::Flush { reply }))
    }
}

impl HostAnswer for HostCalls {
    fn answer(&self, stream: bool) -> BoxFuture<'_, Result<(), Error>> {
        Box::pin(
            self.0
                .call(move |reply| MessagesHostCall::Answer { stream, reply }),
        )
    }

    fn intercept(&self, stream: bool) -> BoxFuture<'_, Result<bool, Error>> {
        Box::pin(
            self.0
                .call(move |reply| MessagesHostCall::Intercept { stream, reply }),
        )
    }
}

struct HostCache {
    services: HostServices<Messages>,
    config: ResponseCacheConfig,
}

impl ResponseCacheService for HostCache {
    fn config(&self) -> &ResponseCacheConfig {
        &self.config
    }

    fn keyed_by_caller(&self) -> bool {
        true
    }

    fn lookup<'a>(
        &'a self,
        _: &'a ResponseCacheRequest,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, litellm_cache::Error>> + Send + 'a>>
    {
        Box::pin(async move {
            self.services
                .call(|reply| MessagesHostCall::Cache(CacheCall::Lookup { reply }))
                .await
                .map_err(litellm_cache::Error::unavailable)?
        })
    }

    fn store<'a>(
        &'a self,
        _: &'a ResponseCacheRequest,
        value: Value,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), litellm_cache::Error>> + Send + 'a>> {
        Box::pin(async move {
            self.services
                .call(|reply| MessagesHostCall::Cache(CacheCall::Store { value, reply }))
                .await
                .map_err(litellm_cache::Error::unavailable)?
        })
    }
}
