use std::{future::Future, marker::PhantomData, pin::Pin, sync::Arc, time::Duration};

use litellm_cache::Error;
use litellm_cache_response::{
    CacheHit, CacheOptions, CachePolicy, CacheScope, PreparedCache, ResponseCacheCall,
    ResponseCacheService, ResponseEnvelope, ScopedCache,
};
use litellm_host::{
    machine::HostServices,
    protocol::{Protocol, Reply},
};
use litellm_inference::caching::{Cachable, CachedOutput};
use serde_json::Value;

const STREAM_EVENTS_KEY: &str = "litellm_cached_anthropic_sse_events";

pub(crate) struct Cached<P>(PhantomData<P>);

impl<P: Protocol> Protocol for Cached<P> {
    type Request = (P::Request, Option<CachePolicy>);
    type Response = P::Response;
    type Error = P::Error;
    type HostCall = CacheCall;
    type Chunk = P::Chunk;
    type StreamHead = P::StreamHead;
}

pub(crate) enum CacheCall {
    Lookup {
        reply: Reply<Result<Option<CacheHit>, Error>>,
    },
    Store {
        value: Value,
        reply: Reply<Result<(), Error>>,
    },
}

struct PythonCacheService<P: Cachable> {
    services: HostServices<Cached<P>>,
    protocol: PhantomData<fn() -> P>,
}

pub(crate) fn scoped<P: Cachable>(services: HostServices<Cached<P>>) -> ScopedCache {
    ScopedCache::new(
        Arc::new(PythonCacheService {
            services,
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
    fn prepare(
        self: Arc<Self>,
        options: CacheOptions,
        _: &str,
        _: &dyn Fn() -> Value,
    ) -> PreparedCache {
        PreparedCache {
            policy: options.policy,
            max_entry_bytes: usize::MAX,
            call: self,
        }
    }
}

impl<P: Cachable> ResponseCacheCall for PythonCacheService<P> {
    fn lookup<'a>(
        &'a self,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<Option<CacheHit>, Error>> + Send + 'a>> {
        Box::pin(async move {
            self.services
                .call(|reply| CacheCall::Lookup { reply })
                .await
                .map_err(|_| Error::Unavailable)??
                .map(|hit| {
                    Ok(CacheHit {
                        value: from_python::<P>(hit.value)?,
                        key: hit.key,
                    })
                })
                .transpose()
        })
    }

    fn store<'a>(
        &'a self,
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

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use litellm_host::{
        interceptors::ProviderIdentity,
        machine::{CallMachine, Machine, MachineStep},
        protocol::{HostRequest, Protocol},
    };
    use litellm_inference::{
        RouteError,
        caching::{CacheRequest, execute_unary},
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    struct TestRoute;

    impl Protocol for TestRoute {
        type Request = ();
        type Response = Value;
        type Error = RouteError;
        type HostCall = Infallible;
        type Chunk = ();
        type StreamHead = ();
    }

    impl Cachable for TestRoute {
        const SURFACE: &'static str = "test";
    }

    #[rstest]
    #[case::hit(true)]
    #[case::miss(false)]
    #[tokio::test]
    async fn python_cache_calls_never_materialize_native_identity(#[case] hit: bool) {
        let mut machine = CallMachine::<Cached<TestRoute>>::new(None, |host| {
            Box::pin(async move {
                let cache = scoped::<TestRoute>(host.services);
                let request = CacheRequest::prepare::<TestRoute>(
                    ProviderIdentity {
                        model: "test".into(),
                        provider: "test".into(),
                    },
                    Some(cache.service.clone()),
                    Some(cache.options(None)),
                    &|_| panic!("Python caching must not construct native identity"),
                );
                execute_unary::<TestRoute, _, _>(request, &(), None, || async {
                    Ok(json!({"source": "provider"}))
                })
                .await
            })
        });
        let MachineStep::Suspended(HostRequest::HostCall(CacheCall::Lookup { reply })) =
            machine.resume().await.unwrap()
        else {
            panic!("expected Python lookup");
        };
        reply.send(Ok(hit.then(|| CacheHit {
            value: json!({"source": "cache"}),
            key: "python-owned-key".into(),
        })));
        if !hit {
            let MachineStep::Suspended(HostRequest::HostCall(CacheCall::Store { value, reply })) =
                machine.resume().await.unwrap()
            else {
                panic!("expected Python store on a miss");
            };
            assert_eq!(value, json!({"source": "provider"}));
            reply.send(Ok(()));
        }
        let MachineStep::Complete(response) = machine.resume().await.unwrap() else {
            panic!("expected completed call");
        };
        assert_eq!(
            response,
            json!({"source": if hit { "cache" } else { "provider" }})
        );
    }

    #[rstest]
    #[tokio::test]
    async fn python_hit_keeps_the_host_key() {
        let mut machine = CallMachine::<Cached<TestRoute>, CacheHit>::new(None, |host| {
            Box::pin(async move {
                let cache = scoped::<TestRoute>(host.services);
                let options = cache.options(None);
                let prepared = cache.service.prepare(options, TestRoute::SURFACE, &|| {
                    panic!("Python caching must not construct native identity")
                });
                Ok(prepared.call.lookup(Duration::ZERO).await.unwrap().unwrap())
            })
        });
        let MachineStep::Suspended(HostRequest::HostCall(CacheCall::Lookup { reply })) =
            machine.resume().await.unwrap()
        else {
            panic!("expected Python lookup");
        };
        reply.send(Ok(Some(CacheHit {
            value: json!({"answer": 42}),
            key: "python-owned-key".into(),
        })));
        let MachineStep::Complete(hit) = machine.resume().await.unwrap() else {
            panic!("expected completed lookup");
        };
        assert_eq!(hit.key, "python-owned-key");
        assert_eq!(
            from_python::<TestRoute>(json!({"answer": 42})).unwrap(),
            hit.value
        );
    }
}
