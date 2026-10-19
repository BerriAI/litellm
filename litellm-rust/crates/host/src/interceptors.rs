use std::future::Future;

use serde::{Serialize, de::DeserializeOwned};

use crate::{
    machine::MachineFault,
    protocol::{InterceptRequest, Reply, reply},
};

use serde_json::Value;

/// The provider request as it is about to leave, offered to the host for rewriting.
#[derive(Clone, Debug, PartialEq)]
pub struct WireRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

/// What the route knows about the request it is sending, for a host that logs it. The
/// route owns these facts; a host reads them beside the wire request and never rewrites
/// them.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestContext {
    pub model: String,
    pub custom_llm_provider: String,
    /// The route's parameters before the provider transformation.
    pub optional_params: Value,
    /// Optional-param names that carry credentials and must be redacted when logged.
    pub secret_fields: Vec<String>,
    /// The credential the route resolved for the provider call.
    pub api_key: Option<litellm_auth::SecretValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawResponse {
    pub body: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderIdentity {
    pub model: String,
    pub provider: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResultSource {
    Provider,
    Cache { key: String },
}

/// What the call cost, settled once by the route. A host records it; it never re-derives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cost {
    /// The amount the provider itself billed, taken from its response.
    Reported { amount: serde_json::Number },
    /// Served from the cache, so nothing was billed.
    CacheHit,
    /// No provider figure: the host estimates from usage and its price table.
    Deferred,
}

impl Cost {
    pub fn settle(source: &ResultSource, reported: Option<serde_json::Number>) -> Self {
        match (source, reported) {
            (ResultSource::Cache { .. }, _) => Self::CacheHit,
            (ResultSource::Provider, Some(amount)) => Self::Reported { amount },
            (ResultSource::Provider, None) => Self::Deferred,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionFacts {
    pub provider: ProviderIdentity,
    pub source: ResultSource,
    pub cost: Cost,
}

/// Answers the hook points a call raises. `InterceptRequest` lists every point; an
/// implementer handles the ones it cares about and passes the rest through.
pub trait Interceptors<E>: Send + Sync {
    fn handle(&self, request: InterceptRequest) -> impl Future<Output = Result<(), E>> + Send {
        request.pass_through();
        async { Ok(()) }
    }

    /// Raises one point and waits for its answer. A host channel yields it to the host.
    fn intercept<A: Send>(
        &self,
        ask: impl FnOnce(Reply<A>) -> InterceptRequest + Send,
    ) -> impl Future<Output = Result<A, E>> + Send
    where
        E: From<MachineFault>,
    {
        let (reply, answer) = reply();
        let handled = self.handle(ask(reply));
        async move {
            handled.await?;
            answer.await.map_err(|_| MachineFault::Abandoned.into())
        }
    }
}

impl<E, T: Interceptors<E> + ?Sized> Interceptors<E> for &T {
    fn handle(&self, request: InterceptRequest) -> impl Future<Output = Result<(), E>> + Send {
        (**self).handle(request)
    }

    fn intercept<A: Send>(
        &self,
        ask: impl FnOnce(Reply<A>) -> InterceptRequest + Send,
    ) -> impl Future<Output = Result<A, E>> + Send
    where
        E: From<MachineFault>,
    {
        (**self).intercept(ask)
    }
}

impl<E> Interceptors<E> for () {}

/// The typed form of each point, for the routes that raise them.
pub trait InterceptPoints<E>: Interceptors<E>
where
    E: From<MachineFault>,
{
    fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> impl Future<Output = Result<WireRequest, E>> + Send {
        self.intercept(|reply| InterceptRequest::BeforeProviderRequest {
            wire: Box::new(wire),
            context: Box::new(context),
            reply,
        })
    }

    fn after_provider_response(
        &self,
        raw: RawResponse,
    ) -> impl Future<Output = Result<(), E>> + Send {
        self.intercept(|reply| InterceptRequest::AfterProviderResponse { raw, reply })
    }

    fn result_ready(&self, facts: ExecutionFacts) -> impl Future<Output = Result<(), E>> + Send {
        self.intercept(|reply| InterceptRequest::ResultReady { facts, reply })
    }

    /// Offers a complete provider response before it is cached or delivered, and returns the
    /// response the call keeps. Never raised for a cache hit.
    fn provider_result<R>(&self, response: R) -> impl Future<Output = Result<R, E>> + Send
    where
        R: Serialize + DeserializeOwned + Send,
    {
        let value = serde_json::to_value(&response);
        async move {
            let value = value.map_err(|_| MachineFault::UnreadableAnswer)?;
            let answer = self
                .intercept(|reply| InterceptRequest::ProviderResult {
                    response: value,
                    reply,
                })
                .await?;
            serde_json::from_value(answer).map_err(|_| MachineFault::UnreadableAnswer.into())
        }
    }
}

impl<E: From<MachineFault>, T: Interceptors<E> + ?Sized> InterceptPoints<E> for T {}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use serde_json::json;

    use super::*;
    use crate::{
        machine::{CallMachine, Machine, MachineFault, MachineStep},
        protocol::{HostRequest, Protocol},
    };

    struct Unit;

    #[derive(Clone, Debug)]
    struct Fault;

    impl Protocol for Unit {
        type Response = (WireRequest, ());
        type Error = Fault;
        type Request = ();
        type HostCall = Infallible;
        type Chunk = Infallible;
        type StreamHead = Infallible;
    }

    impl From<MachineFault> for Fault {
        fn from(_: MachineFault) -> Self {
            Fault
        }
    }

    fn wire(url: &str) -> WireRequest {
        WireRequest {
            url: url.into(),
            headers: Vec::new(),
            body: json!({}),
        }
    }

    fn context() -> RequestContext {
        RequestContext {
            model: "m".into(),
            custom_llm_provider: "p".into(),
            optional_params: json!({}),
            secret_fields: Vec::new(),
            api_key: None,
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn the_channel_yields_each_hook_as_its_op_and_returns_the_answer() {
        let mut machine = CallMachine::<Unit>::new(None, |channel| {
            Box::pin(async move {
                let sent = InterceptPoints::before_provider_request(
                    &channel.interceptors,
                    wire("prepared"),
                    context(),
                )
                .await?;
                InterceptPoints::after_provider_response(
                    &channel.interceptors,
                    RawResponse { body: "raw".into() },
                )
                .await?;
                Ok((sent, ()))
            })
        });

        let Ok(MachineStep::Suspended(HostRequest::Intercept(
            InterceptRequest::BeforeProviderRequest { wire, reply, .. },
        ))) = machine.resume().await
        else {
            panic!("before_provider_request yields BeforeSend");
        };
        assert_eq!(wire.url, "prepared");
        reply.send(WireRequest {
            url: "rewritten".into(),
            ..*wire
        });

        let Ok(MachineStep::Suspended(HostRequest::Intercept(
            InterceptRequest::AfterProviderResponse { raw, reply },
        ))) = machine.resume().await
        else {
            panic!("on_event yields Emit");
        };
        assert_eq!(raw.body, "raw");
        reply.send(());

        let Ok(MachineStep::Complete((sent, ()))) = machine.resume().await else {
            panic!("the call completes with the answers");
        };
        assert_eq!(sent.url, "rewritten");
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn handling_on_the_channel_suspends_for_the_host_and_answers_the_original_reply() {
        let mut machine = CallMachine::<Unit>::new(None, |channel| {
            Box::pin(async move {
                let (reply, answer) = reply();
                channel
                    .interceptors
                    .handle(InterceptRequest::BeforeProviderRequest {
                        wire: Box::new(wire("prepared")),
                        context: Box::new(context()),
                        reply,
                    })
                    .await?;
                Ok((answer.await.map_err(|_| Fault)?, ()))
            })
        });

        let Ok(MachineStep::Suspended(HostRequest::Intercept(
            InterceptRequest::BeforeProviderRequest { wire, reply, .. },
        ))) = machine.resume().await
        else {
            panic!("handling on the channel yields the request to the host");
        };
        assert_eq!(wire.url, "prepared");
        reply.send(WireRequest {
            url: "rewritten".into(),
            ..*wire
        });

        let Ok(MachineStep::Complete((sent, ()))) = machine.resume().await else {
            panic!("the call completes with the host's answer");
        };
        assert_eq!(sent.url, "rewritten");
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn no_hooks_pass_the_wire_request_through() {
        let sent =
            InterceptPoints::<Fault>::before_provider_request(&(), wire("prepared"), context())
                .await
                .unwrap();
        assert_eq!(sent.url, "prepared");
    }
}
