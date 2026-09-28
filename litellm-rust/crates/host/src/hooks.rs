use std::future::Future;

use crate::event::{MachineEvent, RequestContext, WireRequest};

/// What a route reaches for mid-call: the send-time rewrite and the events it reports.
/// Python's `logging_obj.pre_call` and `post_call`, in that order.
pub trait RouteHooks<E>: Send + Sync {
    fn observer(&self) -> Option<std::sync::Arc<dyn crate::lifecycle::CallObserver>> {
        None
    }

    fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> impl Future<Output = Result<WireRequest, E>> + Send;

    fn on_event(&self, event: MachineEvent) -> impl Future<Output = Result<(), E>> + Send;
}

impl<E, T: RouteHooks<E> + ?Sized> RouteHooks<E> for &T {
    fn observer(&self) -> Option<std::sync::Arc<dyn crate::lifecycle::CallObserver>> {
        (**self).observer()
    }

    fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> impl Future<Output = Result<WireRequest, E>> + Send {
        (**self).before_provider_request(wire, context)
    }

    fn on_event(&self, event: MachineEvent) -> impl Future<Output = Result<(), E>> + Send {
        (**self).on_event(event)
    }
}

/// No host: the wire request goes out as prepared and nothing observes the call.
impl<E> RouteHooks<E> for () {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, E> {
        Ok(wire)
    }

    async fn on_event(&self, _: MachineEvent) -> Result<(), E> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use serde_json::json;

    use super::*;
    use crate::protocol::HookRequest;
    use crate::{
        event::RawResponse,
        machine::MachineFault,
        machine::{CallMachine, Machine, MachineStep},
        protocol::HostRequest,
        protocol::Protocol,
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
        let mut machine = CallMachine::<Unit>::new(|channel| {
            Box::pin(async move {
                let sent = RouteHooks::before_provider_request(
                    &channel.hooks,
                    wire("prepared"),
                    context(),
                )
                .await?;
                RouteHooks::on_event(
                    &channel.hooks,
                    MachineEvent::ResponseReceived {
                        raw: RawResponse { body: "raw".into() },
                    },
                )
                .await?;
                Ok((sent, ()))
            })
        });

        let Ok(MachineStep::Suspended(HostRequest::Hook(HookRequest::BeforeProviderRequest {
            wire,
            reply,
            ..
        }))) = machine.resume().await
        else {
            panic!("before_provider_request yields BeforeSend");
        };
        assert_eq!(wire.url, "prepared");
        reply.send(WireRequest {
            url: "rewritten".into(),
            ..*wire
        });

        let Ok(MachineStep::Suspended(HostRequest::Hook(HookRequest::Event(event, reply)))) =
            machine.resume().await
        else {
            panic!("on_event yields Emit");
        };
        assert!(matches!(event, MachineEvent::ResponseReceived { .. }));
        reply.send(());

        let Ok(MachineStep::Complete((sent, ()))) = machine.resume().await else {
            panic!("the call completes with the answers");
        };
        assert_eq!(sent.url, "rewritten");
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn no_hooks_pass_the_wire_request_through() {
        let sent = RouteHooks::<Fault>::before_provider_request(&(), wire("prepared"), context())
            .await
            .unwrap();
        assert_eq!(sent.url, "prepared");
    }
}
