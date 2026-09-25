use std::future::Future;

use crate::{
    event::{MachineEvent, RequestContext, WireRequest},
    MachineFault,
    machine::HostChannel,
    protocol::Protocol,
};

/// What a route reaches for mid-call: the send-time rewrite and the events it reports.
/// Python's `logging_obj.pre_call` and `post_call`, in that order.
pub trait RouteHooks<E>: Send + Sync {
    fn before_send(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> impl Future<Output = Result<WireRequest, E>> + Send;

    fn emit(&self, event: MachineEvent) -> impl Future<Output = Result<(), E>> + Send;
}

/// No host: the wire request goes out as prepared and nothing observes the call.
impl<E> RouteHooks<E> for () {
    async fn before_send(&self, wire: WireRequest, _: RequestContext) -> Result<WireRequest, E> {
        Ok(wire)
    }

    async fn emit(&self, _: MachineEvent) -> Result<(), E> {
        Ok(())
    }
}

impl<R: Protocol> RouteHooks<R::Error> for HostChannel<R>
where
    R::Error: From<MachineFault>,
{
    async fn before_send(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, R::Error> {
        HostChannel::before_send(self, wire, context).await
    }

    async fn emit(&self, event: MachineEvent) -> Result<(), R::Error> {
        HostChannel::emit(self, event).await
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use serde_json::json;

    use super::*;
    use crate::{
        event::RawResponse,
        host::HostOp,
        machine::{CallMachine, Machine, MachineStep},
    };

    struct Unit;

    #[derive(Clone, Debug)]
    struct Fault;

    impl Protocol for Unit {
        type Response = (WireRequest, ());
        type Error = Fault;
        type Projection = ();
        type Op = Infallible;
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

    #[tokio::test]
    async fn the_channel_yields_each_hook_as_its_op_and_returns_the_answer() {
        let mut machine = CallMachine::<Unit>::new(|channel| {
            Box::pin(async move {
                let sent = RouteHooks::before_send(&channel, wire("prepared"), context()).await?;
                RouteHooks::emit(
                    &channel,
                    MachineEvent::ResponseReceived {
                        raw: RawResponse { body: "raw".into() },
                    },
                )
                .await?;
                Ok((sent, ()))
            })
        });

        let Ok(MachineStep::Host(HostOp::BeforeSend { wire, reply, .. })) = machine.resume().await
        else {
            panic!("before_send yields BeforeSend");
        };
        assert_eq!(wire.url, "prepared");
        reply.send(WireRequest {
            url: "rewritten".into(),
            ..*wire
        });

        let Ok(MachineStep::Host(HostOp::Emit(event, reply))) = machine.resume().await else {
            panic!("emit yields Emit");
        };
        assert!(matches!(event, MachineEvent::ResponseReceived { .. }));
        reply.send(());

        let Ok(MachineStep::Complete((sent, ()))) = machine.resume().await else {
            panic!("the call completes with the answers");
        };
        assert_eq!(sent.url, "rewritten");
    }

    #[tokio::test]
    async fn no_hooks_pass_the_wire_request_through() {
        let sent = RouteHooks::<Fault>::before_send(&(), wire("prepared"), context())
            .await
            .unwrap();
        assert_eq!(sent.url, "prepared");
    }
}
