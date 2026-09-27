use std::{convert::Infallible, sync::Arc};

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use litellm_host::{
    call::{CallOutput, HostedCompletion, HostedMachine, observe_call},
    hooks::RouteHooks,
    host::{Demand, HostOp, Reply},
    machine::{Machine, MachineFault, MachineStep},
    protocol::Protocol,
};

use crate::{Error, HttpAdapter};

type ProtocolOf<A> = <A as HttpAdapter>::Protocol;
type ErrorOf<A> = <ProtocolOf<A> as Protocol>::Error;
type StepOf<A> =
    MachineStep<ProtocolOf<A>, HostedCompletion<<ProtocolOf<A> as Protocol>::Response>>;
type Output<E> = CallOutput<Response, http::Response<()>, Bytes, E>;

pub async fn serve<A, H>(
    machine: HostedMachine<A::Protocol>,
    request: <A::Protocol as Protocol>::Projection,
    adapter: A,
    hooks: H,
) -> Result<Response, Error<ErrorOf<A>>>
where
    A: HttpAdapter,
    ErrorOf<A>: From<MachineFault>,
    H: RouteHooks<ErrorOf<A>> + 'static,
{
    let adapter = Arc::new(adapter);
    let observer = hooks.observer();
    let driver = Driver {
        machine,
        request: Some(request),
        adapter: adapter.clone(),
        hooks,
        demand: None,
    };
    match observe_call(observer, driver.start()).await? {
        CallOutput::Complete(response) => Ok(response),
        CallOutput::Stream { head, chunks } => {
            let body = chunks.map(move |chunk| {
                Ok::<_, Infallible>(chunk.unwrap_or_else(|error| adapter.stream_error(error)))
            });
            Ok(head.map(|()| Body::from_stream(body)))
        }
    }
}

struct Driver<A: HttpAdapter, H> {
    machine: HostedMachine<A::Protocol>,
    request: Option<<A::Protocol as Protocol>::Projection>,
    adapter: Arc<A>,
    hooks: H,
    demand: Option<Reply<Demand>>,
}

impl<A, H> Driver<A, H>
where
    A: HttpAdapter,
    ErrorOf<A>: From<MachineFault>,
    H: RouteHooks<ErrorOf<A>> + 'static,
{
    async fn advance(&mut self) -> Result<StepOf<A>, Error<ErrorOf<A>>> {
        if let Some(reply) = self.demand.take() {
            reply.send(Demand::More);
        }
        loop {
            match self.machine.resume().await.map_err(Error::Call)? {
                MachineStep::Host(HostOp::Project(reply)) => {
                    reply.send(self.request.take().ok_or(Error::Protocol)?);
                }
                MachineStep::Host(HostOp::BeforeSend {
                    wire,
                    context,
                    reply,
                }) => reply.send(
                    self.hooks
                        .before_send(*wire, *context)
                        .await
                        .map_err(Error::Call)?,
                ),
                MachineStep::Host(HostOp::Emit(event, reply)) => {
                    self.hooks.emit(event).await.map_err(Error::Call)?;
                    reply.send(());
                }
                MachineStep::Host(HostOp::Custom(op)) => {
                    self.adapter.custom_op(op).await.map_err(Error::Call)?
                }
                boundary => return Ok(boundary),
            }
        }
    }

    async fn start(mut self) -> Result<Output<Error<ErrorOf<A>>>, Error<ErrorOf<A>>> {
        match self.advance().await? {
            MachineStep::Complete(HostedCompletion::Complete(response)) => self
                .adapter
                .complete(response)
                .map(CallOutput::Complete)
                .map_err(Error::Call),
            MachineStep::Host(HostOp::Open(head, reply)) => {
                let head = self.adapter.head(head).map_err(Error::Call)?;
                self.demand = Some(reply);
                let chunks = stream::try_unfold(self, |mut driver| async move {
                    match driver.advance().await? {
                        MachineStep::Host(HostOp::Deliver(chunk, reply)) => {
                            let bytes = driver.adapter.chunk(chunk).map_err(Error::Call)?;
                            driver.demand = Some(reply);
                            Ok(Some((bytes, driver)))
                        }
                        MachineStep::Complete(HostedCompletion::StreamEnded) => Ok(None),
                        _ => Err(Error::Protocol),
                    }
                })
                .boxed();
                Ok(CallOutput::Stream { head, chunks })
            }
            _ => Err(Error::Protocol),
        }
    }
}
