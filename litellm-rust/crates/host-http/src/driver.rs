use std::{convert::Infallible, sync::Arc};

use axum::{
    body::Body,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use litellm_host::{
    call::{CallOutput, HostedCompletion, HostedMachine, observe_call, observe_unary},
    hooks::RouteHooks,
    host::{Demand, HostOp, Reply},
    machine::{Machine, MachineFault, MachineStep},
    protocol::Protocol,
};

use crate::{Error, StreamAdapter};

type StepOf<P> = MachineStep<P, HostedCompletion<<P as Protocol>::Response>>;
type Output<E> = CallOutput<Response, http::Response<()>, Bytes, E>;

pub async fn serve_unary<P, H, R>(
    machine: HostedMachine<P>,
    request: P::Projection,
    hooks: H,
    response: impl FnOnce(P::Response) -> R,
) -> Result<Response, Error<P::Error>>
where
    P: Protocol<Op = Infallible, Chunk = Infallible, StreamHead = Infallible>,
    P::Error: From<MachineFault>,
    H: RouteHooks<P::Error>,
    R: IntoResponse,
{
    let observer = hooks.observer();
    let mut driver = Driver::new(machine, request, hooks);
    observe_unary(observer, async move {
        match driver.advance().await? {
            MachineStep::Complete(HostedCompletion::Complete(value)) => {
                Ok(response(value).into_response())
            }
            _ => Err(Error::Protocol),
        }
    })
    .await
}

pub async fn serve<P, A, H, R>(
    machine: HostedMachine<P>,
    request: P::Projection,
    hooks: H,
    response: impl FnOnce(P::Response) -> R,
    adapter: A,
) -> Result<Response, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamAdapter<Protocol = P>,
    H: RouteHooks<P::Error> + 'static,
    R: IntoResponse,
{
    let adapter = Arc::new(adapter);
    let observer = hooks.observer();
    let driver = Driver::new(machine, request, hooks);
    match observe_call(observer, driver.start(adapter.clone(), response)).await? {
        CallOutput::Complete(response) => Ok(response),
        CallOutput::Stream { head, chunks } => {
            let body = chunks.map(move |chunk| {
                Ok::<_, Infallible>(chunk.unwrap_or_else(|error| adapter.stream_error(error)))
            });
            Ok(head.map(|()| Body::from_stream(body)))
        }
    }
}

struct Driver<P: Protocol, H> {
    machine: HostedMachine<P>,
    request: Option<P::Projection>,
    hooks: H,
    demand: Option<Reply<Demand>>,
}

impl<P, H> Driver<P, H>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    H: RouteHooks<P::Error>,
{
    fn new(machine: HostedMachine<P>, request: P::Projection, hooks: H) -> Self {
        Self {
            machine,
            request: Some(request),
            hooks,
            demand: None,
        }
    }

    async fn advance(&mut self) -> Result<StepOf<P>, Error<P::Error>> {
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
                boundary => return Ok(boundary),
            }
        }
    }

    async fn advance_stream(
        &mut self,
        adapter: &impl StreamAdapter<Protocol = P>,
    ) -> Result<StepOf<P>, Error<P::Error>> {
        loop {
            match self.advance().await? {
                MachineStep::Host(HostOp::Custom(op)) => {
                    adapter.custom_op(op).await.map_err(Error::Call)?
                }
                boundary => return Ok(boundary),
            }
        }
    }

    async fn start<A, R>(
        mut self,
        adapter: Arc<A>,
        response: impl FnOnce(P::Response) -> R,
    ) -> Result<Output<Error<P::Error>>, Error<P::Error>>
    where
        A: StreamAdapter<Protocol = P>,
        H: 'static,
        R: IntoResponse,
    {
        match self.advance_stream(adapter.as_ref()).await? {
            MachineStep::Complete(HostedCompletion::Complete(value)) => {
                Ok(CallOutput::Complete(response(value).into_response()))
            }
            MachineStep::Host(HostOp::Open(head, reply)) => {
                let head = adapter.head(head).map_err(Error::Call)?;
                self.demand = Some(reply);
                let chunks =
                    stream::try_unfold((self, adapter), |(mut driver, adapter)| async move {
                        match driver.advance_stream(adapter.as_ref()).await? {
                            MachineStep::Host(HostOp::Deliver(chunk, reply)) => {
                                let bytes = adapter.chunk(chunk).map_err(Error::Call)?;
                                driver.demand = Some(reply);
                                Ok(Some((bytes, (driver, adapter))))
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
