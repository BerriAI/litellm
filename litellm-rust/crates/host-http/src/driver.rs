use std::{convert::Infallible, sync::Arc};

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::{StreamExt, stream};

use litellm_host::{
    call::{CallOutput, HostedCompletion, HostedMachine},
    hooks::RouteHooks,
    lifecycle::{observe_call, observe_unary},
    machine::{Machine, MachineFault, MachineStep},
    protocol::{Demand, HookRequest, Protocol, Reply, StreamDelivery, Suspension},
    services::HostCallHandler,
};

use crate::{Error, ResponseEncoder, StreamEncoder};

type StepOf<P> = MachineStep<P, HostedCompletion<<P as Protocol>::Response>>;
type Output<E> = CallOutput<Response, http::Response<()>, Bytes, E>;

pub async fn serve_unary<P, A, H, S>(
    machine: HostedMachine<P>,
    services: S,
    hooks: H,
    encoder: A,
) -> Result<Response, Error<P::Error>>
where
    P: Protocol<Chunk = Infallible, StreamHead = Infallible>,
    P::Error: From<MachineFault>,
    H: RouteHooks<P::Error>,
    S: HostCallHandler<P>,
    A: ResponseEncoder<Protocol = P>,
{
    let observer = hooks.observer();
    let mut driver = Driver::new(machine, services, hooks);
    observe_unary(observer, async move {
        match driver.advance().await? {
            MachineStep::Complete(HostedCompletion::Complete(value)) => {
                encoder.encode_response(value).map_err(Error::Call)
            }
            _ => Err(Error::Protocol),
        }
    })
    .await
}

pub async fn serve<P, A, H, S>(
    machine: HostedMachine<P>,
    services: S,
    hooks: H,
    encoder: A,
) -> Result<Response, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamEncoder<Protocol = P>,
    H: RouteHooks<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
{
    let encoder = Arc::new(encoder);
    let observer = hooks.observer();
    let driver = Driver::new(machine, services, hooks);
    match observe_call(observer, driver.start(encoder.clone())).await? {
        CallOutput::Complete(response) => Ok(response),
        CallOutput::Stream { head, chunks } => {
            let body = chunks.map(move |chunk| {
                Ok::<_, Infallible>(
                    chunk.unwrap_or_else(|error| encoder.encode_stream_error(error)),
                )
            });
            Ok(head.map(|()| Body::from_stream(body)))
        }
    }
}

struct Driver<P: Protocol, H, S> {
    machine: HostedMachine<P>,
    services: S,
    hooks: H,
    demand: Option<Reply<Demand>>,
}

impl<P, H, S> Driver<P, H, S>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    H: RouteHooks<P::Error>,
    S: HostCallHandler<P>,
{
    fn new(machine: HostedMachine<P>, services: S, hooks: H) -> Self {
        Self {
            machine,
            services,
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
                MachineStep::Suspended(Suspension::Hook(HookRequest::BeforeProviderRequest {
                    wire,
                    context,
                    reply,
                })) => reply.send(
                    self.hooks
                        .before_provider_request(*wire, *context)
                        .await
                        .map_err(Error::Call)?,
                ),
                MachineStep::Suspended(Suspension::Hook(HookRequest::Event(event, reply))) => {
                    self.hooks.on_event(event).await.map_err(Error::Call)?;
                    reply.send(());
                }
                MachineStep::Suspended(Suspension::HostCall(op)) => {
                    self.services
                        .handle_host_call(op)
                        .await
                        .map_err(Error::Call)?;
                }
                boundary => return Ok(boundary),
            }
        }
    }

    async fn start<A>(mut self, encoder: Arc<A>) -> Result<Output<Error<P::Error>>, Error<P::Error>>
    where
        A: StreamEncoder<Protocol = P>,
        H: 'static,
        S: 'static,
    {
        match self.advance().await? {
            MachineStep::Complete(HostedCompletion::Complete(value)) => encoder
                .encode_response(value)
                .map(CallOutput::Complete)
                .map_err(Error::Call),
            MachineStep::Suspended(Suspension::Stream(StreamDelivery::Open(head, reply))) => {
                let head = encoder.encode_stream_head(head).map_err(Error::Call)?;
                self.demand = Some(reply);
                let chunks =
                    stream::try_unfold((self, encoder), |(mut driver, encoder)| async move {
                        match driver.advance().await? {
                            MachineStep::Suspended(Suspension::Stream(StreamDelivery::Chunk(
                                chunk,
                                reply,
                            ))) => {
                                let bytes = encoder.encode_chunk(chunk).map_err(Error::Call)?;
                                driver.demand = Some(reply);
                                Ok(Some((bytes, (driver, encoder))))
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
