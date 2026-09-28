use std::ops::ControlFlow;

use litellm_host::{
    hooks::RouteHooks,
    machine::{HostFailure, Machine, MachineStep},
    protocol::{HookRequest, HostRequest, Protocol, Reply, StreamDelivery},
};

use crate::services::HostCallHandler;

type ProtocolOf<M> = <M as Machine>::Protocol;
type ErrorOf<M> = <ProtocolOf<M> as Protocol>::Error;

pub enum Boundary<M: Machine> {
    Complete(M::Complete),
    Open(<ProtocolOf<M> as Protocol>::StreamHead),
    Chunk(<ProtocolOf<M> as Protocol>::Chunk),
}

/// Answers host calls and hooks inline and stops at each stream delivery, holding its demand
/// reply until the consumer advances again. Dropping the driver drops the in-flight call.
pub struct Driver<M: Machine, S, H> {
    machine: M,
    services: S,
    hooks: H,
    demand: Option<Reply<ControlFlow<()>>>,
}

impl<M, S, H> Driver<M, S, H>
where
    M: Machine,
    S: HostCallHandler<ProtocolOf<M>>,
    H: RouteHooks<ErrorOf<M>>,
{
    pub fn new(machine: M, services: S, hooks: H) -> Self {
        Self {
            machine,
            services,
            hooks,
            demand: None,
        }
    }

    pub async fn advance(&mut self) -> Result<Boundary<M>, ErrorOf<M>> {
        self.resume(ControlFlow::Continue(())).await
    }

    pub async fn detach(&mut self) -> Result<Boundary<M>, ErrorOf<M>> {
        self.resume(ControlFlow::Break(())).await
    }

    /// Interrupts the machine with a failure the consumer hit at the last stream boundary,
    /// dropping the held demand reply unanswered
    pub async fn fail(&mut self, error: ErrorOf<M>) -> Result<M::Complete, ErrorOf<M>> {
        self.demand = None;
        self.machine.interrupt(HostFailure::Error(error)).await
    }

    async fn resume(&mut self, demand: ControlFlow<()>) -> Result<Boundary<M>, ErrorOf<M>> {
        if let Some(reply) = self.demand.take() {
            reply.send(demand);
        }
        loop {
            let request = match self.machine.resume().await? {
                MachineStep::Complete(complete) => return Ok(Boundary::Complete(complete)),
                MachineStep::Suspended(request) => request,
            };
            let answered = match request {
                HostRequest::HostCall(call) => self.services.handle_host_call(call).await,
                HostRequest::Hook(HookRequest::BeforeProviderRequest {
                    wire,
                    context,
                    reply,
                }) => self
                    .hooks
                    .before_provider_request(*wire, *context)
                    .await
                    .map(|wire| reply.send(wire)),
                HostRequest::Hook(HookRequest::Event(event, reply)) => {
                    self.hooks.on_event(event).await.map(|()| reply.send(()))
                }
                HostRequest::Stream(StreamDelivery::Open(head, reply)) => {
                    self.demand = Some(reply);
                    return Ok(Boundary::Open(head));
                }
                HostRequest::Stream(StreamDelivery::Chunk(chunk, reply)) => {
                    self.demand = Some(reply);
                    return Ok(Boundary::Chunk(chunk));
                }
            };
            if let Err(error) = answered {
                return self.fail(error).await.map(Boundary::Complete);
            }
        }
    }
}
