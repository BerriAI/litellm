use std::future::Future;

use litellm_host::{
    call::{HostedCompletion, HostedMachine},
    event::{CallEvent, FailureOrigin, Timing, epoch_seconds},
    hooks::RouteHooks,
    lifecycle::CallObserver,
    machine::{Machine, MachineFault},
    protocol::{Demand, Protocol},
    services::HostCallHandler,
};

use crate::driver::{Boundary, Driver};

pub trait StreamConsumer<P: Protocol>: Send + Sync {
    fn open_stream(
        &self,
        head: P::StreamHead,
    ) -> impl Future<Output = Result<Demand, P::Error>> + Send;
    fn send_chunk(&self, chunk: P::Chunk) -> impl Future<Output = Result<Demand, P::Error>> + Send;
}

impl<P: Protocol> StreamConsumer<P> for () {
    async fn open_stream(&self, _: P::StreamHead) -> Result<Demand, P::Error> {
        Ok(Demand::More)
    }
    async fn send_chunk(&self, _: P::Chunk) -> Result<Demand, P::Error> {
        Ok(Demand::More)
    }
}

pub struct Host<'a, S, H, C> {
    pub services: &'a S,
    pub hooks: &'a H,
    pub stream: &'a C,
    pub observer: Option<&'a dyn CallObserver>,
}

pub async fn run<M, S, H, C>(
    machine: M,
    host: Host<'_, S, H, C>,
) -> Result<M::Complete, <M::Protocol as Protocol>::Error>
where
    M: Machine,
    S: HostCallHandler<M::Protocol>,
    H: RouteHooks<<M::Protocol as Protocol>::Error>,
    C: StreamConsumer<M::Protocol>,
{
    run_with_completion(machine, host, |_| false).await
}

pub async fn run_hosted<P, S, H, C>(
    machine: HostedMachine<P>,
    host: Host<'_, S, H, C>,
) -> Result<HostedCompletion<P::Response>, P::Error>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    S: HostCallHandler<P>,
    H: RouteHooks<P::Error>,
    C: StreamConsumer<P>,
{
    run_with_completion(machine, host, |completion| {
        matches!(completion, HostedCompletion::Detached)
    })
    .await
}

async fn run_with_completion<M, S, H, C>(
    machine: M,
    host: Host<'_, S, H, C>,
    detached: impl Fn(&M::Complete) -> bool,
) -> Result<M::Complete, <M::Protocol as Protocol>::Error>
where
    M: Machine,
    S: HostCallHandler<M::Protocol>,
    H: RouteHooks<<M::Protocol as Protocol>::Error>,
    C: StreamConsumer<M::Protocol>,
{
    let start_time = epoch_seconds();
    if let Some(observer) = host.observer {
        observer.observe(CallEvent::Started { start_time });
    }
    let outcome = consume(Driver::new(machine, host.services, host.hooks), host.stream).await;
    let timing = Timing {
        start_time,
        end_time: epoch_seconds(),
    };
    let terminal = match &outcome {
        Ok(completion) if detached(completion) => CallEvent::Cancelled { timing },
        Ok(_) => CallEvent::Succeeded { timing },
        Err(_) => CallEvent::Failed {
            timing,
            origin: FailureOrigin::Call,
        },
    };
    if let Some(observer) = host.observer {
        observer.observe(terminal);
    }
    outcome
}

async fn consume<M, S, H, C>(
    mut driver: Driver<M, &S, &H>,
    stream: &C,
) -> Result<M::Complete, <M::Protocol as Protocol>::Error>
where
    M: Machine,
    S: HostCallHandler<M::Protocol>,
    H: RouteHooks<<M::Protocol as Protocol>::Error>,
    C: StreamConsumer<M::Protocol>,
{
    let mut demand = Demand::More;
    loop {
        let boundary = match demand {
            Demand::More => driver.advance().await?,
            Demand::Detached => driver.detach().await?,
        };
        demand = match boundary {
            Boundary::Complete(complete) => return Ok(complete),
            Boundary::Open(head) => stream.open_stream(head).await?,
            Boundary::Chunk(chunk) => stream.send_chunk(chunk).await?,
        };
    }
}
