use std::{future::Future, ops::ControlFlow};

use litellm_host::{
    call::{HostedCompletion, HostedMachine},
    error::HookError,
    hooks::NativeHooks,
    lifecycle::{CallEvent, FailureOrigin, Timing, epoch_seconds},
    machine::{Machine, MachineFault},
    protocol::Protocol,
};

use crate::{
    driver::{Boundary, Driver},
    services::HostCallHandler,
};

pub trait StreamConsumer<P: Protocol>: Send + Sync {
    fn open_stream(
        &self,
        head: P::StreamHead,
    ) -> impl Future<Output = Result<ControlFlow<()>, P::Error>> + Send;
    fn send_chunk(
        &self,
        chunk: P::Chunk,
    ) -> impl Future<Output = Result<ControlFlow<()>, P::Error>> + Send;
}

impl<P: Protocol> StreamConsumer<P> for () {
    async fn open_stream(&self, _: P::StreamHead) -> Result<ControlFlow<()>, P::Error> {
        Ok(ControlFlow::Continue(()))
    }
    async fn send_chunk(&self, _: P::Chunk) -> Result<ControlFlow<()>, P::Error> {
        Ok(ControlFlow::Continue(()))
    }
}

pub struct Host<'a, S, H, C> {
    pub services: &'a S,
    pub hooks: &'a H,
    pub stream: &'a C,
}

pub async fn run<M, S, H, C>(
    machine: M,
    host: Host<'_, S, H, C>,
) -> Result<M::Complete, <M::Protocol as Protocol>::Error>
where
    M: Machine,
    <M::Protocol as Protocol>::Error: From<HookError>,
    S: HostCallHandler<M::Protocol>,
    H: NativeHooks,
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
    P::Error: From<MachineFault> + From<HookError>,
    S: HostCallHandler<P>,
    H: NativeHooks,
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
    <M::Protocol as Protocol>::Error: From<HookError>,
    S: HostCallHandler<M::Protocol>,
    H: NativeHooks,
    C: StreamConsumer<M::Protocol>,
{
    let start_time = epoch_seconds();
    host.hooks.on_event(&CallEvent::Started { start_time });
    let outcome = consume(Driver::new(machine, host.services, host.hooks), host.stream).await;
    let timing = Timing {
        start_time,
        end_time: epoch_seconds(),
    };
    let terminal = match &outcome {
        Ok(completion) if detached(completion) => CallEvent::Cancelled { timing },
        Ok(_) => CallEvent::Succeeded {
            timing,
            response: (),
        },
        Err(_) => CallEvent::Failed {
            timing,
            origin: FailureOrigin::Call,
            error: (),
        },
    };
    host.hooks.on_event(&terminal);
    outcome
}

async fn consume<M, S, H, C>(
    mut driver: Driver<M, &S, &H>,
    stream: &C,
) -> Result<M::Complete, <M::Protocol as Protocol>::Error>
where
    M: Machine,
    <M::Protocol as Protocol>::Error: From<HookError>,
    S: HostCallHandler<M::Protocol>,
    H: NativeHooks,
    C: StreamConsumer<M::Protocol>,
{
    let mut demand = ControlFlow::Continue(());
    loop {
        let boundary = match demand {
            ControlFlow::Continue(()) => driver.advance().await?,
            ControlFlow::Break(()) => driver.detach().await?,
        };
        let delivered = match boundary {
            Boundary::Complete(complete) => return Ok(complete),
            Boundary::Open(head) => stream.open_stream(head).await,
            Boundary::Chunk(chunk) => stream.send_chunk(chunk).await,
        };
        demand = match delivered {
            Ok(demand) => demand,
            Err(error) => return driver.fail(error).await,
        };
    }
}
