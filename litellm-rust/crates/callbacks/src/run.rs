use crate::event::{CallEvent, FailureOrigin, Timing, epoch_seconds};
use crate::host::{Host, HostOp, HostResult};
use crate::machine::{HostFailure, Machine, MachineStep};
use crate::route::Route;

/// Drives a machine to completion against an in-process host and emits exactly one
/// terminal event.
pub async fn run<M, H>(mut machine: M, host: &H) -> Result<M::Complete, <M::Route as Route>::Error>
where
    M: Machine,
    H: Host<M::Route>,
{
    let start_time = epoch_seconds();
    let mut result = None;
    let outcome = loop {
        let step = match machine.resume(result.take()).await {
            Ok(MachineStep::Complete(complete)) => break Ok(complete),
            Ok(MachineStep::Host(op)) => op,
            Err(error) => break Err(error),
        };
        let answer = match step {
            HostOp::Route(op) => host.route(op).await.map(HostResult::Route),
            HostOp::BeforeSend(wire) => host
                .before_send(*wire)
                .await
                .map(|wire| HostResult::BeforeSend(Box::new(wire))),
            HostOp::Emit(event) => host.emit(&event).await.map(|()| HostResult::Emitted),
        };
        match answer {
            Ok(answer) => result = Some(answer),
            Err(error) => break machine.interrupt(HostFailure::Error(error)).await,
        }
    };
    let timing = Timing {
        start_time,
        end_time: epoch_seconds(),
    };
    let terminal = match &outcome {
        Ok(_) => CallEvent::Succeeded { timing },
        Err(_) => CallEvent::Failed {
            timing,
            origin: FailureOrigin::Call,
        },
    };
    let _ = host.emit(&terminal).await;
    outcome
}
