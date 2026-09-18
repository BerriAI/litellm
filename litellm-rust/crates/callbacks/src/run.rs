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
    let _ = host.emit(&CallEvent::Started { start_time }).await;
    let mut result = None;
    let outcome = loop {
        let step = match machine.resume(result.take()).await {
            Ok(MachineStep::Complete(complete)) => break Ok(complete),
            Ok(MachineStep::Host(op)) => op,
            Err(error) => break Err(error),
        };
        let answer = match step {
            HostOp::Route(op) => host.route(op).await.map(HostResult::Route),
            HostOp::BeforeSend { wire, context } => host
                .before_send(*wire, &context)
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::machine::{Interrupted, Step};

    struct Unit;

    impl Route for Unit {
        type Response = ();
        type Error = &'static str;
        type Op = &'static str;
        type OpResult = ();
    }

    struct Scripted {
        ops: Vec<&'static str>,
        outcome: Result<(), &'static str>,
    }

    impl Machine for Scripted {
        type Route = Unit;
        type Complete = ();

        fn resume(&mut self, _: Option<HostResult<Unit>>) -> Step<'_, Self> {
            Box::pin(async move {
                if !self.ops.is_empty() {
                    return Ok(MachineStep::Host(HostOp::Route(self.ops.remove(0))));
                }
                self.outcome.map(MachineStep::Complete)
            })
        }

        fn interrupt(&mut self, failure: HostFailure<&'static str>) -> Interrupted<'_, Self> {
            Box::pin(async move { Err(failure.into_error()) })
        }
    }

    #[derive(Default)]
    struct Recording {
        seen: Mutex<Vec<String>>,
        fail: Option<&'static str>,
    }

    impl Host<Unit> for Recording {
        async fn route(&self, op: &'static str) -> Result<(), &'static str> {
            self.seen.lock().unwrap().push(format!("route:{op}"));
            match self.fail {
                Some(failing) if failing == op => Err("host failed"),
                _ => Ok(()),
            }
        }

        async fn emit(&self, event: &CallEvent) -> Result<(), &'static str> {
            self.seen.lock().unwrap().push(match event {
                CallEvent::Started { .. } => "started".into(),
                CallEvent::Succeeded { .. } => "succeeded".into(),
                CallEvent::Failed { .. } => "failed".into(),
                other => format!("{other:?}"),
            });
            Ok(())
        }
    }

    fn scripted(ops: &[&'static str], outcome: Result<(), &'static str>) -> Scripted {
        Scripted {
            ops: ops.to_vec(),
            outcome,
        }
    }

    #[tokio::test]
    async fn forwards_every_op_then_emits_one_succeeded() {
        let host = Recording::default();
        let outcome = run(scripted(&["project", "send"], Ok(())), &host).await;
        assert_eq!(outcome, Ok(()));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["started", "route:project", "route:send", "succeeded"]
        );
    }

    #[tokio::test]
    async fn errors_and_host_failures_each_emit_failed_once() {
        let host = Recording::default();
        let outcome = run(scripted(&[], Err("boom")), &host).await;
        assert_eq!(outcome, Err("boom"));
        assert_eq!(*host.seen.lock().unwrap(), ["started", "failed"]);

        let host = Recording {
            fail: Some("send"),
            ..Recording::default()
        };
        let outcome = run(scripted(&["project", "send", "never"], Ok(())), &host).await;
        assert_eq!(outcome, Err("host failed"));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["started", "route:project", "route:send", "failed"]
        );
    }

    struct StartTimes(Mutex<Vec<f64>>);

    impl Host<Unit> for StartTimes {
        async fn route(&self, _: &'static str) -> Result<(), &'static str> {
            Ok(())
        }

        async fn emit(&self, event: &CallEvent) -> Result<(), &'static str> {
            if let CallEvent::Started { start_time }
            | CallEvent::Succeeded {
                timing: Timing { start_time, .. },
            } = event
            {
                self.0.lock().unwrap().push(*start_time);
            }
            Err("observer failed")
        }
    }

    #[tokio::test]
    async fn started_opens_the_call_at_the_terminal_start_time_and_cannot_fail_it() {
        let host = StartTimes(Mutex::default());
        assert_eq!(run(scripted(&["project"], Ok(())), &host).await, Ok(()));
        let times = host.0.lock().unwrap();
        assert_eq!(times.len(), 2);
        assert_eq!(times[0], times[1]);
    }
}
