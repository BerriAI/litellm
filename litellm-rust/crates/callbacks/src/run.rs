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
        let answer = match machine.resume(result.take()).await {
            Ok(MachineStep::Complete(complete)) => break Ok(complete),
            Ok(MachineStep::Yield(chunk)) => {
                host.consume(chunk).await.map(|()| HostResult::Consumed)
            }
            Ok(MachineStep::Host(HostOp::Route(op))) => host.route(op).await.map(HostResult::Route),
            Ok(MachineStep::Host(HostOp::BeforeSend { wire, context })) => host
                .before_send(*wire, &context)
                .await
                .map(|wire| HostResult::BeforeSend(Box::new(wire))),
            Ok(MachineStep::Host(HostOp::Emit(event))) => {
                host.emit(&event).await.map(|()| HostResult::Emitted)
            }
            Ok(MachineStep::Host(HostOp::AttemptFailed {
                attempt,
                class,
                error,
            })) => host
                .attempt_failed(&attempt, class, &error)
                .await
                .map(|()| HostResult::Recorded),
            Err(error) => break Err(error),
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
        Ok(complete) if M::succeeded(complete) => CallEvent::Succeeded { timing },
        Ok(_) | Err(_) => CallEvent::Failed {
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
        type Chunk = &'static str;
    }

    struct Flag(bool);

    struct Scripted {
        ops: Vec<&'static str>,
        chunks: Vec<&'static str>,
        outcome: Result<bool, &'static str>,
    }

    impl Machine for Scripted {
        type Route = Unit;
        type Complete = Flag;

        fn resume(&mut self, _: Option<HostResult<Unit>>) -> Step<'_, Self> {
            Box::pin(async move {
                if !self.ops.is_empty() {
                    return Ok(MachineStep::Host(HostOp::Route(self.ops.remove(0))));
                }
                if !self.chunks.is_empty() {
                    return Ok(MachineStep::Yield(self.chunks.remove(0)));
                }
                self.outcome.map(|flag| MachineStep::Complete(Flag(flag)))
            })
        }

        fn interrupt(&mut self, failure: HostFailure<&'static str>) -> Interrupted<'_, Self> {
            Box::pin(async move { Err(failure.into_error()) })
        }

        fn succeeded(complete: &Flag) -> bool {
            complete.0
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
                CallEvent::Succeeded { .. } => "succeeded".into(),
                CallEvent::Failed { .. } => "failed".into(),
                other => format!("{other:?}"),
            });
            Ok(())
        }

        async fn consume(&self, chunk: &'static str) -> Result<(), &'static str> {
            self.seen.lock().unwrap().push(format!("chunk:{chunk}"));
            Ok(())
        }
    }

    fn scripted(ops: &[&'static str], outcome: Result<bool, &'static str>) -> Scripted {
        Scripted {
            ops: ops.to_vec(),
            chunks: Vec::new(),
            outcome,
        }
    }

    #[tokio::test]
    async fn forwards_every_op_then_emits_one_succeeded() {
        let host = Recording::default();
        let outcome = run(scripted(&["project", "send"], Ok(true)), &host).await;
        assert!(outcome.is_ok_and(|flag| flag.0));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["route:project", "route:send", "succeeded"]
        );
    }

    #[tokio::test]
    async fn chunks_reach_the_consumer_in_order_before_the_terminal() {
        let host = Recording::default();
        let machine = Scripted {
            ops: vec!["send"],
            chunks: vec!["a", "b"],
            outcome: Ok(true),
        };
        let outcome = run(machine, &host).await;
        assert!(outcome.is_ok_and(|flag| flag.0));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["route:send", "chunk:a", "chunk:b", "succeeded"]
        );
    }

    #[tokio::test]
    async fn failed_completions_errors_and_host_failures_each_emit_failed_once() {
        let host = Recording::default();
        let outcome = run(scripted(&[], Ok(false)), &host).await;
        assert!(outcome.is_ok_and(|flag| !flag.0));
        assert_eq!(*host.seen.lock().unwrap(), ["failed"]);

        let host = Recording::default();
        let outcome = run(scripted(&[], Err("boom")), &host).await;
        assert_eq!(outcome.map(|_| ()), Err("boom"));
        assert_eq!(*host.seen.lock().unwrap(), ["failed"]);

        let host = Recording {
            fail: Some("send"),
            ..Recording::default()
        };
        let outcome = run(scripted(&["project", "send", "never"], Ok(true)), &host).await;
        assert_eq!(outcome.map(|_| ()), Err("host failed"));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["route:project", "route:send", "failed"]
        );
    }
}
