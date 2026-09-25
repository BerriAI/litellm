use crate::event::{CallEvent, FailureOrigin, Timing, epoch_seconds};
use crate::host::{Host, HostOp};
use crate::machine::{HostFailure, Machine, MachineStep};
use crate::protocol::Protocol;

/// Drives a machine to completion against an in-process host and emits exactly one
/// terminal event.
pub async fn run<M, H>(
    mut machine: M,
    host: &H,
) -> Result<M::Complete, <M::Protocol as Protocol>::Error>
where
    M: Machine,
    <M::Protocol as Protocol>::Error: From<crate::MachineFault>,
    H: Host<M::Protocol>,
{
    let start_time = epoch_seconds();
    let _ = host.emit(&CallEvent::Started { start_time }).await;
    let outcome = loop {
        let op = match machine.resume().await {
            Ok(MachineStep::Complete(complete)) => break Ok(complete),
            Ok(MachineStep::Host(op)) => op,
            Err(error) => break Err(error),
        };
        if let Err(error) = perform(host, op).await {
            break machine.interrupt(HostFailure::Error(error)).await;
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

async fn perform<R: Protocol<Error: From<crate::MachineFault>>, H: Host<R>>(
    host: &H,
    op: HostOp<R>,
) -> Result<(), R::Error> {
    match op {
        HostOp::Project(reply) => host
            .project()
            .await
            .map(|projection| reply.send(projection)),
        HostOp::Custom(op) => host.custom_op(op).await,
        HostOp::PreRequest { request, reply } => host
            .pre_request(*request)
            .await
            .map(|params| reply.send(params)),
        HostOp::BeforeSend {
            wire,
            context,
            reply,
        } => host
            .before_send(*wire, &context)
            .await
            .map(|wire| reply.send(wire)),
        HostOp::Emit(event, reply) => host
            .emit(&CallEvent::Machine(event))
            .await
            .map(|()| reply.send(())),
        HostOp::AfterResponse { response, reply } => host
            .after_response(*response)
            .await
            .map(|verdict| reply.send(verdict)),
        HostOp::Open(head, reply) => host.open(head).await.map(|demand| reply.send(demand)),
        HostOp::Deliver(chunk, reply) => host.deliver(chunk).await.map(|demand| reply.send(demand)),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;
    use crate::event::PublicRequest;
    use crate::host::{Reply, Verdict};
    use crate::{MachineFault, machine::CallMachine};

    struct Unit;

    impl Protocol for Unit {
        type Response = ();
        type Error = &'static str;
        type Projection = ();
        type Op = (&'static str, Reply<()>);
        type Chunk = ();
        type StreamHead = ();
    }

    impl From<MachineFault> for &'static str {
        fn from(_: MachineFault) -> Self {
            "machine fault"
        }
    }

    #[derive(Default)]
    struct Recording {
        seen: Mutex<Vec<String>>,
        fail: Option<&'static str>,
    }

    impl Host<Unit> for Recording {
        async fn project(&self) -> Result<(), &'static str> {
            self.seen.lock().unwrap().push("project".into());
            Ok(())
        }

        async fn custom_op(
            &self,
            (op, reply): (&'static str, Reply<()>),
        ) -> Result<(), &'static str> {
            self.seen.lock().unwrap().push(format!("op:{op}"));
            if self.fail == Some(op) {
                return Err("host failed");
            }
            reply.send(());
            Ok(())
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

    fn scripted(
        ops: &'static [&'static str],
        outcome: Result<(), &'static str>,
    ) -> CallMachine<Unit> {
        CallMachine::new(move |host| {
            Box::pin(async move {
                host.project().await?;
                for op in ops {
                    host.custom_op(|reply| (*op, reply)).await?;
                }
                outcome
            })
        })
    }

    #[tokio::test]
    async fn a_host_without_stream_support_refuses_instead_of_discarding_output() {
        let host = Recording::default();
        assert_eq!(host.open(()).await, Err("machine fault"));
        assert_eq!(host.deliver(()).await, Err("machine fault"));
    }

    #[tokio::test]
    async fn forwards_every_op_then_emits_one_succeeded() {
        let host = Recording::default();
        let outcome = run(scripted(&["sign", "send"], Ok(())), &host).await;
        assert_eq!(outcome, Ok(()));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["started", "project", "op:sign", "op:send", "succeeded"]
        );
    }

    #[tokio::test]
    async fn errors_and_host_failures_each_emit_failed_once() {
        let host = Recording::default();
        let outcome = run(scripted(&[], Err("boom")), &host).await;
        assert_eq!(outcome, Err("boom"));
        assert_eq!(*host.seen.lock().unwrap(), ["started", "project", "failed"]);

        let host = Recording {
            fail: Some("send"),
            ..Recording::default()
        };
        let outcome = run(scripted(&["sign", "send", "never"], Ok(())), &host).await;
        assert_eq!(outcome, Err("host failed"));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["started", "project", "op:sign", "op:send", "failed"]
        );
    }

    #[tokio::test]
    async fn hook_ops_default_to_handing_back_what_the_machine_offered() {
        let host = Recording::default();
        let machine = CallMachine::<Unit>::new(|host| {
            Box::pin(async move {
                host.project().await?;
                let request = PublicRequest {
                    model: "model".into(),
                    custom_llm_provider: "provider".into(),
                    messages: json!([]),
                    params: [("tools".to_string(), json!(["a"]))].into_iter().collect(),
                    fields: &["tools"],
                };
                if host.pre_request(request.clone()).await? != request.params {
                    return Err("pre_request altered the params");
                }
                match host.after_response(()).await? {
                    Verdict::Return(()) => Ok(()),
                    Verdict::Resend(_) => Err("after_response asked to resend"),
                }
            })
        });
        assert_eq!(run(machine, &host).await, Ok(()));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["started", "project", "succeeded"]
        );
    }

    struct StartTimes(Mutex<Vec<f64>>);

    impl Host<Unit> for StartTimes {
        async fn project(&self) -> Result<(), &'static str> {
            Ok(())
        }

        async fn custom_op(
            &self,
            (_, reply): (&'static str, Reply<()>),
        ) -> Result<(), &'static str> {
            reply.send(());
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
        assert_eq!(run(scripted(&["send"], Ok(())), &host).await, Ok(()));
        let times = host.0.lock().unwrap();
        assert_eq!(times.len(), 2);
        assert_eq!(times[0], times[1]);
    }
}
