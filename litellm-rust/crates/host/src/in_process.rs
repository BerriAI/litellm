use crate::{
    event::{CallEvent, FailureOrigin, Timing, epoch_seconds},
    hooks::RouteHooks,
    lifecycle::CallObserver,
    machine::{HostFailure, Machine, MachineStep},
    protocol::{Demand, HookRequest, Protocol, StreamDelivery, Suspension},
    services::HostCallHandler,
};
use std::future::Future;

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
    machine: crate::call::HostedMachine<P>,
    host: Host<'_, S, H, C>,
) -> Result<crate::call::HostedCompletion<P::Response>, P::Error>
where
    P: Protocol,
    P::Error: From<crate::machine::MachineFault>,
    S: HostCallHandler<P>,
    H: RouteHooks<P::Error>,
    C: StreamConsumer<P>,
{
    run_with_completion(machine, host, |completion| {
        matches!(completion, crate::call::HostedCompletion::Detached)
    })
    .await
}

async fn run_with_completion<M, S, H, C>(
    mut machine: M,
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
    let outcome = loop {
        let suspension = match machine.resume().await {
            Ok(MachineStep::Complete(complete)) => break Ok(complete),
            Ok(MachineStep::Suspended(suspension)) => suspension,
            Err(error) => break Err(error),
        };
        let result = match suspension {
            Suspension::HostCall(call) => host.services.handle_host_call(call).await,
            Suspension::Hook(HookRequest::BeforeProviderRequest {
                wire,
                context,
                reply,
            }) => host
                .hooks
                .before_provider_request(*wire, *context)
                .await
                .map(|wire| reply.send(wire)),
            Suspension::Hook(HookRequest::Event(event, reply)) => {
                host.hooks.on_event(event).await.map(|()| reply.send(()))
            }
            Suspension::Stream(StreamDelivery::Open(head, reply)) => host
                .stream
                .open_stream(head)
                .await
                .map(|demand| reply.send(demand)),
            Suspension::Stream(StreamDelivery::Chunk(chunk, reply)) => host
                .stream
                .send_chunk(chunk)
                .await
                .map(|demand| reply.send(demand)),
        };
        if let Err(error) = result {
            break machine.interrupt(HostFailure::Error(error)).await;
        }
    };
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::machine::{CallMachine, MachineFault};
    use crate::protocol::Reply;

    struct Unit;

    impl Protocol for Unit {
        type Response = ();
        type Error = &'static str;
        type Request = ();
        type HostCall = (&'static str, Reply<()>);
        type Chunk = std::convert::Infallible;
        type StreamHead = std::convert::Infallible;
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

    impl Recording {
        pub fn runtime(&self) -> crate::in_process::Host<'_, Self, Self, ()> {
            crate::in_process::Host {
                services: self,
                hooks: self,
                stream: &(),
                observer: Some(self),
            }
        }
    }
    impl crate::services::HostCallHandler<Unit> for Recording {
        async fn handle_host_call(
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
    }

    impl crate::lifecycle::CallObserver for Recording {
        fn observe(&self, event: crate::event::CallEvent) {
            self.seen.lock().unwrap().push(match event {
                CallEvent::Started { .. } => "started".into(),
                CallEvent::Succeeded { .. } => "succeeded".into(),
                CallEvent::Failed { .. } => "failed".into(),
                other => format!("{other:?}"),
            });
        }
    }
    impl crate::hooks::RouteHooks<<Unit as crate::protocol::Protocol>::Error> for Recording {
        async fn before_provider_request(
            &self,
            wire: crate::event::WireRequest,
            _: crate::event::RequestContext,
        ) -> Result<crate::event::WireRequest, <Unit as crate::protocol::Protocol>::Error> {
            Ok(wire)
        }
        async fn on_event(
            &self,
            event: crate::event::MachineEvent,
        ) -> Result<(), <Unit as crate::protocol::Protocol>::Error> {
            crate::lifecycle::CallObserver::observe(self, crate::event::CallEvent::Machine(event));
            Ok(())
        }
    }

    fn scripted(
        ops: &'static [&'static str],
        outcome: Result<(), &'static str>,
    ) -> CallMachine<Unit> {
        CallMachine::new(move |host| {
            Box::pin(async move {
                for op in ops {
                    host.services.call(|reply| (*op, reply)).await?;
                }
                outcome
            })
        })
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn forwards_every_op_then_emits_one_succeeded() {
        let host = Recording::default();
        let outcome = run(scripted(&["sign", "send"], Ok(())), host.runtime()).await;
        assert_eq!(outcome, Ok(()));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["started", "op:sign", "op:send", "succeeded"]
        );
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn errors_and_host_failures_each_emit_failed_once() {
        let host = Recording::default();
        let outcome = run(scripted(&[], Err("boom")), host.runtime()).await;
        assert_eq!(outcome, Err("boom"));
        assert_eq!(*host.seen.lock().unwrap(), ["started", "failed"]);

        let host = Recording {
            fail: Some("send"),
            ..Recording::default()
        };
        let outcome = run(scripted(&["sign", "send", "never"], Ok(())), host.runtime()).await;
        assert_eq!(outcome, Err("host failed"));
        assert_eq!(
            *host.seen.lock().unwrap(),
            ["started", "op:sign", "op:send", "failed"]
        );
    }

    struct StartTimes(Mutex<Vec<f64>>);

    impl StartTimes {
        pub fn runtime(&self) -> crate::in_process::Host<'_, Self, Self, ()> {
            crate::in_process::Host {
                services: self,
                hooks: self,
                stream: &(),
                observer: Some(self),
            }
        }
    }
    impl crate::services::HostCallHandler<Unit> for StartTimes {
        async fn handle_host_call(
            &self,
            (_, reply): (&'static str, Reply<()>),
        ) -> Result<(), &'static str> {
            reply.send(());
            Ok(())
        }
    }

    impl crate::lifecycle::CallObserver for StartTimes {
        fn observe(&self, event: crate::event::CallEvent) {
            if let CallEvent::Started { start_time }
            | CallEvent::Succeeded {
                timing: Timing { start_time, .. },
            } = event
            {
                self.0.lock().unwrap().push(start_time);
            }
        }
    }
    impl crate::hooks::RouteHooks<<Unit as crate::protocol::Protocol>::Error> for StartTimes {
        async fn before_provider_request(
            &self,
            wire: crate::event::WireRequest,
            _: crate::event::RequestContext,
        ) -> Result<crate::event::WireRequest, <Unit as crate::protocol::Protocol>::Error> {
            Ok(wire)
        }
        async fn on_event(
            &self,
            event: crate::event::MachineEvent,
        ) -> Result<(), <Unit as crate::protocol::Protocol>::Error> {
            crate::lifecycle::CallObserver::observe(self, crate::event::CallEvent::Machine(event));
            Ok(())
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn started_opens_the_call_at_the_terminal_start_time_and_cannot_fail_it() {
        let host = StartTimes(Mutex::default());
        assert_eq!(
            run(scripted(&["send"], Ok(())), host.runtime()).await,
            Ok(())
        );
        let times = host.0.lock().unwrap();
        assert_eq!(times.len(), 2);
        assert_eq!(times[0], times[1]);
    }
}
