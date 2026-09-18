use crate::event::{CallEvent, FailureOrigin, Timing, epoch_seconds};
use crate::host::{HostOp, HostResult};
use crate::layer::Layer;
use crate::machine::{HostFailure, Interrupted, Machine, MachineStep, Step};
use crate::route::Route;

/// Whether a completed value is a success. A router's report completes the machine even
/// when the logical call failed, so the terminal event has to ask.
pub trait CallOutcome {
    fn succeeded(&self) -> bool;
}

impl CallOutcome for () {
    fn succeeded(&self) -> bool {
        true
    }
}

/// Emits exactly one terminal event per logical call, then completes with the inner value.
///
/// The event is a host op like any other, so it reaches the adapter inline. An interrupt
/// cannot yield ops, so on that path the driver is responsible for the `Failed` event.
pub struct Terminal<M: Machine> {
    inner: M,
    start_time: f64,
    state: State<M>,
}

enum State<M: Machine> {
    Running,
    Emitting(Result<M::Complete, <M::Route as Route>::Error>),
    Done,
}

impl<M: Machine> Terminal<M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            start_time: epoch_seconds(),
            state: State::Running,
        }
    }

    fn timing(&self) -> Timing {
        Timing {
            start_time: self.start_time,
            end_time: epoch_seconds(),
        }
    }

    async fn step(
        &mut self,
        result: Option<HostResult<M::Route>>,
    ) -> Result<MachineStep<M::Route, M::Complete>, <M::Route as Route>::Error>
    where
        M::Complete: CallOutcome,
    {
        match std::mem::replace(&mut self.state, State::Done) {
            State::Running => match self.inner.resume(result).await {
                Ok(MachineStep::Host(op)) => {
                    self.state = State::Running;
                    Ok(MachineStep::Host(op))
                }
                Ok(MachineStep::Complete(complete)) => {
                    let timing = self.timing();
                    let event = if complete.succeeded() {
                        CallEvent::Succeeded { timing }
                    } else {
                        CallEvent::Failed {
                            timing,
                            origin: FailureOrigin::Call,
                        }
                    };
                    self.state = State::Emitting(Ok(complete));
                    Ok(MachineStep::Host(HostOp::Emit(event)))
                }
                Err(error) => {
                    let timing = self.timing();
                    self.state = State::Emitting(Err(error));
                    Ok(MachineStep::Host(HostOp::Emit(CallEvent::Failed {
                        timing,
                        origin: FailureOrigin::Call,
                    })))
                }
            },
            State::Emitting(outcome) => outcome.map(MachineStep::Complete),
            State::Done => unreachable!("terminal machine resumed after completion"),
        }
    }
}

impl<M> Machine for Terminal<M>
where
    M: Machine,
    M::Complete: CallOutcome,
{
    type Route = M::Route;
    type Complete = M::Complete;

    fn resume(&mut self, result: Option<HostResult<Self::Route>>) -> Step<'_, Self> {
        Box::pin(self.step(result))
    }

    fn interrupt(
        &mut self,
        failure: HostFailure<<Self::Route as Route>::Error>,
    ) -> Interrupted<'_, Self> {
        self.state = State::Done;
        self.inner.interrupt(failure)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TerminalLayer;

impl<M> Layer<M> for TerminalLayer
where
    M: Machine,
    M::Complete: CallOutcome,
{
    type Output = Terminal<M>;

    fn layer(&self, inner: M) -> Terminal<M> {
        Terminal::new(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Stack;

    struct Unit;

    impl Route for Unit {
        type Response = ();
        type Error = &'static str;
        type Op = &'static str;
        type OpResult = ();
    }

    struct Scripted {
        ops: Vec<&'static str>,
        outcome: Result<bool, &'static str>,
    }

    struct Flag(bool);

    impl CallOutcome for Flag {
        fn succeeded(&self) -> bool {
            self.0
        }
    }

    impl Machine for Scripted {
        type Route = Unit;
        type Complete = Flag;

        fn resume(&mut self, _: Option<HostResult<Unit>>) -> Step<'_, Self> {
            Box::pin(async move {
                if !self.ops.is_empty() {
                    return Ok(MachineStep::Host(HostOp::Route(self.ops.remove(0))));
                }
                self.outcome.map(|flag| MachineStep::Complete(Flag(flag)))
            })
        }

        fn interrupt(&mut self, failure: HostFailure<&'static str>) -> Interrupted<'_, Self> {
            Box::pin(async move { Err(failure.into_error()) })
        }
    }

    async fn drain(machine: &mut Terminal<Scripted>) -> (Vec<String>, Result<bool, &'static str>) {
        let mut seen = Vec::new();
        let mut result = None;
        loop {
            match machine.resume(result.take()).await {
                Ok(MachineStep::Host(HostOp::Route(op))) => {
                    seen.push(format!("route:{op}"));
                    result = Some(HostResult::Route(()));
                }
                Ok(MachineStep::Host(HostOp::Emit(event))) => {
                    seen.push(match event {
                        CallEvent::Succeeded { .. } => "succeeded".into(),
                        CallEvent::Failed { .. } => "failed".into(),
                        other => format!("{other:?}"),
                    });
                    result = Some(HostResult::Emitted);
                }
                Ok(MachineStep::Host(HostOp::BeforeSend(_))) => unreachable!(),
                Ok(MachineStep::Complete(flag)) => return (seen, Ok(flag.0)),
                Err(error) => return (seen, Err(error)),
            }
        }
    }

    #[tokio::test]
    async fn emits_one_terminal_after_forwarding_inner_ops() {
        let mut machine = Stack::new(Scripted {
            ops: vec!["project", "send"],
            outcome: Ok(true),
        })
        .layer(TerminalLayer)
        .build();
        let (seen, outcome) = drain(&mut machine).await;
        assert_eq!(seen, ["route:project", "route:send", "succeeded"]);
        assert_eq!(outcome, Ok(true));
    }

    #[tokio::test]
    async fn failed_completions_and_errors_both_emit_failed_once() {
        let mut machine = Terminal::new(Scripted {
            ops: vec![],
            outcome: Ok(false),
        });
        let (seen, outcome) = drain(&mut machine).await;
        assert_eq!(seen, ["failed"]);
        assert_eq!(outcome, Ok(false));

        let mut machine = Terminal::new(Scripted {
            ops: vec![],
            outcome: Err("boom"),
        });
        let (seen, outcome) = drain(&mut machine).await;
        assert_eq!(seen, ["failed"]);
        assert_eq!(outcome, Err("boom"));
    }
}
