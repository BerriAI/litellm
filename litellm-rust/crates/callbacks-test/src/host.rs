use litellm_callbacks::host::{HostOp, HostResult};
use litellm_callbacks::machine::{HostFailure, Machine, MachineStep};
use litellm_callbacks::route::{Layered, LayeredOp, LayeredResult, Route};

use crate::route::TestRoute;
use crate::trace::{Observed, Trace, event_name};

/// How the host answers a route op: the answer string, or fail the op with this error.
pub enum Answer<E> {
    Value(String),
    Fail(E),
}

/// The host side of one route: answers its ops and owns the trace. [`drain`] handles
/// `BeforeSend` and `Emit` itself, so a host only decides route ops.
pub trait Answers<R: Route> {
    fn trace(&mut self) -> &mut Trace;

    fn route(&mut self, op: R::Op) -> Result<R::OpResult, R::Error>;
}

/// Answers route ops through a closure and records everything, so a test controls the
/// host side of the conversation as precisely as the script controls the machine side.
pub struct RecordingHost<E, F = fn(&str) -> Answer<E>> {
    answer: F,
    pub trace: Trace,
    _error: std::marker::PhantomData<E>,
}

impl<E> RecordingHost<E> {
    /// Echoes every route op back as its own answer.
    pub fn echo() -> Self {
        RecordingHost::with(|op| Answer::Value(op.to_string()))
    }
}

impl<E, F: FnMut(&str) -> Answer<E>> RecordingHost<E, F> {
    pub fn with(answer: F) -> Self {
        Self {
            answer,
            trace: Trace::default(),
            _error: std::marker::PhantomData,
        }
    }
}

impl<E, F> Answers<TestRoute<E>> for RecordingHost<E, F>
where
    E: Clone + Send + Sync + 'static,
    F: FnMut(&str) -> Answer<E>,
{
    fn trace(&mut self) -> &mut Trace {
        &mut self.trace
    }

    fn route(&mut self, op: String) -> Result<String, E> {
        self.trace.0.push(Observed::Route(op.clone()));
        match (self.answer)(&op) {
            Answer::Value(value) => Ok(value),
            Answer::Fail(error) => Err(error),
        }
    }
}

/// Answers a [`Layered`] route: the outer half through `outer`, which sees the shared
/// trace so it can record what it was asked, the inner half through another host.
pub struct WithOuter<F, H> {
    pub outer: F,
    pub inner: H,
}

impl<O, I, F, H> Answers<Layered<O, I>> for WithOuter<F, H>
where
    O: Route,
    I: Route,
    F: FnMut(&mut Trace, O::Op) -> Result<O::OpResult, I::Error>,
    H: Answers<I>,
{
    fn trace(&mut self) -> &mut Trace {
        self.inner.trace()
    }

    fn route(
        &mut self,
        op: LayeredOp<O::Op, I::Op>,
    ) -> Result<LayeredResult<O::OpResult, I::OpResult>, I::Error> {
        match op {
            LayeredOp::Outer(op) => (self.outer)(self.inner.trace(), op).map(LayeredResult::Outer),
            LayeredOp::Inner(op) => self.inner.route(op).map(LayeredResult::Inner),
        }
    }
}

/// Drives a machine to completion against the host, recording the trace. Like a language
/// driver, it interrupts the machine when the host fails an op and emits the one terminal
/// event itself once the machine is done.
pub async fn drain<M, H>(
    machine: &mut M,
    host: &mut H,
) -> Result<M::Complete, <M::Route as Route>::Error>
where
    M: Machine,
    H: Answers<M::Route>,
{
    let outcome = drive(machine, host).await;
    let terminal = match &outcome {
        Ok(complete) if M::succeeded(complete) => "succeeded",
        Ok(_) | Err(_) => "failed",
    };
    host.trace().0.push(Observed::Emit(terminal));
    outcome
}

async fn drive<M, H>(
    machine: &mut M,
    host: &mut H,
) -> Result<M::Complete, <M::Route as Route>::Error>
where
    M: Machine,
    H: Answers<M::Route>,
{
    let mut result = None;
    loop {
        let op = match machine.resume(result.take()).await {
            Ok(MachineStep::Complete(complete)) => return Ok(complete),
            Ok(MachineStep::Host(op)) => op,
            Err(error) => return Err(error),
        };
        result = Some(match op {
            HostOp::Route(op) => match host.route(op) {
                Ok(answer) => HostResult::Route(answer),
                Err(error) => return machine.interrupt(HostFailure::Error(error)).await,
            },
            HostOp::BeforeSend { wire, .. } => {
                host.trace().0.push(Observed::BeforeSend);
                HostResult::BeforeSend(wire)
            }
            HostOp::Emit(event) => {
                host.trace().0.push(Observed::Emit(event_name(&event)));
                HostResult::Emitted
            }
        });
    }
}
