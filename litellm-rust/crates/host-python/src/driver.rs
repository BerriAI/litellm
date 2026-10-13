use std::ops::ControlFlow;

use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use litellm_host::{
    call::HostedCompletion,
    interceptors::WireRequest,
    lifecycle::{CallEvent, ExecutionEvent, FailureOrigin, Timing, epoch_seconds},
    machine::{HostFailure, Machine, MachineStep},
    observation::ObservationSender,
    protocol::{HostRequest, InterceptRequest, Protocol, Reply, StreamDelivery},
};

use crate::PythonHostCalls;
use crate::handle::{Execution, ExecutionBody, ExecutionStep, PythonLifecycle};
use crate::hooks::{HookResume, HookStep, PythonCallEvent, PythonCallHooks};
use crate::native::{NativeMachine, NativePoll};
use crate::{InvokeError, PythonBinding, missing_state};

type ProtocolOf<H> = <H as PythonBinding>::Protocol;
type ErrorOf<H> = <ProtocolOf<H> as Protocol>::Error;
type ResponseOf<H> = <ProtocolOf<H> as Protocol>::Response;
type NativeStep<H> = MachineStep<ProtocolOf<H>, HostedCompletion<ResponseOf<H>>>;
type NativeResult<H> = Result<NativeStep<H>, ErrorOf<H>>;
type Interruption<H> = Option<HostFailure<ErrorOf<H>>>;
type StartMachine<P, M> = Box<
    dyn FnOnce(Python<'_>, &Bound<'_, PyDict>, <P as Protocol>::Request) -> PyResult<M>
        + Send
        + Sync,
>;

pub struct CallOptions {
    pub asynchronous: bool,
    pub lifecycle: PythonLifecycle,
    pub observers: Option<ObservationSender>,
}

impl CallOptions {
    pub fn new(asynchronous: bool, lifecycle: PythonLifecycle) -> Self {
        Self {
            asynchronous,
            lifecycle,
            observers: None,
        }
    }
}

enum Stage {
    Begin,
    Call,
    Streaming,
    AfterSuccess,
    Succeeded(Py<PyAny>),
    Failed(Py<PyBaseException>),
}

enum EventNext {
    Started,
    Emitted(Reply<()>),
    Terminal,
}

enum Pending<L> {
    Host,
    Native,
    Arguments(HookResume<L, Py<PyDict>>),
    Wire(HookResume<L, Box<WireRequest>>, Reply<WireRequest>),
    Response(HookResume<L, Py<PyAny>>),
    Event(HookResume<L, ()>, EventNext),
    Consumer(Reply<ControlFlow<()>>),
}

/// A route answer as the driver resumes on it: a Python exception interrupts the call as
/// raised, a native rejection resumes the machine with it.
fn answered<E>(answer: Result<(), InvokeError<E>>) -> PyResult<Result<(), E>> {
    match answer {
        Ok(()) => Ok(Ok(())),
        Err(InvokeError::Native(error)) => Ok(Err(error)),
        Err(InvokeError::Python(error)) => Err(error),
    }
}

enum Next<H: PythonBinding + PythonHostCalls<H::Protocol>> {
    Return(ExecutionStep),
    Continue(NativePoll<NativeResult<H>>),
}

struct PythonDriver<H, M, L>
where
    L: PythonCallHooks + 'static,
    H: PythonBinding + PythonHostCalls<H::Protocol>,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<ResponseOf<H>>>,
{
    binding: H,
    hooks: L,
    native: NativeMachine<M>,
    start: Option<StartMachine<ProtocolOf<H>, M>>,
    closed: bool,
    arguments: Option<Py<PyDict>>,
    started_at: f64,
    ended_at: Option<f64>,
    stage: Stage,
    pending: Option<Pending<L>>,
    interrupted: Option<Py<PyBaseException>>,
    observers: Option<ObservationSender>,
    observing: bool,
    terminal_observation: Option<CallEvent>,
}

pub fn run_call<H, M, L>(
    py: Python<'_>,
    start: impl FnOnce(
        Python<'_>,
        &Bound<'_, PyDict>,
        <H::Protocol as Protocol>::Request,
    ) -> PyResult<M>
    + Send
    + Sync
    + 'static,
    binding: H,
    hooks: L,
    arguments: Py<PyDict>,
    options: CallOptions,
) -> PyResult<Py<PyAny>>
where
    L: PythonCallHooks + 'static,
    H: PythonBinding + PythonHostCalls<H::Protocol> + 'static,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<ResponseOf<H>>>,
{
    let mut driver = PythonDriver {
        binding,
        hooks,
        native: NativeMachine::new(options.asynchronous),
        start: Some(Box::new(start)),
        closed: false,
        arguments: Some(arguments),
        started_at: 0.0,
        ended_at: None,
        stage: Stage::Begin,
        pending: None,
        interrupted: None,
        observers: options.observers,
        observing: false,
        terminal_observation: None,
    };
    if options.asynchronous {
        return Execution::new(driver, options.lifecycle)
            .into_coroutine(py)
            .map(Bound::unbind);
    }
    match driver.resume(None)? {
        ExecutionStep::Return(value) => Ok(value),
        ExecutionStep::Open(head) => Execution::suspended(driver, options.lifecycle)
            .into_sync_stream(py, head)
            .map(Bound::unbind),
        ExecutionStep::Await(_) | ExecutionStep::Yield(_) => {
            Err(PyRuntimeError::new_err("sync call suspended"))
        }
    }
}

fn is_cancellation(py: Python<'_>, error: &PyErr) -> bool {
    !error.is_instance_of::<PyException>(py)
}

impl<H, M, L> PythonDriver<H, M, L>
where
    L: PythonCallHooks + 'static,
    H: PythonBinding + PythonHostCalls<H::Protocol>,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<ResponseOf<H>>>,
{
    fn timing(&self) -> Timing {
        Timing {
            start_time: self.started_at,
            end_time: self.ended_at.unwrap_or_else(epoch_seconds),
        }
    }

    fn drive(
        &mut self,
        py: Python<'_>,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<ExecutionStep> {
        match (self.pending.take(), result) {
            (None, None) => {
                self.started_at = epoch_seconds();
                let started = PythonCallEvent::Started {
                    start_time: self.started_at,
                };
                self.observing = true;
                self.observe(&started);
                match self.hooks.on_event(py, started) {
                    Ok(step) => self.on_event(py, step, EventNext::Started),
                    Err(error) => self.hook_failed(py, error),
                }
            }
            (Some(Pending::Host), Some(result)) => {
                match self.binding.resume_host_call(py, result) {
                    Ok(Some(awaitable)) => {
                        self.pending = Some(Pending::Host);
                        Ok(ExecutionStep::Await(awaitable))
                    }
                    Ok(None) => self.resume_machine(py, None),
                    Err(InvokeError::Python(error)) => self.interrupt(py, error),
                    Err(InvokeError::Native(error)) => {
                        self.resume_machine(py, Some(HostFailure::Error(error)))
                    }
                }
            }
            (Some(Pending::Native), Some(Ok(_))) => {
                let result = self.native.take_result()?;
                self.run_steps(py, NativePoll::Ready(result))
            }
            (Some(Pending::Native), Some(Err(error))) => self.interrupt(py, error),
            (Some(Pending::Consumer(reply)), Some(read)) => {
                reply.send(if read.is_ok() {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                });
                self.resume_machine(py, None)
            }
            (Some(Pending::Arguments(resume)), Some(result)) => {
                let step = resume(&mut self.hooks, py, result);
                match step {
                    Ok(step) => self.on_arguments(py, step),
                    Err(error) => self.hook_failed(py, error),
                }
            }
            (Some(Pending::Wire(resume, reply)), Some(result)) => {
                let step = resume(&mut self.hooks, py, result);
                match step {
                    Ok(step) => self.on_wire(py, step, reply),
                    Err(error) => self.hook_failed(py, error),
                }
            }
            (Some(Pending::Response(resume)), Some(result)) => {
                let step = resume(&mut self.hooks, py, result);
                match step {
                    Ok(step) => self.on_response(py, step),
                    Err(error) => self.hook_failed(py, error),
                }
            }
            (Some(Pending::Event(resume, next)), Some(result)) => {
                let step = resume(&mut self.hooks, py, result);
                match step {
                    Ok(step) => self.on_event(py, step, next),
                    Err(error) => self.hook_failed(py, error),
                }
            }
            _ => Err(missing_state()),
        }
    }

    fn on_arguments(
        &mut self,
        py: Python<'_>,
        step: HookStep<L, Py<PyDict>>,
    ) -> PyResult<ExecutionStep> {
        match step {
            HookStep::Await(awaitable, resume) => {
                self.pending = Some(Pending::Arguments(resume));
                Ok(ExecutionStep::Await(awaitable))
            }
            HookStep::Ready(arguments) => {
                if let Err(error) = self.hooks.arguments_prepared(py, &arguments) {
                    return self.hook_failed(py, error);
                }
                let decoded = self.binding.decode_request(py, arguments.bind(py));
                self.arguments = Some(arguments);
                let request = match decoded {
                    Ok(request) => request,
                    Err(InvokeError::Native(error)) => return self.machine_failed(py, error),
                    Err(InvokeError::Python(error)) => {
                        return self.failure(py, error, FailureOrigin::Call);
                    }
                };
                let start = self.start.take().ok_or_else(missing_state)?;
                let arguments = self.arguments.as_ref().ok_or_else(missing_state)?;
                let machine = match start(py, arguments.bind(py), request) {
                    Ok(machine) => machine,
                    Err(error) => return self.failure(py, error, FailureOrigin::Host),
                };
                self.native.start(machine);
                self.stage = Stage::Call;
                self.resume_machine(py, None)
            }
        }
    }

    fn on_wire(
        &mut self,
        py: Python<'_>,
        step: HookStep<L, Box<WireRequest>>,
        reply: Reply<WireRequest>,
    ) -> PyResult<ExecutionStep> {
        match step {
            HookStep::Await(awaitable, resume) => {
                self.pending = Some(Pending::Wire(resume, reply));
                Ok(ExecutionStep::Await(awaitable))
            }
            HookStep::Ready(wire) => {
                reply.send(*wire);
                self.resume_machine(py, None)
            }
        }
    }

    fn on_response(
        &mut self,
        py: Python<'_>,
        step: HookStep<L, Py<PyAny>>,
    ) -> PyResult<ExecutionStep> {
        match step {
            HookStep::Await(awaitable, resume) => {
                self.pending = Some(Pending::Response(resume));
                Ok(ExecutionStep::Await(awaitable))
            }
            HookStep::Ready(response) => self.succeeded(py, response),
        }
    }

    fn on_event(
        &mut self,
        py: Python<'_>,
        step: HookStep<L, ()>,
        next: EventNext,
    ) -> PyResult<ExecutionStep> {
        match step {
            HookStep::Await(awaitable, resume) => {
                self.pending = Some(Pending::Event(resume, next));
                Ok(ExecutionStep::Await(awaitable))
            }
            HookStep::Ready(()) => match next {
                EventNext::Started => self.begin(py),
                EventNext::Emitted(reply) => {
                    reply.send(());
                    self.resume_machine(py, None)
                }
                EventNext::Terminal => {
                    if let Some(event) = self.terminal_observation.take()
                        && let Some(observers) = &self.observers
                    {
                        observers.emit(event);
                    }
                    self.observing = false;
                    match &self.stage {
                        Stage::Succeeded(response) => {
                            Ok(ExecutionStep::Return(response.clone_ref(py)))
                        }
                        Stage::Failed(error) => {
                            Err(PyErr::from_value(error.bind(py).clone().into_any()))
                        }
                        _ => Err(missing_state()),
                    }
                }
            },
        }
    }

    fn begin(&mut self, py: Python<'_>) -> PyResult<ExecutionStep> {
        let arguments = self.arguments.take().ok_or_else(missing_state)?;
        match self.hooks.prepare_arguments(py, arguments, self.started_at) {
            Ok(step) => self.on_arguments(py, step),
            Err(error) => self.hook_failed(py, error),
        }
    }

    fn hook_failed(&mut self, py: Python<'_>, error: PyErr) -> PyResult<ExecutionStep> {
        match self.stage {
            Stage::Begin | Stage::AfterSuccess => self.failure(py, error, FailureOrigin::Host),
            Stage::Call | Stage::Streaming => self.interrupt(py, error),
            Stage::Succeeded(_) | Stage::Failed(_) => Err(error),
        }
    }

    fn resume_machine(
        &mut self,
        py: Python<'_>,
        interruption: Interruption<H>,
    ) -> PyResult<ExecutionStep> {
        let step = self.native.resume(py, interruption)?;
        self.run_steps(py, step)
    }

    fn run_steps(
        &mut self,
        py: Python<'_>,
        mut step: NativePoll<NativeResult<H>>,
    ) -> PyResult<ExecutionStep> {
        loop {
            let result = match step {
                NativePoll::Suspend(awaitable) => {
                    self.pending = Some(Pending::Native);
                    return Ok(ExecutionStep::Await(awaitable));
                }
                NativePoll::Ready(result) => result,
            };
            step = match self.handle_native(py, result)? {
                Next::Return(step) => return Ok(step),
                Next::Continue(step) => step,
            };
        }
    }

    /// Answers one machine step: performs the op it asked for, or finishes the call.
    fn handle_native(&mut self, py: Python<'_>, result: NativeResult<H>) -> PyResult<Next<H>> {
        let op = match result {
            Ok(MachineStep::Suspended(op)) => op,
            Ok(MachineStep::Complete(response)) => {
                return self.completed(py, response).map(Next::Return);
            }
            Err(error) => return self.machine_failed(py, error).map(Next::Return),
        };
        let answered = match op {
            HostRequest::HostCall(op) => match self.binding.begin_host_call(py, op) {
                Ok(Some(awaitable)) => {
                    self.pending = Some(Pending::Host);
                    return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                }
                result => answered(result.map(|_| ())),
            },
            HostRequest::Intercept(InterceptRequest::BeforeProviderRequest {
                wire,
                context,
                reply,
            }) => match self.hooks.before_provider_request(py, wire, &context) {
                Ok(HookStep::Ready(wire)) => {
                    reply.send(*wire);
                    Ok(Ok(()))
                }
                Ok(HookStep::Await(awaitable, resume)) => {
                    self.pending = Some(Pending::Wire(resume, reply));
                    return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                }
                Err(error) => Err(error),
            },
            HostRequest::Stream(StreamDelivery::Open(head, reply)) => {
                return self.opened(py, head, reply).map(Next::Return);
            }
            HostRequest::Stream(StreamDelivery::Chunk(chunk, reply)) => {
                return self.delivered(py, chunk, reply).map(Next::Return);
            }
            HostRequest::Intercept(InterceptRequest::ResultReady { facts, reply }) => {
                let event = PythonCallEvent::Execution(ExecutionEvent::ResultReady { facts });
                self.observe(&event);
                match self.hooks.on_event(py, event) {
                    Ok(HookStep::Ready(())) => {
                        reply.send(());
                        Ok(Ok(()))
                    }
                    Ok(HookStep::Await(awaitable, resume)) => {
                        self.pending = Some(Pending::Event(resume, EventNext::Emitted(reply)));
                        return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                    }
                    Err(error) => Err(error),
                }
            }
            HostRequest::Intercept(InterceptRequest::AfterProviderResponse { raw, reply }) => {
                let event = PythonCallEvent::Execution(ExecutionEvent::ProviderResponseReceived {
                    raw: &raw,
                });
                self.observe(&event);
                match self.hooks.on_event(py, event) {
                    Ok(HookStep::Ready(())) => {
                        reply.send(());
                        Ok(Ok(()))
                    }
                    Ok(HookStep::Await(awaitable, resume)) => {
                        self.pending = Some(Pending::Event(resume, EventNext::Emitted(reply)));
                        return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                    }
                    Err(error) => Err(error),
                }
            }
        };
        match answered {
            Ok(Ok(())) => self.native.resume(py, None).map(Next::Continue),
            Ok(Err(native)) => self
                .native
                .resume(py, Some(HostFailure::Error(native)))
                .map(Next::Continue),
            Err(error) => self.interrupt(py, error).map(Next::Return),
        }
    }

    fn opened(
        &mut self,
        py: Python<'_>,
        head: <ProtocolOf<H> as Protocol>::StreamHead,
        reply: Reply<ControlFlow<()>>,
    ) -> PyResult<ExecutionStep> {
        self.stage = Stage::Streaming;
        let head = match self.binding.encode_stream_head(py, head) {
            Ok(head) => head,
            Err(error) => return self.interrupt(py, error),
        };
        match self.hooks.on_stream_open(py, &head) {
            Ok(()) => {
                self.pending = Some(Pending::Consumer(reply));
                Ok(ExecutionStep::Open(head))
            }
            Err(error) => self.interrupt(py, error),
        }
    }

    fn delivered(
        &mut self,
        py: Python<'_>,
        chunk: <ProtocolOf<H> as Protocol>::Chunk,
        reply: Reply<ControlFlow<()>>,
    ) -> PyResult<ExecutionStep> {
        let chunk = match self.binding.encode_chunk(py, chunk) {
            Ok(chunk) => chunk,
            Err(error) => return self.interrupt(py, error),
        };
        match self.hooks.on_stream_chunk(py, &chunk) {
            Ok(()) => {
                self.pending = Some(Pending::Consumer(reply));
                Ok(ExecutionStep::Yield(chunk))
            }
            Err(error) => self.interrupt(py, error),
        }
    }

    fn interrupt(&mut self, py: Python<'_>, error: PyErr) -> PyResult<ExecutionStep> {
        let cancelled = is_cancellation(py, &error);
        let native = H::host_error(&error);
        self.interrupted = Some(error.into_value(py));
        let failure = if cancelled {
            HostFailure::Cancelled(native)
        } else {
            HostFailure::Error(native)
        };
        self.resume_machine(py, Some(failure))
    }

    fn completed(
        &mut self,
        py: Python<'_>,
        response: HostedCompletion<ResponseOf<H>>,
    ) -> PyResult<ExecutionStep> {
        self.ended_at = Some(epoch_seconds());
        let response = match response {
            HostedCompletion::Complete(response) => response,
            HostedCompletion::StreamEnded | HostedCompletion::Detached => {
                return self.succeeded(py, py.None());
            }
        };
        let public = match self.binding.encode_response(py, response) {
            Ok(public) => public,
            Err(error) => return self.failure(py, error, FailureOrigin::Call),
        };
        if let Stage::Streaming = self.stage {
            return self.succeeded(py, public);
        }
        self.stage = Stage::AfterSuccess;
        match self.hooks.transform_response(py, public, self.timing()) {
            Ok(step) => self.on_response(py, step),
            Err(error) => self.failure(py, error, FailureOrigin::Host),
        }
    }

    fn machine_failed(&mut self, py: Python<'_>, error: ErrorOf<H>) -> PyResult<ExecutionStep> {
        self.ended_at.get_or_insert_with(epoch_seconds);
        let error = match self.interrupted.take() {
            Some(retained) => PyErr::from_value(retained.into_bound(py).into_any()),
            None => self.classified(py, error),
        };
        self.failure(py, error, FailureOrigin::Call)
    }

    /// The route's public exception for a native failure. When classification itself
    /// fails, that failure is raised with the native error's text as its `__context__`.
    fn classified(&self, py: Python<'_>, error: ErrorOf<H>) -> PyErr {
        let native = error.to_string();
        let classifier_error = match self.binding.map_error(py, error) {
            Ok(failure) => return failure.into(),
            Err(classifier_error) => classifier_error,
        };
        classifier_error.set_context(py, Some(PyRuntimeError::new_err(native)));
        classifier_error
    }

    fn succeeded(&mut self, py: Python<'_>, response: Py<PyAny>) -> PyResult<ExecutionStep> {
        let event = PythonCallEvent::Succeeded {
            timing: self.timing(),
            response: &response,
        };
        self.terminal_observation = self.observers.as_ref().map(|_| event.snapshot());
        let step = self.hooks.on_event(py, event)?;
        self.stage = Stage::Succeeded(response);
        self.on_event(py, step, EventNext::Terminal)
    }

    fn failure(
        &mut self,
        py: Python<'_>,
        error: PyErr,
        origin: FailureOrigin,
    ) -> PyResult<ExecutionStep> {
        self.ended_at.get_or_insert_with(epoch_seconds);
        if is_cancellation(py, &error) {
            return Err(error);
        }
        let event = PythonCallEvent::Failed {
            timing: self.timing(),
            origin,
            error: &error,
        };
        self.terminal_observation = self.observers.as_ref().map(|_| event.snapshot());
        let step = self.hooks.on_event(py, event)?;
        self.stage = Stage::Failed(error.into_value(py));
        self.on_event(py, step, EventNext::Terminal)
    }

    fn observe(&self, event: &PythonCallEvent<'_>) {
        if let Some(observers) = &self.observers {
            observers.emit(event.snapshot());
        }
    }

    fn clear(&mut self) {
        if !self.closed {
            self.closed = true;
            if self.observing {
                self.observe(&PythonCallEvent::Cancelled {
                    timing: self.timing(),
                });
                self.observing = false;
            }
            self.native.close();
            self.start = None;
            Python::attach(|py| {
                self.hooks.close(py);
                self.binding.close(py);
            });
        }
    }
}

impl<H, M, L> ExecutionBody for PythonDriver<H, M, L>
where
    L: PythonCallHooks + 'static,
    H: PythonBinding + PythonHostCalls<H::Protocol>,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<ResponseOf<H>>>,
{
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| {
            let outcome = self.drive(py, result);
            if let Err(error) = &outcome
                && self.observing
            {
                let event = if is_cancellation(py, error) {
                    PythonCallEvent::Cancelled {
                        timing: self.timing(),
                    }
                } else {
                    PythonCallEvent::Failed {
                        timing: self.timing(),
                        origin: FailureOrigin::Host,
                        error,
                    }
                };
                self.observe(&event);
                self.observing = false;
                self.terminal_observation = None;
            }
            outcome
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.binding.traverse(visit)?;
        self.hooks.traverse(visit)?;
        visit.call(&self.arguments)?;
        visit.call(&self.interrupted)?;
        match &self.stage {
            Stage::Succeeded(response) => visit.call(response),
            Stage::Failed(error) => visit.call(error),
            _ => Ok(()),
        }
    }
}

impl<H, M, L> Drop for PythonDriver<H, M, L>
where
    L: PythonCallHooks + 'static,
    H: PythonBinding + PythonHostCalls<H::Protocol>,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<ResponseOf<H>>>,
{
    fn drop(&mut self) {
        self.clear();
    }
}
