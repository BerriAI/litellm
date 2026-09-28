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
            HostRequest::HostCall(op) => answered(self.binding.handle_host_call(py, op)),
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
        match self.hooks.on_stream_open(py) {
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use litellm_host::{
        interceptors::{RawResponse, RequestContext},
        machine::{CallMachine, MachineFault},
    };
    use pyo3::exceptions::{PyBaseException, PyValueError};
    use pyo3::types::PyDict;

    use super::*;
    use crate::{PythonOwned, PythonRuntime};
    use litellm_host::hooks::CallHooks;
    use litellm_host::interceptors::Interceptors;

    static PYTHON_GLOBALS: Mutex<()> = Mutex::new(());

    fn lifecycle_binding(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        py.import("host_test_lifecycle")
    }

    fn call_options(asynchronous: bool) -> CallOptions {
        CallOptions::new(asynchronous, lifecycle_binding)
    }

    fn install_lifecycle_module(py: Python<'_>) -> Bound<'_, PyModule> {
        let source =
            std::ffi::CString::new(include_str!("../../../../litellm/rust_bridge/lifecycle.py"))
                .unwrap();
        PyModule::from_code(
            py,
            &source,
            pyo3::ffi::c_str!("lifecycle.py"),
            pyo3::ffi::c_str!("host_test_lifecycle"),
        )
        .unwrap()
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Error(String);

    impl std::fmt::Display for Error {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    impl From<MachineFault> for Error {
        fn from(fault: MachineFault) -> Self {
            Self(format!("{fault:?}"))
        }
    }

    struct Synthetic;

    impl Protocol for Synthetic {
        type Response = String;
        type Error = Error;
        type Request = String;
        type HostCall = (&'static str, Reply<String>);
        type Chunk = std::convert::Infallible;
        type StreamHead = std::convert::Infallible;
    }

    fn wire() -> WireRequest {
        WireRequest {
            url: "https://example.invalid".into(),
            headers: Vec::new(),
            body: serde_json::json!({}),
        }
    }

    fn context() -> RequestContext {
        RequestContext {
            model: "model".into(),
            custom_llm_provider: "provider".into(),
            optional_params: serde_json::json!({}),
            secret_fields: Vec::new(),
            api_key: None,
        }
    }

    #[derive(Default)]
    struct Log(Arc<Mutex<Vec<String>>>);

    impl Log {
        fn push(&self, entry: impl Into<String>) {
            self.0.lock().unwrap().push(entry.into());
        }

        fn entries(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    #[derive(Clone, Copy)]
    enum OpScript {
        Answer,
        RaisePython,
        RejectNatively,
        RejectRequestNatively,
        RaiseRequestPython,
    }

    struct SyntheticBinding {
        log: Log,
        op: OpScript,
        classifier_fails: bool,
    }

    /// The fake route's public exception, kept as a value so a test sees what `classify`
    /// produced before the driver raises it.
    #[derive(Debug, PartialEq, Eq)]
    struct Classified(String);

    impl From<Classified> for PyErr {
        fn from(classified: Classified) -> Self {
            PyValueError::new_err(format!("classified: {}", classified.0))
        }
    }

    impl SyntheticBinding {
        fn answer(&self, value: impl FnOnce() -> String) -> Result<String, InvokeError<Error>> {
            match self.op {
                OpScript::Answer => Ok(value()),
                OpScript::RaisePython | OpScript::RaiseRequestPython => {
                    Err(PyValueError::new_err("op failed").into())
                }
                OpScript::RejectNatively | OpScript::RejectRequestNatively => {
                    Err(InvokeError::Native(Error("op rejected".into())))
                }
            }
        }
    }

    impl PythonBinding for SyntheticBinding {
        type Protocol = Synthetic;
        type Failure = Classified;

        fn decode_request(
            &mut self,
            _: Python<'_>,
            arguments: &Bound<'_, PyDict>,
        ) -> Result<String, InvokeError<Error>> {
            self.log.push("project");
            match self.op {
                OpScript::RejectRequestNatively | OpScript::RaiseRequestPython => {
                    self.answer(String::new)
                }
                _ => Ok(format!("project:{}", arguments.len())),
            }
        }

        fn encode_stream_head(
            &mut self,
            _: Python<'_>,
            head: std::convert::Infallible,
        ) -> PyResult<Py<PyAny>> {
            match head {}
        }

        fn encode_chunk(
            &mut self,
            _: Python<'_>,
            chunk: std::convert::Infallible,
        ) -> PyResult<Py<PyAny>> {
            match chunk {}
        }

        fn encode_response(&mut self, py: Python<'_>, response: String) -> PyResult<Py<PyAny>> {
            self.log.push("complete");
            Ok(pyo3::types::PyString::new(py, &response)
                .into_any()
                .unbind())
        }

        fn map_error(&self, _: Python<'_>, error: Error) -> PyResult<Classified> {
            self.log.push(format!("classify:{error}"));
            if self.classifier_fails {
                return Err(pyo3::exceptions::PyTypeError::new_err("classifier failed"));
            }
            Ok(Classified(error.0))
        }

        fn host_error(error: &PyErr) -> Error {
            Error(error.to_string())
        }
    }

    impl PythonHostCalls<Synthetic> for SyntheticBinding {
        fn handle_host_call(
            &mut self,
            _: Python<'_>,
            (op, reply): (&'static str, Reply<String>),
        ) -> Result<(), InvokeError<Error>> {
            self.log.push(format!("op:{op}"));
            self.answer(|| op.to_string())
                .map(|answer| reply.send(answer))
        }
    }

    impl PythonOwned for SyntheticBinding {
        fn close(&mut self, _: Python<'_>) {
            self.log.push("host.close");
        }
        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    #[derive(Clone, Copy)]
    enum HookScript {
        Plain,
        RewriteArguments,
        ObserveArguments,
        FailBegin,
        ReplaceResponse,
        FailAfterSuccess,
        CancelTerminal,
        FailTerminal,
    }

    struct SyntheticHooks {
        log: Log,
        script: HookScript,
    }

    impl CallHooks<PythonRuntime> for SyntheticHooks {
        fn prepare_arguments(
            &mut self,
            _: Python<'_>,
            arguments: Py<PyDict>,
            _: f64,
        ) -> PyResult<HookStep<Self, Py<PyDict>>> {
            self.log.push("begin");
            if matches!(self.script, HookScript::FailBegin) {
                return Err(PyValueError::new_err("begin failed"));
            }
            if matches!(self.script, HookScript::RewriteArguments) {
                let prepared = Python::attach(|py| -> PyResult<Py<PyDict>> {
                    let prepared = arguments.bind(py).copy()?;
                    prepared.set_item("prepared", "hook")?;
                    prepared.set_item("api_key", "hook-key")?;
                    Ok(prepared.unbind())
                })?;
                return Ok(HookStep::Ready(prepared));
            }
            Ok(HookStep::Ready(arguments))
        }

        fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
            if matches!(self.script, HookScript::ObserveArguments) {
                let value: String = arguments
                    .bind(py)
                    .get_item("prepared")?
                    .unwrap()
                    .extract()?;
                self.log.push(format!("adopted:{value}"));
            }
            Ok(())
        }

        fn before_provider_request(
            &mut self,
            _: Python<'_>,
            wire: Box<WireRequest>,
            _: &RequestContext,
        ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
            self.log.push("before_provider_request");
            Ok(HookStep::Ready(Box::new(WireRequest {
                url: "rewritten".into(),
                ..*wire
            })))
        }

        fn transform_response(
            &mut self,
            py: Python<'_>,
            response: Py<PyAny>,
            _: Timing,
        ) -> PyResult<HookStep<Self, Py<PyAny>>> {
            self.log.push("after_success");
            match self.script {
                HookScript::ReplaceResponse => Ok(HookStep::Ready(
                    "replaced".into_pyobject(py)?.into_any().unbind(),
                )),
                HookScript::FailAfterSuccess => Err(PyValueError::new_err("after_success failed")),
                HookScript::Plain
                | HookScript::ObserveArguments
                | HookScript::RewriteArguments
                | HookScript::FailBegin
                | HookScript::CancelTerminal
                | HookScript::FailTerminal => Ok(HookStep::Ready(response)),
            }
        }

        fn on_event(
            &mut self,
            py: Python<'_>,
            event: PythonCallEvent<'_>,
        ) -> PyResult<HookStep<Self, ()>> {
            if matches!(self.script, HookScript::CancelTerminal)
                && matches!(
                    event,
                    PythonCallEvent::Succeeded { .. } | PythonCallEvent::Failed { .. }
                )
            {
                return Err(pyo3::exceptions::asyncio::CancelledError::new_err(
                    "callback cancelled",
                ));
            }
            if matches!(self.script, HookScript::FailTerminal)
                && matches!(
                    event,
                    PythonCallEvent::Succeeded { .. } | PythonCallEvent::Failed { .. }
                )
            {
                return Err(PyValueError::new_err("callback failed"));
            }
            self.log.push(match event {
                PythonCallEvent::Started { .. } => "started".into(),
                PythonCallEvent::Cancelled { .. } => "cancelled".into(),
                PythonCallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => {
                    format!("response:{}", raw.body)
                }
                PythonCallEvent::Succeeded { response, .. } => {
                    format!("succeeded:{}", response.bind(py))
                }
                PythonCallEvent::Failed { origin, error, .. } => {
                    format!("failed:{origin:?}:{}", error.value(py))
                }
            });
            Ok(HookStep::Ready(()))
        }

        fn on_stream_open(&mut self, _: Python<'_>) -> PyResult<()> {
            self.log.push("opened");
            Ok(())
        }

        fn on_stream_chunk(&mut self, _: Python<'_>, _: &Py<PyAny>) -> PyResult<()> {
            self.log.push("delivered");
            Ok(())
        }
    }

    impl PythonOwned for SyntheticHooks {
        fn close(&mut self, _: Python<'_>) {
            self.log.push("adapter.close");
        }
        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    fn run_scripted(
        py: Python<'_>,
        machine: impl FnOnce(String) -> CallMachine<Synthetic> + Send + Sync + 'static,
        op: OpScript,
        script: HookScript,
        asynchronous: bool,
    ) -> (PyResult<Py<PyAny>>, Vec<String>) {
        run_hosted(
            py,
            machine,
            SyntheticBinding {
                log: Log::default(),
                op,
                classifier_fails: false,
            },
            script,
            asynchronous,
        )
    }

    fn run_hosted(
        py: Python<'_>,
        machine: impl FnOnce(String) -> CallMachine<Synthetic> + Send + Sync + 'static,
        host: SyntheticBinding,
        script: HookScript,
        asynchronous: bool,
    ) -> (PyResult<Py<PyAny>>, Vec<String>) {
        run_composed(
            py,
            machine,
            host,
            script,
            std::convert::identity,
            call_options(asynchronous),
        )
    }

    #[rstest::rstest]
    #[case::synchronous(false)]
    #[case::asynchronous(true)]
    fn composed_hooks_share_prepared_arguments_and_one_execution(#[case] asynchronous: bool) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let log = Log::default();
            let hook = |script| SyntheticHooks {
                log: Log(log.0.clone()),
                script,
            };
            let hooks = crate::HookChain::new(
                hook(HookScript::ObserveArguments),
                crate::HookChain::new(
                    hook(HookScript::RewriteArguments),
                    hook(HookScript::ReplaceResponse),
                ),
            );
            let result = run_call(
                py,
                move |_, _, request| Ok(success_machine()(request)),
                SyntheticBinding {
                    log: Log(log.0.clone()),
                    op: OpScript::Answer,
                    classifier_fails: false,
                },
                hooks,
                PyDict::new(py).unbind(),
                call_options(asynchronous),
            )
            .unwrap();
            let response = if asynchronous {
                assert!(log.entries().is_empty());
                let completion = result.call_method1(py, "send", (py.None(),)).unwrap_err();
                assert!(completion.is_instance_of::<pyo3::exceptions::PyStopIteration>(py));
                completion.value(py).getattr("value").unwrap().unbind()
            } else {
                result
            };
            assert_eq!(response.extract::<String>(py).unwrap(), "replaced");
            let entries = log.entries();
            assert!(entries.iter().any(|entry| entry == "adopted:hook"));
            assert_eq!(
                entries
                    .iter()
                    .filter(|entry| entry.as_str() == "project")
                    .count(),
                1
            );
            assert_eq!(
                entries
                    .iter()
                    .filter(|entry| entry.as_str() == "op:sign")
                    .count(),
                1
            );
            assert_eq!(
                entries
                    .iter()
                    .filter(|entry| entry.as_str() == "succeeded:replaced")
                    .count(),
                3
            );
            assert!(!entries.iter().any(|entry| entry.starts_with("failed:")));
        });
    }

    fn run_composed<L: PythonCallHooks + 'static>(
        py: Python<'_>,
        machine: impl FnOnce(String) -> CallMachine<Synthetic> + Send + Sync + 'static,
        host: SyntheticBinding,
        script: HookScript,
        compose: impl FnOnce(SyntheticHooks) -> L,
        options: CallOptions,
    ) -> (PyResult<Py<PyAny>>, Vec<String>) {
        let asynchronous = options.asynchronous;
        let log = Log(host.log.0.clone());
        let adapter = SyntheticHooks {
            log: Log(log.0.clone()),
            script,
        };
        let arguments = PyDict::new(py);
        arguments.set_item("model", "m").unwrap();
        let result = run_call(
            py,
            move |_, _, request| Ok(machine(request)),
            host,
            compose(adapter),
            arguments.unbind(),
            options,
        );
        let result = if asynchronous {
            result.and_then(|coroutine| {
                let completed = coroutine
                    .call_method1(py, "send", (py.None(),))
                    .unwrap_err();
                if !completed.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                    return Err(completed);
                }
                completed.value(py).getattr("value").map(Bound::unbind)
            })
        } else {
            result
        };
        (result, log.entries())
    }

    /// Answers to projection, to the route op and to `before_provider_request` all reach the
    /// response, so a driver that misroutes a reply changes what the call returns.
    fn success_machine() -> impl FnOnce(String) -> CallMachine<Synthetic> + Send + Sync {
        move |projected| {
            CallMachine::new(None, move |host| {
                Box::pin(async move {
                    let signed = host.services.call(|reply| ("sign", reply)).await?;
                    let wire = host
                        .interceptors
                        .before_provider_request(wire(), context())
                        .await?;
                    host.interceptors
                        .after_provider_response(RawResponse { body: "raw".into() })
                        .await?;
                    Ok(format!("{projected}|{signed}|{}", wire.url))
                })
            })
        }
    }

    #[rstest::rstest]
    #[case::draining(4, false)]
    #[case::full(1, false)]
    #[case::closed(4, true)]
    fn observation_delivery_preserves_callback_order_and_response(
        #[case] capacity: usize,
        #[case] closed: bool,
        #[values(false, true)] asynchronous: bool,
    ) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let (expected, expected_log) = run_scripted(
                py,
                success_machine(),
                OpScript::Answer,
                HookScript::ReplaceResponse,
                asynchronous,
            );
            let (sender, mut receiver) = litellm_host::observation::observation_channel(
                std::num::NonZeroUsize::new(capacity).unwrap(),
            );
            if closed {
                receiver.close();
            }
            let (actual, actual_log) = run_composed(
                py,
                success_machine(),
                SyntheticBinding {
                    log: Log::default(),
                    op: OpScript::Answer,
                    classifier_fails: false,
                },
                HookScript::ReplaceResponse,
                std::convert::identity,
                CallOptions {
                    asynchronous,
                    lifecycle: lifecycle_binding,
                    observers: Some(sender.clone()),
                },
            );
            assert_eq!(
                actual.unwrap().extract::<String>(py).unwrap(),
                expected.unwrap().extract::<String>(py).unwrap()
            );
            assert_eq!(actual_log, expected_log);
            let events: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
            assert_eq!(events.len() as u64 + sender.dropped_events(), 3);
            if !closed && capacity >= 3 {
                let [
                    CallEvent::Started { start_time },
                    CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }),
                    CallEvent::Succeeded { timing, .. },
                ] = events.as_slice()
                else {
                    panic!("expected start, provider response and success: {events:?}");
                };
                assert_eq!(raw.body, "raw");
                assert_eq!(*start_time, timing.start_time);
                assert!(timing.end_time >= timing.start_time);
            }
        });
    }

    #[rstest::rstest]
    #[case::preparation(HookScript::FailBegin, false, 1)]
    #[case::transformation(HookScript::FailAfterSuccess, false, 1)]
    #[case::terminal_cancellation(HookScript::CancelTerminal, true, 0)]
    #[case::terminal_failure(HookScript::FailTerminal, false, 0)]
    fn observation_reports_the_final_outcome_without_replaying_callbacks(
        #[case] script: HookScript,
        #[case] cancelled: bool,
        #[case] failure_callbacks: usize,
        #[values(false, true)] asynchronous: bool,
    ) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let (sender, mut receiver) = litellm_host::observation::observation_channel(
                std::num::NonZeroUsize::new(4).unwrap(),
            );
            let (result, log) = run_composed(
                py,
                success_machine(),
                SyntheticBinding {
                    log: Log::default(),
                    op: OpScript::Answer,
                    classifier_fails: false,
                },
                script,
                std::convert::identity,
                CallOptions {
                    asynchronous,
                    lifecycle: lifecycle_binding,
                    observers: Some(sender),
                },
            );
            let error = result.unwrap_err();
            assert_eq!(
                error.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py),
                cancelled
            );
            let events: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
            assert!(matches!(events.first(), Some(CallEvent::Started { .. })));
            assert_eq!(
                matches!(events.last(), Some(CallEvent::Cancelled { .. })),
                cancelled
            );
            assert_eq!(
                matches!(
                    events.last(),
                    Some(CallEvent::Failed {
                        origin: FailureOrigin::Host,
                        ..
                    })
                ),
                !cancelled
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(
                        event,
                        CallEvent::Failed { .. }
                            | CallEvent::Succeeded { .. }
                            | CallEvent::Cancelled { .. }
                    ))
                    .count(),
                1
            );
            assert_eq!(
                log.iter()
                    .filter(|entry| entry.starts_with("failed:"))
                    .count(),
                failure_callbacks
            );
            assert!(
                !log.iter()
                    .any(|entry| entry.starts_with("succeeded:") || entry == "cancelled")
            );
        });
    }

    #[rstest::rstest]
    #[case::preparation(HookScript::FailBegin, OpScript::Answer)]
    #[case::native_decode(HookScript::Plain, OpScript::RejectRequestNatively)]
    #[case::python_decode(HookScript::Plain, OpScript::RaiseRequestPython)]
    fn startup_failures_release_the_factory_without_constructing_a_machine(
        #[case] script: HookScript,
        #[case] op: OpScript,
        #[values(false, true)] asynchronous: bool,
    ) {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Release(Arc<AtomicBool>);
        impl Drop for Release {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let constructed = Arc::new(AtomicBool::new(false));
            let did_construct = constructed.clone();
            let released = Arc::new(AtomicBool::new(false));
            let release = Release(released.clone());
            let (result, log) = run_scripted(
                py,
                move |request| {
                    let _release = release;
                    did_construct.store(true, Ordering::SeqCst);
                    success_machine()(request)
                },
                op,
                script,
                asynchronous,
            );
            assert!(result.is_err());
            assert!(!constructed.load(Ordering::SeqCst));
            assert!(released.load(Ordering::SeqCst));
            assert_eq!(
                log.iter()
                    .filter(|event| event.starts_with("failed:"))
                    .count(),
                1
            );
            assert_eq!(
                log.iter().filter(|event| *event == "adapter.close").count(),
                1
            );
            assert_eq!(log.iter().filter(|event| *event == "host.close").count(), 1);
        });
    }

    #[rstest::rstest]
    fn success_runs_every_step_in_order_and_returns_the_public_response() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let (result, log) = run_scripted(
                    py,
                    success_machine(),
                    OpScript::Answer,
                    HookScript::Plain,
                    asynchronous,
                );
                assert_eq!(
                    result.unwrap().extract::<String>(py).unwrap(),
                    "project:1|sign|rewritten"
                );
                assert_eq!(
                    log,
                    [
                        "started",
                        "begin",
                        "project",
                        "op:sign",
                        "before_provider_request",
                        "response:raw",
                        "complete",
                        "after_success",
                        "succeeded:project:1|sign|rewritten",
                        "adapter.close",
                        "host.close",
                    ]
                );
            }
        });
    }

    struct Streaming;

    impl Protocol for Streaming {
        type Response = ();
        type Error = Error;
        type Request = ();
        type HostCall = std::convert::Infallible;
        type Chunk = &'static str;
        type StreamHead = Vec<(&'static str, &'static str)>;
    }

    struct StreamingBinding;

    impl PythonBinding for StreamingBinding {
        type Protocol = Streaming;
        type Failure = Classified;

        fn decode_request(
            &mut self,
            _: Python<'_>,
            _: &Bound<'_, PyDict>,
        ) -> Result<(), InvokeError<Error>> {
            Ok(())
        }

        fn encode_stream_head(
            &mut self,
            py: Python<'_>,
            head: Vec<(&'static str, &'static str)>,
        ) -> PyResult<Py<PyAny>> {
            let headers = PyDict::new(py);
            for (name, value) in head {
                headers.set_item(name, value)?;
            }
            let hidden = PyDict::new(py);
            hidden.set_item("additional_headers", headers)?;
            Ok(hidden.into_any().unbind())
        }

        fn encode_chunk(&mut self, py: Python<'_>, chunk: &'static str) -> PyResult<Py<PyAny>> {
            Ok(pyo3::types::PyString::new(py, chunk).into_any().unbind())
        }

        fn encode_response(&mut self, py: Python<'_>, (): ()) -> PyResult<Py<PyAny>> {
            Ok(py.None())
        }

        fn map_error(&self, _: Python<'_>, error: Error) -> PyResult<Classified> {
            Ok(Classified(error.0))
        }

        fn host_error(error: &PyErr) -> Error {
            Error(error.to_string())
        }
    }

    impl PythonHostCalls<Streaming> for StreamingBinding {
        fn handle_host_call(
            &mut self,
            _: Python<'_>,
            op: std::convert::Infallible,
        ) -> Result<(), InvokeError<Error>> {
            match op {}
        }
    }

    impl PythonOwned for StreamingBinding {
        fn close(&mut self, _: Python<'_>) {}
        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    fn streaming_machine()
    -> impl FnOnce(()) -> litellm_host::call::HostedMachine<Streaming> + Send + Sync {
        move |()| {
            litellm_host::call::hosted_call((), None, |(), _, _, _observations| async {
                Ok(litellm_host::call::CallOutput::Stream {
                    head: vec![("request-id", "req_1")],
                    chunks: Box::pin(futures_util::stream::iter([Ok("first"), Ok("second")])),
                })
            })
        }
    }

    #[rstest::rstest]
    #[case::sync(false)]
    #[case::asynchronous(true)]
    fn explicitly_closing_a_stream_dispatches_success_for_delivered_chunks(
        #[case] asynchronous: bool,
    ) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let log = Log::default();
            let (sender, mut receiver) = litellm_host::observation::observation_channel(
                std::num::NonZeroUsize::new(3).unwrap(),
            );
            let handed = run_call(
                py,
                |_, _, request| Ok(streaming_machine()(request)),
                StreamingBinding,
                SyntheticHooks {
                    log: Log(log.0.clone()),
                    script: HookScript::Plain,
                },
                PyDict::new(py).unbind(),
                CallOptions {
                    asynchronous,
                    lifecycle: lifecycle_binding,
                    observers: Some(sender),
                },
            )
            .unwrap();
            let stream = if asynchronous {
                let stop = handed.call_method1(py, "send", (py.None(),)).unwrap_err();
                stop.value(py).getattr("value").unwrap()
            } else {
                handed.into_bound(py)
            };
            if asynchronous {
                for method in ["__anext__", "aclose", "aclose"] {
                    let stop = stream
                        .call_method0(method)
                        .unwrap()
                        .call_method1("send", (py.None(),))
                        .unwrap_err();
                    assert!(stop.is_instance_of::<pyo3::exceptions::PyStopIteration>(py));
                }
            } else {
                assert_eq!(
                    stream
                        .call_method0("__next__")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "first"
                );
                stream.call_method0("close").unwrap();
                stream.call_method0("close").unwrap();
            }
            let events: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
            assert!(matches!(
                events.as_slice(),
                [CallEvent::Started { .. }, CallEvent::Succeeded { .. }]
            ));
            assert_eq!(
                log.entries(),
                [
                    "started",
                    "begin",
                    "opened",
                    "delivered",
                    "succeeded:None",
                    "adapter.close"
                ]
            );
        });
    }

    /// Drives a `Stream` (async) or `SyncStream` to completion from a sync test.
    fn read_all(py: Python<'_>, stream: &Bound<'_, PyAny>, asynchronous: bool) -> Vec<String> {
        if !asynchronous {
            return stream
                .try_iter()
                .unwrap()
                .map(|chunk| chunk.unwrap().extract().unwrap())
                .collect();
        }
        std::iter::from_fn(|| {
            let stop = stream
                .call_method0("__anext__")
                .unwrap()
                .call_method1("send", (py.None(),))
                .unwrap_err();
            if stop.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                return None;
            }
            assert!(stop.is_instance_of::<pyo3::exceptions::PyStopIteration>(py));
            Some(stop.value(py).getattr("value").unwrap().extract().unwrap())
        })
        .collect()
    }

    #[rstest::rstest]
    fn a_stream_carries_its_head_before_the_first_chunk() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let log = Log::default();
                let adapter = SyntheticHooks {
                    log: Log(log.0.clone()),
                    script: HookScript::Plain,
                };
                let handed = run_call(
                    py,
                    |_, _, request| Ok(streaming_machine()(request)),
                    StreamingBinding,
                    adapter,
                    PyDict::new(py).unbind(),
                    call_options(asynchronous),
                )
                .unwrap();
                let stream = if asynchronous {
                    let stop = handed.call_method1(py, "send", (py.None(),)).unwrap_err();
                    stop.value(py).getattr("value").unwrap()
                } else {
                    handed.into_bound(py)
                };
                let hidden: std::collections::HashMap<
                    String,
                    std::collections::HashMap<String, String>,
                > = stream.getattr("head").unwrap().extract().unwrap();
                assert_eq!(
                    hidden["additional_headers"],
                    std::collections::HashMap::from([(
                        "request-id".to_string(),
                        "req_1".to_string()
                    )])
                );
                assert_eq!(log.entries(), ["started", "begin", "opened"]);
                assert_eq!(read_all(py, &stream, asynchronous), ["first", "second"]);
            }
        });
    }

    fn failing_machine() -> impl FnOnce(String) -> CallMachine<Synthetic> + Send + Sync {
        move |_| {
            CallMachine::new(None, |_| {
                Box::pin(async move { Err(Error("provider exploded".into())) })
            })
        }
    }

    #[rstest::rstest]
    fn a_native_failure_is_classified_once_and_reported_classified() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let (result, log) = run_scripted(
                    py,
                    failing_machine(),
                    OpScript::Answer,
                    HookScript::Plain,
                    asynchronous,
                );
                let error = result.unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(error.value(py).to_string(), "classified: provider exploded");
                assert_eq!(
                    log,
                    [
                        "started",
                        "begin",
                        "project",
                        "classify:provider exploded",
                        "failed:Call:classified: provider exploded",
                        "adapter.close",
                        "host.close",
                    ]
                );
            }
        });
    }

    #[rstest::rstest]
    fn a_native_rejection_from_a_host_operation_is_classified_once() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (result, log) = run_scripted(
                py,
                success_machine(),
                OpScript::RejectNatively,
                HookScript::Plain,
                false,
            );
            assert_eq!(
                result.unwrap_err().value(py).to_string(),
                "classified: op rejected"
            );
            assert_eq!(
                log,
                [
                    "started",
                    "begin",
                    "project",
                    "op:sign",
                    "classify:op rejected",
                    "failed:Call:classified: op rejected",
                    "adapter.close",
                    "host.close",
                ]
            );
        });
    }

    #[rstest::rstest]
    fn a_python_exception_from_a_host_operation_is_reported_as_raised() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (result, log) = run_scripted(
                py,
                success_machine(),
                OpScript::RaisePython,
                HookScript::Plain,
                false,
            );
            let error = result.unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.value(py).to_string(), "op failed");
            assert_eq!(
                log,
                [
                    "started",
                    "begin",
                    "project",
                    "op:sign",
                    "failed:Call:op failed",
                    "adapter.close",
                    "host.close",
                ]
            );
        });
    }

    #[rstest::rstest]
    fn a_failing_classifier_surfaces_with_the_native_error_as_context() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (result, log) = run_hosted(
                py,
                failing_machine(),
                SyntheticBinding {
                    log: Log::default(),
                    op: OpScript::Answer,
                    classifier_fails: true,
                },
                HookScript::Plain,
                false,
            );
            let error = result.unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert_eq!(error.value(py).to_string(), "classifier failed");
            let context = error.context(py).unwrap();
            assert!(context.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(context.value(py).to_string(), "provider exploded");
            assert_eq!(
                log,
                [
                    "started",
                    "begin",
                    "project",
                    "classify:provider exploded",
                    "failed:Call:classifier failed",
                    "adapter.close",
                    "host.close",
                ]
            );
        });
    }

    #[rstest::rstest]
    fn begin_failures_are_host_failures_without_provider_mapping() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (result, log) = run_scripted(
                py,
                success_machine(),
                OpScript::Answer,
                HookScript::FailBegin,
                false,
            );
            let error = result.unwrap_err();
            assert_eq!(error.value(py).to_string(), "begin failed");
            assert_eq!(
                log,
                [
                    "started",
                    "begin",
                    "failed:Host:begin failed",
                    "adapter.close",
                    "host.close"
                ]
            );
        });
    }

    enum ArgumentPolicy {
        Inherit,
        Reject(Py<PyBaseException>),
    }

    impl CallHooks<PythonRuntime> for ArgumentPolicy {
        fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
            match self {
                Self::Inherit => arguments.bind(py).set_item("api_key", "inherited"),
                Self::Reject(error) => Err(PyErr::from_value(error.bind(py).clone().into_any())),
            }
        }
    }

    impl PythonOwned for ArgumentPolicy {
        fn close(&mut self, _: Python<'_>) {}

        fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            match self {
                Self::Inherit => Ok(()),
                Self::Reject(error) => visit.call(error),
            }
        }
    }

    #[rstest::rstest]
    #[case::sync_success(false, false)]
    #[case::async_success(true, false)]
    #[case::sync_failure(false, true)]
    #[case::async_failure(true, true)]
    fn resource_setup_uses_prepared_arguments_and_failures_are_terminal(
        #[case] asynchronous: bool,
        #[case] fail_setup: bool,
    ) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let log = Log::default();
            let setup_log = Log(log.0.clone());
            let error = PyValueError::new_err("setup failed").into_value(py);
            let setup_error = error.clone_ref(py);
            let arguments = PyDict::new(py);
            arguments.set_item("api_key", "original").unwrap();
            let result = run_call(
                py,
                move |py, prepared, request| {
                    let source: String = prepared.get_item("prepared")?.unwrap().extract()?;
                    let key: String = prepared.get_item("api_key")?.unwrap().extract()?;
                    setup_log.push(format!("setup:{source}:{key}"));
                    if fail_setup {
                        return Err(PyErr::from_value(setup_error.into_bound(py).into_any()));
                    }
                    Ok(success_machine()(request))
                },
                SyntheticBinding {
                    log: Log(log.0.clone()),
                    op: OpScript::Answer,
                    classifier_fails: false,
                },
                crate::HookChain::new(
                    SyntheticHooks {
                        log: Log(log.0.clone()),
                        script: HookScript::RewriteArguments,
                    },
                    ArgumentPolicy::Inherit,
                ),
                arguments.clone().unbind(),
                call_options(asynchronous),
            );
            let settled = if asynchronous {
                assert!(log.entries().is_empty());
                let completion = result
                    .unwrap()
                    .call_method1(py, "send", (py.None(),))
                    .unwrap_err();
                if completion.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                    completion.value(py).getattr("value").map(Bound::unbind)
                } else {
                    Err(completion)
                }
            } else {
                result
            };
            assert_eq!(
                arguments
                    .get_item("api_key")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "original"
            );
            assert_eq!(
                &log.entries()[..4],
                ["started", "begin", "project", "setup:hook:inherited"]
            );
            if fail_setup {
                assert!(settled.unwrap_err().value(py).is(error.bind(py)));
                assert_eq!(
                    log.entries(),
                    [
                        "started",
                        "begin",
                        "project",
                        "setup:hook:inherited",
                        "failed:Host:setup failed",
                        "adapter.close",
                        "host.close",
                    ]
                );
            } else {
                assert!(settled.is_ok());
                assert!(
                    log.entries()
                        .iter()
                        .any(|entry| entry.starts_with("succeeded:"))
                );
            }
        });
    }

    #[rstest::rstest]
    fn closing_an_unstarted_call_never_initializes_resources() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let log = Log::default();
            let setup_log = Log(log.0.clone());
            let pending = run_call(
                py,
                move |_, _, request| {
                    setup_log.push("setup");
                    Ok(success_machine()(request))
                },
                SyntheticBinding {
                    log: Log(log.0.clone()),
                    op: OpScript::Answer,
                    classifier_fails: false,
                },
                SyntheticHooks {
                    log: Log(log.0.clone()),
                    script: HookScript::Plain,
                },
                PyDict::new(py).unbind(),
                call_options(true),
            )
            .unwrap();
            assert!(log.entries().is_empty());
            pending.call_method0(py, "close").unwrap();
            drop(pending);
            assert_eq!(log.entries(), ["adapter.close", "host.close"]);
        });
    }

    #[rstest::rstest]
    #[case::synchronous(false)]
    #[case::asynchronous(true)]
    fn an_argument_policy_rejection_is_the_callers_error_and_the_machine_never_starts(
        #[case] asynchronous: bool,
    ) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let raised = PyValueError::new_err("over budget").into_value(py);
            let (result, log) = run_composed(
                py,
                success_machine(),
                SyntheticBinding {
                    log: Log::default(),
                    op: OpScript::Answer,
                    classifier_fails: false,
                },
                HookScript::Plain,
                |hooks| crate::HookChain::new(hooks, ArgumentPolicy::Reject(raised.clone_ref(py))),
                call_options(asynchronous),
            );
            assert!(result.unwrap_err().value(py).is(raised.bind(py)));
            assert_eq!(
                log,
                [
                    "started",
                    "begin",
                    "failed:Host:over budget",
                    "adapter.close",
                    "host.close"
                ]
            );
        });
    }

    #[rstest::rstest]
    #[case::synchronous(false)]
    #[case::asynchronous(true)]
    fn the_host_projects_from_the_keyword_view_the_argument_policy_rewrote(
        #[case] asynchronous: bool,
    ) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let (result, _) = run_composed(
                py,
                success_machine(),
                SyntheticBinding {
                    log: Log::default(),
                    op: OpScript::Answer,
                    classifier_fails: false,
                },
                HookScript::Plain,
                |hooks| crate::HookChain::new(hooks, ArgumentPolicy::Inherit),
                call_options(asynchronous),
            );
            assert_eq!(
                result.unwrap().extract::<String>(py).unwrap(),
                "project:2|sign|rewritten"
            );
        });
    }

    #[rstest::rstest]
    fn the_adapters_finalized_response_is_what_the_call_returns_and_reports() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let (result, log) = run_scripted(
                    py,
                    success_machine(),
                    OpScript::Answer,
                    HookScript::ReplaceResponse,
                    asynchronous,
                );
                assert_eq!(result.unwrap().extract::<String>(py).unwrap(), "replaced");
                assert!(log.contains(&"succeeded:replaced".to_string()));
                assert!(!log.contains(&"succeeded:project:1|rewritten".to_string()));
            }
        });
    }

    #[rstest::rstest]
    fn a_failure_while_finalizing_fails_the_call_instead_of_succeeding() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let (result, log) = run_scripted(
                    py,
                    success_machine(),
                    OpScript::Answer,
                    HookScript::FailAfterSuccess,
                    asynchronous,
                );
                let error = result.unwrap_err();
                assert_eq!(error.value(py).to_string(), "after_success failed");
                assert_eq!(
                    &log[log.len() - 4..],
                    [
                        "after_success",
                        "failed:Host:after_success failed",
                        "adapter.close",
                        "host.close"
                    ]
                );
                assert!(!log.iter().any(|entry| entry.starts_with("succeeded")));
            }
        });
    }

    #[rstest::rstest]
    fn cancellation_ends_the_call_without_terminal_dispatch() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            struct Cancelling(Log);
            impl PythonBinding for Cancelling {
                type Protocol = Synthetic;
                type Failure = Classified;
                fn decode_request(
                    &mut self,
                    _: Python<'_>,
                    _: &Bound<'_, PyDict>,
                ) -> Result<String, InvokeError<Error>> {
                    self.0.push("project");
                    Err(pyo3::exceptions::asyncio::CancelledError::new_err(()).into())
                }

                fn encode_stream_head(
                    &mut self,
                    _: Python<'_>,
                    head: std::convert::Infallible,
                ) -> PyResult<Py<PyAny>> {
                    match head {}
                }
                fn encode_chunk(
                    &mut self,
                    _: Python<'_>,
                    chunk: std::convert::Infallible,
                ) -> PyResult<Py<PyAny>> {
                    match chunk {}
                }
                fn encode_response(&mut self, _: Python<'_>, _: String) -> PyResult<Py<PyAny>> {
                    Err(missing_state())
                }
                fn map_error(&self, _: Python<'_>, error: Error) -> PyResult<Classified> {
                    self.0.push("classify");
                    Ok(Classified(error.0))
                }
                fn host_error(error: &PyErr) -> Error {
                    Error(error.to_string())
                }
            }

            impl PythonHostCalls<Synthetic> for Cancelling {
                fn handle_host_call(
                    &mut self,
                    _: Python<'_>,
                    _: (&'static str, Reply<String>),
                ) -> Result<(), InvokeError<Error>> {
                    Err(missing_state().into())
                }
            }

            impl PythonOwned for Cancelling {
                fn close(&mut self, _: Python<'_>) {}
                fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
                    Ok(())
                }
            }
            let log = Log::default();
            let host = Cancelling(Log(log.0.clone()));
            let adapter = SyntheticHooks {
                log: Log(log.0.clone()),
                script: HookScript::Plain,
            };
            let error = run_call(
                py,
                |_, _, request| Ok(success_machine()(request)),
                host,
                adapter,
                PyDict::new(py).unbind(),
                call_options(false),
            )
            .unwrap_err();
            assert!(!error.is_instance_of::<pyo3::exceptions::PyException>(py));
            assert_eq!(
                log.entries(),
                ["started", "begin", "project", "adapter.close"]
            );
        });
    }

    #[rstest::rstest]
    fn python_driver_preserves_inline_await_and_native_ownership() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            py.import("asyncio").unwrap();
            let module = install_lifecycle_module(py);
            let locals = PyDict::new(py);
            locals
                .set_item("drive", module.getattr("drive").unwrap())
                .unwrap();
            locals
                .set_item(
                    "await_execution",
                    wrap_pyfunction!(await_execution, py).unwrap(),
                )
                .unwrap();
            locals
                .set_item(
                    "calling_execution",
                    wrap_pyfunction!(calling_execution, py).unwrap(),
                )
                .unwrap();
            let probe = std::ffi::CString::new(include_str!("../tests/lifecycle.py")).unwrap();
            py.run(&probe, Some(&locals), Some(&locals)).unwrap();
        });
    }
    struct RetainingHost {
        retained: Option<Py<PyAny>>,
    }

    impl ExecutionBody for RetainingHost {
        fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
            Python::attach(|py| Ok(ExecutionStep::Return(py.None())))
        }

        fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.retained)
        }
    }

    #[pyfunction]
    fn retaining_coroutine(py: Python<'_>, retained: Py<PyAny>) -> PyResult<Py<Execution>> {
        Py::new(
            py,
            Execution::new(
                RetainingHost {
                    retained: Some(retained),
                },
                lifecycle_binding,
            ),
        )
    }

    struct AwaitBody(Option<Py<PyAny>>);

    impl ExecutionBody for AwaitBody {
        fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
            match self.0.take() {
                Some(awaitable) => Ok(ExecutionStep::Await(awaitable)),
                None => result
                    .expect("selected await completed")
                    .map(ExecutionStep::Return),
            }
        }

        fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.0)
        }
    }

    #[pyfunction]
    fn await_execution(awaitable: Py<PyAny>) -> Execution {
        Execution::new(AwaitBody(Some(awaitable)), lifecycle_binding)
    }

    struct CallingBody(Py<PyAny>);

    impl ExecutionBody for CallingBody {
        fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
            Python::attach(|py| self.0.call0(py).map(ExecutionStep::Return))
        }

        fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.0)
        }
    }

    #[pyfunction]
    fn calling_execution(callback: Py<PyAny>) -> Execution {
        Execution::new(CallingBody(callback), lifecycle_binding)
    }

    struct ErrorBody(Option<Py<PyBaseException>>);

    impl ExecutionBody for ErrorBody {
        fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
            Python::attach(|py| {
                Err(PyErr::from_value(
                    self.0.take().unwrap().into_bound(py).into_any(),
                ))
            })
        }

        fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.0)
        }
    }

    #[pyfunction]
    fn error_execution(error: Bound<'_, PyBaseException>) -> Execution {
        Execution::new(ErrorBody(Some(error.unbind())), lifecycle_binding)
    }

    #[rstest::rstest]
    fn retained_exception_frames_are_collectable() {
        crate::initialize_python();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "error_execution",
                    wrap_pyfunction!(error_execution, py).unwrap(),
                )
                .unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import gc
import weakref

class Retained:
    pass

def cycle():
    retained = Retained()
    try:
        raise ValueError('retained traceback')
    except ValueError as error:
        retained.owner = error_execution(error)
    return weakref.ref(retained)

reference = cycle()
gc.collect()
assert reference() is None
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
        });
    }

    #[rstest::rstest]
    fn coroutine_collects_cycles_retained_by_bridge_host() {
        crate::initialize_python();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "retaining_coroutine",
                    wrap_pyfunction!(retaining_coroutine, py).unwrap(),
                )
                .unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import gc
import weakref

class Retained:
    pass

def cycle():
    retained = Retained()
    coroutine = retaining_coroutine(retained)
    retained.coroutine = coroutine
    return weakref.ref(retained)

retained_ref = cycle()
gc.collect()
assert retained_ref() is None
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
        });
    }
}
