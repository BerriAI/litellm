use std::sync::Arc;
use std::task::Poll;

use futures_util::future::{AbortHandle, Abortable};
use litellm_host::event::WireRequest;
use litellm_host::event::{FailureOrigin, Timing, epoch_seconds};
use litellm_host::host::{Demand, HostOp, HostStep, Reply, Verdict};
use litellm_host::machine::{HostFailure, Machine, MachineStep};
use litellm_host::protocol::Protocol;
use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::Mutex;

use crate::adapter::{
    InvokeError, LifecycleEvent, LifecycleStep, Preflight, ProtocolHost, PythonLifecycle,
    missing_state,
};
use crate::execution::{poll_async_value, run_async_value, run_sync_value};
use crate::handle::{Execution, ExecutionBody, ExecutionStep};

type ProtocolOf<H> = <H as ProtocolHost>::Protocol;
type ErrorOf<H> = <ProtocolOf<H> as Protocol>::Error;
type ResponseOf<H> = <ProtocolOf<H> as Protocol>::Response;
type NativeStep<H> = MachineStep<ProtocolOf<H>, ResponseOf<H>>;
type NativeResult<H> = Result<NativeStep<H>, ErrorOf<H>>;
type Interruption<H> = Option<HostFailure<ErrorOf<H>>>;

type MachineResult<M> = Result<
    MachineStep<<M as Machine>::Protocol, <M as Machine>::Complete>,
    <<M as Machine>::Protocol as Protocol>::Error,
>;

struct MachineState<M: Machine> {
    machine: M,
    result: Option<MachineResult<M>>,
}

enum Stage {
    Begin,
    Call,
    Streaming,
    AfterSuccess,
    Succeeded(Py<PyAny>),
    Failed(Py<PyBaseException>),
}

enum Expect {
    Started,
    Arguments,
    Answer(Answer),
    Response,
    Terminal,
}

/// A machine op the adapter answers, with the reply its answer goes through.
enum Answer {
    Params(Reply<serde_json::Map<String, serde_json::Value>>),
    Wire(Reply<WireRequest>),
    Emitted(Reply<()>),
}

impl Answer {
    fn send(self, step: LifecycleStep) -> PyResult<()> {
        match (self, step) {
            (Self::Params(reply), LifecycleStep::Params(params)) => reply.send(params),
            (Self::Wire(reply), LifecycleStep::Wire(wire)) => reply.send(*wire),
            (Self::Emitted(reply), LifecycleStep::Done) => reply.send(()),
            _ => return Err(missing_state()),
        }
        Ok(())
    }
}

enum Pending {
    Native,
    Adapter(Expect),
    /// The stream handed to the caller waits for its next read or its close.
    Consumer(Reply<Demand>),
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

enum Next<H: ProtocolHost> {
    Return(ExecutionStep),
    Continue(HostStep<NativeResult<H>, Py<PyAny>>),
}

struct PythonDriver<H, M>
where
    H: ProtocolHost,
    M: Machine<Protocol = H::Protocol, Complete = ResponseOf<H>> + 'static,
{
    host: H,
    adapter: Box<dyn PythonLifecycle>,
    preflight: Preflight,
    machine: Option<Arc<Mutex<MachineState<M>>>>,
    arguments: Option<Py<PyDict>>,
    started_at: f64,
    ended_at: Option<f64>,
    stage: Stage,
    pending: Option<Pending>,
    native_abort: Option<AbortHandle>,
    interrupted: Option<Py<PyBaseException>>,
    asynchronous: bool,
}

/// Runs one native call for Python: synchronously, or as a coroutine that awaits every
/// host suspension inline in the caller's task. `preflight` runs once, on the keyword view
/// the adapter's `begin` returned, before the host projects from it.
pub fn run_call<H, M>(
    py: Python<'_>,
    machine: M,
    host: H,
    adapter: Box<dyn PythonLifecycle>,
    preflight: Preflight,
    arguments: Py<PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>>
where
    H: ProtocolHost + 'static,
    M: Machine<Protocol = H::Protocol, Complete = ResponseOf<H>> + 'static,
{
    let mut driver = PythonDriver {
        host,
        adapter,
        preflight,
        machine: Some(Arc::new(Mutex::new(MachineState {
            machine,
            result: None,
        }))),
        arguments: Some(arguments),
        started_at: 0.0,
        ended_at: None,
        stage: Stage::Begin,
        pending: None,
        native_abort: None,
        interrupted: None,
        asynchronous,
    };
    if asynchronous {
        let execution = Py::new(py, Execution::new(driver))?;
        return py
            .import("litellm.rust_bridge.lifecycle")?
            .getattr("drive")?
            .call1((execution,))
            .map(Bound::unbind);
    }
    match driver.resume(None)? {
        ExecutionStep::Return(value) => Ok(value),
        ExecutionStep::Open(head) => py
            .import("litellm.rust_bridge.lifecycle")?
            .getattr("SyncStream")?
            .call1((Py::new(py, Execution::suspended(driver))?, head))
            .map(Bound::unbind),
        ExecutionStep::Await(_) | ExecutionStep::Yield(_) => {
            Err(PyRuntimeError::new_err("sync call suspended"))
        }
    }
}

fn is_cancellation(py: Python<'_>, error: &PyErr) -> bool {
    !error.is_instance_of::<PyException>(py)
}

impl<H, M> PythonDriver<H, M>
where
    H: ProtocolHost,
    M: Machine<Protocol = H::Protocol, Complete = ResponseOf<H>> + 'static,
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
                let started = LifecycleEvent::Started {
                    start_time: self.started_at,
                };
                match self.adapter.emit(py, started) {
                    Ok(step) => self.on_adapter(py, step, Expect::Started),
                    Err(error) => self.adapter_failed(py, error),
                }
            }
            (Some(Pending::Native), Some(Ok(_))) => {
                let result = self.take_native_result()?;
                self.run_steps(py, HostStep::Ready(result))
            }
            (Some(Pending::Native), Some(Err(error))) => self.interrupt(py, error),
            (Some(Pending::Consumer(reply)), Some(read)) => {
                reply.send(if read.is_ok() {
                    Demand::More
                } else {
                    Demand::Detached
                });
                self.resume_machine(py, None)
            }
            (Some(Pending::Adapter(expect)), Some(result)) => {
                match self.adapter.resume(py, result) {
                    Ok(step) => self.on_adapter(py, step, expect),
                    Err(error) => self.adapter_failed(py, error),
                }
            }
            _ => Err(missing_state()),
        }
    }

    fn on_adapter(
        &mut self,
        py: Python<'_>,
        step: LifecycleStep,
        expect: Expect,
    ) -> PyResult<ExecutionStep> {
        if let LifecycleStep::Await(awaitable) = step {
            self.pending = Some(Pending::Adapter(expect));
            return Ok(ExecutionStep::Await(awaitable));
        }
        match (expect, step) {
            (Expect::Started, LifecycleStep::Done) => self.begin(py),
            (Expect::Arguments, LifecycleStep::Arguments(arguments)) => {
                if let Err(error) = (self.preflight)(py, arguments.bind(py)) {
                    return self.adapter_failed(py, error);
                }
                self.arguments = Some(arguments);
                self.stage = Stage::Call;
                self.resume_machine(py, None)
            }
            (Expect::Answer(answer), step) => {
                answer.send(step)?;
                self.resume_machine(py, None)
            }
            (Expect::Response, LifecycleStep::Response(response)) => self.succeeded(py, response),
            (Expect::Terminal, LifecycleStep::Done) => match &self.stage {
                Stage::Succeeded(response) => Ok(ExecutionStep::Return(response.clone_ref(py))),
                Stage::Failed(error) => Err(PyErr::from_value(error.bind(py).clone().into_any())),
                _ => Err(missing_state()),
            },
            _ => Err(missing_state()),
        }
    }

    fn begin(&mut self, py: Python<'_>) -> PyResult<ExecutionStep> {
        let arguments = self.arguments.take().ok_or_else(missing_state)?;
        match self.adapter.begin(py, arguments, self.started_at) {
            Ok(step) => self.on_adapter(py, step, Expect::Arguments),
            Err(error) => self.adapter_failed(py, error),
        }
    }

    fn adapter_failed(&mut self, py: Python<'_>, error: PyErr) -> PyResult<ExecutionStep> {
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
        let step = self.resume_core(py, interruption)?;
        self.run_steps(py, step)
    }

    fn run_steps(
        &mut self,
        py: Python<'_>,
        mut step: HostStep<NativeResult<H>, Py<PyAny>>,
    ) -> PyResult<ExecutionStep> {
        loop {
            let result = match step {
                HostStep::Suspend(awaitable) => {
                    self.pending = Some(Pending::Native);
                    return Ok(ExecutionStep::Await(awaitable));
                }
                HostStep::Ready(result) => result,
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
            Ok(MachineStep::Host(op)) => op,
            Ok(MachineStep::Complete(response)) => {
                return self.completed(py, response).map(Next::Return);
            }
            Err(error) => return self.machine_failed(py, error).map(Next::Return),
        };
        let answered = match op {
            HostOp::Project(reply) => {
                let arguments = self.arguments.as_ref().ok_or_else(missing_state)?;
                let projected = self.host.project(py, arguments.bind(py));
                answered(projected.map(|projection| reply.send(projection)))
            }
            HostOp::Custom(op) => answered(self.host.invoke(py, op)),
            HostOp::PreRequest { request, reply } => {
                let step = self.adapter.pre_request(py, *request);
                return self.asked(py, step, Answer::Params(reply));
            }
            HostOp::AfterResponse { response, reply } => {
                reply.send(Verdict::Return(*response));
                Ok(Ok(()))
            }
            HostOp::BeforeSend {
                wire,
                context,
                reply,
            } => {
                let step = self.adapter.before_send(py, wire, &context);
                return self.asked(py, step, Answer::Wire(reply));
            }
            HostOp::Open(head, reply) => return self.opened(py, head, reply).map(Next::Return),
            HostOp::Deliver(chunk, reply) => {
                return self.delivered(py, chunk, reply).map(Next::Return);
            }
            HostOp::Emit(event, reply) => {
                let step = self.adapter.emit(py, LifecycleEvent::Machine(&event));
                return self.asked(py, step, Answer::Emitted(reply));
            }
        };
        match answered {
            Ok(Ok(())) => self.resume_core(py, None).map(Next::Continue),
            Ok(Err(native)) => self
                .resume_core(py, Some(HostFailure::Error(native)))
                .map(Next::Continue),
            Err(error) => self.interrupt(py, error).map(Next::Return),
        }
    }

    /// Settles a machine op the adapter was asked: sends its answer and continues the
    /// machine, suspends on the awaitable it returned, or interrupts with what it raised.
    fn asked(
        &mut self,
        py: Python<'_>,
        step: PyResult<LifecycleStep>,
        answer: Answer,
    ) -> PyResult<Next<H>> {
        match step {
            Ok(LifecycleStep::Await(awaitable)) => {
                self.pending = Some(Pending::Adapter(Expect::Answer(answer)));
                Ok(Next::Return(ExecutionStep::Await(awaitable)))
            }
            Ok(step) => {
                answer.send(step)?;
                self.resume_core(py, None).map(Next::Continue)
            }
            Err(error) => self.interrupt(py, error).map(Next::Return),
        }
    }

    fn opened(
        &mut self,
        py: Python<'_>,
        head: <ProtocolOf<H> as Protocol>::StreamHead,
        reply: Reply<Demand>,
    ) -> PyResult<ExecutionStep> {
        self.stage = Stage::Streaming;
        let head = match self.host.head(py, head) {
            Ok(head) => head,
            Err(error) => return self.interrupt(py, error),
        };
        match self.adapter.opened(py) {
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
        reply: Reply<Demand>,
    ) -> PyResult<ExecutionStep> {
        let chunk = match self.host.chunk(py, chunk) {
            Ok(chunk) => chunk,
            Err(error) => return self.interrupt(py, error),
        };
        match self.adapter.delivered(py, &chunk) {
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

    fn resume_core(
        &mut self,
        py: Python<'_>,
        interruption: Interruption<H>,
    ) -> PyResult<HostStep<NativeResult<H>, Py<PyAny>>> {
        let state = Arc::clone(self.machine.as_ref().ok_or_else(missing_state)?);
        let future = async move {
            let mut state = state.lock().await;
            let result = match interruption {
                Some(failure) => state
                    .machine
                    .interrupt(failure)
                    .await
                    .map(MachineStep::Complete),
                None => state.machine.resume().await,
            };
            state.result = Some(result);
            Ok(())
        };
        if self.asynchronous {
            let mut future = Box::pin(future);
            if let Poll::Ready(()) = poll_async_value(py, future.as_mut())? {
                return Ok(HostStep::Ready(self.take_native_result()?));
            }
            let (abort, registration) = AbortHandle::new_pair();
            self.native_abort = Some(abort);
            Ok(HostStep::Suspend(
                run_async_value(py, async move {
                    Abortable::new(future, registration)
                        .await
                        .map_err(|_| PyRuntimeError::new_err("native execution closed"))?
                })?
                .unbind(),
            ))
        } else {
            run_sync_value(py, future)?;
            Ok(HostStep::Ready(self.take_native_result()?))
        }
    }

    fn take_native_result(&self) -> PyResult<NativeResult<H>> {
        self.machine
            .as_ref()
            .ok_or_else(missing_state)?
            .try_lock()
            .map_err(|_| missing_state())?
            .result
            .take()
            .ok_or_else(missing_state)
    }

    fn completed(&mut self, py: Python<'_>, response: ResponseOf<H>) -> PyResult<ExecutionStep> {
        self.ended_at = Some(epoch_seconds());
        let public = match self.host.complete(py, response) {
            Ok(public) => public,
            Err(error) => return self.failure(py, error, FailureOrigin::Call),
        };
        if let Stage::Streaming = self.stage {
            return self.succeeded(py, public);
        }
        self.stage = Stage::AfterSuccess;
        match self.adapter.after_success(py, public, self.timing()) {
            Ok(step) => self.on_adapter(py, step, Expect::Response),
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
        let classifier_error = match self.host.classify(py, error) {
            Ok(failure) => return failure.into(),
            Err(classifier_error) => classifier_error,
        };
        classifier_error.set_context(py, Some(PyRuntimeError::new_err(native)));
        classifier_error
    }

    fn succeeded(&mut self, py: Python<'_>, response: Py<PyAny>) -> PyResult<ExecutionStep> {
        let event = LifecycleEvent::Succeeded {
            timing: self.timing(),
            response: &response,
        };
        let step = self.adapter.emit(py, event)?;
        self.stage = Stage::Succeeded(response);
        self.on_adapter(py, step, Expect::Terminal)
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
        let event = LifecycleEvent::Failed {
            timing: self.timing(),
            origin,
            error: &error,
        };
        let step = self.adapter.emit(py, event)?;
        self.stage = Stage::Failed(error.into_value(py));
        self.on_adapter(py, step, Expect::Terminal)
    }

    fn clear(&mut self) {
        if let Some(abort) = self.native_abort.take() {
            abort.abort();
        }
        if self.machine.take().is_some() {
            Python::attach(|py| {
                self.adapter.close(py);
                self.host.close(py);
            });
        }
    }
}

impl<H, M> ExecutionBody for PythonDriver<H, M>
where
    H: ProtocolHost,
    M: Machine<Protocol = H::Protocol, Complete = ResponseOf<H>> + 'static,
{
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| self.drive(py, result))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.host.traverse(visit)?;
        self.adapter.traverse(visit)?;
        visit.call(&self.arguments)?;
        visit.call(&self.interrupted)?;
        match &self.stage {
            Stage::Succeeded(response) => visit.call(response),
            Stage::Failed(error) => visit.call(error),
            _ => Ok(()),
        }
    }
}

impl<H, M> Drop for PythonDriver<H, M>
where
    H: ProtocolHost,
    M: Machine<Protocol = H::Protocol, Complete = ResponseOf<H>> + 'static,
{
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use litellm_host::event::{MachineEvent, PublicRequest, RawResponse, RequestContext};
    use litellm_host::{MachineFault, machine::CallMachine};
    use pyo3::exceptions::{PyBaseException, PyValueError};
    use pyo3::types::PyDict;

    use super::*;

    static PYTHON_GLOBALS: Mutex<()> = Mutex::new(());

    fn install_lifecycle_module(py: Python<'_>) -> Bound<'_, PyModule> {
        py.run(
            pyo3::ffi::c_str!(
                r#"
import sys
import types

sys.modules.setdefault('litellm', types.ModuleType('litellm'))
sys.modules.setdefault('litellm.rust_bridge', types.ModuleType('litellm.rust_bridge'))
"#
            ),
            None,
            None,
        )
        .unwrap();
        let source =
            std::ffi::CString::new(include_str!("../../../../litellm/rust_bridge/lifecycle.py"))
                .unwrap();
        PyModule::from_code(
            py,
            &source,
            pyo3::ffi::c_str!("lifecycle.py"),
            pyo3::ffi::c_str!("litellm.rust_bridge.lifecycle"),
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
        type Projection = String;
        type Op = (&'static str, Reply<String>);
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
    }

    struct SyntheticHost {
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

    impl SyntheticHost {
        fn answer(&self, value: impl FnOnce() -> String) -> Result<String, InvokeError<Error>> {
            match self.op {
                OpScript::Answer => Ok(value()),
                OpScript::RaisePython => Err(PyValueError::new_err("op failed").into()),
                OpScript::RejectNatively => Err(InvokeError::Native(Error("op rejected".into()))),
            }
        }
    }

    impl ProtocolHost for SyntheticHost {
        type Protocol = Synthetic;
        type Failure = Classified;

        fn project(
            &mut self,
            _: Python<'_>,
            arguments: &Bound<'_, PyDict>,
        ) -> Result<String, InvokeError<Error>> {
            self.log.push("project");
            self.answer(|| format!("project:{}", arguments.len()))
        }

        fn invoke(
            &mut self,
            _: Python<'_>,
            (op, reply): (&'static str, Reply<String>),
        ) -> Result<(), InvokeError<Error>> {
            self.log.push(format!("op:{op}"));
            self.answer(|| op.to_string())
                .map(|answer| reply.send(answer))
        }

        fn head(&mut self, _: Python<'_>, head: std::convert::Infallible) -> PyResult<Py<PyAny>> {
            match head {}
        }

        fn chunk(&mut self, _: Python<'_>, chunk: std::convert::Infallible) -> PyResult<Py<PyAny>> {
            match chunk {}
        }

        fn complete(&mut self, py: Python<'_>, response: String) -> PyResult<Py<PyAny>> {
            self.log.push("complete");
            Ok(pyo3::types::PyString::new(py, &response)
                .into_any()
                .unbind())
        }

        fn classify(&self, _: Python<'_>, error: Error) -> PyResult<Classified> {
            self.log.push(format!("classify:{error}"));
            if self.classifier_fails {
                return Err(pyo3::exceptions::PyTypeError::new_err("classifier failed"));
            }
            Ok(Classified(error.0))
        }

        fn host_error(error: &PyErr) -> Error {
            Error(error.to_string())
        }

        fn close(&mut self, _: Python<'_>) {
            self.log.push("host.close");
        }

        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    #[derive(Clone, Copy)]
    enum AdapterScript {
        Plain,
        FailBegin,
        ReplaceResponse,
        FailAfterSuccess,
        AwaitPreRequest,
    }

    fn rewritten_params() -> serde_json::Map<String, serde_json::Value> {
        [("tools".to_string(), serde_json::json!("rewritten-tools"))]
            .into_iter()
            .collect()
    }

    fn completed_awaitable(py: Python<'_>) -> PyResult<Py<PyAny>> {
        let locals = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!("async def done():\n    return 'hook-result'\ncoroutine = done()"),
            Some(&locals),
            Some(&locals),
        )?;
        Ok(locals.get_item("coroutine")?.expect("coroutine").unbind())
    }

    struct SyntheticAdapter {
        log: Log,
        script: AdapterScript,
    }

    impl PythonLifecycle for SyntheticAdapter {
        fn begin(
            &mut self,
            _: Python<'_>,
            arguments: Py<PyDict>,
            _: f64,
        ) -> PyResult<LifecycleStep> {
            self.log.push("begin");
            if matches!(self.script, AdapterScript::FailBegin) {
                return Err(PyValueError::new_err("begin failed"));
            }
            Ok(LifecycleStep::Arguments(arguments))
        }

        fn pre_request(&mut self, py: Python<'_>, _: PublicRequest) -> PyResult<LifecycleStep> {
            self.log.push("pre_request");
            match self.script {
                AdapterScript::AwaitPreRequest => {
                    Ok(LifecycleStep::Await(completed_awaitable(py)?))
                }
                _ => Ok(LifecycleStep::Params(rewritten_params())),
            }
        }

        fn before_send(
            &mut self,
            _: Python<'_>,
            wire: Box<WireRequest>,
            _: &RequestContext,
        ) -> PyResult<LifecycleStep> {
            self.log.push("before_send");
            Ok(LifecycleStep::Wire(Box::new(WireRequest {
                url: "rewritten".into(),
                ..*wire
            })))
        }

        fn after_success(
            &mut self,
            py: Python<'_>,
            response: Py<PyAny>,
            _: Timing,
        ) -> PyResult<LifecycleStep> {
            self.log.push("after_success");
            match self.script {
                AdapterScript::ReplaceResponse => Ok(LifecycleStep::Response(
                    "replaced".into_pyobject(py)?.into_any().unbind(),
                )),
                AdapterScript::FailAfterSuccess => {
                    Err(PyValueError::new_err("after_success failed"))
                }
                AdapterScript::Plain
                | AdapterScript::FailBegin
                | AdapterScript::AwaitPreRequest => Ok(LifecycleStep::Response(response)),
            }
        }

        fn emit(&mut self, py: Python<'_>, event: LifecycleEvent<'_>) -> PyResult<LifecycleStep> {
            self.log.push(match event {
                LifecycleEvent::Started { .. } => "started".into(),
                LifecycleEvent::Machine(MachineEvent::ResponseReceived { raw }) => {
                    format!("response:{}", raw.body)
                }
                LifecycleEvent::Succeeded { response, .. } => {
                    format!("succeeded:{}", response.bind(py))
                }
                LifecycleEvent::Failed { origin, error, .. } => {
                    format!("failed:{origin:?}:{}", error.value(py))
                }
            });
            Ok(LifecycleStep::Done)
        }

        fn opened(&mut self, _: Python<'_>) -> PyResult<()> {
            self.log.push("opened");
            Ok(())
        }

        fn delivered(&mut self, _: Python<'_>, _: &Py<PyAny>) -> PyResult<()> {
            self.log.push("delivered");
            Ok(())
        }

        fn resume(
            &mut self,
            py: Python<'_>,
            result: PyResult<Py<PyAny>>,
        ) -> PyResult<LifecycleStep> {
            if !matches!(self.script, AdapterScript::AwaitPreRequest) {
                return Err(missing_state());
            }
            self.log.push(format!("resumed:{}", result?.bind(py)));
            Ok(LifecycleStep::Params(rewritten_params()))
        }

        fn close(&mut self, _: Python<'_>) {
            self.log.push("adapter.close");
        }

        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    fn run_scripted(
        py: Python<'_>,
        machine: CallMachine<Synthetic>,
        op: OpScript,
        script: AdapterScript,
        asynchronous: bool,
    ) -> (PyResult<Py<PyAny>>, Vec<String>) {
        run_hosted(
            py,
            machine,
            SyntheticHost {
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
        machine: CallMachine<Synthetic>,
        host: SyntheticHost,
        script: AdapterScript,
        asynchronous: bool,
    ) -> (PyResult<Py<PyAny>>, Vec<String>) {
        run_preflighted(py, machine, host, script, no_preflight, asynchronous)
    }

    fn no_preflight(_: Python<'_>, _: &Bound<'_, PyDict>) -> PyResult<()> {
        Ok(())
    }

    fn run_preflighted(
        py: Python<'_>,
        machine: CallMachine<Synthetic>,
        host: SyntheticHost,
        script: AdapterScript,
        preflight: Preflight,
        asynchronous: bool,
    ) -> (PyResult<Py<PyAny>>, Vec<String>) {
        let log = Log(host.log.0.clone());
        let adapter = SyntheticAdapter {
            log: Log(log.0.clone()),
            script,
        };
        let arguments = PyDict::new(py);
        arguments.set_item("model", "m").unwrap();
        let result = run_call(
            py,
            machine,
            host,
            Box::new(adapter),
            preflight,
            arguments.unbind(),
            asynchronous,
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

    /// Answers to projection, to the route op and to `before_send` all reach the
    /// response, so a driver that misroutes a reply changes what the call returns.
    fn success_machine() -> CallMachine<Synthetic> {
        CallMachine::new(|host| {
            Box::pin(async move {
                let projected = host.project().await?;
                let signed = host.custom_op(|reply| ("sign", reply)).await?;
                let wire = host.before_send(wire(), context()).await?;
                host.emit(MachineEvent::ResponseReceived {
                    raw: RawResponse { body: "raw".into() },
                })
                .await?;
                Ok(format!("{projected}|{signed}|{}", wire.url))
            })
        })
    }

    #[test]
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
                    AdapterScript::Plain,
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
                        "before_send",
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
        type Projection = ();
        type Op = std::convert::Infallible;
        type Chunk = &'static str;
        type StreamHead = Vec<(&'static str, &'static str)>;
    }

    struct StreamingHost;

    impl ProtocolHost for StreamingHost {
        type Protocol = Streaming;
        type Failure = Classified;

        fn project(
            &mut self,
            _: Python<'_>,
            _: &Bound<'_, PyDict>,
        ) -> Result<(), InvokeError<Error>> {
            Ok(())
        }

        fn invoke(
            &mut self,
            _: Python<'_>,
            op: std::convert::Infallible,
        ) -> Result<(), InvokeError<Error>> {
            match op {}
        }

        fn head(
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

        fn chunk(&mut self, py: Python<'_>, chunk: &'static str) -> PyResult<Py<PyAny>> {
            Ok(pyo3::types::PyString::new(py, chunk).into_any().unbind())
        }

        fn complete(&mut self, py: Python<'_>, (): ()) -> PyResult<Py<PyAny>> {
            Ok(py.None())
        }

        fn classify(&self, _: Python<'_>, error: Error) -> PyResult<Classified> {
            Ok(Classified(error.0))
        }

        fn host_error(error: &PyErr) -> Error {
            Error(error.to_string())
        }

        fn close(&mut self, _: Python<'_>) {}

        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    fn streaming_machine() -> CallMachine<Streaming> {
        CallMachine::new(|host| {
            Box::pin(async move {
                host.project().await?;
                if host.open(vec![("request-id", "req_1")]).await? == Demand::Detached {
                    return Ok(());
                }
                for chunk in ["first", "second"] {
                    if host.deliver(chunk).await? == Demand::Detached {
                        break;
                    }
                }
                Ok(())
            })
        })
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

    #[test]
    fn a_stream_carries_its_head_as_hidden_params_before_the_first_chunk() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let log = Log::default();
                let adapter = SyntheticAdapter {
                    log: Log(log.0.clone()),
                    script: AdapterScript::Plain,
                };
                let handed = run_call(
                    py,
                    streaming_machine(),
                    StreamingHost,
                    Box::new(adapter),
                    no_preflight,
                    PyDict::new(py).unbind(),
                    asynchronous,
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
                > = stream.getattr("_hidden_params").unwrap().extract().unwrap();
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

    fn hooked_machine() -> CallMachine<Synthetic> {
        CallMachine::new(|host| {
            Box::pin(async move {
                let projected = host.project().await?;
                let params = host
                    .pre_request(PublicRequest {
                        model: "m".into(),
                        custom_llm_provider: "p".into(),
                        messages: serde_json::json!([]),
                        params: [("tools".to_string(), serde_json::json!("original-tools"))]
                            .into_iter()
                            .collect(),
                        fields: &["tools"],
                    })
                    .await?;
                let wire = host.before_send(wire(), context()).await?;
                let tools = params["tools"].as_str().unwrap_or("not-a-string");
                Ok(format!("{projected}|{tools}|{}", wire.url))
            })
        })
    }

    #[test]
    fn the_pre_request_step_runs_between_projection_and_before_send() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let (result, log) = run_scripted(
                    py,
                    hooked_machine(),
                    OpScript::Answer,
                    AdapterScript::Plain,
                    asynchronous,
                );
                assert_eq!(
                    result.unwrap().extract::<String>(py).unwrap(),
                    "project:1|rewritten-tools|rewritten"
                );
                assert_eq!(
                    log,
                    [
                        "started",
                        "begin",
                        "project",
                        "pre_request",
                        "before_send",
                        "complete",
                        "after_success",
                        "succeeded:project:1|rewritten-tools|rewritten",
                        "adapter.close",
                        "host.close",
                    ]
                );
            }
        });
    }

    #[test]
    fn a_pre_request_step_that_awaits_is_resumed_with_the_awaited_value() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let (result, log) = run_scripted(
                py,
                hooked_machine(),
                OpScript::Answer,
                AdapterScript::AwaitPreRequest,
                true,
            );
            assert_eq!(
                result.unwrap().extract::<String>(py).unwrap(),
                "project:1|rewritten-tools|rewritten"
            );
            assert_eq!(
                log,
                [
                    "started",
                    "begin",
                    "project",
                    "pre_request",
                    "resumed:hook-result",
                    "before_send",
                    "complete",
                    "after_success",
                    "succeeded:project:1|rewritten-tools|rewritten",
                    "adapter.close",
                    "host.close",
                ]
            );
        });
    }

    fn failing_machine() -> CallMachine<Synthetic> {
        CallMachine::new(|host| {
            Box::pin(async move {
                host.project().await?;
                Err(Error("provider exploded".into()))
            })
        })
    }

    #[test]
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
                    AdapterScript::Plain,
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

    #[test]
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
                AdapterScript::Plain,
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
                    "classify:op rejected",
                    "failed:Call:classified: op rejected",
                    "adapter.close",
                    "host.close",
                ]
            );
        });
    }

    #[test]
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
                AdapterScript::Plain,
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
                    "failed:Call:op failed",
                    "adapter.close",
                    "host.close",
                ]
            );
        });
    }

    #[test]
    fn a_failing_classifier_surfaces_with_the_native_error_as_context() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (result, log) = run_hosted(
                py,
                failing_machine(),
                SyntheticHost {
                    log: Log::default(),
                    op: OpScript::Answer,
                    classifier_fails: true,
                },
                AdapterScript::Plain,
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

    #[test]
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
                AdapterScript::FailBegin,
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

    /// The rejection a preflight raised, kept so a test can check the caller receives that
    /// exact object. A `Preflight` is a plain `fn`, so it cannot capture one itself.
    static REJECTION: Mutex<Option<Py<PyBaseException>>> = Mutex::new(None);

    fn rejecting_preflight(py: Python<'_>, _: &Bound<'_, PyDict>) -> PyResult<()> {
        let error = PyValueError::new_err("over budget");
        *REJECTION.lock().unwrap() = Some(error.value(py).clone().unbind());
        Err(error)
    }

    fn inheriting_preflight(_: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<()> {
        arguments.set_item("api_key", "inherited")
    }

    #[test]
    fn a_preflight_rejection_is_the_callers_error_and_the_machine_never_starts() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let (result, log) = run_preflighted(
                    py,
                    success_machine(),
                    SyntheticHost {
                        log: Log::default(),
                        op: OpScript::Answer,
                        classifier_fails: false,
                    },
                    AdapterScript::Plain,
                    rejecting_preflight,
                    asynchronous,
                );
                let error = result.unwrap_err();
                let raised = REJECTION.lock().unwrap().take().unwrap();
                assert!(error.value(py).is(&raised));
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
            }
        });
    }

    #[test]
    fn the_host_projects_from_the_keyword_view_the_preflight_rewrote() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            for asynchronous in [false, true] {
                let (result, _) = run_preflighted(
                    py,
                    success_machine(),
                    SyntheticHost {
                        log: Log::default(),
                        op: OpScript::Answer,
                        classifier_fails: false,
                    },
                    AdapterScript::Plain,
                    inheriting_preflight,
                    asynchronous,
                );
                assert_eq!(
                    result.unwrap().extract::<String>(py).unwrap(),
                    "project:2|sign|rewritten"
                );
            }
        });
    }

    #[test]
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
                    AdapterScript::ReplaceResponse,
                    asynchronous,
                );
                assert_eq!(result.unwrap().extract::<String>(py).unwrap(), "replaced");
                assert!(log.contains(&"succeeded:replaced".to_string()));
                assert!(!log.contains(&"succeeded:project:1|rewritten".to_string()));
            }
        });
    }

    #[test]
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
                    AdapterScript::FailAfterSuccess,
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

    #[test]
    fn cancellation_ends_the_call_without_terminal_dispatch() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            struct Cancelling(Log);
            impl ProtocolHost for Cancelling {
                type Protocol = Synthetic;
                type Failure = Classified;
                fn project(
                    &mut self,
                    _: Python<'_>,
                    _: &Bound<'_, PyDict>,
                ) -> Result<String, InvokeError<Error>> {
                    self.0.push("project");
                    Err(pyo3::exceptions::asyncio::CancelledError::new_err(()).into())
                }
                fn invoke(
                    &mut self,
                    _: Python<'_>,
                    _: (&'static str, Reply<String>),
                ) -> Result<(), InvokeError<Error>> {
                    Err(missing_state().into())
                }
                fn head(
                    &mut self,
                    _: Python<'_>,
                    head: std::convert::Infallible,
                ) -> PyResult<Py<PyAny>> {
                    match head {}
                }
                fn chunk(
                    &mut self,
                    _: Python<'_>,
                    chunk: std::convert::Infallible,
                ) -> PyResult<Py<PyAny>> {
                    match chunk {}
                }
                fn complete(&mut self, _: Python<'_>, _: String) -> PyResult<Py<PyAny>> {
                    Err(missing_state())
                }
                fn classify(&self, _: Python<'_>, error: Error) -> PyResult<Classified> {
                    self.0.push("classify");
                    Ok(Classified(error.0))
                }
                fn host_error(error: &PyErr) -> Error {
                    Error(error.to_string())
                }
                fn close(&mut self, _: Python<'_>) {}
                fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
                    Ok(())
                }
            }
            let log = Log::default();
            let host = Cancelling(Log(log.0.clone()));
            let adapter = SyntheticAdapter {
                log: Log(log.0.clone()),
                script: AdapterScript::Plain,
            };
            let error = run_call(
                py,
                success_machine(),
                host,
                Box::new(adapter),
                no_preflight,
                PyDict::new(py).unbind(),
                false,
            )
            .unwrap_err();
            assert!(!error.is_instance_of::<pyo3::exceptions::PyException>(py));
            assert_eq!(
                log.entries(),
                ["started", "begin", "project", "adapter.close"]
            );
        });
    }

    #[test]
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
            Execution::new(RetainingHost {
                retained: Some(retained),
            }),
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
        Execution::new(AwaitBody(Some(awaitable)))
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
        Execution::new(CallingBody(callback))
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
        Execution::new(ErrorBody(Some(error.unbind())))
    }

    #[test]
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

    #[test]
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
