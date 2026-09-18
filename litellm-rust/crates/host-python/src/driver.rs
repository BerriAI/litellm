use std::sync::Arc;
use std::task::Poll;

use futures_util::future::{AbortHandle, Abortable};
use litellm_callbacks::event::{FailureOrigin, Timing, epoch_seconds};
use litellm_callbacks::host::{Demand, HostOp, HostResult, HostStep};
use litellm_callbacks::machine::{HostFailure, Machine, MachineStep};
use litellm_callbacks::route::Route;
use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::Mutex;

use crate::adapter::{
    HostOpError, LifecycleEvent, LifecycleStep, PythonLifecycle, RouteHost, missing_state,
};
use crate::execution::{poll_async_value, run_async_value, run_sync_value};
use crate::handle::{Execution, ExecutionBody, ExecutionStep};

type RouteOf<H> = <H as RouteHost>::Route;
type ErrorOf<H> = <RouteOf<H> as Route>::Error;
type ResponseOf<H> = <RouteOf<H> as Route>::Response;
type NativeStep<H> = MachineStep<RouteOf<H>, ResponseOf<H>>;
type NativeResult<H> = Result<NativeStep<H>, ErrorOf<H>>;
type NativeResume<H> = Option<Result<HostResult<RouteOf<H>>, HostFailure<ErrorOf<H>>>>;

type MachineResult<M> = Result<
    MachineStep<<M as Machine>::Route, <M as Machine>::Complete>,
    <<M as Machine>::Route as Route>::Error,
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

#[derive(Clone, Copy)]
enum Expect {
    Started,
    Arguments,
    Wire,
    Emitted,
    Response,
    Terminal,
}

enum Pending {
    Native,
    Adapter(Expect),
    /// The stream handed to the caller waits for its next read or its close.
    Consumer,
}

enum Next<H: RouteHost> {
    Return(ExecutionStep),
    Continue(HostStep<NativeResult<H>, Py<PyAny>>),
}

struct PythonDriver<H, M>
where
    H: RouteHost,
    M: Machine<Route = H::Route, Complete = ResponseOf<H>> + 'static,
{
    route: H,
    adapter: Box<dyn PythonLifecycle>,
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
/// host suspension inline in the caller's task.
pub fn run_call<H, M>(
    py: Python<'_>,
    machine: M,
    route: H,
    adapter: Box<dyn PythonLifecycle>,
    arguments: Py<PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>>
where
    H: RouteHost + 'static,
    M: Machine<Route = H::Route, Complete = ResponseOf<H>> + 'static,
{
    let mut driver = PythonDriver {
        route,
        adapter,
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
        ExecutionStep::Open => py
            .import("litellm.rust_bridge.lifecycle")?
            .getattr("SyncStream")?
            .call1((Py::new(py, Execution::suspended(driver))?,))
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
    H: RouteHost,
    M: Machine<Route = H::Route, Complete = ResponseOf<H>> + 'static,
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
            (Some(Pending::Consumer), Some(read)) => {
                let demand = if read.is_ok() {
                    Demand::More
                } else {
                    Demand::Detached
                };
                self.resume_machine(py, Some(Ok(HostResult::Demand(demand))))
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
        match (expect, step) {
            (_, LifecycleStep::Await(awaitable)) => {
                self.pending = Some(Pending::Adapter(expect));
                Ok(ExecutionStep::Await(awaitable))
            }
            (Expect::Started, LifecycleStep::Done) => self.begin(py),
            (Expect::Arguments, LifecycleStep::Arguments(arguments)) => {
                self.arguments = Some(arguments);
                self.stage = Stage::Call;
                self.resume_machine(py, None)
            }
            (Expect::Wire, LifecycleStep::Wire(wire)) => {
                self.resume_machine(py, Some(Ok(HostResult::BeforeSend(wire))))
            }
            (Expect::Emitted, LifecycleStep::Done) => {
                self.resume_machine(py, Some(Ok(HostResult::Emitted)))
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
        result: NativeResume<H>,
    ) -> PyResult<ExecutionStep> {
        let step = self.resume_core(py, result)?;
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
        let answer = match op {
            HostOp::Route(op) => {
                let arguments = self.arguments.as_ref().ok_or_else(missing_state)?;
                match self.route.invoke(py, arguments.bind(py), op) {
                    Ok(result) => Ok(HostResult::Route(result)),
                    Err(HostOpError::Native(error)) => {
                        return self
                            .resume_core(py, Some(Err(HostFailure::Error(error))))
                            .map(Next::Continue);
                    }
                    Err(HostOpError::Python(error)) => Err(error),
                }
            }
            HostOp::BeforeSend { wire, context } => {
                match self.adapter.before_send(py, wire, &context) {
                    Ok(LifecycleStep::Wire(wire)) => Ok(HostResult::BeforeSend(wire)),
                    Ok(LifecycleStep::Await(awaitable)) => {
                        self.pending = Some(Pending::Adapter(Expect::Wire));
                        return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                    }
                    Ok(_) => return Err(missing_state()),
                    Err(error) => Err(error),
                }
            }
            HostOp::Open(_) => return self.opened(py).map(Next::Return),
            HostOp::Deliver(chunk) => return self.delivered(py, chunk).map(Next::Return),
            HostOp::Emit(event) => match self.adapter.emit(py, LifecycleEvent::Machine(&event)) {
                Ok(LifecycleStep::Done) => Ok(HostResult::Emitted),
                Ok(LifecycleStep::Await(awaitable)) => {
                    self.pending = Some(Pending::Adapter(Expect::Emitted));
                    return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                }
                Ok(_) => return Err(missing_state()),
                Err(error) => Err(error),
            },
        };
        match answer {
            Ok(answer) => self.resume_core(py, Some(Ok(answer))).map(Next::Continue),
            Err(error) => self.interrupt(py, error).map(Next::Return),
        }
    }

    fn opened(&mut self, py: Python<'_>) -> PyResult<ExecutionStep> {
        self.stage = Stage::Streaming;
        match self.adapter.opened(py) {
            Ok(()) => {
                self.pending = Some(Pending::Consumer);
                Ok(ExecutionStep::Open)
            }
            Err(error) => self.interrupt(py, error),
        }
    }

    fn delivered(
        &mut self,
        py: Python<'_>,
        chunk: <RouteOf<H> as Route>::Chunk,
    ) -> PyResult<ExecutionStep> {
        let chunk = match self.route.chunk(py, chunk) {
            Ok(chunk) => chunk,
            Err(error) => return self.interrupt(py, error),
        };
        match self.adapter.delivered(py, &chunk) {
            Ok(()) => {
                self.pending = Some(Pending::Consumer);
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
        self.resume_machine(py, Some(Err(failure)))
    }

    fn resume_core(
        &mut self,
        py: Python<'_>,
        result: NativeResume<H>,
    ) -> PyResult<HostStep<NativeResult<H>, Py<PyAny>>> {
        let state = Arc::clone(self.machine.as_ref().ok_or_else(missing_state)?);
        let future = async move {
            let mut state = state.lock().await;
            let result = match result {
                Some(Err(failure)) => state
                    .machine
                    .interrupt(failure)
                    .await
                    .map(MachineStep::Complete),
                Some(Ok(result)) => state.machine.resume(Some(result)).await,
                None => state.machine.resume(None).await,
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
        let public = match self.route.complete(py, response) {
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
        let classifier_error = match self.route.classify(py, error) {
            Ok(failure) => return failure.into(),
            Err(classifier_error) => classifier_error,
        };
        let attached = classifier_error.value(py).setattr(
            "__context__",
            PyRuntimeError::new_err(native).into_value(py),
        );
        match attached {
            Ok(()) => classifier_error,
            Err(error) => error,
        }
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
                self.route.close(py);
            });
        }
    }
}

impl<H, M> ExecutionBody for PythonDriver<H, M>
where
    H: RouteHost,
    M: Machine<Route = H::Route, Complete = ResponseOf<H>> + 'static,
{
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| self.drive(py, result))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.route.traverse(visit)?;
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
    H: RouteHost,
    M: Machine<Route = H::Route, Complete = ResponseOf<H>> + 'static,
{
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use litellm_callbacks::event::{MachineEvent, RequestContext, WireRequest};
    use litellm_callbacks::machine::{Interrupted, Step};
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

    struct Synthetic;

    impl Route for Synthetic {
        type Response = String;
        type Error = Error;
        type Op = &'static str;
        type OpResult = String;
        type Chunk = std::convert::Infallible;
        type StreamHead = std::convert::Infallible;
    }

    /// Yields the scripted ops in order, then completes or fails as scripted.
    struct ScriptedMachine {
        ops: Vec<HostOp<Synthetic>>,
        outcome: Option<Result<String, Error>>,
        answers: Vec<String>,
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

    impl Machine for ScriptedMachine {
        type Route = Synthetic;
        type Complete = String;

        fn resume(&mut self, result: Option<HostResult<Synthetic>>) -> Step<'_, Self> {
            Box::pin(async move {
                if let Some(result) = result {
                    self.answers.push(match result {
                        HostResult::Route(value) => value,
                        HostResult::BeforeSend(wire) => wire.url,
                        HostResult::Emitted => "emitted".into(),
                        HostResult::Demand(demand) => format!("{demand:?}"),
                    });
                }
                if !self.ops.is_empty() {
                    return Ok(MachineStep::Host(self.ops.remove(0)));
                }
                self.outcome
                    .take()
                    .ok_or_else(|| Error("resumed after completion".into()))?
                    .map(MachineStep::Complete)
            })
        }

        fn interrupt(&mut self, failure: HostFailure<Error>) -> Interrupted<'_, Self> {
            self.ops.clear();
            self.outcome = None;
            Box::pin(async move { Err(failure.into_error()) })
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

    impl RouteHost for SyntheticHost {
        type Route = Synthetic;
        type Failure = Classified;

        fn invoke(
            &mut self,
            _: Python<'_>,
            arguments: &Bound<'_, PyDict>,
            op: &'static str,
        ) -> Result<String, HostOpError<Error>> {
            self.log.push(format!("route:{op}"));
            match self.op {
                OpScript::Answer => Ok(format!("{op}:{}", arguments.len())),
                OpScript::RaisePython => Err(PyValueError::new_err("op failed").into()),
                OpScript::RejectNatively => Err(HostOpError::Native(Error("op rejected".into()))),
            }
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
            self.log.push("route.close");
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
                AdapterScript::Plain | AdapterScript::FailBegin => {
                    Ok(LifecycleStep::Response(response))
                }
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

        fn resume(&mut self, _: Python<'_>, _: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep> {
            Err(missing_state())
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
        machine: ScriptedMachine,
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
        machine: ScriptedMachine,
        route: SyntheticHost,
        script: AdapterScript,
        asynchronous: bool,
    ) -> (PyResult<Py<PyAny>>, Vec<String>) {
        let log = Log(route.log.0.clone());
        let adapter = SyntheticAdapter {
            log: Log(log.0.clone()),
            script,
        };
        let arguments = PyDict::new(py);
        arguments.set_item("model", "m").unwrap();
        let result = run_call(
            py,
            machine,
            route,
            Box::new(adapter),
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

    fn success_machine() -> ScriptedMachine {
        ScriptedMachine {
            ops: vec![
                HostOp::Route("project"),
                HostOp::BeforeSend {
                    wire: Box::new(wire()),
                    context: Box::new(context()),
                },
                HostOp::Emit(MachineEvent::ResponseReceived {
                    raw: litellm_callbacks::event::RawResponse { body: "raw".into() },
                }),
            ],
            outcome: Some(Ok("done".into())),
            answers: Vec::new(),
        }
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
                assert_eq!(result.unwrap().extract::<String>(py).unwrap(), "done");
                assert_eq!(
                    log,
                    [
                        "started",
                        "begin",
                        "route:project",
                        "before_send",
                        "response:raw",
                        "complete",
                        "after_success",
                        "succeeded:done",
                        "adapter.close",
                        "route.close",
                    ]
                );
            }
        });
    }

    fn failing_machine() -> ScriptedMachine {
        ScriptedMachine {
            ops: vec![HostOp::Route("project")],
            outcome: Some(Err(Error("provider exploded".into()))),
            answers: Vec::new(),
        }
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
                        "route:project",
                        "classify:provider exploded",
                        "failed:Call:classified: provider exploded",
                        "adapter.close",
                        "route.close",
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
                    "route:project",
                    "classify:op rejected",
                    "failed:Call:classified: op rejected",
                    "adapter.close",
                    "route.close",
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
                    "route:project",
                    "failed:Call:op failed",
                    "adapter.close",
                    "route.close",
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
            let context = error.value(py).getattr("__context__").unwrap();
            assert!(context.is_instance_of::<PyRuntimeError>());
            assert_eq!(context.str().unwrap().to_string(), "provider exploded");
            assert_eq!(
                log,
                [
                    "started",
                    "begin",
                    "route:project",
                    "classify:provider exploded",
                    "failed:Call:classifier failed",
                    "adapter.close",
                    "route.close",
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
                    "route.close"
                ]
            );
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
                assert!(!log.contains(&"succeeded:done".to_string()));
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
                        "route.close"
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
            impl RouteHost for Cancelling {
                type Route = Synthetic;
                type Failure = Classified;
                fn invoke(
                    &mut self,
                    py: Python<'_>,
                    _: &Bound<'_, PyDict>,
                    _: &'static str,
                ) -> Result<String, HostOpError<Error>> {
                    self.0.push("route");
                    Err(PyErr::from_value(
                        py.import("asyncio")
                            .unwrap()
                            .getattr("CancelledError")
                            .unwrap()
                            .call0()
                            .unwrap(),
                    )
                    .into())
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
            let route = Cancelling(Log(log.0.clone()));
            let adapter = SyntheticAdapter {
                log: Log(log.0.clone()),
                script: AdapterScript::Plain,
            };
            let error = run_call(
                py,
                success_machine(),
                route,
                Box::new(adapter),
                PyDict::new(py).unbind(),
                false,
            )
            .unwrap_err();
            assert!(!error.is_instance_of::<pyo3::exceptions::PyException>(py));
            assert_eq!(
                log.entries(),
                ["started", "begin", "route", "adapter.close"]
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
