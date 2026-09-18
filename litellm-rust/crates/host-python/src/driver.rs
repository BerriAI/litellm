use std::sync::Arc;
use std::task::Poll;

use futures_util::future::{AbortHandle, Abortable};
use litellm_callbacks::event::{CallEvent, FailureOrigin, Timing, epoch_seconds};
use litellm_callbacks::host::{HostOp, HostResult, HostStep};
use litellm_callbacks::machine::{HostFailure, Machine, MachineStep};
use litellm_callbacks::route::Route;
use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::Mutex;

use crate::adapter::{AdapterStep, CallbackAdapter, PublicValue, RouteHost, missing_state};
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
    AfterSuccess,
    Succeeded(Py<PyAny>),
    Failed(Py<PyBaseException>),
}

#[derive(Clone, Copy)]
enum Expect {
    Arguments,
    Wire,
    Emitted,
    Recorded,
    Response,
    Terminal,
}

enum Pending {
    Native,
    /// A chunk is with the consumer; the machine resumes when it asks for the next one.
    Yielded,
    Adapter(Expect),
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
    adapter: Box<dyn CallbackAdapter>,
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

fn driver<H, M>(
    machine: M,
    route: H,
    adapter: Box<dyn CallbackAdapter>,
    arguments: Py<PyDict>,
    asynchronous: bool,
) -> PythonDriver<H, M>
where
    H: RouteHost + 'static,
    M: Machine<Route = H::Route, Complete = ResponseOf<H>> + 'static,
{
    PythonDriver {
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
    }
}

/// Runs one native call for Python: synchronously, or as a coroutine that awaits every
/// host suspension inline in the caller's task.
pub fn run_call<H, M>(
    py: Python<'_>,
    machine: M,
    route: H,
    adapter: Box<dyn CallbackAdapter>,
    arguments: Py<PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>>
where
    H: RouteHost + 'static,
    M: Machine<Route = H::Route, Complete = ResponseOf<H>> + 'static,
{
    let mut driver = driver(machine, route, adapter, arguments, asynchronous);
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
        ExecutionStep::Await(_) => Err(PyRuntimeError::new_err("sync call suspended")),
        ExecutionStep::Yield(_) => Err(PyRuntimeError::new_err("sync call yielded a chunk")),
    }
}

/// Runs one streaming native call for Python as an async iterator that yields each chunk
/// in the caller's task and completes, with its terminal callbacks, after the last one.
pub fn run_stream<H, M>(
    py: Python<'_>,
    machine: M,
    route: H,
    adapter: Box<dyn CallbackAdapter>,
    arguments: Py<PyDict>,
) -> PyResult<Py<PyAny>>
where
    H: RouteHost + 'static,
    M: Machine<Route = H::Route, Complete = ResponseOf<H>> + 'static,
{
    let execution = Py::new(
        py,
        Execution::new(driver(machine, route, adapter, arguments, true)),
    )?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("drive_stream")?
        .call1((execution,))
        .map(Bound::unbind)
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
                let arguments = self.arguments.take().ok_or_else(missing_state)?;
                match self.adapter.begin(py, arguments, self.started_at) {
                    Ok(step) => self.on_adapter(py, step, Expect::Arguments),
                    Err(error) => self.adapter_failed(py, error),
                }
            }
            (Some(Pending::Native), Some(Ok(_))) => {
                let result = self.take_native_result()?;
                self.run_steps(py, HostStep::Ready(result))
            }
            (Some(Pending::Native), Some(Err(error))) => self.interrupt(py, error),
            (Some(Pending::Yielded), Some(Ok(_))) => {
                self.resume_machine(py, Some(Ok(HostResult::Consumed)))
            }
            (Some(Pending::Yielded), Some(Err(error))) => self.interrupt(py, error),
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
        step: AdapterStep,
        expect: Expect,
    ) -> PyResult<ExecutionStep> {
        match (expect, step) {
            (_, AdapterStep::Await(awaitable)) => {
                self.pending = Some(Pending::Adapter(expect));
                Ok(ExecutionStep::Await(awaitable))
            }
            (Expect::Arguments, AdapterStep::Arguments(arguments)) => {
                self.arguments = Some(arguments);
                self.stage = Stage::Call;
                self.resume_machine(py, None)
            }
            (Expect::Wire, AdapterStep::Wire(wire)) => {
                self.resume_machine(py, Some(Ok(HostResult::BeforeSend(wire))))
            }
            (Expect::Emitted, AdapterStep::Done) => {
                self.resume_machine(py, Some(Ok(HostResult::Emitted)))
            }
            (Expect::Emitted, AdapterStep::Arguments(arguments)) => {
                self.arguments = Some(arguments);
                self.resume_machine(py, Some(Ok(HostResult::Emitted)))
            }
            (Expect::Recorded, AdapterStep::Done) => {
                self.resume_machine(py, Some(Ok(HostResult::Recorded)))
            }
            (Expect::Response, AdapterStep::Response(response)) => self.succeeded(py, response),
            (Expect::Terminal, AdapterStep::Done) => match &self.stage {
                Stage::Succeeded(response) => Ok(ExecutionStep::Return(response.clone_ref(py))),
                Stage::Failed(error) => Err(PyErr::from_value(error.bind(py).clone().into_any())),
                _ => Err(missing_state()),
            },
            _ => Err(missing_state()),
        }
    }

    fn adapter_failed(&mut self, py: Python<'_>, error: PyErr) -> PyResult<ExecutionStep> {
        match self.stage {
            Stage::Begin | Stage::AfterSuccess => self.failure(py, error, FailureOrigin::Host),
            Stage::Call => self.interrupt(py, error),
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
            Ok(MachineStep::Yield(chunk)) => {
                return match self.route.chunk(py, chunk) {
                    Ok(chunk) => {
                        self.pending = Some(Pending::Yielded);
                        Ok(Next::Return(ExecutionStep::Yield(chunk)))
                    }
                    Err(error) => self.interrupt(py, error).map(Next::Return),
                };
            }
            Ok(MachineStep::Complete(response)) => {
                return self.completed(py, response).map(Next::Return);
            }
            Err(error) => return self.machine_failed(py, error).map(Next::Return),
        };
        let answer = match op {
            HostOp::Route(op) => {
                let arguments = self.arguments.as_ref().ok_or_else(missing_state)?;
                self.route
                    .invoke(py, arguments.bind(py), op)
                    .map(HostResult::Route)
            }
            HostOp::BeforeSend { wire, context } => {
                match self.adapter.before_send(py, wire, &context) {
                    Ok(AdapterStep::Wire(wire)) => Ok(HostResult::BeforeSend(wire)),
                    Ok(AdapterStep::Await(awaitable)) => {
                        self.pending = Some(Pending::Adapter(Expect::Wire));
                        return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                    }
                    Ok(_) => return Err(missing_state()),
                    Err(error) => Err(error),
                }
            }
            HostOp::Emit(event) => match self.adapter.emit(py, &event, None) {
                Ok(AdapterStep::Done) => Ok(HostResult::Emitted),
                Ok(AdapterStep::Arguments(arguments)) => {
                    self.arguments = Some(arguments);
                    Ok(HostResult::Emitted)
                }
                Ok(AdapterStep::Await(awaitable)) => {
                    self.pending = Some(Pending::Adapter(Expect::Emitted));
                    return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                }
                Ok(_) => return Err(missing_state()),
                Err(error) => Err(error),
            },
            HostOp::AttemptFailed {
                attempt,
                class,
                error,
            } => {
                let native = H::native_error(error);
                let public = self.route.map_failure(py, &native).unwrap_or(native);
                match self.adapter.attempt_failed(py, &attempt, class, &public) {
                    Ok(AdapterStep::Done) => Ok(HostResult::Recorded),
                    Ok(AdapterStep::Await(awaitable)) => {
                        self.pending = Some(Pending::Adapter(Expect::Recorded));
                        return Ok(Next::Return(ExecutionStep::Await(awaitable)));
                    }
                    Ok(_) => return Err(missing_state()),
                    Err(error) => Err(error),
                }
            }
        };
        match answer {
            Ok(answer) => self.resume_core(py, Some(Ok(answer))).map(Next::Continue),
            Err(error) => self.interrupt(py, error).map(Next::Return),
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
            None => H::native_error(error),
        };
        self.failure(py, error, FailureOrigin::Call)
    }

    fn succeeded(&mut self, py: Python<'_>, response: Py<PyAny>) -> PyResult<ExecutionStep> {
        let event = CallEvent::Succeeded {
            timing: self.timing(),
        };
        let step = self
            .adapter
            .emit(py, &event, Some(PublicValue::Response(&response)))?;
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
        let public = match origin {
            FailureOrigin::Call => self.route.map_failure(py, &error).unwrap_or(error),
            FailureOrigin::Host => error,
        };
        let event = CallEvent::Failed {
            timing: self.timing(),
            origin,
        };
        let step = self
            .adapter
            .emit(py, &event, Some(PublicValue::Error(&public)))?;
        self.stage = Stage::Failed(public.into_value(py));
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

    use litellm_callbacks::event::{Passthrough, RequestContext, WireRequest};
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

    struct Synthetic;

    impl Route for Synthetic {
        type Response = String;
        type Error = Error;
        type Op = &'static str;
        type OpResult = String;
        type Chunk = String;
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
            passthrough_fields: Passthrough::default(),
            secret_fields: Vec::new(),
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
                        HostResult::Recorded => "recorded".into(),
                        HostResult::Consumed => "consumed".into(),
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

    struct SyntheticHost {
        log: Log,
        fail_op: bool,
    }

    impl RouteHost for SyntheticHost {
        type Route = Synthetic;

        fn invoke(
            &mut self,
            _: Python<'_>,
            arguments: &Bound<'_, PyDict>,
            op: &'static str,
        ) -> PyResult<String> {
            self.log.push(format!("route:{op}"));
            if arguments.contains("attempt")? {
                self.log.push("route.saw_attempt_arguments");
            }
            if self.fail_op {
                return Err(PyValueError::new_err("op failed"));
            }
            Ok(format!("{op}:{}", arguments.len()))
        }

        fn complete(&mut self, py: Python<'_>, response: String) -> PyResult<Py<PyAny>> {
            self.log.push("complete");
            Ok(pyo3::types::PyString::new(py, &response)
                .into_any()
                .unbind())
        }

        fn chunk(&mut self, py: Python<'_>, chunk: String) -> PyResult<Py<PyAny>> {
            self.log.push(format!("chunk:{chunk}"));
            Ok(pyo3::types::PyString::new(py, &chunk).into_any().unbind())
        }

        fn native_error(error: Error) -> PyErr {
            PyValueError::new_err(error.0)
        }

        fn host_error(error: &PyErr) -> Error {
            Error(error.to_string())
        }

        fn map_failure(&self, py: Python<'_>, error: &PyErr) -> PyResult<PyErr> {
            self.log.push("map_failure");
            Ok(PyValueError::new_err(format!(
                "mapped: {}",
                error.value(py)
            )))
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

    impl CallbackAdapter for SyntheticAdapter {
        fn begin(&mut self, _: Python<'_>, arguments: Py<PyDict>, _: f64) -> PyResult<AdapterStep> {
            self.log.push("begin");
            if matches!(self.script, AdapterScript::FailBegin) {
                return Err(PyValueError::new_err("begin failed"));
            }
            Ok(AdapterStep::Arguments(arguments))
        }

        fn before_send(
            &mut self,
            _: Python<'_>,
            wire: Box<WireRequest>,
            _: &RequestContext,
        ) -> PyResult<AdapterStep> {
            self.log.push("before_send");
            Ok(AdapterStep::Wire(Box::new(WireRequest {
                url: "rewritten".into(),
                ..*wire
            })))
        }

        fn after_success(
            &mut self,
            py: Python<'_>,
            response: Py<PyAny>,
            _: Timing,
        ) -> PyResult<AdapterStep> {
            self.log.push("after_success");
            match self.script {
                AdapterScript::ReplaceResponse => Ok(AdapterStep::Response(
                    "replaced".into_pyobject(py)?.into_any().unbind(),
                )),
                AdapterScript::FailAfterSuccess => {
                    Err(PyValueError::new_err("after_success failed"))
                }
                AdapterScript::Plain | AdapterScript::FailBegin => {
                    Ok(AdapterStep::Response(response))
                }
            }
        }

        fn emit(
            &mut self,
            py: Python<'_>,
            event: &CallEvent,
            public: Option<PublicValue<'_>>,
        ) -> PyResult<AdapterStep> {
            self.log.push(match (event, public) {
                (CallEvent::ResponseReceived { raw }, None) => format!("response:{}", raw.body),
                (CallEvent::Succeeded { .. }, Some(PublicValue::Response(value))) => {
                    format!("succeeded:{}", value.bind(py))
                }
                (CallEvent::Failed { origin, .. }, Some(PublicValue::Error(error))) => {
                    format!("failed:{origin:?}:{}", error.value(py))
                }
                (CallEvent::AttemptStarted { attempt }, None) => {
                    let arguments = PyDict::new(py);
                    arguments.set_item("attempt", attempt.index)?;
                    self.log.push(format!("attempt_started:{}", attempt.index));
                    return Ok(AdapterStep::Arguments(arguments.unbind()));
                }
                _ => "unexpected".into(),
            });
            Ok(AdapterStep::Done)
        }

        fn attempt_failed(
            &mut self,
            py: Python<'_>,
            attempt: &litellm_callbacks::event::AttemptInfo,
            class: litellm_callbacks::failure::FailureClass,
            error: &PyErr,
        ) -> PyResult<AdapterStep> {
            self.log.push(format!(
                "attempt_failed:{}:{class}:{}",
                attempt.index,
                error.value(py)
            ));
            Ok(AdapterStep::Done)
        }

        fn resume(&mut self, _: Python<'_>, _: PyResult<Py<PyAny>>) -> PyResult<AdapterStep> {
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
        fail_op: bool,
        script: AdapterScript,
        asynchronous: bool,
    ) -> (PyResult<Py<PyAny>>, Vec<String>) {
        let log = Log::default();
        let route = SyntheticHost {
            log: Log(log.0.clone()),
            fail_op,
        };
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
                HostOp::Emit(CallEvent::ResponseReceived {
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
                    false,
                    AdapterScript::Plain,
                    asynchronous,
                );
                assert_eq!(result.unwrap().extract::<String>(py).unwrap(), "done");
                assert_eq!(
                    log,
                    [
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

    /// Yields its chunks after one route op, then completes.
    struct StreamingMachine {
        chunks: Vec<String>,
        sent: bool,
    }

    impl Machine for StreamingMachine {
        type Route = Synthetic;
        type Complete = String;

        fn resume(&mut self, _: Option<HostResult<Synthetic>>) -> Step<'_, Self> {
            Box::pin(async move {
                if !self.sent {
                    self.sent = true;
                    return Ok(MachineStep::Host(HostOp::Route("send")));
                }
                if !self.chunks.is_empty() {
                    return Ok(MachineStep::Yield(self.chunks.remove(0)));
                }
                Ok(MachineStep::Complete("done".into()))
            })
        }

        fn interrupt(&mut self, failure: HostFailure<Error>) -> Interrupted<'_, Self> {
            self.chunks.clear();
            Box::pin(async move { Err(failure.into_error()) })
        }
    }

    fn attempt(index: u32) -> litellm_callbacks::event::AttemptInfo {
        litellm_callbacks::event::AttemptInfo {
            trace_id: "trace".into(),
            index,
            group: 0,
            deployment: 1,
        }
    }

    #[test]
    fn streaming_calls_yield_each_chunk_to_the_consumer_then_complete_once() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let log = Log::default();
            let machine = || StreamingMachine {
                chunks: vec!["a".into(), "b".into()],
                sent: false,
            };
            let host = || SyntheticHost {
                log: Log(log.0.clone()),
                fail_op: false,
            };
            let adapter = || SyntheticAdapter {
                log: Log(log.0.clone()),
                script: AdapterScript::Plain,
            };
            let arguments = PyDict::new(py);

            let stream = run_stream(
                py,
                machine(),
                host(),
                Box::new(adapter()),
                arguments.clone().unbind(),
            )
            .unwrap();
            let mut chunks = Vec::new();
            loop {
                let next = stream.call_method0(py, "__anext__").unwrap();
                let stopped = next.call_method1(py, "send", (py.None(),)).unwrap_err();
                if stopped.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                    break;
                }
                assert!(stopped.is_instance_of::<pyo3::exceptions::PyStopIteration>(py));
                chunks.push(
                    stopped
                        .value(py)
                        .getattr("value")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                );
            }

            assert_eq!(chunks, ["a", "b"]);
            assert_eq!(
                log.entries(),
                [
                    "begin",
                    "route:send",
                    "chunk:a",
                    "chunk:b",
                    "complete",
                    "after_success",
                    "succeeded:done",
                    "adapter.close",
                    "route.close",
                ]
            );

            let sync = run_call(
                py,
                machine(),
                host(),
                Box::new(adapter()),
                arguments.unbind(),
                false,
            );
            assert_eq!(
                sync.unwrap_err().value(py).to_string(),
                "sync call yielded a chunk"
            );
        });
    }

    #[test]
    fn attempt_failures_are_mapped_and_recorded_before_the_machine_continues() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let machine = ScriptedMachine {
                ops: vec![
                    HostOp::Emit(CallEvent::AttemptStarted {
                        attempt: attempt(0),
                    }),
                    HostOp::Route("project"),
                    HostOp::AttemptFailed {
                        attempt: attempt(0),
                        class: litellm_callbacks::failure::FailureClass::RateLimited,
                        error: Error("rate limited".into()),
                    },
                    HostOp::Emit(CallEvent::AttemptStarted {
                        attempt: attempt(1),
                    }),
                    HostOp::Route("send"),
                ],
                outcome: Some(Ok("done".into())),
                answers: Vec::new(),
            };
            let (result, log) = run_scripted(py, machine, false, AdapterScript::Plain, true);
            assert_eq!(result.unwrap().extract::<String>(py).unwrap(), "done");
            assert_eq!(
                log,
                [
                    "begin",
                    "attempt_started:0",
                    "route:project",
                    "route.saw_attempt_arguments",
                    "map_failure",
                    "attempt_failed:0:RateLimited:mapped: rate limited",
                    "attempt_started:1",
                    "route:send",
                    "route.saw_attempt_arguments",
                    "complete",
                    "after_success",
                    "succeeded:done",
                    "adapter.close",
                    "route.close",
                ]
            );
        });
    }

    #[test]
    fn machine_failures_are_mapped_and_dispatched_once_as_call_failures() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let machine = ScriptedMachine {
                ops: vec![HostOp::Route("project")],
                outcome: Some(Err(Error("provider exploded".into()))),
                answers: Vec::new(),
            };
            let (result, log) = run_scripted(py, machine, false, AdapterScript::Plain, false);
            let error = result.unwrap_err();
            assert_eq!(error.value(py).to_string(), "mapped: provider exploded");
            assert_eq!(
                log,
                [
                    "begin",
                    "route:project",
                    "map_failure",
                    "failed:Call:mapped: provider exploded",
                    "adapter.close",
                    "route.close",
                ]
            );
        });
    }

    #[test]
    fn host_operation_failures_interrupt_the_call_and_keep_the_python_exception() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (result, log) =
                run_scripted(py, success_machine(), true, AdapterScript::Plain, false);
            let error = result.unwrap_err();
            assert_eq!(error.value(py).to_string(), "mapped: op failed");
            assert!(!log.contains(&"before_send".to_string()));
            assert!(log.contains(&"failed:Call:mapped: op failed".to_string()));
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
                false,
                AdapterScript::FailBegin,
                false,
            );
            let error = result.unwrap_err();
            assert_eq!(error.value(py).to_string(), "begin failed");
            assert_eq!(
                log,
                [
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
                    false,
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
                    false,
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
                fn invoke(
                    &mut self,
                    py: Python<'_>,
                    _: &Bound<'_, PyDict>,
                    _: &'static str,
                ) -> PyResult<String> {
                    self.0.push("route");
                    Err(PyErr::from_value(
                        py.import("asyncio")
                            .unwrap()
                            .getattr("CancelledError")
                            .unwrap()
                            .call0()
                            .unwrap(),
                    ))
                }
                fn complete(&mut self, _: Python<'_>, _: String) -> PyResult<Py<PyAny>> {
                    Err(missing_state())
                }
                fn native_error(error: Error) -> PyErr {
                    PyValueError::new_err(error.0)
                }
                fn host_error(error: &PyErr) -> Error {
                    Error(error.to_string())
                }
                fn map_failure(&self, _: Python<'_>, _: &PyErr) -> PyResult<PyErr> {
                    self.0.push("map_failure");
                    Err(missing_state())
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
            assert_eq!(log.entries(), ["begin", "route", "adapter.close"]);
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
