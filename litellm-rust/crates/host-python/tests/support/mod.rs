#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    hooks::{ArgumentHooks, Cancelled, OutcomeHooks, ProviderHooks, ResponseHooks},
    interceptors::{ExecutionFacts, Interceptors, RawResponse, RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing},
    machine::{CallMachine, MachineFault},
    protocol::{Protocol, Reply},
};
use litellm_host_python::{
    Awaiter, CallOptions, InvokeError, PythonBinding, PythonGroup, PythonHostCalls, PythonOwned,
    PythonSurface, report_unless_cancelled, run_call,
};
use pyo3::{
    exceptions::{PyBaseException, PyTypeError, PyValueError, asyncio::CancelledError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyString},
};
use rstest::fixture;

static PYTHON_GLOBALS: Mutex<()> = Mutex::new(());

/// One initialized interpreter with the real lifecycle protocol installed, held for the
/// length of a test so tests sharing the interpreter run one at a time.
pub struct Interpreter(MutexGuard<'static, ()>);

#[fixture]
pub fn interpreter() -> Interpreter {
    let guard = PYTHON_GLOBALS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    Python::initialize();
    Python::attach(|py| {
        py.import("asyncio").unwrap();
        install_lifecycle_module(py);
    });
    Interpreter(guard)
}

pub fn lifecycle_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("host_test_lifecycle")
}

/// Hands a call's execution to its caller the way the product entrypoints do: an async call
/// gets the coroutine from `drive`, a sync call the value or stream `run` settles to.
pub fn handed(
    py: Python<'_>,
    execution: PyResult<Py<PyAny>>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let execution = execution?;
    let module = lifecycle_module(py)?;
    let handed = if asynchronous {
        module.getattr("drive")?.call1((execution,))?
    } else {
        module
            .getattr("run")?
            .call1((execution, module.getattr("SyncStream")?))?
    };
    Ok(handed.unbind())
}

pub fn call_options(asynchronous: bool) -> CallOptions {
    CallOptions::new(asynchronous, lifecycle_module)
}

pub fn install_lifecycle_module(py: Python<'_>) -> Bound<'_, PyModule> {
    let source = std::ffi::CString::new(concat!(
        include_str!("../../../../../litellm/rust_bridge/execution.py"),
        "\n_drive = drive\n\n\nasync def drive(execution):\n    return await _drive(execution, AsyncStream)\n",
    ))
    .unwrap();
    PyModule::from_code(
        py,
        &source,
        pyo3::ffi::c_str!("execution.py"),
        pyo3::ffi::c_str!("host_test_lifecycle"),
    )
    .unwrap()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

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

pub struct Synthetic;

impl Protocol for Synthetic {
    type Response = String;
    type Error = Error;
    type Request = String;
    type HostCall = (&'static str, Reply<String>);
    type Chunk = std::convert::Infallible;
    type StreamHead = std::convert::Infallible;
}

pub fn wire() -> WireRequest {
    WireRequest {
        url: "https://example.invalid".into(),
        headers: Vec::new(),
        body: serde_json::json!({}),
    }
}

pub fn context() -> RequestContext {
    RequestContext {
        model: "model".into(),
        custom_llm_provider: "provider".into(),
        optional_params: serde_json::json!({}),
        secret_fields: Vec::new(),
        api_key: None,
    }
}

#[derive(Clone, Default)]
pub struct Log(Arc<Mutex<Vec<String>>>);

impl Log {
    pub fn push(&self, entry: impl Into<String>) {
        self.0.lock().unwrap().push(entry.into());
    }

    pub fn entries(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum OpScript {
    Answer,
    RaisePython,
    RejectNatively,
    RejectRequestNatively,
    RaiseRequestPython,
    CancelRequest,
    AwaitAnswer,
    AwaitFailure,
}

/// The fake route's public exception, kept as a value so a test sees what `classify`
/// produced before the driver raises it.
#[derive(Debug, PartialEq, Eq)]
pub struct Classified(pub String);

impl From<Classified> for PyErr {
    fn from(classified: Classified) -> Self {
        PyValueError::new_err(format!("classified: {}", classified.0))
    }
}

pub struct SyntheticBinding {
    log: Log,
    op: OpScript,
    classifier_fails: bool,
}

pub fn binding(log: &Log, op: OpScript) -> SyntheticBinding {
    SyntheticBinding {
        log: log.clone(),
        op,
        classifier_fails: false,
    }
}

impl SyntheticBinding {
    pub fn with_failing_classifier(self) -> Self {
        Self {
            classifier_fails: true,
            ..self
        }
    }

    fn answer(&self, value: impl FnOnce() -> String) -> Result<String, InvokeError<Error>> {
        match self.op {
            OpScript::Answer | OpScript::AwaitAnswer | OpScript::AwaitFailure => Ok(value()),
            OpScript::RaisePython | OpScript::RaiseRequestPython => {
                Err(PyValueError::new_err("op failed").into())
            }
            OpScript::RejectNatively | OpScript::RejectRequestNatively => {
                Err(InvokeError::Native(Error("op rejected".into())))
            }
            OpScript::CancelRequest => Err(CancelledError::new_err(()).into()),
        }
    }
}

impl PythonBinding for SyntheticBinding {
    type Protocol = Synthetic;
    type Failure = Classified;

    fn decode_request(
        &self,
        _: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<String, InvokeError<Error>> {
        self.log.push("project");
        match self.op {
            OpScript::RejectRequestNatively
            | OpScript::RaiseRequestPython
            | OpScript::CancelRequest => self.answer(String::new),
            _ => Ok(format!("project:{}", arguments.len())),
        }
    }

    fn encode_stream_head(
        &self,
        _: Python<'_>,
        head: std::convert::Infallible,
    ) -> PyResult<Py<PyAny>> {
        match head {}
    }

    fn encode_chunk(&self, _: Python<'_>, chunk: std::convert::Infallible) -> PyResult<Py<PyAny>> {
        match chunk {}
    }

    fn encode_response(&self, py: Python<'_>, response: String) -> PyResult<Py<PyAny>> {
        self.log.push("complete");
        Ok(PyString::new(py, &response).into_any().unbind())
    }

    fn map_error(&self, _: Python<'_>, error: Error) -> PyResult<Classified> {
        self.log.push(format!("classify:{error}"));
        if self.classifier_fails {
            return Err(PyTypeError::new_err("classifier failed"));
        }
        Ok(Classified(error.0))
    }

    fn host_error(error: &PyErr) -> Error {
        Error(error.to_string())
    }
}

impl PythonHostCalls<Synthetic> for SyntheticBinding {
    async fn handle_host_call(
        &self,
        task: &dyn Awaiter,
        (op, reply): (&'static str, Reply<String>),
    ) -> Result<(), InvokeError<Error>> {
        self.log.push(format!("op:{op}"));
        if !matches!(self.op, OpScript::AwaitAnswer | OpScript::AwaitFailure) {
            return self
                .answer(|| op.to_string())
                .map(|answer| reply.send(answer));
        }
        let awaitable = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let module = PyModule::from_code(
                py,
                pyo3::ffi::c_str!(
                    "async def answer(fail):\n    if fail:\n        raise LookupError('async host failed')\n    return 'awaited'\n"
                ),
                pyo3::ffi::c_str!("host_op.py"),
                pyo3::ffi::c_str!("host_op"),
            )?;
            Ok(module
                .getattr("answer")?
                .call1((matches!(self.op, OpScript::AwaitFailure),))?
                .unbind())
        })?;
        let answer = task.awaited(awaitable).await?;
        reply.send(Python::attach(|py| answer.extract::<String>(py))?);
        Ok(())
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

#[derive(Clone, Copy, Debug)]
pub enum HookScript {
    Plain,
    RewriteArguments,
    ObserveArguments,
    FailBegin,
    ReplaceResponse,
    FailAfterSuccess,
    CancelTerminal,
    FailTerminal,
}

pub struct SyntheticHooks {
    log: Log,
    script: HookScript,
}

pub fn hooks(log: &Log, script: HookScript) -> SyntheticHooks {
    SyntheticHooks {
        log: log.clone(),
        script,
    }
}

impl SyntheticHooks {
    /// What a terminal hook does with the outcome it was handed: cancels the call, raises
    /// an ordinary error that is only reported, or records it.
    fn outcome(&self, entry: String) -> Result<(), Cancelled<PyErr>> {
        match self.script {
            HookScript::CancelTerminal => {
                Err(Cancelled(CancelledError::new_err("callback cancelled")))
            }
            HookScript::FailTerminal => {
                report_unless_cancelled(Err(PyValueError::new_err("callback failed")))
            }
            _ => {
                self.log.push(entry);
                Ok(())
            }
        }
    }
}

impl ArgumentHooks<PythonSurface> for SyntheticHooks {
    async fn prepare<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        _: f64,
    ) -> PyResult<Py<PyDict>> {
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
            return Ok(prepared);
        }
        Ok(arguments)
    }

    fn prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        if matches!(self.script, HookScript::ObserveArguments) {
            let value: String =
                Python::attach(|py| arguments.bind(py).get_item("prepared")?.unwrap().extract())?;
            self.log.push(format!("adopted:{value}"));
        }
        Ok(())
    }
}

impl ProviderHooks<PythonSurface> for SyntheticHooks {
    async fn before_provider_request<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        _: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        self.log.push("before_provider_request");
        Ok(Box::new(WireRequest {
            url: "rewritten".into(),
            ..*wire
        }))
    }

    async fn after_provider_response<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        raw: &RawResponse,
    ) -> PyResult<()> {
        self.log.push(format!("response:{}", raw.body));
        Ok(())
    }

    async fn result_ready<'a>(&'a self, _: &'a dyn Awaiter, _: &ExecutionFacts) -> PyResult<()> {
        self.log.push("cache_hit");
        Ok(())
    }
}

impl ResponseHooks<PythonSurface> for SyntheticHooks {
    async fn transform<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<Py<PyAny>> {
        self.log.push("after_success");
        match self.script {
            HookScript::ReplaceResponse => {
                Python::attach(|py| Ok("replaced".into_pyobject(py)?.into_any().unbind()))
            }
            HookScript::FailAfterSuccess => Err(PyValueError::new_err("after_success failed")),
            _ => Ok(response),
        }
    }

    fn stream_opened(&self, _: &Py<PyAny>) -> PyResult<()> {
        self.log.push("opened");
        Ok(())
    }

    fn stream_chunk(&self, _: &Py<PyAny>) -> PyResult<()> {
        self.log.push("delivered");
        Ok(())
    }
}

impl OutcomeHooks<PythonSurface> for SyntheticHooks {
    async fn started<'a>(&'a self, _: &'a dyn Awaiter, _: f64) -> Result<(), Cancelled<PyErr>> {
        self.log.push("started");
        Ok(())
    }

    async fn succeeded<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        _: Timing,
        response: &Py<PyAny>,
    ) -> Result<(), Cancelled<PyErr>> {
        self.outcome(Python::attach(|py| {
            format!("succeeded:{}", response.bind(py))
        }))
    }

    async fn failed<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        _: Timing,
        origin: FailureOrigin,
        error: &Py<PyBaseException>,
    ) -> Result<(), Cancelled<PyErr>> {
        self.outcome(Python::attach(|py| {
            format!("failed:{origin:?}:{}", error.bind(py))
        }))
    }

    async fn cancelled<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        _: Timing,
    ) -> Result<(), Cancelled<PyErr>> {
        self.log.push("cancelled");
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

pub type MachineFactory = Box<dyn FnOnce(String) -> CallMachine<Synthetic> + Send + Sync>;

/// Answers to projection, to the route op and to `before_provider_request` all reach the
/// response, so a driver that misroutes a reply changes what the call returns.
pub fn success_machine() -> MachineFactory {
    Box::new(move |projected| {
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
    })
}

pub fn failing_machine() -> MachineFactory {
    Box::new(move |_| {
        CallMachine::new(None, |_| {
            Box::pin(async move { Err(Error("provider exploded".into())) })
        })
    })
}

/// Drives an asynchronous call's coroutine to its settled value from a sync test. A
/// synchronous call is already settled.
pub fn settled(
    py: Python<'_>,
    result: PyResult<Py<PyAny>>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    if !asynchronous {
        return result;
    }
    let coroutine = result?;
    let completed = coroutine
        .call_method1(py, "send", (py.None(),))
        .unwrap_err();
    if !completed.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
        return Err(completed);
    }
    completed.value(py).getattr("value").map(Bound::unbind)
}

pub fn run_scripted(
    py: Python<'_>,
    machine: MachineFactory,
    op: OpScript,
    script: HookScript,
    asynchronous: bool,
) -> (PyResult<Py<PyAny>>, Vec<String>) {
    let log = Log::default();
    run_composed(
        py,
        machine,
        binding(&log, op),
        hooks(&log, script),
        std::convert::identity,
        call_options(asynchronous),
    )
}

pub fn run_composed<L: PythonGroup + 'static>(
    py: Python<'_>,
    machine: MachineFactory,
    binding: SyntheticBinding,
    hooks: SyntheticHooks,
    compose: impl FnOnce(SyntheticHooks) -> L,
    options: CallOptions,
) -> (PyResult<Py<PyAny>>, Vec<String>) {
    let asynchronous = options.asynchronous;
    let log = binding.log.clone();
    let arguments = PyDict::new(py);
    arguments.set_item("model", "m").unwrap();
    let result = handed(
        py,
        run_call(
            py,
            move |_, _, request| Ok(machine(request)),
            binding,
            compose(hooks),
            arguments.unbind(),
            options,
        ),
        asynchronous,
    );
    (settled(py, result, asynchronous), log.entries())
}

pub struct Streaming;

impl Protocol for Streaming {
    type Response = ();
    type Error = Error;
    type Request = ();
    type HostCall = std::convert::Infallible;
    type Chunk = &'static str;
    type StreamHead = Vec<(&'static str, &'static str)>;
}

pub struct StreamingBinding;

impl PythonBinding for StreamingBinding {
    type Protocol = Streaming;
    type Failure = Classified;

    fn decode_request(
        &self,
        _: Python<'_>,
        _: &Bound<'_, PyDict>,
    ) -> Result<(), InvokeError<Error>> {
        Ok(())
    }

    fn encode_stream_head(
        &self,
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

    fn encode_chunk(&self, py: Python<'_>, chunk: &'static str) -> PyResult<Py<PyAny>> {
        Ok(PyString::new(py, chunk).into_any().unbind())
    }

    fn encode_response(&self, py: Python<'_>, (): ()) -> PyResult<Py<PyAny>> {
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
    async fn handle_host_call(
        &self,
        _: &dyn Awaiter,
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

pub fn streaming_machine() -> impl FnOnce(()) -> HostedMachine<Streaming> + Send + Sync {
    move |()| {
        hosted_call((), None, |(), _, _, _observations| async {
            Ok(CallOutput::Stream {
                head: vec![("request-id", "req_1")],
                chunks: Box::pin(futures_util::stream::iter([Ok("first"), Ok("second")])),
            })
        })
    }
}

/// Starts a streaming call and returns the caller's stream object.
pub fn open_stream(
    py: Python<'_>,
    hooks: SyntheticHooks,
    options: CallOptions,
) -> Bound<'_, PyAny> {
    let asynchronous = options.asynchronous;
    let handed = handed(
        py,
        run_call(
            py,
            |_, _, request| Ok(streaming_machine()(request)),
            StreamingBinding,
            hooks,
            PyDict::new(py).unbind(),
            options,
        ),
        asynchronous,
    )
    .unwrap();
    if asynchronous {
        let stop = handed.call_method1(py, "send", (py.None(),)).unwrap_err();
        stop.value(py).getattr("value").unwrap()
    } else {
        handed.into_bound(py)
    }
}

/// Drives an `AsyncStream` or `SyncStream` to completion from a sync test.
pub fn read_all(py: Python<'_>, stream: &Bound<'_, PyAny>, asynchronous: bool) -> Vec<String> {
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
