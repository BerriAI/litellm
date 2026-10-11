use std::{
    future::Future,
    ops::ControlFlow,
    pin::Pin,
    sync::{Arc, Mutex, PoisonError},
};

use litellm_host::{
    call::HostedCompletion,
    hooks::StreamPolicy,
    lifecycle::{CallEvent, ExecutionEvent, FailureOrigin, Timing, epoch_seconds},
    machine::{HostFailure, Machine, MachineStep},
    observation::ObservationSender,
    protocol::{HostRequest, InterceptRequest, Protocol, StreamDelivery},
};
use pyo3::{
    exceptions::{PyBaseException, PyException, PyRuntimeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use tokio::sync::Mutex as PartsMutex;

use crate::{
    InvokeError, PythonBinding, PythonHostCalls,
    handle::{Execution, ExecutionBody, ExecutionStep, PythonLifecycle, Suspension},
    hooks::{PythonCallEvent, PythonCallHooks},
    missing_state,
    native::NativeMachine,
    task::{PythonTask, Resumed},
};

type ProtocolOf<H> = <H as PythonBinding>::Protocol;
type ErrorOf<H> = <ProtocolOf<H> as Protocol>::Error;
type ResponseOf<H> = <ProtocolOf<H> as Protocol>::Response;
type ChunkOf<H> = <ProtocolOf<H> as Protocol>::Chunk;
type StartMachine<P, M> = Box<
    dyn FnOnce(Python<'_>, &Bound<'_, PyDict>, <P as Protocol>::Request) -> PyResult<M>
        + Send
        + Sync,
>;
type Body = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

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

fn is_cancellation(py: Python<'_>, error: &PyErr) -> bool {
    !error.is_instance_of::<PyException>(py)
}

struct Observation {
    observers: Option<ObservationSender>,
    observing: bool,
    started_at: f64,
    ended_at: Option<f64>,
    terminal: Option<CallEvent>,
}

impl Observation {
    fn timing(&self) -> Timing {
        Timing {
            start_time: self.started_at,
            end_time: self.ended_at.unwrap_or_else(epoch_seconds),
        }
    }

    fn emit(&self, event: &PythonCallEvent<'_>) {
        if let Some(observers) = &self.observers {
            observers.emit(event.snapshot());
        }
    }

    fn retain_terminal(&mut self, event: &PythonCallEvent<'_>) {
        self.terminal = self.observers.as_ref().map(|_| event.snapshot());
    }

    fn finish(&mut self) {
        if let Some(event) = self.terminal.take()
            && let Some(observers) = &self.observers
        {
            observers.emit(event);
        }
        self.observing = false;
    }
}

/// What the call shares between its running body and its handle: the binding and hooks the
/// handle traverses and closes, the keyword view, and the observation state the handle
/// reports a cancelled or abandoned call from.
struct Parts<H, L> {
    binding: H,
    hooks: L,
    arguments: Option<Py<PyDict>>,
    observation: Observation,
}

type SharedParts<H, L> = Arc<PartsMutex<Parts<H, L>>>;

enum Setup<E> {
    Host(PyErr),
    Decode(InvokeError<E>),
}

/// The call as one async body on the caller's task. Every hook and host call is awaited in
/// place; the parts lock is held across those awaits and released around every native step
/// and stream boundary, so the handle can traverse the call while it waits on the runtime
/// or on its consumer.
struct Running<H, M, L>
where
    H: PythonBinding,
    M: Machine,
{
    task: PythonTask,
    parts: SharedParts<H, L>,
    native: NativeMachine<M>,
    start: Option<StartMachine<ProtocolOf<H>, M>>,
    interrupted: Option<Py<PyBaseException>>,
    /// A relayed stream is open: chunks go to the consumer as they arrive and the response
    /// transformation never runs.
    streaming: bool,
    /// The caller's stream has been opened, by relay, hold-back or synthesis.
    opened: bool,
    stream_policy: StreamPolicy,
    /// The chunks of a held-back stream, until the stream ends and they are assembled.
    held: Option<Vec<ChunkOf<H>>>,
}

/// The route's public exception for a native failure. When classification itself fails,
/// that failure is raised with the native error's text as its `__context__`.
fn classified<H: PythonBinding>(py: Python<'_>, binding: &H, error: ErrorOf<H>) -> PyErr {
    let native = error.to_string();
    let classifier_error = match binding.map_error(py, error) {
        Ok(failure) => return failure.into(),
        Err(classifier_error) => classifier_error,
    };
    classifier_error.set_context(py, Some(PyRuntimeError::new_err(native)));
    classifier_error
}

impl<H, M, L> Running<H, M, L>
where
    L: PythonCallHooks + 'static,
    H: PythonBinding + PythonHostCalls<H::Protocol> + 'static,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<ResponseOf<H>>>,
{
    async fn run(mut self) -> PyResult<Py<PyAny>> {
        match self.begin().await {
            Ok(()) => self.execute().await,
            Err(Setup::Host(error)) => self.failure(error, FailureOrigin::Host).await,
            Err(Setup::Decode(InvokeError::Python(error))) => {
                self.failure(error, FailureOrigin::Call).await
            }
            Err(Setup::Decode(InvokeError::Native(error))) => self.machine_failed(error).await,
        }
    }

    async fn begin(&mut self) -> Result<(), Setup<ErrorOf<H>>> {
        let started_at = epoch_seconds();
        let mut parts = self.parts.lock().await;
        parts.observation.started_at = started_at;
        parts.observation.observing = true;
        let started = PythonCallEvent::Started {
            start_time: started_at,
        };
        parts.observation.emit(&started);
        parts
            .hooks
            .on_event(&self.task, started)
            .await
            .map_err(Setup::Host)?;
        let arguments = parts
            .arguments
            .take()
            .ok_or_else(|| Setup::Host(missing_state()))?;
        let prepared = parts
            .hooks
            .prepare_arguments(&self.task, arguments, started_at)
            .await
            .map_err(Setup::Host)?;
        let prepared = parts
            .hooks
            .prepare_request(&self.task, prepared)
            .await
            .map_err(Setup::Host)?;
        parts
            .hooks
            .arguments_prepared(&self.task, &prepared)
            .await
            .map_err(Setup::Host)?;
        self.stream_policy = parts
            .hooks
            .stream_policy(&self.task, &prepared)
            .await
            .map_err(Setup::Host)?;
        let Parts {
            binding, arguments, ..
        } = &mut *parts;
        let decoded = Python::attach(|py| binding.decode_request(py, prepared.bind(py)));
        let arguments = arguments.insert(prepared);
        let request = decoded.map_err(Setup::Decode)?;
        let start = self
            .start
            .take()
            .ok_or_else(|| Setup::Host(missing_state()))?;
        let machine =
            Python::attach(|py| start(py, arguments.bind(py), request)).map_err(Setup::Host)?;
        self.native.start(machine);
        Ok(())
    }

    async fn execute(&mut self) -> PyResult<Py<PyAny>> {
        let mut interruption = None;
        loop {
            let step = match self.native.resume(&self.task, interruption.take()).await {
                Ok(step) => step,
                Err(error) if self.interrupted.is_none() => {
                    interruption = Some(self.interrupt(error));
                    continue;
                }
                Err(error) => return Err(error),
            };
            let op = match step {
                Ok(MachineStep::Suspended(op)) => op,
                Ok(MachineStep::Complete(completion)) => return self.completed(completion).await,
                Err(error) => return self.machine_failed(error).await,
            };
            interruption = match self.answer(op).await {
                Ok(()) => None,
                Err(InvokeError::Native(error)) => Some(HostFailure::Error(error)),
                Err(InvokeError::Python(error)) => Some(self.interrupt(error)),
            };
        }
    }

    async fn answer(
        &mut self,
        op: HostRequest<ProtocolOf<H>>,
    ) -> Result<(), InvokeError<ErrorOf<H>>> {
        match op {
            HostRequest::HostCall(call) => {
                let mut parts = self.parts.lock().await;
                parts.binding.handle_host_call(&self.task, call).await
            }
            HostRequest::Intercept(InterceptRequest::BeforeProviderRequest {
                wire,
                context,
                reply,
            }) => {
                let mut parts = self.parts.lock().await;
                let wire = parts
                    .hooks
                    .before_provider_request(&self.task, wire, &context)
                    .await?;
                reply.send(*wire);
                Ok(())
            }
            HostRequest::Intercept(InterceptRequest::ResultReady { facts, reply }) => {
                self.notify(PythonCallEvent::Execution(ExecutionEvent::ResultReady {
                    facts,
                }))
                .await?;
                reply.send(());
                Ok(())
            }
            HostRequest::Intercept(InterceptRequest::AfterProviderResponse { raw, reply }) => {
                self.notify(PythonCallEvent::Execution(
                    ExecutionEvent::ProviderResponseReceived { raw: &raw },
                ))
                .await?;
                reply.send(());
                Ok(())
            }
            HostRequest::Stream(StreamDelivery::Open(head, reply)) => {
                if self.stream_policy == StreamPolicy::HoldBack {
                    self.held = Some(Vec::new());
                    let ping = {
                        let mut parts = self.parts.lock().await;
                        Python::attach(|py| parts.binding.keepalive_chunk(py))?
                    };
                    self.native.set_keepalive(ping);
                } else {
                    self.streaming = true;
                }
                let head = self.open_head(head).await?;
                reply.send(self.hand_to_consumer(Suspension::Open(head)).await);
                Ok(())
            }
            HostRequest::Stream(StreamDelivery::Chunk(chunk, reply)) => {
                if let Some(held) = &mut self.held {
                    held.push(chunk);
                    reply.send(ControlFlow::Continue(()));
                    return Ok(());
                }
                let chunk = self.deliverable_chunk(chunk).await?;
                reply.send(self.hand_to_consumer(Suspension::Yield(chunk)).await);
                Ok(())
            }
        }
    }

    /// The stream head as the caller receives it, after the hooks saw it.
    async fn open_head(
        &mut self,
        head: <ProtocolOf<H> as Protocol>::StreamHead,
    ) -> PyResult<Py<PyAny>> {
        let mut parts = self.parts.lock().await;
        let Parts { binding, hooks, .. } = &mut *parts;
        let head = Python::attach(|py| binding.encode_stream_head(py, head))?;
        hooks.on_stream_open(&self.task, &head).await?;
        self.opened = true;
        Ok(head)
    }

    /// One chunk as the caller receives it, after the hooks saw it.
    async fn deliverable_chunk(&mut self, chunk: ChunkOf<H>) -> PyResult<Py<PyAny>> {
        let mut parts = self.parts.lock().await;
        let Parts { binding, hooks, .. } = &mut *parts;
        let chunk = Python::attach(|py| binding.encode_chunk(py, chunk))?;
        hooks.on_stream_chunk(&self.task, &chunk).await?;
        Ok(chunk)
    }

    /// Delivers a complete public response to the caller as a stream: the route's replay
    /// of it, chunk by chunk, opening the stream first unless the caller already holds it.
    async fn deliver_as_stream(&mut self, response: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let replay = {
            let mut parts = self.parts.lock().await;
            Python::attach(|py| parts.binding.stream_of_response(py, &response))
        };
        let replay = match replay {
            Ok(replay) => replay,
            Err(error) => return self.failure(error, FailureOrigin::Host).await,
        };
        let head = replay.head;
        let chunks = replay.chunks;
        if !self.opened {
            let head = match self.open_head(head).await {
                Ok(head) => head,
                Err(error) => return self.failure(error, FailureOrigin::Host).await,
            };
            if self
                .hand_to_consumer(Suspension::Open(head))
                .await
                .is_break()
            {
                return self.succeeded(Python::attach(|py| py.None())).await;
            }
        }
        for chunk in chunks {
            let chunk = match self.deliverable_chunk(chunk).await {
                Ok(chunk) => chunk,
                Err(error) => return self.failure(error, FailureOrigin::Host).await,
            };
            if self
                .hand_to_consumer(Suspension::Yield(chunk))
                .await
                .is_break()
            {
                break;
            }
        }
        self.succeeded(Python::attach(|py| py.None())).await
    }

    async fn notify(&mut self, event: PythonCallEvent<'_>) -> PyResult<()> {
        let mut parts = self.parts.lock().await;
        parts.observation.emit(&event);
        parts.hooks.on_event(&self.task, event).await
    }

    /// Hands a stream boundary to the consumer. Reading on is demand for the next chunk;
    /// anything else, closing the stream included, detaches the call from it.
    async fn hand_to_consumer(&mut self, suspension: Suspension) -> ControlFlow<()> {
        match self.task.suspend(suspension).await {
            Ok(_) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        }
    }

    fn interrupt(&mut self, error: PyErr) -> HostFailure<ErrorOf<H>> {
        let (cancelled, native) =
            Python::attach(|py| (is_cancellation(py, &error), H::host_error(&error)));
        self.interrupted = Some(Python::attach(|py| error.into_value(py)));
        if cancelled {
            HostFailure::Cancelled(native)
        } else {
            HostFailure::Error(native)
        }
    }

    async fn completed(
        &mut self,
        completion: HostedCompletion<ResponseOf<H>>,
    ) -> PyResult<Py<PyAny>> {
        let held = self.held.take();
        self.native.set_keepalive(None);
        let transformed = {
            let mut parts = self.parts.lock().await;
            parts.observation.ended_at = Some(epoch_seconds());
            let response = match (completion, held) {
                (HostedCompletion::Complete(response), _) => response,
                (HostedCompletion::StreamEnded, Some(chunks)) => {
                    match Python::attach(|py| parts.binding.assemble_stream(py, chunks)) {
                        Ok(response) => response,
                        Err(InvokeError::Python(error)) => {
                            drop(parts);
                            return self.failure(error, FailureOrigin::Call).await;
                        }
                        Err(InvokeError::Native(error)) => {
                            drop(parts);
                            return self.machine_failed(error).await;
                        }
                    }
                }
                (HostedCompletion::StreamEnded | HostedCompletion::Detached, _) => {
                    drop(parts);
                    return self.succeeded(Python::attach(|py| py.None())).await;
                }
            };
            let public = match Python::attach(|py| parts.binding.encode_response(py, response)) {
                Ok(public) => public,
                Err(error) => {
                    drop(parts);
                    return self.failure(error, FailureOrigin::Call).await;
                }
            };
            if self.streaming {
                Ok(public)
            } else {
                let timing = parts.observation.timing();
                parts
                    .hooks
                    .transform_response(&self.task, public, timing)
                    .await
            }
        };
        let response = match transformed {
            Ok(response) => response,
            Err(error) => return self.failure(error, FailureOrigin::Host).await,
        };
        match self.stream_policy {
            StreamPolicy::Relay => self.succeeded(response).await,
            StreamPolicy::HoldBack | StreamPolicy::Synthesize if !self.streaming => {
                self.deliver_as_stream(response).await
            }
            StreamPolicy::HoldBack | StreamPolicy::Synthesize => self.succeeded(response).await,
        }
    }

    async fn machine_failed(&mut self, error: ErrorOf<H>) -> PyResult<Py<PyAny>> {
        let error = {
            let mut parts = self.parts.lock().await;
            parts.observation.ended_at.get_or_insert_with(epoch_seconds);
            match self.interrupted.take() {
                Some(retained) => {
                    Python::attach(|py| PyErr::from_value(retained.into_bound(py).into_any()))
                }
                None => Python::attach(|py| classified(py, &parts.binding, error)),
            }
        };
        self.failure(error, FailureOrigin::Call).await
    }

    async fn succeeded(&mut self, response: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let mut parts = self.parts.lock().await;
        let event = PythonCallEvent::Succeeded {
            timing: parts.observation.timing(),
            response: &response,
        };
        parts.observation.retain_terminal(&event);
        parts.hooks.on_event(&self.task, event).await?;
        parts.observation.finish();
        Ok(response)
    }

    async fn failure(&mut self, error: PyErr, origin: FailureOrigin) -> PyResult<Py<PyAny>> {
        let mut parts = self.parts.lock().await;
        parts.observation.ended_at.get_or_insert_with(epoch_seconds);
        if Python::attach(|py| is_cancellation(py, &error)) {
            return Err(error);
        }
        let event = PythonCallEvent::Failed {
            timing: parts.observation.timing(),
            origin,
            error: &error,
        };
        parts.observation.retain_terminal(&event);
        parts.hooks.on_event(&self.task, event).await?;
        parts.observation.finish();
        Err(error)
    }
}

struct PythonDriver<H, L>
where
    H: PythonBinding,
    L: PythonCallHooks,
{
    task: PythonTask,
    body: Mutex<Option<Body>>,
    parts: SharedParts<H, L>,
}

impl<H, L> PythonDriver<H, L>
where
    H: PythonBinding + 'static,
    L: PythonCallHooks + 'static,
{
    fn body(&mut self) -> &mut Option<Body> {
        self.body.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    /// A failure the body raised past its terminal hooks, or a cancellation, is the last
    /// thing the observers hear about the call.
    fn observe_host_failure(&self, error: &PyErr) {
        let Ok(mut parts) = self.parts.try_lock() else {
            return;
        };
        if !parts.observation.observing {
            return;
        }
        let timing = parts.observation.timing();
        Python::attach(|py| {
            let event = if is_cancellation(py, error) {
                PythonCallEvent::Cancelled { timing }
            } else {
                PythonCallEvent::Failed {
                    timing,
                    origin: FailureOrigin::Host,
                    error,
                }
            };
            parts.observation.emit(&event);
        });
        parts.observation.observing = false;
        parts.observation.terminal = None;
    }
}

impl<H, L> ExecutionBody for PythonDriver<H, L>
where
    H: PythonBinding + 'static,
    L: PythonCallHooks + 'static,
{
    fn resume(&mut self, answer: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        let task = self.task.clone();
        let body = self.body();
        let Some(running) = body.as_mut() else {
            return Err(missing_state());
        };
        let outcome = match task.resume(running.as_mut(), answer) {
            Ok(Resumed::Suspended(suspension)) => return Ok(ExecutionStep::Suspend(suspension)),
            Ok(Resumed::Complete(outcome)) => outcome,
            Err(error) => Err(error),
        };
        *body = None;
        match outcome {
            Ok(value) => Ok(ExecutionStep::Return(value)),
            Err(error) => {
                self.observe_host_failure(&error);
                Err(error)
            }
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Ok(parts) = self.parts.try_lock() else {
            return Ok(());
        };
        parts.binding.traverse(visit)?;
        parts.hooks.traverse(visit)?;
        visit.call(&parts.arguments)
    }
}

impl<H, L> Drop for PythonDriver<H, L>
where
    H: PythonBinding,
    L: PythonCallHooks,
{
    fn drop(&mut self) {
        *self.body.get_mut().unwrap_or_else(PoisonError::into_inner) = None;
        let Ok(mut parts) = self.parts.try_lock() else {
            return;
        };
        Python::attach(|py| {
            if parts.observation.observing {
                let timing = parts.observation.timing();
                parts
                    .observation
                    .emit(&PythonCallEvent::Cancelled { timing });
                parts.observation.observing = false;
            }
            parts.hooks.close(py);
            parts.binding.close(py);
        });
    }
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
    let task = PythonTask::new(options.asynchronous);
    let parts = Arc::new(PartsMutex::new(Parts {
        binding,
        hooks,
        arguments: Some(arguments),
        observation: Observation {
            observers: options.observers,
            observing: false,
            started_at: 0.0,
            ended_at: None,
            terminal: None,
        },
    }));
    let running = Running {
        task: task.clone(),
        parts: Arc::clone(&parts),
        native: NativeMachine::new(options.asynchronous),
        start: Some(Box::new(start)),
        interrupted: None,
        streaming: false,
        opened: false,
        stream_policy: StreamPolicy::Relay,
        held: None,
    };
    let mut driver = PythonDriver {
        task,
        body: Mutex::new(Some(Box::pin(running.run()))),
        parts,
    };
    if options.asynchronous {
        return Execution::new(driver, options.lifecycle)
            .into_coroutine(py)
            .map(Bound::unbind);
    }
    match driver.resume(None)? {
        ExecutionStep::Return(value) => Ok(value),
        ExecutionStep::Suspend(Suspension::Open(head)) => {
            Execution::suspended(driver, options.lifecycle)
                .into_sync_stream(py, head)
                .map(Bound::unbind)
        }
        ExecutionStep::Suspend(
            Suspension::Await(_) | Suspension::Keepalive { .. } | Suspension::Yield(_),
        ) => Err(PyRuntimeError::new_err("sync call suspended")),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use litellm_host::{
        hooks::CallHooks,
        interceptors::{Interceptors, RawResponse, RequestContext, WireRequest},
        machine::{CallMachine, MachineFault},
        protocol::Reply,
    };
    use pyo3::exceptions::{PyBaseException, PyValueError};
    use pyo3::types::PyDict;

    use super::*;
    use crate::{PythonOwned, PythonRuntime};

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
        AwaitAnswer,
        AwaitFailure,
        AwaitTwice,
        AwaitCancellation,
        AwaitNativeFailure,
    }

    impl OpScript {
        fn awaits(self) -> bool {
            matches!(
                self,
                Self::AwaitAnswer
                    | Self::AwaitFailure
                    | Self::AwaitTwice
                    | Self::AwaitCancellation
                    | Self::AwaitNativeFailure
            )
        }
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
                OpScript::Answer
                | OpScript::AwaitAnswer
                | OpScript::AwaitFailure
                | OpScript::AwaitTwice
                | OpScript::AwaitCancellation
                | OpScript::AwaitNativeFailure => Ok(value()),
                OpScript::RaisePython | OpScript::RaiseRequestPython => {
                    Err(PyValueError::new_err("op failed").into())
                }
                OpScript::RejectNatively | OpScript::RejectRequestNatively => {
                    Err(InvokeError::Native(Error("op rejected".into())))
                }
            }
        }

        fn host_op<'py>(&self, py: Python<'py>, function: &str) -> PyResult<Bound<'py, PyAny>> {
            PyModule::from_code(
                py,
                pyo3::ffi::c_str!(
                    "import asyncio\nasync def answer(fail, cancel):\n    if cancel:\n        raise asyncio.CancelledError('host cancelled')\n    if fail:\n        raise LookupError('async host failed')\n    return 'awaited'\nasync def echo(value):\n    return value\n"
                ),
                pyo3::ffi::c_str!("host_op.py"),
                pyo3::ffi::c_str!("host_op"),
            )?
            .getattr(function)
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
        async fn handle_host_call(
            &mut self,
            task: &PythonTask,
            (op, reply): (&'static str, Reply<String>),
        ) -> Result<(), InvokeError<Error>> {
            self.log.push(format!("op:{op}"));
            if !self.op.awaits() {
                return self
                    .answer(|| op.to_string())
                    .map(|answer| reply.send(answer));
            }
            let awaitable = Python::attach(|py| {
                self.host_op(py, "answer")?
                    .call1((
                        matches!(self.op, OpScript::AwaitFailure),
                        matches!(self.op, OpScript::AwaitCancellation),
                    ))
                    .map(Bound::unbind)
            })?;
            let result = task.settle(Ok(awaitable)).await?;
            self.log.push("host.resume");
            if matches!(self.op, OpScript::AwaitNativeFailure) {
                return Err(InvokeError::Native(Error("awaited rejection".into())));
            }
            let answer = Python::attach(|py| result?.extract::<String>(py))?;
            if !matches!(self.op, OpScript::AwaitTwice) {
                reply.send(answer);
                return Ok(());
            }
            let echoed = Python::attach(|py| {
                self.host_op(py, "echo")?
                    .call1((answer,))
                    .map(Bound::unbind)
            })?;
            let result = task.settle(Ok(echoed)).await?;
            self.log.push("host.resume.again");
            reply.send(Python::attach(|py| result?.extract::<String>(py))?);
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

    #[derive(Clone, Copy)]
    enum HookScript {
        Plain,
        RewriteArguments,
        RewriteRequest,
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
        async fn prepare_arguments(
            &mut self,
            _: &PythonTask,
            arguments: Py<PyDict>,
            _: f64,
        ) -> PyResult<Py<PyDict>> {
            self.log.push("begin");
            if matches!(self.script, HookScript::FailBegin) {
                return Err(PyValueError::new_err("begin failed"));
            }
            if matches!(self.script, HookScript::RewriteArguments) {
                return Python::attach(|py| {
                    let prepared = arguments.bind(py).copy()?;
                    prepared.set_item("prepared", "hook")?;
                    prepared.set_item("api_key", "hook-key")?;
                    Ok(prepared.unbind())
                });
            }
            Ok(arguments)
        }

        async fn prepare_request(
            &mut self,
            _: &PythonTask,
            arguments: Py<PyDict>,
        ) -> PyResult<Py<PyDict>> {
            if !matches!(self.script, HookScript::RewriteRequest) {
                return Ok(arguments);
            }
            self.log.push("request");
            Python::attach(|py| {
                let prepared = arguments.bind(py).copy()?;
                prepared.set_item("prepared", "request")?;
                prepared.set_item("requested", "yes")?;
                Ok(prepared.unbind())
            })
        }

        async fn arguments_prepared(
            &mut self,
            _: &PythonTask,
            arguments: &Py<PyDict>,
        ) -> PyResult<()> {
            if matches!(self.script, HookScript::ObserveArguments) {
                let value: String = Python::attach(|py| {
                    arguments.bind(py).get_item("prepared")?.unwrap().extract()
                })?;
                self.log.push(format!("adopted:{value}"));
            }
            Ok(())
        }

        async fn before_provider_request(
            &mut self,
            _: &PythonTask,
            wire: Box<WireRequest>,
            _: &RequestContext,
        ) -> PyResult<Box<WireRequest>> {
            self.log.push("before_provider_request");
            Ok(Box::new(WireRequest {
                url: "rewritten".into(),
                ..*wire
            }))
        }

        async fn transform_response(
            &mut self,
            _: &PythonTask,
            response: Py<PyAny>,
            _: Timing,
        ) -> PyResult<Py<PyAny>> {
            self.log.push("after_success");
            match self.script {
                HookScript::ReplaceResponse => {
                    Python::attach(|py| Ok("replaced".into_pyobject(py)?.into_any().unbind()))
                }
                HookScript::FailAfterSuccess => Err(PyValueError::new_err("after_success failed")),
                HookScript::Plain
                | HookScript::ObserveArguments
                | HookScript::RewriteArguments
                | HookScript::RewriteRequest
                | HookScript::FailBegin
                | HookScript::CancelTerminal
                | HookScript::FailTerminal => Ok(response),
            }
        }

        async fn on_event(&mut self, _: &PythonTask, event: PythonCallEvent<'_>) -> PyResult<()> {
            let terminal = matches!(
                event,
                PythonCallEvent::Succeeded { .. } | PythonCallEvent::Failed { .. }
            );
            if terminal && matches!(self.script, HookScript::CancelTerminal) {
                return Err(pyo3::exceptions::asyncio::CancelledError::new_err(
                    "callback cancelled",
                ));
            }
            if terminal && matches!(self.script, HookScript::FailTerminal) {
                return Err(PyValueError::new_err("callback failed"));
            }
            let entry = Python::attach(|py| match event {
                PythonCallEvent::Execution(ExecutionEvent::ResultReady { .. }) => {
                    "cache_hit".to_string()
                }
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
            self.log.push(entry);
            Ok(())
        }

        async fn on_stream_open(&mut self, _: &PythonTask, _: &Py<PyAny>) -> PyResult<()> {
            self.log.push("opened");
            Ok(())
        }

        async fn on_stream_chunk(&mut self, _: &PythonTask, _: &Py<PyAny>) -> PyResult<()> {
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
    #[case::success(OpScript::AwaitAnswer)]
    #[case::failure(OpScript::AwaitFailure)]
    #[case::repeated_suspension(OpScript::AwaitTwice)]
    #[case::cancellation(OpScript::AwaitCancellation)]
    #[case::native_failure(OpScript::AwaitNativeFailure)]
    fn asynchronous_host_operations_resume_the_same_machine(#[case] op: OpScript) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Python::initialize();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let (result, log) = run_scripted(
                py,
                |_| {
                    CallMachine::<Synthetic>::new(None, |host| {
                        Box::pin(async move { host.services.call(|reply| ("read", reply)).await })
                    })
                },
                op,
                HookScript::Plain,
                true,
            );
            match op {
                OpScript::AwaitAnswer | OpScript::AwaitTwice => {
                    assert_eq!(result.unwrap().extract::<String>(py).unwrap(), "awaited");
                    assert_eq!(
                        log.iter().filter(|entry| *entry == "host.resume").count(),
                        1
                    );
                    assert_eq!(
                        log.iter()
                            .filter(|entry| *entry == "host.resume.again")
                            .count(),
                        usize::from(matches!(op, OpScript::AwaitTwice))
                    );
                }
                OpScript::AwaitNativeFailure => {
                    assert_eq!(
                        result.unwrap_err().value(py).to_string(),
                        "classified: awaited rejection"
                    );
                    assert_eq!(
                        log.iter()
                            .filter(|entry| entry.starts_with("classify:"))
                            .count(),
                        1
                    );
                    assert_eq!(
                        log.iter()
                            .filter(|entry| entry.starts_with("failed:"))
                            .count(),
                        1
                    );
                    assert!(!log.iter().any(|entry| entry.starts_with("succeeded:")));
                }
                OpScript::AwaitCancellation => {
                    let error = result.unwrap_err();
                    assert!(error.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py));
                    assert_eq!(error.value(py).to_string(), "host cancelled");
                    assert!(!log.iter().any(|entry| entry.starts_with("host.resume")
                        || entry.starts_with("failed:")
                        || entry.starts_with("succeeded:")
                        || entry == "after_success"));
                    assert_eq!(&log[log.len() - 2..], ["adapter.close", "host.close"]);
                }
                OpScript::AwaitFailure => {
                    assert!(
                        result
                            .unwrap_err()
                            .is_instance_of::<pyo3::exceptions::PyLookupError>(py)
                    );
                    assert_eq!(
                        log.iter()
                            .filter(|entry| entry.starts_with("failed:"))
                            .count(),
                        1
                    );
                    assert!(!log.iter().any(|entry| entry.starts_with("succeeded:")));
                }
                _ => unreachable!(),
            }
        });
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
            let hooks = crate::HookChain::new()
                .with(hook(HookScript::ObserveArguments))
                .with(hook(HookScript::RewriteArguments))
                .with(hook(HookScript::ReplaceResponse));
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

    /// The request stage runs after every hook's argument stage, whatever the chain order,
    /// and the binding decodes the view it produced.
    #[rstest::rstest]
    #[case::synchronous(false)]
    #[case::asynchronous(true)]
    fn request_rewrites_follow_every_argument_rewrite_and_reach_the_binding(
        #[case] asynchronous: bool,
    ) {
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
            let hooks = crate::HookChain::new()
                .with(hook(HookScript::RewriteRequest))
                .with(hook(HookScript::ObserveArguments))
                .with(hook(HookScript::RewriteArguments));
            let (result, entries) = run_composed(
                py,
                success_machine(),
                SyntheticBinding {
                    log: Log(log.0.clone()),
                    op: OpScript::Answer,
                    classifier_fails: false,
                },
                HookScript::Plain,
                move |_| hooks,
                call_options(asynchronous),
            );
            assert_eq!(
                result.unwrap().extract::<String>(py).unwrap(),
                "project:4|sign|rewritten"
            );
            let stages: Vec<&str> = entries
                .iter()
                .map(String::as_str)
                .filter(|entry| {
                    matches!(*entry, "begin" | "request" | "project")
                        || entry.starts_with("adopted:")
                })
                .collect();
            assert_eq!(
                stages,
                [
                    "begin",
                    "begin",
                    "begin",
                    "request",
                    "adopted:request",
                    "project"
                ]
            );
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
        async fn handle_host_call(
            &mut self,
            _: &PythonTask,
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
    /// A route whose complete response is the chunks joined with `+`, and whose replay of a
    /// response splits it on `+` again, so what the caller streams shows exactly what the
    /// hooks saw and changed.
    struct Held;

    impl Protocol for Held {
        type Response = String;
        type Error = Error;
        type Request = ();
        type HostCall = std::convert::Infallible;
        type Chunk = &'static str;
        type StreamHead = Vec<(&'static str, &'static str)>;
    }

    struct HeldBinding;

    impl PythonBinding for HeldBinding {
        type Protocol = Held;
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
            Ok(headers.into_any().unbind())
        }

        fn encode_chunk(&mut self, py: Python<'_>, chunk: &'static str) -> PyResult<Py<PyAny>> {
            Ok(pyo3::types::PyString::new(py, chunk).into_any().unbind())
        }

        fn encode_response(&mut self, py: Python<'_>, response: String) -> PyResult<Py<PyAny>> {
            Ok(pyo3::types::PyString::new(py, &response)
                .into_any()
                .unbind())
        }

        fn assemble_stream(
            &mut self,
            _: Python<'_>,
            chunks: Vec<&'static str>,
        ) -> Result<String, InvokeError<Error>> {
            Ok(chunks.join("+"))
        }

        fn stream_of_response(
            &mut self,
            py: Python<'_>,
            response: &Py<PyAny>,
        ) -> PyResult<crate::StreamReplay<Self::Protocol>> {
            let text: String = response.extract(py)?;
            let chunks = text
                .split('+')
                .map(|piece| &*Box::leak(piece.to_owned().into_boxed_str()))
                .collect();
            Ok(crate::StreamReplay {
                head: vec![("replayed", "true")],
                chunks,
            })
        }

        fn keepalive_chunk(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
            Ok(Some(
                pyo3::types::PyString::new(py, "ping").into_any().unbind(),
            ))
        }

        fn map_error(&self, _: Python<'_>, error: Error) -> PyResult<Classified> {
            Ok(Classified(error.0))
        }

        fn host_error(error: &PyErr) -> Error {
            Error(error.to_string())
        }
    }

    impl PythonHostCalls<Held> for HeldBinding {
        async fn handle_host_call(
            &mut self,
            _: &PythonTask,
            op: std::convert::Infallible,
        ) -> Result<(), InvokeError<Error>> {
            match op {}
        }
    }

    impl PythonOwned for HeldBinding {
        fn close(&mut self, _: Python<'_>) {}
        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    /// Hooks that pick a stream policy and mark every response they transform with `!`.
    struct PolicyHooks {
        log: Log,
        policy: StreamPolicy,
    }

    impl CallHooks<PythonRuntime> for PolicyHooks {
        async fn stream_policy(
            &mut self,
            _: &PythonTask,
            _: &Py<PyDict>,
        ) -> PyResult<StreamPolicy> {
            Ok(self.policy)
        }

        async fn transform_response(
            &mut self,
            _: &PythonTask,
            response: Py<PyAny>,
            _: Timing,
        ) -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                let text: String = response.extract(py)?;
                self.log.push(format!("transform:{text}"));
                Ok(pyo3::types::PyString::new(py, &format!("{text}!"))
                    .into_any()
                    .unbind())
            })
        }

        async fn on_event(&mut self, _: &PythonTask, event: PythonCallEvent<'_>) -> PyResult<()> {
            if let PythonCallEvent::Succeeded { response, .. } = event {
                self.log.push(Python::attach(|py| {
                    format!("succeeded:{}", response.bind(py))
                }));
            }
            Ok(())
        }

        async fn on_stream_open(&mut self, _: &PythonTask, head: &Py<PyAny>) -> PyResult<()> {
            self.log
                .push(Python::attach(|py| format!("opened:{}", head.bind(py))));
            Ok(())
        }

        async fn on_stream_chunk(&mut self, _: &PythonTask, chunk: &Py<PyAny>) -> PyResult<()> {
            self.log
                .push(Python::attach(|py| format!("delivered:{}", chunk.bind(py))));
            Ok(())
        }
    }

    impl PythonOwned for PolicyHooks {
        fn close(&mut self, _: Python<'_>) {}
        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    fn held_stream_machine()
    -> impl FnOnce(()) -> litellm_host::call::HostedMachine<Held> + Send + Sync {
        move |()| {
            litellm_host::call::hosted_call((), None, |(), _, _, _| async {
                Ok(litellm_host::call::CallOutput::Stream {
                    head: vec![("request-id", "req_1")],
                    chunks: Box::pin(futures_util::stream::iter([Ok("first"), Ok("second")])),
                })
            })
        }
    }

    fn held_complete_machine()
    -> impl FnOnce(()) -> litellm_host::call::HostedMachine<Held> + Send + Sync {
        move |()| {
            litellm_host::call::hosted_call((), None, |(), _, _, _| async {
                Ok(litellm_host::call::CallOutput::Complete("a+b".to_string()))
            })
        }
    }

    fn run_held(
        py: Python<'_>,
        machine: impl FnOnce(()) -> litellm_host::call::HostedMachine<Held> + Send + Sync + 'static,
        policy: StreamPolicy,
        asynchronous: bool,
    ) -> (Py<PyAny>, Log) {
        install_lifecycle_module(py);
        let log = Log::default();
        let handed = run_call(
            py,
            move |_, _, request| Ok(machine(request)),
            HeldBinding,
            PolicyHooks {
                log: Log(log.0.clone()),
                policy,
            },
            PyDict::new(py).unbind(),
            call_options(asynchronous),
        )
        .unwrap();
        let handed = if asynchronous {
            let completion = handed.call_method1(py, "send", (py.None(),)).unwrap_err();
            assert!(completion.is_instance_of::<pyo3::exceptions::PyStopIteration>(py));
            completion.value(py).getattr("value").unwrap().unbind()
        } else {
            handed
        };
        (handed, log)
    }

    /// A held-back stream reaches the caller as the replay of the transformed response: the
    /// provider's chunks are never delivered, the hooks transform what they assembled to,
    /// and the stream hooks see what the caller got.
    #[rstest::rstest]
    #[case::sync(false)]
    #[case::asynchronous(true)]
    fn a_held_back_stream_delivers_the_transformed_replay_instead_of_the_provider_chunks(
        #[case] asynchronous: bool,
    ) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (stream, log) = run_held(
                py,
                held_stream_machine(),
                StreamPolicy::HoldBack,
                asynchronous,
            );
            let head = stream.getattr(py, "head").unwrap();
            assert_eq!(head.bind(py).to_string(), "{'request-id': 'req_1'}");
            assert_eq!(
                read_all(py, stream.bind(py), asynchronous),
                ["first", "second!"]
            );
            assert_eq!(
                log.entries(),
                [
                    "opened:{'request-id': 'req_1'}",
                    "transform:first+second",
                    "delivered:first",
                    "delivered:second!",
                    "succeeded:None",
                ]
            );
        });
    }

    /// A caller who asked for a stream gets one even when the hooks made the provider call
    /// complete: the complete response is transformed, then replayed as chunks.
    #[rstest::rstest]
    #[case::sync(false)]
    #[case::asynchronous(true)]
    fn a_synthesized_stream_replays_the_transformed_complete_response(#[case] asynchronous: bool) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (stream, log) = run_held(
                py,
                held_complete_machine(),
                StreamPolicy::Synthesize,
                asynchronous,
            );
            let head = stream.getattr(py, "head").unwrap();
            assert_eq!(head.bind(py).to_string(), "{'replayed': 'true'}");
            assert_eq!(read_all(py, stream.bind(py), asynchronous), ["a", "b!"]);
            assert_eq!(
                log.entries(),
                [
                    "transform:a+b",
                    "opened:{'replayed': 'true'}",
                    "delivered:a",
                    "delivered:b!",
                    "succeeded:None",
                ]
            );
        });
    }

    /// Relaying is untouched by the policy stage: live chunks reach the caller and no
    /// response transformation runs.
    #[rstest::rstest]
    #[case::sync(false)]
    #[case::asynchronous(true)]
    fn a_relayed_stream_runs_no_response_transformation(#[case] asynchronous: bool) {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (stream, log) =
                run_held(py, held_stream_machine(), StreamPolicy::Relay, asynchronous);
            assert_eq!(
                read_all(py, stream.bind(py), asynchronous),
                ["first", "second"]
            );
            assert!(
                !log.entries()
                    .iter()
                    .any(|entry| entry.starts_with("transform:"))
            );
            assert_eq!(
                log.entries(),
                [
                    "opened:{'request-id': 'req_1'}",
                    "delivered:first",
                    "delivered:second",
                    "succeeded:None",
                ]
            );
        });
    }

    /// Closing a held-back stream before it delivers runs no transformation and reports
    /// success for nothing delivered, exactly as closing a relayed stream does.
    #[rstest::rstest]
    fn closing_a_held_back_stream_early_delivers_nothing() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            let (stream, log) = run_held(py, held_stream_machine(), StreamPolicy::HoldBack, false);
            stream.call_method0(py, "close").unwrap();
            assert_eq!(
                log.entries(),
                ["opened:{'request-id': 'req_1'}", "succeeded:None"]
            );
        });
    }

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
        async fn arguments_prepared(
            &mut self,
            _: &PythonTask,
            arguments: &Py<PyDict>,
        ) -> PyResult<()> {
            Python::attach(|py| match self {
                Self::Inherit => arguments.bind(py).set_item("api_key", "inherited"),
                Self::Reject(error) => Err(PyErr::from_value(error.bind(py).clone().into_any())),
            })
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
                crate::HookChain::new()
                    .with(SyntheticHooks {
                        log: Log(log.0.clone()),
                        script: HookScript::RewriteArguments,
                    })
                    .with(ArgumentPolicy::Inherit),
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
                |hooks| {
                    crate::HookChain::new()
                        .with(hooks)
                        .with(ArgumentPolicy::Reject(raised.clone_ref(py)))
                },
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
                |hooks| {
                    crate::HookChain::new()
                        .with(hooks)
                        .with(ArgumentPolicy::Inherit)
                },
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
                async fn handle_host_call(
                    &mut self,
                    _: &PythonTask,
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
                Some(awaitable) => Ok(ExecutionStep::Suspend(Suspension::Await(awaitable))),
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

    struct Retaining {
        retained: Option<Py<PyAny>>,
        log: Log,
    }

    impl CallHooks<PythonRuntime> for Retaining {}

    impl PythonOwned for Retaining {
        fn close(&mut self, _: Python<'_>) {
            self.log.push("retaining.close");
            self.retained = None;
        }

        fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            visit.call(&self.retained)
        }
    }

    /// A stream the caller abandons is a cycle: the stream owns the execution, a hook
    /// retains an object that owns the stream. The handle traverses the hooks while the
    /// call waits at the stream boundary, so the collector finds the cycle and closes it.
    #[rstest::rstest]
    fn an_abandoned_stream_retained_by_a_hook_is_collected() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        crate::initialize_python();
        Python::attach(|py| {
            install_lifecycle_module(py);
            let log = Log::default();
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!("import gc, weakref\nclass Retained:\n    pass\nretained = Retained()\nreference = weakref.ref(retained)\n"),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let retained = locals.get_item("retained").unwrap().unwrap().unbind();
            let handed = run_call(
                py,
                |_, _, request| Ok(streaming_machine()(request)),
                StreamingBinding,
                crate::HookChain::new().with(Retaining {
                    retained: Some(retained),
                    log: Log(log.0.clone()),
                }),
                PyDict::new(py).unbind(),
                call_options(true),
            )
            .unwrap();
            let stop = handed.call_method1(py, "send", (py.None(),)).unwrap_err();
            let stream = stop.value(py).getattr("value").unwrap();
            locals.set_item("stream", stream).unwrap();
            drop(stop);
            drop(handed);
            py.run(
                pyo3::ffi::c_str!("retained.stream = stream\ndel retained, stream\ngc.collect()\nassert reference() is None\n"),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            assert_eq!(log.entries(), ["retaining.close"]);
        });
    }
}
