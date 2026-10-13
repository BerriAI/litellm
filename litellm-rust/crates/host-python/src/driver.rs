use std::ops::ControlFlow;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::{Mutex, MutexGuard};
use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use litellm_coroutine::{Coroutine, CoroutineState};
use litellm_host::{
    call::HostedCompletion,
    lifecycle::{CallEvent, ExecutionEvent, FailureOrigin, Timing, epoch_seconds},
    machine::{HostFailure, Machine, MachineStep},
    observation::ObservationSender,
    protocol::{HostRequest, InterceptRequest, Protocol, Reply, StreamDelivery},
};

use crate::PythonHostCalls;
use crate::handle::{Execution, ExecutionBody, ExecutionStep, PythonLifecycle, Suspension};
use crate::hooks::{HookStep, PythonCallEvent, PythonCallHooks};
use crate::native::{NativeMachine, NativePoll};
use crate::task::{Answer, PythonTask, Yielded};
use crate::{InvokeError, PythonBinding, missing_state};

type ProtocolOf<H> = <H as PythonBinding>::Protocol;
type ErrorOf<H> = <ProtocolOf<H> as Protocol>::Error;
type ResponseOf<H> = <ProtocolOf<H> as Protocol>::Response;
type Interruption<H> = Option<HostFailure<ErrorOf<H>>>;
type StartMachine<P, M> = Box<
    dyn FnOnce(Python<'_>, &Bound<'_, PyDict>, <P as Protocol>::Request) -> PyResult<M>
        + Send
        + Sync,
>;
type Body = Coroutine<Yielded, PyResult<Py<PyAny>>>;

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

/// What the running call shares with its handle: the binding and hooks the handle
/// traverses and closes, and the Python values the call retains across a suspension.
struct Parts<H, L> {
    binding: H,
    hooks: L,
    arguments: Option<Py<PyDict>>,
    interrupted: Option<Py<PyBaseException>>,
    selected: Option<Py<PyAny>>,
    observation: Observation,
}

type SharedParts<H, L> = Arc<Mutex<Parts<H, L>>>;

// parking_lot's send_guard would let the body keep parts locked while suspended, hiding them from GC traverse
static_assertions::assert_not_impl_any!(MutexGuard<'static, ()>: Send);

enum Setup<E> {
    Host(PyErr),
    Decode(InvokeError<E>),
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

/// The call as one async body. Every hook step and host call is awaited in place on the
/// caller's task; the parts lock is never held across an await.
struct Running<H, M, L>
where
    H: PythonBinding,
    M: Machine,
{
    task: PythonTask,
    parts: SharedParts<H, L>,
    native: NativeMachine<M>,
    start: Option<StartMachine<ProtocolOf<H>, M>>,
    streaming: bool,
}

impl<H, M, L> Running<H, M, L>
where
    L: PythonCallHooks + 'static,
    H: PythonBinding + PythonHostCalls<H::Protocol> + 'static,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<ResponseOf<H>>>,
{
    fn parts(&self) -> MutexGuard<'_, Parts<H, L>> {
        self.parts.lock()
    }

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
        let started = {
            let mut parts = self.parts();
            parts.observation.started_at = started_at;
            parts.observation.observing = true;
            let event = PythonCallEvent::Started {
                start_time: started_at,
            };
            parts.observation.emit(&event);
            Python::attach(|py| parts.hooks.on_event(py, event)).map_err(Setup::Host)?
        };
        self.settle(started).await.map_err(Setup::Host)?;
        let arguments = self
            .parts()
            .arguments
            .take()
            .ok_or_else(|| Setup::Host(missing_state()))?;
        let prepare = {
            let mut parts = self.parts();
            Python::attach(|py| parts.hooks.prepare_arguments(py, arguments, started_at))
                .map_err(Setup::Host)?
        };
        let prepared = self.settle(prepare).await.map_err(Setup::Host)?;
        let request = {
            let mut parts = self.parts();
            let Parts {
                binding,
                hooks,
                arguments,
                ..
            } = &mut *parts;
            Python::attach(|py| {
                hooks
                    .arguments_prepared(py, &prepared)
                    .map_err(Setup::Host)?;
                let decoded = binding.decode_request(py, prepared.bind(py));
                *arguments = Some(prepared);
                decoded.map_err(Setup::Decode)
            })?
        };
        let start = self
            .start
            .take()
            .ok_or_else(|| Setup::Host(missing_state()))?;
        let machine = {
            let parts = self.parts();
            let arguments = parts
                .arguments
                .as_ref()
                .ok_or_else(|| Setup::Host(missing_state()))?;
            Python::attach(|py| start(py, arguments.bind(py), request)).map_err(Setup::Host)?
        };
        self.native.start(machine);
        Ok(())
    }

    /// Runs one hook step to its value, handing each awaitable it produces to the caller.
    async fn settle<T>(&self, step: HookStep<L, T>) -> PyResult<T> {
        let mut step = step;
        loop {
            let (awaitable, resume) = match step {
                HookStep::Ready(value) => return Ok(value),
                HookStep::Await(awaitable, resume) => (awaitable, resume),
            };
            let answer = self.task.suspend(Suspension::Await(awaitable)).await;
            step = {
                let mut parts = self.parts();
                Python::attach(|py| resume(&mut parts.hooks, py, answer))?
            };
        }
    }

    async fn execute(&mut self) -> PyResult<Py<PyAny>> {
        let mut interruption: Interruption<H> = None;
        loop {
            let poll = Python::attach(|py| self.native.resume(py, interruption.take()))?;
            let result = match poll {
                NativePoll::Ready(result) => result,
                NativePoll::Suspend(awaitable) => {
                    match self.task.suspend(Suspension::Await(awaitable)).await {
                        Ok(_) => self.native.take_result()?,
                        Err(error) if self.parts().interrupted.is_none() => {
                            interruption = Some(self.interrupt(error));
                            continue;
                        }
                        Err(error) => return Err(error),
                    }
                }
            };
            let op = match result {
                Ok(MachineStep::Suspended(op)) => op,
                Ok(MachineStep::Complete(completion)) => {
                    return self.completed(completion).await;
                }
                Err(error) => return self.machine_failed(error).await,
            };
            interruption = match self.answer(op).await {
                Ok(()) => None,
                Err(InvokeError::Native(error)) => Some(HostFailure::Error(error)),
                Err(InvokeError::Python(error)) => Some(self.interrupt(error)),
            };
        }
    }

    fn interrupt(&self, error: PyErr) -> HostFailure<ErrorOf<H>> {
        Python::attach(|py| {
            let cancelled = is_cancellation(py, &error);
            let native = H::host_error(&error);
            self.parts().interrupted = Some(error.into_value(py));
            if cancelled {
                HostFailure::Cancelled(native)
            } else {
                HostFailure::Error(native)
            }
        })
    }

    /// Answers one machine op. A Python error interrupts the call as raised, a native
    /// rejection resumes the machine with it.
    async fn answer(
        &mut self,
        op: HostRequest<ProtocolOf<H>>,
    ) -> Result<(), InvokeError<ErrorOf<H>>> {
        match op {
            HostRequest::HostCall(call) => self.host_call(call).await,
            HostRequest::Intercept(InterceptRequest::BeforeProviderRequest {
                wire,
                context,
                reply,
            }) => {
                let step = {
                    let mut parts = self.parts();
                    Python::attach(|py| parts.hooks.before_provider_request(py, wire, &context))?
                };
                let wire = self.settle(step).await?;
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
                self.streaming = true;
                let head = {
                    let mut parts = self.parts();
                    let Parts { binding, hooks, .. } = &mut *parts;
                    Python::attach(|py| {
                        let head = binding.encode_stream_head(py, head)?;
                        hooks.on_stream_open(py, &head)?;
                        Ok::<_, PyErr>(head)
                    })?
                };
                self.consumed(Suspension::Open(head), reply).await;
                Ok(())
            }
            HostRequest::Stream(StreamDelivery::Chunk(chunk, reply)) => {
                let chunk = {
                    let mut parts = self.parts();
                    let Parts { binding, hooks, .. } = &mut *parts;
                    Python::attach(|py| {
                        let chunk = binding.encode_chunk(py, chunk)?;
                        hooks.on_stream_chunk(py, &chunk)?;
                        Ok::<_, PyErr>(chunk)
                    })?
                };
                self.consumed(Suspension::Yield(chunk), reply).await;
                Ok(())
            }
        }
    }

    async fn host_call(
        &self,
        call: <ProtocolOf<H> as Protocol>::HostCall,
    ) -> Result<(), InvokeError<ErrorOf<H>>> {
        let mut pending = {
            let mut parts = self.parts();
            Python::attach(|py| parts.binding.begin_host_call(py, call))?
        };
        while let Some(awaitable) = pending {
            let result = self.task.suspend(Suspension::Await(awaitable)).await;
            pending = {
                let mut parts = self.parts();
                Python::attach(|py| parts.binding.resume_host_call(py, result))?
            };
        }
        Ok(())
    }

    async fn notify(&self, event: PythonCallEvent<'_>) -> PyResult<()> {
        let step = {
            let mut parts = self.parts();
            parts.observation.emit(&event);
            Python::attach(|py| parts.hooks.on_event(py, event))?
        };
        self.settle(step).await
    }

    /// Hands a stream boundary to the consumer. Reading on is demand for the next chunk;
    /// anything else, closing the stream included, detaches the call from it.
    async fn consumed(&self, suspension: Suspension, reply: Reply<ControlFlow<()>>) {
        let read = self.task.suspend(suspension).await;
        reply.send(if read.is_ok() {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        });
    }

    async fn completed(
        &mut self,
        completion: HostedCompletion<ResponseOf<H>>,
    ) -> PyResult<Py<PyAny>> {
        self.parts().observation.ended_at = Some(epoch_seconds());
        let response = match completion {
            HostedCompletion::Complete(response) => response,
            HostedCompletion::StreamEnded | HostedCompletion::Detached => {
                let none = Python::attach(|py| py.None());
                return self.succeeded(none).await;
            }
        };
        let encoded = {
            let mut parts = self.parts();
            Python::attach(|py| parts.binding.encode_response(py, response))
        };
        let public = match encoded {
            Ok(public) => public,
            Err(error) => return self.failure(error, FailureOrigin::Call).await,
        };
        if self.streaming {
            return self.succeeded(public).await;
        }
        let transform = {
            let mut parts = self.parts();
            let timing = parts.observation.timing();
            Python::attach(|py| parts.hooks.transform_response(py, public, timing))
        };
        let step = match transform {
            Ok(step) => step,
            Err(error) => return self.failure(error, FailureOrigin::Host).await,
        };
        match self.settle(step).await {
            Ok(response) => self.succeeded(response).await,
            Err(error) => self.failure(error, FailureOrigin::Host).await,
        }
    }

    async fn machine_failed(&mut self, error: ErrorOf<H>) -> PyResult<Py<PyAny>> {
        let error = {
            let mut parts = self.parts();
            parts.observation.ended_at.get_or_insert_with(epoch_seconds);
            Python::attach(|py| match parts.interrupted.take() {
                Some(retained) => PyErr::from_value(retained.into_bound(py).into_any()),
                None => classified(py, &parts.binding, error),
            })
        };
        self.failure(error, FailureOrigin::Call).await
    }

    async fn succeeded(&mut self, response: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let step = {
            let mut parts = self.parts();
            let timing = parts.observation.timing();
            let event = PythonCallEvent::Succeeded {
                timing,
                response: &response,
            };
            parts.observation.retain_terminal(&event);
            Python::attach(|py| {
                let step = parts.hooks.on_event(py, event)?;
                parts.selected = Some(response.clone_ref(py));
                Ok::<_, PyErr>(step)
            })?
        };
        self.settle(step).await?;
        let mut parts = self.parts();
        parts.selected = None;
        parts.observation.finish();
        Ok(response)
    }

    async fn failure(&mut self, error: PyErr, origin: FailureOrigin) -> PyResult<Py<PyAny>> {
        let step = {
            let mut parts = self.parts();
            parts.observation.ended_at.get_or_insert_with(epoch_seconds);
            let timing = parts.observation.timing();
            Python::attach(|py| {
                if is_cancellation(py, &error) {
                    return Ok::<_, PyErr>(None);
                }
                let event = PythonCallEvent::Failed {
                    timing,
                    origin,
                    error: &error,
                };
                parts.observation.retain_terminal(&event);
                let step = parts.hooks.on_event(py, event)?;
                parts.selected = Some(error.value(py).clone().into_any().unbind());
                Ok(Some(step))
            })?
        };
        let Some(step) = step else {
            return Err(error);
        };
        self.settle(step).await?;
        let mut parts = self.parts();
        parts.selected = None;
        parts.observation.finish();
        Err(error)
    }
}

struct Driver<H, L>
where
    H: PythonBinding,
    L: PythonCallHooks,
{
    body: Mutex<Body>,
    reply: Option<Reply<Answer>>,
    parts: SharedParts<H, L>,
}

impl<H, L> Driver<H, L>
where
    H: PythonBinding + 'static,
    L: PythonCallHooks + 'static,
{
    /// A failure the body raised past its terminal hooks, or a cancellation, is the last
    /// thing the observers hear about the call.
    fn observe_host_failure(&self, error: &PyErr) {
        let mut parts = self.parts.lock();
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

impl<H, L> ExecutionBody for Driver<H, L>
where
    H: PythonBinding + 'static,
    L: PythonCallHooks + 'static,
{
    fn resume(&mut self, answer: Option<Answer>) -> PyResult<ExecutionStep> {
        match (self.reply.take(), answer) {
            (Some(reply), Some(answer)) => reply.send(answer),
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => return Err(missing_state()),
        }
        let step = {
            let body = self.body.get_mut();
            let mut resume = pin!(body.resume());
            match resume
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
            {
                Poll::Ready(step) => step,
                Poll::Pending => return Err(missing_state()),
            }
        };
        match step {
            Ok(CoroutineState::Yielded(Yielded { suspension, reply })) => {
                self.reply = Some(reply);
                Ok(ExecutionStep::Suspended(suspension))
            }
            Ok(CoroutineState::Complete(Ok(value))) => Ok(ExecutionStep::Return(value)),
            Ok(CoroutineState::Complete(Err(error))) => {
                self.observe_host_failure(&error);
                Err(error)
            }
            Err(_) => Err(missing_state()),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(parts) = self.parts.try_lock() else {
            return Ok(());
        };
        parts.binding.traverse(visit)?;
        parts.hooks.traverse(visit)?;
        visit.call(&parts.arguments)?;
        visit.call(&parts.interrupted)?;
        visit.call(&parts.selected)
    }
}

impl<H, L> Drop for Driver<H, L>
where
    H: PythonBinding,
    L: PythonCallHooks,
{
    fn drop(&mut self) {
        self.body.get_mut().cancel();
        self.reply = None;
        let mut parts = self.parts.lock();
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
    let parts = Arc::new(Mutex::new(Parts {
        binding,
        hooks,
        arguments: Some(arguments),
        interrupted: None,
        selected: None,
        observation: Observation {
            observers: options.observers,
            observing: false,
            started_at: 0.0,
            ended_at: None,
            terminal: None,
        },
    }));
    let body = Coroutine::new({
        let parts = Arc::clone(&parts);
        let asynchronous = options.asynchronous;
        move |co| {
            Running {
                task: PythonTask::new(co),
                parts,
                native: NativeMachine::new(asynchronous),
                start: Some(Box::new(start)),
                streaming: false,
            }
            .run()
        }
    });
    let mut driver = Driver {
        body: Mutex::new(body),
        reply: None,
        parts,
    };
    if options.asynchronous {
        return Execution::new(driver, options.lifecycle)
            .into_coroutine(py)
            .map(Bound::unbind);
    }
    match driver.resume(None)? {
        ExecutionStep::Return(value) => Ok(value),
        ExecutionStep::Suspended(Suspension::Open(head)) => {
            Execution::suspended(driver, options.lifecycle)
                .into_sync_stream(py, head)
                .map(Bound::unbind)
        }
        ExecutionStep::Suspended(Suspension::Await(_) | Suspension::Yield(_)) => {
            Err(PyRuntimeError::new_err("sync call suspended"))
        }
    }
}
