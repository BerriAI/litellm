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
use crate::caller::{Answer, PythonCaller, Yielded};
use crate::handle::{Execution, ExecutionBody, ExecutionStep, StepTypes};
use crate::hooks::{PythonCallEvent, PythonCallHooks};
use crate::native::{NativeMachine, NativePoll};
use crate::{Error, InvokeError, PythonBinding};

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
    pub steps: StepTypes,
    pub observers: Option<ObservationSender>,
}

impl CallOptions {
    pub fn new(asynchronous: bool, steps: StepTypes) -> Self {
        Self {
            asynchronous,
            steps,
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
/// Binding and hooks lock their own state only around Python work, so the handle can
/// traverse them while a hook or host call is suspended.
struct Shared<H, L> {
    binding: H,
    hooks: L,
    parts: Mutex<Parts>,
}

struct Parts {
    arguments: Option<Py<PyDict>>,
    interrupted: Option<Py<PyBaseException>>,
    selected: Option<Selected>,
    observation: Observation,
}

/// The call's outcome while its terminal hooks run. The body borrows it for the event and
/// holds no Python reference of its own, so the one reference is the traversed one.
enum Selected {
    Response(Arc<Py<PyAny>>),
    Exception(Arc<Py<PyBaseException>>),
}

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

/// The call as one async body. Every hook and host call is awaited in place on the
/// caller's task; the parts lock is never held across an await.
struct CallBody<H, M, L>
where
    H: PythonBinding,
    M: Machine,
{
    caller: PythonCaller,
    shared: Arc<Shared<H, L>>,
    native: NativeMachine<M>,
    start: Option<StartMachine<ProtocolOf<H>, M>>,
    streaming: bool,
}

impl<H, M, L> CallBody<H, M, L>
where
    L: PythonCallHooks + 'static,
    H: PythonBinding + PythonHostCalls<H::Protocol> + 'static,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<ResponseOf<H>>>,
{
    fn parts(&self) -> MutexGuard<'_, Parts> {
        self.shared.parts.lock()
    }

    fn hooks(&self) -> &L {
        &self.shared.hooks
    }

    fn binding(&self) -> &H {
        &self.shared.binding
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
        let started = PythonCallEvent::Started {
            start_time: started_at,
        };
        {
            let mut parts = self.parts();
            parts.observation.started_at = started_at;
            parts.observation.observing = true;
            parts.observation.emit(&started);
        }
        self.hooks()
            .on_event(&self.caller, started)
            .await
            .map_err(Setup::Host)?;
        let arguments = self
            .parts()
            .arguments
            .take()
            .ok_or_else(|| Setup::Host(Error::MissingState.into()))?;
        let prepared = self
            .hooks()
            .prepare_arguments(&self.caller, arguments, started_at)
            .await
            .map_err(Setup::Host)?;
        self.hooks()
            .arguments_prepared(&prepared)
            .map_err(Setup::Host)?;
        let decoded = Python::attach(|py| self.binding().decode_request(py, prepared.bind(py)));
        self.parts().arguments = Some(prepared);
        let request = decoded.map_err(Setup::Decode)?;
        let start = self
            .start
            .take()
            .ok_or_else(|| Setup::Host(Error::MissingState.into()))?;
        let machine = {
            let parts = self.parts();
            let arguments = parts
                .arguments
                .as_ref()
                .ok_or_else(|| Setup::Host(Error::MissingState.into()))?;
            Python::attach(|py| start(py, arguments.bind(py), request)).map_err(Setup::Host)?
        };
        self.native.start(machine);
        Ok(())
    }

    async fn execute(&mut self) -> PyResult<Py<PyAny>> {
        let mut interruption: Interruption<H> = None;
        loop {
            let poll = Python::attach(|py| self.native.resume(py, interruption.take()))?;
            let result = match poll {
                NativePoll::Ready(result) => result,
                NativePoll::Suspend(awaitable) => match self.caller.await_python(awaitable).await {
                    Ok(_) => self.native.take_result()?,
                    Err(error) if self.parts().interrupted.is_none() => {
                        interruption = Some(self.interrupt(error));
                        continue;
                    }
                    Err(error) => return Err(error),
                },
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
            HostRequest::HostCall(call) => {
                self.binding().handle_host_call(&self.caller, call).await
            }
            HostRequest::Intercept(InterceptRequest::BeforeProviderRequest {
                wire,
                context,
                reply,
            }) => {
                let wire = self
                    .hooks()
                    .before_provider_request(&self.caller, wire, &context)
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
                self.streaming = true;
                let head = Python::attach(|py| self.binding().encode_stream_head(py, head))?;
                self.hooks().on_stream_open(&head)?;
                reply.send(self.caller.open_stream(head).await);
                Ok(())
            }
            HostRequest::Stream(StreamDelivery::Chunk(chunk, reply)) => {
                let chunk = Python::attach(|py| self.binding().encode_chunk(py, chunk))?;
                self.hooks().on_stream_chunk(&chunk)?;
                reply.send(self.caller.yield_chunk(chunk).await);
                Ok(())
            }
        }
    }

    async fn notify(&self, event: PythonCallEvent<'_>) -> PyResult<()> {
        self.parts().observation.emit(&event);
        self.hooks().on_event(&self.caller, event).await
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
        let public = match Python::attach(|py| self.binding().encode_response(py, response)) {
            Ok(public) => public,
            Err(error) => return self.failure(error, FailureOrigin::Call).await,
        };
        if self.streaming {
            return self.succeeded(public).await;
        }
        let timing = self.parts().observation.timing();
        match self
            .hooks()
            .transform_response(&self.caller, public, timing)
            .await
        {
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
                None => classified(py, self.binding(), error),
            })
        };
        self.failure(error, FailureOrigin::Call).await
    }

    async fn succeeded(&mut self, response: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let response = Arc::new(response);
        let timing = self.parts().observation.timing();
        let event = PythonCallEvent::Succeeded {
            timing,
            response: &response,
        };
        {
            let mut parts = self.parts();
            parts.observation.retain_terminal(&event);
            parts.selected = Some(Selected::Response(Arc::clone(&response)));
        }
        let notified = self.hooks().on_event(&self.caller, event).await;
        {
            let mut parts = self.parts();
            parts.selected = None;
            notified?;
            parts.observation.finish();
        }
        Arc::into_inner(response).ok_or_else(|| Error::MissingState.into())
    }

    async fn failure(&mut self, error: PyErr, origin: FailureOrigin) -> PyResult<Py<PyAny>> {
        let timing = {
            let mut parts = self.parts();
            parts.observation.ended_at.get_or_insert_with(epoch_seconds);
            parts.observation.timing()
        };
        let exception = Python::attach(|py| {
            if is_cancellation(py, &error) {
                return Err(error);
            }
            Ok(Arc::new(error.into_value(py)))
        })?;
        let event = PythonCallEvent::Failed {
            timing,
            origin,
            error: &exception,
        };
        {
            let mut parts = self.parts();
            parts.observation.retain_terminal(&event);
            parts.selected = Some(Selected::Exception(Arc::clone(&exception)));
        }
        let notified = self.hooks().on_event(&self.caller, event).await;
        {
            let mut parts = self.parts();
            parts.selected = None;
            notified?;
            parts.observation.finish();
        }
        let exception = Arc::into_inner(exception).ok_or(Error::MissingState)?;
        Err(Python::attach(|py| {
            PyErr::from_value(exception.into_bound(py).into_any())
        }))
    }
}

struct CallDriver<H, L>
where
    H: PythonBinding,
    L: PythonCallHooks,
{
    body: Mutex<Body>,
    reply: Option<Reply<Answer>>,
    shared: Arc<Shared<H, L>>,
}

impl<H, L> CallDriver<H, L>
where
    H: PythonBinding + 'static,
    L: PythonCallHooks + 'static,
{
    /// A failure the body raised past its terminal hooks, or a cancellation, is the last
    /// thing the observers hear about the call.
    fn observe_host_failure(&self, error: &PyErr) {
        let mut parts = self.shared.parts.lock();
        if !parts.observation.observing {
            return;
        }
        let timing = parts.observation.timing();
        Python::attach(|py| {
            let exception = error.value(py).clone().unbind();
            let event = if is_cancellation(py, error) {
                PythonCallEvent::Cancelled { timing }
            } else {
                PythonCallEvent::Failed {
                    timing,
                    origin: FailureOrigin::Host,
                    error: &exception,
                }
            };
            parts.observation.emit(&event);
        });
        parts.observation.observing = false;
        parts.observation.terminal = None;
    }
}

impl<H, L> ExecutionBody for CallDriver<H, L>
where
    H: PythonBinding + 'static,
    L: PythonCallHooks + 'static,
{
    fn resume(&mut self, answer: Option<Answer>) -> PyResult<ExecutionStep> {
        match (self.reply.take(), answer) {
            (Some(reply), Some(answer)) => reply.send(answer),
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => return Err(Error::MissingState.into()),
        }
        let step = {
            let body = self.body.get_mut();
            let mut resume = pin!(body.resume());
            match resume
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
            {
                Poll::Ready(step) => step,
                Poll::Pending => return Err(Error::MissingState.into()),
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
            Err(_) => Err(Error::MissingState.into()),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.shared.binding.traverse(visit)?;
        self.shared.hooks.traverse(visit)?;
        let Some(parts) = self.shared.parts.try_lock() else {
            return Ok(());
        };
        visit.call(&parts.arguments)?;
        visit.call(&parts.interrupted)?;
        match &parts.selected {
            Some(Selected::Response(response)) => visit.call(&**response),
            Some(Selected::Exception(exception)) => visit.call(&**exception),
            None => Ok(()),
        }
    }
}

impl<H, L> Drop for CallDriver<H, L>
where
    H: PythonBinding,
    L: PythonCallHooks,
{
    fn drop(&mut self) {
        self.body.get_mut().cancel();
        self.reply = None;
        let Some(Shared {
            binding,
            hooks,
            parts,
        }) = Arc::get_mut(&mut self.shared)
        else {
            return;
        };
        let observation = &mut parts.get_mut().observation;
        Python::attach(|py| {
            if observation.observing {
                let timing = observation.timing();
                observation.emit(&PythonCallEvent::Cancelled { timing });
                observation.observing = false;
            }
            hooks.close(py);
            binding.close(py);
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
    let shared = Arc::new(Shared {
        binding,
        hooks,
        parts: Mutex::new(Parts {
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
        }),
    });
    let body = Coroutine::new({
        let shared = Arc::clone(&shared);
        let asynchronous = options.asynchronous;
        move |co| {
            CallBody {
                caller: PythonCaller::new(co),
                shared,
                native: NativeMachine::new(asynchronous),
                start: Some(Box::new(start)),
                streaming: false,
            }
            .run()
        }
    });
    let driver = CallDriver {
        body: Mutex::new(body),
        reply: None,
        shared,
    };
    Py::new(py, Execution::new(driver, options.steps)).map(Py::into_any)
}
