//! Drives a hosted coroutine from Python: every host call the body yields is answered by the method
//! of that name on the host object, awaited in the caller's task, and the body's output is turned
//! into the Python value the execution returns.

use std::{
    any::Any,
    future::Future,
    mem,
    panic::AssertUnwindSafe,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::{
    FutureExt,
    future::{AbortHandle, Abortable},
};
use litellm_callbacks::{
    CallbackError, ErasedCallback, HookInput, HookOutput, Host, HostCallError, HostException,
    HostFault, HostReply, HostRequest, Hosted, RequestState, hosted,
};
use litellm_coroutine::{CoroutineState, Reply, ResumeError};
use litellm_host_python::{
    Execution, ExecutionBody, ExecutionStep, Suspension, json_from_py, json_to_py,
    poll_async_value, run_async_value,
};
use pyo3::{
    exceptions::{PyException, PyRuntimeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
};

use crate::{execution_steps, panic_error, payload, verdict_object};

type Handled = Result<HookOutput, Box<dyn Any + Send>>;
type Outcome<C> = Result<CoroutineState<HostRequest, C>, ResumeError>;
type Finish<C> = Box<dyn FnOnce(Python<'_>, C) -> PyResult<Py<PyAny>> + Send + Sync>;

struct Slot<C> {
    body: Option<Hosted<C>>,
    outcome: Option<Outcome<C>>,
}

enum Waiting {
    Nothing,
    Rust(AbortHandle),
    Host(Reply<HostReply>),
}

enum Next {
    Step(ExecutionStep),
    Again,
}

/// One hosted coroutine run as a Python execution: the body's host calls go to `host`, and
/// `finish` turns the body's output into the execution's return value.
pub(crate) struct HostedExecution<C: Send + 'static> {
    slot: Arc<Mutex<Slot<C>>>,
    host: Py<PyAny>,
    waiting: Waiting,
    finish: Option<Finish<C>>,
}

fn out_of_order() -> PyErr {
    PyRuntimeError::new_err("the rust callback execution was resumed out of order")
}

/// The Python error a failed host call surfaces as: the exception the host raised, when it raised
/// one, else a `RuntimeError` naming the call.
pub fn host_call_error(error: &HostCallError) -> PyErr {
    if let HostCallError::Raised { exception, .. } = error
        && let Some(original) = exception.original::<PyErr>()
    {
        return Python::attach(|py| original.clone_ref(py));
    }
    PyRuntimeError::new_err(error.to_string())
}

pub(crate) fn callback_error(error: CallbackError) -> PyErr {
    match &error {
        CallbackError::Host(host) => host_call_error(host),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

fn host_reply(py: Python<'_>, answer: PyResult<Py<PyAny>>) -> HostReply {
    match answer {
        Ok(value) => {
            json_from_py(value.bind(py)).map_err(|error| HostFault::Malformed(error.to_string()))
        }
        Err(error) => Err(HostFault::Raised(HostException::new(
            error.to_string(),
            error,
        ))),
    }
}

fn is_awaitable(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value
        .py()
        .import("inspect")?
        .call_method1("isawaitable", (value,))?
        .is_truthy()
}

/// Runs `run` as a Python execution whose host calls `host` answers. A panic in the body raises a
/// `RuntimeError`; otherwise `finish` turns the body's output into the value the execution returns.
pub fn run_hosted<C, F>(
    run: impl FnOnce(Host) -> F,
    host: Py<PyAny>,
    finish: impl FnOnce(Python<'_>, C) -> PyResult<Py<PyAny>> + Send + Sync + 'static,
) -> Execution
where
    C: Send + 'static,
    F: Future<Output = C> + Send + 'static,
{
    let body = hosted(move |host| AssertUnwindSafe(run(host)).catch_unwind());
    Execution::new(
        HostedExecution::new(
            body,
            host,
            Box::new(
                move |py, output: Result<C, Box<dyn Any + Send>>| match output {
                    Ok(output) => finish(py, output),
                    Err(panic) => Err(panic_error(panic)),
                },
            ),
        ),
        execution_steps,
    )
}

/// `run_hosted` for one hook run a host starts itself: the verdict is the execution's return
/// value, as it is for `RustCallback.start`.
pub fn run_hosted_hook<F>(run: impl FnOnce(Host) -> F, host: Py<PyAny>) -> Execution
where
    F: Future<Output = HookOutput> + Send + 'static,
{
    run_hosted(run, host, |py, handled: HookOutput| match handled {
        Ok(verdict) => verdict_object(py, verdict),
        Err(error) => Err(callback_error(error)),
    })
}

/// The execution behind `RustCallback.start`: the callback's hosted handler, its verdict returned.
pub(crate) fn callback_execution(
    callback: Arc<dyn ErasedCallback>,
    input: HookInput,
    state: Option<RequestState>,
    render_loop: Option<Arc<Py<PyAny>>>,
    host: Py<PyAny>,
) -> HostedExecution<Handled> {
    let body = hosted(move |host| async move {
        payload::on_render_loop(
            render_loop,
            AssertUnwindSafe(async { callback.handle_hosted(&input, state.as_ref(), host).await })
                .catch_unwind(),
        )
        .await
    });
    HostedExecution::new(
        body,
        host,
        Box::new(|py, handled: Handled| match handled {
            Ok(Ok(verdict)) => verdict_object(py, verdict),
            Ok(Err(error)) => Err(callback_error(error)),
            Err(panic) => Err(panic_error(panic)),
        }),
    )
}

impl<C: Send + 'static> HostedExecution<C> {
    fn new(body: Hosted<C>, host: Py<PyAny>, finish: Finish<C>) -> Self {
        Self {
            slot: Arc::new(Mutex::new(Slot {
                body: Some(body),
                outcome: None,
            })),
            host,
            waiting: Waiting::Nothing,
            finish: Some(finish),
        }
    }

    fn slot(&self) -> std::sync::MutexGuard<'_, Slot<C>> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn landed(&self) -> PyResult<Outcome<C>> {
        self.slot().outcome.take().ok_or_else(out_of_order)
    }

    fn cancel(&mut self) {
        if let Waiting::Rust(abort) = mem::replace(&mut self.waiting, Waiting::Nothing) {
            abort.abort();
        }
        let body = self.slot().body.take();
        drop(body);
    }

    fn poll(&mut self, py: Python<'_>) -> PyResult<Option<ExecutionStep>> {
        let mut body = self.slot().body.take().ok_or_else(out_of_order)?;
        let slot = Arc::clone(&self.slot);
        let mut resumed = Box::pin(async move {
            let outcome = body.resume().await;
            let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
            slot.body = Some(body);
            slot.outcome = Some(outcome);
            Ok::<_, PyErr>(())
        });
        if poll_async_value(py, resumed.as_mut())?.is_ready() {
            return Ok(None);
        }
        let (abort, registration) = AbortHandle::new_pair();
        let awaitable = run_async_value(py, async move {
            Abortable::new(resumed, registration)
                .await
                .map_err(|_| PyRuntimeError::new_err("the rust callback execution was closed"))?
        })?;
        self.waiting = Waiting::Rust(abort);
        Ok(Some(ExecutionStep::Suspended(Suspension::Await(
            awaitable.unbind(),
        ))))
    }

    fn run(&mut self, py: Python<'_>, landed: Option<Outcome<C>>) -> PyResult<ExecutionStep> {
        let mut landed = landed;
        loop {
            let outcome = match landed.take() {
                Some(outcome) => outcome,
                None => match self.poll(py)? {
                    Some(step) => return Ok(step),
                    None => self.landed()?,
                },
            };
            if let Next::Step(step) = self.settle(py, outcome)? {
                return Ok(step);
            }
        }
    }

    fn settle(&mut self, py: Python<'_>, outcome: Outcome<C>) -> PyResult<Next> {
        match outcome {
            Ok(CoroutineState::Yielded(HostRequest {
                call,
                request,
                reply,
            })) => {
                let returned = json_to_py(py, &request)
                    .and_then(|request| self.host.bind(py).call_method1(call, (request,)));
                match returned {
                    Ok(value) if is_awaitable(&value)? => {
                        self.waiting = Waiting::Host(reply);
                        Ok(Next::Step(ExecutionStep::Suspended(Suspension::Await(
                            value.unbind(),
                        ))))
                    }
                    returned => {
                        self.answer(py, reply, returned.map(Bound::unbind))?;
                        Ok(Next::Again)
                    }
                }
            }
            Ok(CoroutineState::Complete(output)) => {
                let finish = self.finish.take().ok_or_else(out_of_order)?;
                Ok(Next::Step(ExecutionStep::Return(finish(py, output)?)))
            }
            Err(error) => Err(PyRuntimeError::new_err(error.to_string())),
        }
    }

    fn answer(
        &mut self,
        py: Python<'_>,
        reply: Reply<HostReply>,
        answer: PyResult<Py<PyAny>>,
    ) -> PyResult<()> {
        match answer {
            Err(error) if !error.is_instance_of::<PyException>(py) => {
                drop(reply);
                self.cancel();
                Err(error)
            }
            answer => {
                reply.send(host_reply(py, answer));
                Ok(())
            }
        }
    }
}

impl<C: Send + 'static> ExecutionBody for HostedExecution<C> {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(
            |py| match (mem::replace(&mut self.waiting, Waiting::Nothing), result) {
                (Waiting::Nothing, None) => self.run(py, None),
                (Waiting::Host(reply), Some(answer)) => {
                    self.answer(py, reply, answer)?;
                    self.run(py, None)
                }
                (Waiting::Rust(_), Some(Ok(_))) => {
                    let outcome = self.landed()?;
                    self.run(py, Some(outcome))
                }
                (Waiting::Rust(abort), Some(Err(error))) => {
                    self.waiting = Waiting::Rust(abort);
                    self.cancel();
                    Err(error)
                }
                (waiting, _) => {
                    self.waiting = waiting;
                    self.cancel();
                    Err(out_of_order())
                }
            },
        )
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.host)
    }
}

impl<C: Send + 'static> Drop for HostedExecution<C> {
    fn drop(&mut self) {
        self.cancel();
    }
}
