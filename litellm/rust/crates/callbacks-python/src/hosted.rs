use std::{
    any::Any,
    mem,
    panic::AssertUnwindSafe,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::{
    FutureExt,
    future::{AbortHandle, Abortable},
};
use litellm_callbacks::{
    CallbackError, ErasedCallback, HookInput, HookOutput, HostCallError, HostException, HostFault,
    HostReply, HostRequest, Hosted, RequestState, hosted,
};
use litellm_coroutine::{CoroutineState, Reply, ResumeError};
use litellm_host_python::{
    ExecutionBody, ExecutionStep, Suspension, json_from_py, json_to_py, poll_async_value,
    run_async_value,
};
use pyo3::{
    exceptions::{PyException, PyRuntimeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
};

use crate::{panic_error, payload, verdict_object};

type Handled = Result<HookOutput, Box<dyn Any + Send>>;
type Outcome = Result<CoroutineState<HostRequest, Handled>, ResumeError>;

#[derive(Default)]
struct Slot {
    body: Option<Hosted<Handled>>,
    outcome: Option<Outcome>,
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

pub(crate) struct HostedExecution {
    slot: Arc<Mutex<Slot>>,
    host: Py<PyAny>,
    waiting: Waiting,
}

fn out_of_order() -> PyErr {
    PyRuntimeError::new_err("the rust callback execution was resumed out of order")
}

pub(crate) fn callback_error(error: CallbackError) -> PyErr {
    if let CallbackError::Host(HostCallError::Raised { exception, .. }) = &error
        && let Some(original) = exception.original::<PyErr>()
    {
        return Python::attach(|py| original.clone_ref(py));
    }
    PyRuntimeError::new_err(error.to_string())
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

impl HostedExecution {
    pub(crate) fn new(
        callback: Arc<dyn ErasedCallback>,
        input: HookInput,
        state: Option<RequestState>,
        render_loop: Option<Arc<Py<PyAny>>>,
        host: Py<PyAny>,
    ) -> Self {
        let body = hosted(move |host| async move {
            payload::on_render_loop(
                render_loop,
                AssertUnwindSafe(async {
                    callback.handle_hosted(&input, state.as_ref(), host).await
                })
                .catch_unwind(),
            )
            .await
        });
        Self {
            slot: Arc::new(Mutex::new(Slot {
                body: Some(body),
                outcome: None,
            })),
            host,
            waiting: Waiting::Nothing,
        }
    }

    fn slot(&self) -> std::sync::MutexGuard<'_, Slot> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn landed(&self) -> PyResult<Outcome> {
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

    fn run(&mut self, py: Python<'_>, landed: Option<Outcome>) -> PyResult<ExecutionStep> {
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

    fn settle(&mut self, py: Python<'_>, outcome: Outcome) -> PyResult<Next> {
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
            Ok(CoroutineState::Complete(Ok(Ok(verdict)))) => Ok(Next::Step(ExecutionStep::Return(
                verdict_object(py, verdict)?,
            ))),
            Ok(CoroutineState::Complete(Ok(Err(error)))) => Err(callback_error(error)),
            Ok(CoroutineState::Complete(Err(panic))) => Err(panic_error(panic)),
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

impl ExecutionBody for HostedExecution {
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

impl Drop for HostedExecution {
    fn drop(&mut self) {
        self.cancel();
    }
}
