use litellm_host::{
    hooks::CallHooks,
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{CallEvent, ExecutionEvent, Timing},
};
use pyo3::{
    exceptions::{PyBaseException, PyException},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::{
    HookResume, HookStep, PythonCallEvent, PythonCallHooks, PythonOwned, PythonRuntime,
    missing_state,
};

enum Continuation<A, B, T> {
    First(HookResume<A, T>),
    Second(HookResume<B, T>),
}

type Pending<A, B, T, C> = Option<(Continuation<A, B, T>, C)>;
type OwnedEvent = CallEvent<Py<PyAny>, Py<PyBaseException>, RawResponse>;

pub struct HookChain<A, B> {
    first: A,
    second: B,
    arguments: Pending<A, B, Py<PyDict>, f64>,
    wire: Pending<A, B, Box<WireRequest>, RequestContext>,
    response: Pending<A, B, Py<PyAny>, Timing>,
    event: Pending<A, B, (), OwnedEvent>,
}

impl<A, B> HookChain<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self {
            first,
            second,
            arguments: None,
            wire: None,
            response: None,
            event: None,
        }
    }

    fn second_step<T, C>(
        step: HookStep<B, T>,
        context: C,
        pending: &mut Pending<A, B, T, C>,
        resume_chain: HookResume<Self, T>,
    ) -> HookStep<Self, T> {
        match step {
            HookStep::Ready(value) => HookStep::Ready(value),
            HookStep::Await(awaitable, resume) => {
                *pending = Some((Continuation::Second(resume), context));
                HookStep::Await(awaitable, resume_chain)
            }
        }
    }
}

impl<A: PythonCallHooks, B: PythonCallHooks> HookChain<A, B> {
    fn first_arguments(
        &mut self,
        py: Python<'_>,
        step: HookStep<A, Py<PyDict>>,
        context: f64,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        match step {
            HookStep::Ready(value) => {
                let step = self.second.prepare_arguments(py, value, context)?;
                Ok(Self::second_step(
                    step,
                    context,
                    &mut self.arguments,
                    Self::resume_arguments,
                ))
            }
            HookStep::Await(awaitable, resume) => {
                self.arguments = Some((Continuation::First(resume), context));
                Ok(HookStep::Await(awaitable, Self::resume_arguments))
            }
        }
    }

    fn resume_arguments(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        let (continuation, context) = self.arguments.take().ok_or_else(missing_state)?;
        let result = resume_unless_cancelled(py, result)?;
        match continuation {
            Continuation::First(resume) => {
                let step = resume(&mut self.first, py, result)?;
                self.first_arguments(py, step, context)
            }
            Continuation::Second(resume) => {
                let step = resume(&mut self.second, py, result)?;
                Ok(Self::second_step(
                    step,
                    context,
                    &mut self.arguments,
                    Self::resume_arguments,
                ))
            }
        }
    }

    fn first_wire(
        &mut self,
        py: Python<'_>,
        step: HookStep<A, Box<WireRequest>>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        match step {
            HookStep::Ready(value) => {
                let step = self.second.before_provider_request(py, value, context)?;
                match step {
                    HookStep::Ready(wire) => Ok(HookStep::Ready(wire)),
                    step => Ok(Self::second_step(
                        step,
                        context.clone(),
                        &mut self.wire,
                        Self::resume_wire,
                    )),
                }
            }
            HookStep::Await(awaitable, resume) => {
                self.wire = Some((Continuation::First(resume), context.clone()));
                Ok(HookStep::Await(awaitable, Self::resume_wire))
            }
        }
    }

    fn resume_wire(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        let (continuation, context) = self.wire.take().ok_or_else(missing_state)?;
        let result = resume_unless_cancelled(py, result)?;
        match continuation {
            Continuation::First(resume) => {
                let step = resume(&mut self.first, py, result)?;
                self.first_wire(py, step, &context)
            }
            Continuation::Second(resume) => {
                let step = resume(&mut self.second, py, result)?;
                Ok(Self::second_step(
                    step,
                    context,
                    &mut self.wire,
                    Self::resume_wire,
                ))
            }
        }
    }

    fn first_response(
        &mut self,
        py: Python<'_>,
        step: HookStep<A, Py<PyAny>>,
        context: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        match step {
            HookStep::Ready(value) => {
                let step = self.second.transform_response(py, value, context)?;
                Ok(Self::second_step(
                    step,
                    context,
                    &mut self.response,
                    Self::resume_response,
                ))
            }
            HookStep::Await(awaitable, resume) => {
                self.response = Some((Continuation::First(resume), context));
                Ok(HookStep::Await(awaitable, Self::resume_response))
            }
        }
    }

    fn resume_response(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        let (continuation, context) = self.response.take().ok_or_else(missing_state)?;
        let result = resume_unless_cancelled(py, result)?;
        match continuation {
            Continuation::First(resume) => {
                let step = resume(&mut self.first, py, result)?;
                self.first_response(py, step, context)
            }
            Continuation::Second(resume) => {
                let step = resume(&mut self.second, py, result)?;
                Ok(Self::second_step(
                    step,
                    context,
                    &mut self.response,
                    Self::resume_response,
                ))
            }
        }
    }

    fn first_event(
        &mut self,
        py: Python<'_>,
        result: PyResult<HookStep<A, ()>>,
        event: OwnedEvent,
    ) -> PyResult<HookStep<Self, ()>> {
        match notification_result(py, is_terminal(&event), result)? {
            HookStep::Ready(()) => {
                let result = dispatch(py, &mut self.second, &event);
                let step = notification_result(py, is_terminal(&event), result)?;
                Ok(Self::second_step(
                    step,
                    event,
                    &mut self.event,
                    Self::resume_event,
                ))
            }
            HookStep::Await(awaitable, resume) => {
                self.event = Some((Continuation::First(resume), event));
                Ok(HookStep::Await(awaitable, Self::resume_event))
            }
        }
    }

    fn resume_event(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, ()>> {
        let (continuation, event) = self.event.take().ok_or_else(missing_state)?;
        let result = resume_unless_cancelled(py, result)?;
        match continuation {
            Continuation::First(resume) => {
                let result = resume(&mut self.first, py, result);
                self.first_event(py, result, event)
            }
            Continuation::Second(resume) => {
                let result = resume(&mut self.second, py, result);
                let step = notification_result(py, is_terminal(&event), result)?;
                Ok(Self::second_step(
                    step,
                    event,
                    &mut self.event,
                    Self::resume_event,
                ))
            }
        }
    }
}

fn resume_unless_cancelled(
    py: Python<'_>,
    result: PyResult<Py<PyAny>>,
) -> PyResult<PyResult<Py<PyAny>>> {
    match result {
        Err(error) if !error.is_instance_of::<PyException>(py) => Err(error),
        result => Ok(result),
    }
}

fn is_terminal<Response, Error, Raw>(event: &CallEvent<Response, Error, Raw>) -> bool {
    matches!(
        event,
        CallEvent::Succeeded { .. } | CallEvent::Failed { .. }
    )
}

fn notification_result<H>(
    py: Python<'_>,
    terminal: bool,
    result: PyResult<HookStep<H, ()>>,
) -> PyResult<HookStep<H, ()>> {
    match result {
        Err(error) if terminal && error.is_instance_of::<PyException>(py) => {
            error.write_unraisable(py, None);
            Ok(HookStep::Ready(()))
        }
        result => result,
    }
}

fn retain_event(py: Python<'_>, event: PythonCallEvent<'_>) -> OwnedEvent {
    match event {
        CallEvent::Started { start_time } => CallEvent::Started { start_time },
        CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => {
            CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw: raw.clone() })
        }
        CallEvent::Succeeded { timing, response } => CallEvent::Succeeded {
            timing,
            response: response.clone_ref(py),
        },
        CallEvent::Failed {
            timing,
            origin,
            error,
        } => CallEvent::Failed {
            timing,
            origin,
            error: error.clone_ref(py).into_value(py),
        },
        CallEvent::Cancelled { timing } => CallEvent::Cancelled { timing },
    }
}

fn dispatch<H: PythonCallHooks>(
    py: Python<'_>,
    hooks: &mut H,
    event: &OwnedEvent,
) -> PyResult<HookStep<H, ()>> {
    match event {
        CallEvent::Started { start_time } => hooks.on_event(
            py,
            CallEvent::Started {
                start_time: *start_time,
            },
        ),
        CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => hooks.on_event(
            py,
            CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }),
        ),
        CallEvent::Succeeded { timing, response } => hooks.on_event(
            py,
            CallEvent::Succeeded {
                timing: *timing,
                response,
            },
        ),
        CallEvent::Failed {
            timing,
            origin,
            error,
        } => hooks.on_event(
            py,
            CallEvent::Failed {
                timing: *timing,
                origin: *origin,
                error: &PyErr::from_value(error.bind(py).clone().into_any()),
            },
        ),
        CallEvent::Cancelled { timing } => {
            hooks.on_event(py, CallEvent::Cancelled { timing: *timing })
        }
    }
}

impl<A: PythonCallHooks, B: PythonCallHooks> CallHooks<PythonRuntime> for HookChain<A, B> {
    fn prepare_arguments(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        let step = self.first.prepare_arguments(py, arguments, started_at)?;
        self.first_arguments(py, step, started_at)
    }

    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        self.first.arguments_prepared(py, arguments)?;
        self.second.arguments_prepared(py, arguments)
    }

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        let step = self.first.before_provider_request(py, wire, context)?;
        self.first_wire(py, step, context)
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        let step = self.first.transform_response(py, response, timing)?;
        self.first_response(py, step, timing)
    }

    fn on_event(
        &mut self,
        py: Python<'_>,
        event: PythonCallEvent<'_>,
    ) -> PyResult<HookStep<Self, ()>> {
        let terminal = is_terminal(&event);
        let result = self.first.on_event(py, event.clone());
        match notification_result(py, terminal, result)? {
            HookStep::Ready(()) => {
                let result = self.second.on_event(py, event.clone());
                match notification_result(py, terminal, result)? {
                    HookStep::Ready(()) => Ok(HookStep::Ready(())),
                    step => Ok(Self::second_step(
                        step,
                        retain_event(py, event),
                        &mut self.event,
                        Self::resume_event,
                    )),
                }
            }
            HookStep::Await(awaitable, resume) => {
                self.event = Some((Continuation::First(resume), retain_event(py, event)));
                Ok(HookStep::Await(awaitable, Self::resume_event))
            }
        }
    }

    fn on_stream_open(&mut self, py: Python<'_>) -> PyResult<()> {
        self.first.on_stream_open(py)?;
        self.second.on_stream_open(py)
    }

    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        self.first.on_stream_chunk(py, chunk)?;
        self.second.on_stream_chunk(py, chunk)
    }
}

impl<A: PythonOwned, B: PythonOwned> PythonOwned for HookChain<A, B> {
    fn close(&mut self, py: Python<'_>) {
        self.arguments = None;
        self.wire = None;
        self.response = None;
        self.event = None;
        self.first.close(py);
        self.second.close(py);
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.first.traverse(visit)?;
        self.second.traverse(visit)?;
        match &self.event {
            Some((_, CallEvent::Succeeded { response, .. })) => visit.call(response),
            Some((_, CallEvent::Failed { error, .. })) => visit.call(error),
            _ => Ok(()),
        }
    }
}
