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
    HookStep, PythonCallEvent, PythonCallHooks, PythonOwned, PythonRuntime, missing_state,
};

use super::adapter::{ChainHooks, ChainStep, HookAdapter};

type OwnedEvent = CallEvent<Py<PyAny>, Py<PyBaseException>, RawResponse>;

#[derive(Default)]
pub struct HookChain {
    hooks: Vec<Box<dyn ChainHooks>>,
    arguments: Option<(usize, f64)>,
    wire: Option<(usize, RequestContext)>,
    response: Option<(usize, Timing)>,
    event: Option<(usize, OwnedEvent)>,
}

impl HookChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, hooks: impl PythonCallHooks + 'static) -> Self {
        self.hooks.push(Box::new(HookAdapter::new(hooks)));
        self
    }

    fn arguments_from(
        &mut self,
        py: Python<'_>,
        index: usize,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        let Some(hooks) = self.hooks.get_mut(index) else {
            return Ok(HookStep::Ready(arguments));
        };
        let step = hooks.prepare_arguments(py, arguments, started_at)?;
        self.arguments_step(py, index, step, started_at)
    }

    fn arguments_step(
        &mut self,
        py: Python<'_>,
        index: usize,
        step: ChainStep<Py<PyDict>>,
        started_at: f64,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        match step {
            ChainStep::Ready(value) => self.arguments_from(py, index + 1, value, started_at),
            ChainStep::Await(awaitable) => {
                self.arguments = Some((index, started_at));
                Ok(HookStep::Await(awaitable, Self::resume_arguments))
            }
        }
    }

    fn resume_arguments(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        let (index, context) = self.arguments.take().ok_or_else(missing_state)?;
        let result = resume_unless_cancelled(py, result)?;
        let step = self.hooks[index].resume_arguments(py, result)?;
        self.arguments_step(py, index, step, context)
    }

    fn wire_from(
        &mut self,
        py: Python<'_>,
        index: usize,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        let Some(hooks) = self.hooks.get_mut(index) else {
            return Ok(HookStep::Ready(wire));
        };
        let step = hooks.before_provider_request(py, wire, context)?;
        self.wire_step(py, index, step, context)
    }

    fn wire_step(
        &mut self,
        py: Python<'_>,
        index: usize,
        step: ChainStep<Box<WireRequest>>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        match step {
            ChainStep::Ready(value) => self.wire_from(py, index + 1, value, context),
            ChainStep::Await(awaitable) => {
                self.wire = Some((index, context.clone()));
                Ok(HookStep::Await(awaitable, Self::resume_wire))
            }
        }
    }

    fn resume_wire(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        let (index, context) = self.wire.take().ok_or_else(missing_state)?;
        let result = resume_unless_cancelled(py, result)?;
        let step = self.hooks[index].resume_wire(py, result)?;
        self.wire_step(py, index, step, &context)
    }

    fn response_from(
        &mut self,
        py: Python<'_>,
        index: usize,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        let Some(hooks) = self.hooks.get_mut(index) else {
            return Ok(HookStep::Ready(response));
        };
        let step = hooks.transform_response(py, response, timing)?;
        self.response_step(py, index, step, timing)
    }

    fn response_step(
        &mut self,
        py: Python<'_>,
        index: usize,
        step: ChainStep<Py<PyAny>>,
        timing: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        match step {
            ChainStep::Ready(value) => self.response_from(py, index + 1, value, timing),
            ChainStep::Await(awaitable) => {
                self.response = Some((index, timing));
                Ok(HookStep::Await(awaitable, Self::resume_response))
            }
        }
    }

    fn resume_response(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        let (index, context) = self.response.take().ok_or_else(missing_state)?;
        let result = resume_unless_cancelled(py, result)?;
        let step = self.hooks[index].resume_response(py, result)?;
        self.response_step(py, index, step, context)
    }

    fn event_from(
        &mut self,
        py: Python<'_>,
        index: usize,
        event: OwnedEvent,
    ) -> PyResult<HookStep<Self, ()>> {
        let Some(hooks) = self.hooks.get_mut(index) else {
            return Ok(HookStep::Ready(()));
        };
        let result = dispatch(py, hooks.as_mut(), &event);
        let step = notification_result(py, is_terminal(&event), result)?;
        self.event_step(py, index, step, event)
    }

    fn event_step(
        &mut self,
        py: Python<'_>,
        index: usize,
        step: ChainStep<()>,
        event: OwnedEvent,
    ) -> PyResult<HookStep<Self, ()>> {
        match step {
            ChainStep::Ready(()) => self.event_from(py, index + 1, event),
            ChainStep::Await(awaitable) => {
                self.event = Some((index, event));
                Ok(HookStep::Await(awaitable, Self::resume_event))
            }
        }
    }

    fn resume_event(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, ()>> {
        let (index, event) = self.event.take().ok_or_else(missing_state)?;
        let result = resume_unless_cancelled(py, result)?;
        let result = self.hooks[index].resume_event(py, result);
        let step = notification_result(py, is_terminal(&event), result)?;
        self.event_step(py, index, step, event)
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

fn notification_result(
    py: Python<'_>,
    terminal: bool,
    result: PyResult<ChainStep<()>>,
) -> PyResult<ChainStep<()>> {
    match result {
        Err(error) if terminal && error.is_instance_of::<PyException>(py) => {
            error.write_unraisable(py, None);
            Ok(ChainStep::Ready(()))
        }
        result => result,
    }
}

fn retain_event(py: Python<'_>, event: PythonCallEvent<'_>) -> OwnedEvent {
    match event {
        CallEvent::Execution(ExecutionEvent::ResultReady { facts }) => {
            CallEvent::Execution(ExecutionEvent::ResultReady { facts })
        }
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

fn dispatch(
    py: Python<'_>,
    hooks: &mut dyn ChainHooks,
    event: &OwnedEvent,
) -> PyResult<ChainStep<()>> {
    match event {
        CallEvent::Execution(ExecutionEvent::ResultReady { facts }) => hooks.on_event(
            py,
            CallEvent::Execution(ExecutionEvent::ResultReady {
                facts: facts.clone(),
            }),
        ),
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

impl CallHooks<PythonRuntime> for HookChain {
    fn prepare_arguments(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        self.arguments_from(py, 0, arguments, started_at)
    }

    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        self.hooks
            .iter_mut()
            .try_for_each(|hooks| hooks.arguments_prepared(py, arguments))
    }

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        self.wire_from(py, 0, wire, context)
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        self.response_from(py, 0, response, timing)
    }

    fn on_event(
        &mut self,
        py: Python<'_>,
        event: PythonCallEvent<'_>,
    ) -> PyResult<HookStep<Self, ()>> {
        for (index, hooks) in self.hooks.iter_mut().enumerate() {
            let result = hooks.on_event(py, event.clone());
            match notification_result(py, is_terminal(&event), result)? {
                ChainStep::Ready(()) => {}
                ChainStep::Await(awaitable) => {
                    self.event = Some((index, retain_event(py, event)));
                    return Ok(HookStep::Await(awaitable, Self::resume_event));
                }
            }
        }
        Ok(HookStep::Ready(()))
    }

    fn on_stream_open(&mut self, py: Python<'_>, head: &Py<PyAny>) -> PyResult<()> {
        self.hooks
            .iter_mut()
            .try_for_each(|hooks| hooks.on_stream_open(py, head))
    }

    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        self.hooks
            .iter_mut()
            .try_for_each(|hooks| hooks.on_stream_chunk(py, chunk))
    }
}

impl PythonOwned for HookChain {
    fn close(&mut self, py: Python<'_>) {
        self.arguments = None;
        self.wire = None;
        self.response = None;
        self.event = None;
        for hooks in &mut self.hooks {
            hooks.close(py);
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for hooks in &self.hooks {
            hooks.traverse(visit)?;
        }
        match &self.event {
            Some((_, CallEvent::Succeeded { response, .. })) => visit.call(response),
            Some((_, CallEvent::Failed { error, .. })) => visit.call(error),
            _ => Ok(()),
        }
    }
}
