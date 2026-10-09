use litellm_host::{
    hooks::{CallHooks, NativeHooks},
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
use super::native::NativeAdapter;
use super::phase::{Order, Stage, transform};
use crate::hooks::HookLayer;

type OwnedEvent = CallEvent<Py<PyAny>, Py<PyBaseException>, RawResponse>;

/// One hook boxed for a [`HookChain`]. A callback crate hands its hooks back as these so a
/// route composes them without naming their types.
pub struct Hooks(Box<dyn ChainHooks>);

impl Hooks {
    pub fn new(hooks: impl PythonCallHooks + 'static) -> Self {
        Self(Box::new(HookAdapter::new(hooks)))
    }

    pub fn native(hooks: impl NativeHooks + 'static) -> Self {
        Self(Box::new(NativeAdapter(hooks)))
    }
}

/// Hooks composed as an onion. The first hook added is the outermost: it sees the
/// arguments and the wire request first and the response and the terminal events last.
#[derive(Default)]
pub struct HookChain {
    pub(super) hooks: Vec<Box<dyn ChainHooks>>,
    event: Option<(usize, OwnedEvent)>,
}

impl HookChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(self, hooks: impl PythonCallHooks + 'static) -> Self {
        self.with_all([Hooks::new(hooks)])
    }

    pub fn layer<Call>(self, py: Python<'_>, layer: &impl HookLayer<Call>, call: Call) -> Self {
        self.with_all(layer.layer(py, call))
    }

    pub fn with_all(mut self, hooks: impl IntoIterator<Item = Hooks>) -> Self {
        self.hooks
            .extend(hooks.into_iter().map(|Hooks(hooks)| hooks));
        self
    }

    fn event_from(
        &mut self,
        py: Python<'_>,
        index: Option<usize>,
        event: OwnedEvent,
    ) -> PyResult<HookStep<Self, ()>> {
        let Some(index) = index else {
            return Ok(HookStep::Ready(()));
        };
        let result = dispatch(py, self.hooks[index].as_mut(), &event);
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
            ChainStep::Ready(()) => {
                let next = event_order(&event).next(index, self.hooks.len());
                self.event_from(py, next, event)
            }
            ChainStep::Await(awaitable) => {
                self.event = Some((index, event));
                Ok(HookStep::Await(awaitable, Box::new(Self::resume_event)))
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

pub(super) fn resume_unless_cancelled(
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

/// Only the start of a call flows toward the provider; everything after it flows back out.
fn event_order<Response, Error, Raw>(event: &CallEvent<Response, Error, Raw>) -> Order {
    match event {
        CallEvent::Started { .. } => Order::Inbound,
        CallEvent::Execution(_)
        | CallEvent::Succeeded { .. }
        | CallEvent::Failed { .. }
        | CallEvent::Cancelled { .. } => Order::Outbound,
    }
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
        transform(
            py,
            &mut self.hooks,
            Order::Inbound,
            arguments,
            Stage::Wrapper { started_at },
        )
    }

    fn prepare_request(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        transform(py, &mut self.hooks, Order::Inbound, arguments, Stage::Body)
    }

    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        Order::Inbound
            .walk(self.hooks.len())
            .try_for_each(|index| self.hooks[index].arguments_prepared(py, arguments))
    }

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        transform(py, &mut self.hooks, Order::Inbound, wire, context.clone())
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        transform(py, &mut self.hooks, Order::Outbound, response, timing)
    }

    fn on_event(
        &mut self,
        py: Python<'_>,
        event: PythonCallEvent<'_>,
    ) -> PyResult<HookStep<Self, ()>> {
        let order = event_order(&event);
        let terminal = is_terminal(&event);
        let mut index = order.first(self.hooks.len());
        while let Some(current) = index {
            let result = self.hooks[current].on_event(py, event.clone());
            match notification_result(py, terminal, result)? {
                ChainStep::Ready(()) => index = order.next(current, self.hooks.len()),
                ChainStep::Await(awaitable) => {
                    self.event = Some((current, retain_event(py, event)));
                    return Ok(HookStep::Await(awaitable, Box::new(Self::resume_event)));
                }
            }
        }
        Ok(HookStep::Ready(()))
    }

    fn on_stream_open(&mut self, py: Python<'_>, head: &Py<PyAny>) -> PyResult<()> {
        Order::Outbound
            .walk(self.hooks.len())
            .try_for_each(|index| self.hooks[index].on_stream_open(py, head))
    }

    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        Order::Outbound
            .walk(self.hooks.len())
            .try_for_each(|index| self.hooks[index].on_stream_chunk(py, chunk))
    }
}

impl PythonOwned for HookChain {
    fn close(&mut self, py: Python<'_>) {
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
