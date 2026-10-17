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

use crate::{Error, HookStep, PythonCallEvent, PythonCallHooks, PythonOwned, PythonRuntime};

use super::adapter::{ChainHooks, ChainStep, HookAdapter};

type OwnedEvent = CallEvent<Py<PyAny>, Py<PyBaseException>, RawResponse>;

#[derive(Default)]
pub struct HookChain {
    hooks: Vec<Box<dyn ChainHooks>>,
    pending: Option<Pending>,
    event: Option<(usize, OwnedEvent)>,
}

/// Where a suspended transformation resumes: the hook it waits on and the context the rest of
/// the chain still needs.
enum Pending {
    Arguments(usize, f64),
    Wire(usize, RequestContext),
    ProviderResult(usize),
    Response(usize, Timing),
}

/// A stage that feeds each hook's result to the next hook.
trait Transform {
    type Value;
    type Context;

    fn call(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Self::Value,
        context: &Self::Context,
    ) -> PyResult<ChainStep<Self::Value>>;

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Self::Value>>;

    fn suspend(index: usize, context: Self::Context) -> Pending;

    fn resumed(pending: Pending) -> Option<(usize, Self::Context)>;
}

struct Arguments;

impl Transform for Arguments {
    type Value = Py<PyDict>;
    type Context = f64;

    fn call(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Py<PyDict>,
        started_at: &f64,
    ) -> PyResult<ChainStep<Py<PyDict>>> {
        hooks.prepare_arguments(py, value, *started_at)
    }

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyDict>>> {
        hooks.resume_arguments(py, result)
    }

    fn suspend(index: usize, started_at: f64) -> Pending {
        Pending::Arguments(index, started_at)
    }

    fn resumed(pending: Pending) -> Option<(usize, f64)> {
        match pending {
            Pending::Arguments(index, started_at) => Some((index, started_at)),
            _ => None,
        }
    }
}

struct Wire;

impl Transform for Wire {
    type Value = Box<WireRequest>;
    type Context = RequestContext;

    fn call(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<ChainStep<Box<WireRequest>>> {
        hooks.before_provider_request(py, value, context)
    }

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Box<WireRequest>>> {
        hooks.resume_wire(py, result)
    }

    fn suspend(index: usize, context: RequestContext) -> Pending {
        Pending::Wire(index, context)
    }

    fn resumed(pending: Pending) -> Option<(usize, RequestContext)> {
        match pending {
            Pending::Wire(index, context) => Some((index, context)),
            _ => None,
        }
    }
}

struct ProviderResult;

impl Transform for ProviderResult {
    type Value = Py<PyAny>;
    type Context = ();

    fn call(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Py<PyAny>,
        _: &(),
    ) -> PyResult<ChainStep<Py<PyAny>>> {
        hooks.provider_result(py, value)
    }

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyAny>>> {
        hooks.resume_response(py, result)
    }

    fn suspend(index: usize, (): ()) -> Pending {
        Pending::ProviderResult(index)
    }

    fn resumed(pending: Pending) -> Option<(usize, ())> {
        match pending {
            Pending::ProviderResult(index) => Some((index, ())),
            _ => None,
        }
    }
}

struct Response;

impl Transform for Response {
    type Value = Py<PyAny>;
    type Context = Timing;

    fn call(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Py<PyAny>,
        timing: &Timing,
    ) -> PyResult<ChainStep<Py<PyAny>>> {
        hooks.transform_response(py, value, *timing)
    }

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyAny>>> {
        hooks.resume_response(py, result)
    }

    fn suspend(index: usize, timing: Timing) -> Pending {
        Pending::Response(index, timing)
    }

    fn resumed(pending: Pending) -> Option<(usize, Timing)> {
        match pending {
            Pending::Response(index, timing) => Some((index, timing)),
            _ => None,
        }
    }
}

impl HookChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, hooks: impl PythonCallHooks + 'static) -> Self {
        self.hooks.push(Box::new(HookAdapter::new(hooks)));
        self
    }

    fn transform_from<T: Transform>(
        &mut self,
        py: Python<'_>,
        index: usize,
        value: T::Value,
        context: T::Context,
    ) -> PyResult<HookStep<Self, T::Value>> {
        let Some(hooks) = self.hooks.get_mut(index) else {
            return Ok(HookStep::Ready(value));
        };
        let step = T::call(hooks.as_mut(), py, value, &context)?;
        self.transform_step::<T>(py, index, step, context)
    }

    fn transform_step<T: Transform>(
        &mut self,
        py: Python<'_>,
        index: usize,
        step: ChainStep<T::Value>,
        context: T::Context,
    ) -> PyResult<HookStep<Self, T::Value>> {
        match step {
            ChainStep::Ready(value) => self.transform_from::<T>(py, index + 1, value, context),
            ChainStep::Await(awaitable) => {
                self.pending = Some(T::suspend(index, context));
                Ok(HookStep::Await(awaitable, Self::resume_transform::<T>))
            }
        }
    }

    fn resume_transform<T: Transform>(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, T::Value>> {
        let (index, context) = self
            .pending
            .take()
            .and_then(T::resumed)
            .ok_or(Error::MissingState)?;
        let result = resume_unless_cancelled(py, result)?;
        let step = T::resume(self.hooks[index].as_mut(), py, result)?;
        self.transform_step::<T>(py, index, step, context)
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
        let (index, event) = self.event.take().ok_or(Error::MissingState)?;
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
        self.transform_from::<Arguments>(py, 0, arguments, started_at)
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
        self.transform_from::<Wire>(py, 0, wire, context.clone())
    }

    fn reviews_provider_results(&mut self, py: Python<'_>) -> PyResult<bool> {
        for hooks in &mut self.hooks {
            if hooks.reviews_provider_results(py)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn provider_result(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        self.transform_from::<ProviderResult>(py, 0, response, ())
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        self.transform_from::<Response>(py, 0, response, timing)
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
        self.pending = None;
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
