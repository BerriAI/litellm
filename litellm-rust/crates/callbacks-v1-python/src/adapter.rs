//! The v1 lifecycle: each step builds one envelope, hands every subscribed observer a
//! fresh conversion of it, and, before the request is sent, folds every interceptor's
//! patch into the wire request.

use litellm_callbacks_v1::{
    CallFacts, Envelope, ErrorFacts, Event, RequestFacts, Sequencer, WirePatch, apply,
};
use litellm_host::event::{MachineEvent, RequestContext, WireRequest};
use litellm_host_python::{
    LifecycleEvent, LifecycleStep, PythonLifecycle, from_py, is_cancellation, missing_state, to_py,
};
use pyo3::exceptions::PyValueError;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use serde_json::{Map, Value};

use crate::call::V1PythonSurface;
use crate::python::V1Python;
use crate::subscribers::Subscriber;

/// What a step hands back to the driver once every observer of its envelope has run.
enum Continuation {
    Done,
    Arguments(Py<PyDict>),
    Wire(Box<WireRequest>),
}

/// The suspended step; `index` is the subscriber whose awaitable the driver is awaiting.
enum Pending {
    Observe {
        envelope: Envelope,
        index: usize,
        then: Continuation,
    },
    Intercept {
        wire: Box<WireRequest>,
        context: RequestContext,
        index: usize,
    },
}

pub struct V1PythonLifecycle {
    surface: V1PythonSurface,
    subscribers: Vec<Subscriber>,
    asynchronous: bool,
    sequencer: Option<Sequencer>,
    streamed: bool,
    pending: Option<Pending>,
    closed: bool,
}

impl Continuation {
    fn into_step(self) -> LifecycleStep {
        match self {
            Self::Done => LifecycleStep::Done,
            Self::Arguments(arguments) => LifecycleStep::Arguments(arguments),
            Self::Wire(wire) => LifecycleStep::Wire(wire),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Arguments(arguments) => visit.call(arguments),
            Self::Done | Self::Wire(_) => Ok(()),
        }
    }
}

impl V1PythonLifecycle {
    pub(crate) fn new(
        surface: V1PythonSurface,
        subscribers: Vec<Subscriber>,
        asynchronous: bool,
    ) -> Self {
        Self {
            surface,
            subscribers,
            asynchronous,
            sequencer: None,
            streamed: false,
            pending: None,
            closed: false,
        }
    }

    fn envelope(&mut self, event: Event) -> PyResult<Envelope> {
        self.sequencer
            .as_mut()
            .map(|sequencer| sequencer.envelope(event))
            .ok_or_else(missing_state)
    }

    fn report(
        &self,
        py: Python<'_>,
        index: usize,
        envelope: &Envelope,
        error: PyErr,
    ) -> PyResult<()> {
        let name = &self.subscribers[index].name;
        let event: &'static str = envelope.event.kind().into();
        match V1Python::Report.call(py, (name, event, error.value(py))) {
            Err(report_error) if is_cancellation(py, &report_error) => Err(report_error),
            _ => Ok(()),
        }
    }

    fn observe(
        &mut self,
        py: Python<'_>,
        envelope: Envelope,
        from: usize,
        then: Continuation,
    ) -> PyResult<LifecycleStep> {
        for index in from..self.subscribers.len() {
            let Some(handler) = self.subscribers[index]
                .observer(envelope.event.kind())
                .map(|handler| handler.clone_ref(py))
            else {
                continue;
            };
            let outcome = to_py(py, &envelope).and_then(|argument| handler.call(py, argument));
            match outcome {
                Ok(awaitable) if handler.is_async() => {
                    self.pending = Some(Pending::Observe {
                        envelope,
                        index,
                        then,
                    });
                    return Ok(LifecycleStep::Await(awaitable.unbind()));
                }
                Ok(_) => {}
                Err(error) if is_cancellation(py, &error) => return Err(error),
                Err(error) => self.report(py, index, &envelope, error)?,
            }
        }
        Ok(then.into_step())
    }

    fn apply_patch(
        &self,
        py: Python<'_>,
        index: usize,
        wire: WireRequest,
        result: Py<PyAny>,
    ) -> PyResult<WireRequest> {
        let patch = if result.bind(py).is_none() {
            WirePatch::default()
        } else {
            from_py::<WirePatch>(result.bind(py))?
        };
        let name = &self.subscribers[index].name;
        apply(wire, patch)
            .map_err(|error| PyValueError::new_err(format!("callback {name}: {error}")))
    }

    fn intercept(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: RequestContext,
        from: usize,
    ) -> PyResult<LifecycleStep> {
        let mut current = wire;
        for index in from..self.subscribers.len() {
            let Some(handler) = self.subscribers[index]
                .intercept
                .as_ref()
                .map(|handler| handler.clone_ref(py))
            else {
                continue;
            };
            let request = to_py(py, &RequestFacts::new(&current, &context))?;
            let result = handler.call(py, request)?;
            if handler.is_async() {
                self.pending = Some(Pending::Intercept {
                    wire: current,
                    context,
                    index,
                });
                return Ok(LifecycleStep::Await(result.unbind()));
            }
            current = Box::new(self.apply_patch(py, index, *current, result.unbind())?);
        }
        let envelope =
            self.envelope(Event::RequestSending(RequestFacts::new(&current, &context)))?;
        self.observe(py, envelope, 0, Continuation::Wire(current))
    }

    fn metadata(arguments: &Bound<'_, PyDict>) -> PyResult<(Map<String, Value>, Vec<String>)> {
        let Some(value) = arguments.get_item("metadata")? else {
            return Ok((Map::new(), Vec::new()));
        };
        let Ok(metadata) = value.cast::<PyDict>() else {
            return Ok((Map::new(), vec!["metadata".to_string()]));
        };
        let mut included = Map::new();
        let mut dropped = Vec::new();
        for (key, value) in metadata.iter() {
            match key.extract::<String>() {
                Ok(name) => match from_py::<Value>(&value) {
                    Ok(value) => {
                        included.insert(name, value);
                    }
                    Err(_) => dropped.push(name),
                },
                Err(_) => dropped.push(key.str()?.to_string()),
            }
        }
        Ok((included, dropped))
    }

    fn call_id(py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<String> {
        if let Some(value) = arguments.get_item("litellm_call_id")?
            && value.is_instance_of::<PyString>()
        {
            return value.extract();
        }
        V1Python::NewCallId.call(py, ())?.extract()
    }

    fn error_facts(py: Python<'_>, error: &PyErr) -> PyResult<ErrorFacts> {
        let class = error.get_type(py);
        let module = class.getattr("__module__")?.extract::<String>()?;
        let qualified = class.getattr("__qualname__")?.extract::<String>()?;
        let status_code = error
            .value(py)
            .getattr("status_code")
            .ok()
            .and_then(|value| value.extract::<u16>().ok());
        Ok(ErrorFacts {
            class: format!("{module}.{qualified}"),
            message: error.value(py).str()?.to_string(),
            status_code,
        })
    }
}

impl PythonLifecycle for V1PythonLifecycle {
    fn begin(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        start_time: f64,
    ) -> PyResult<LifecycleStep> {
        let call_id = Self::call_id(py, arguments.bind(py))?;
        let (metadata, metadata_dropped) = Self::metadata(arguments.bind(py))?;
        self.sequencer = Some(Sequencer::new(call_id, self.surface.call_type));
        let envelope = self.envelope(Event::CallStarted(CallFacts {
            start_time,
            asynchronous: self.asynchronous,
            metadata,
            metadata_dropped,
        }))?;
        self.observe(py, envelope, 0, Continuation::Arguments(arguments))
    }

    fn before_send(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<LifecycleStep> {
        self.intercept(py, wire, context.clone(), 0)
    }

    fn emit(&mut self, py: Python<'_>, event: LifecycleEvent<'_>) -> PyResult<LifecycleStep> {
        match event {
            LifecycleEvent::Machine(MachineEvent::ResponseReceived { raw }) => {
                let envelope = self.envelope(Event::ResponseReceived {
                    body: raw.body.clone(),
                })?;
                self.observe(py, envelope, 0, Continuation::Done)
            }
            LifecycleEvent::Succeeded { timing, response } => {
                let projected = V1Python::ProjectResponse
                    .call(py, (response,))
                    .and_then(|value| from_py::<Value>(&value));
                let (response, response_error) = match projected {
                    Ok(response) => (response, None),
                    Err(error) if is_cancellation(py, &error) => return Err(error),
                    Err(error) => (Value::Null, Some(error.to_string())),
                };
                let envelope = self.envelope(Event::CallSucceeded {
                    timing: timing.into(),
                    streamed: self.streamed,
                    response,
                    response_error,
                })?;
                self.observe(py, envelope, 0, Continuation::Done)
            }
            LifecycleEvent::Failed {
                timing,
                origin,
                error,
            } => {
                let facts = Self::error_facts(py, error)?;
                let envelope = self.envelope(Event::CallFailed {
                    timing: timing.into(),
                    streamed: self.streamed,
                    origin: origin.into(),
                    error: facts,
                })?;
                self.observe(py, envelope, 0, Continuation::Done)
            }
        }
    }

    fn opened(&mut self, _py: Python<'_>) -> PyResult<()> {
        self.streamed = true;
        Ok(())
    }

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep> {
        match self.pending.take().ok_or_else(missing_state)? {
            Pending::Observe {
                envelope,
                index,
                then,
            } => {
                if let Err(error) = result {
                    if is_cancellation(py, &error) {
                        return Err(error);
                    }
                    self.report(py, index, &envelope, error)?;
                }
                self.observe(py, envelope, index + 1, then)
            }
            Pending::Intercept {
                wire,
                context,
                index,
            } => {
                let patched = self.apply_patch(py, index, *wire, result?)?;
                self.intercept(py, Box::new(patched), context, index + 1)
            }
        }
    }

    fn close(&mut self, _py: Python<'_>) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.pending = None;
        self.subscribers.clear();
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for subscriber in &self.subscribers {
            subscriber.traverse(visit)?;
        }
        match &self.pending {
            Some(Pending::Observe { then, .. }) => then.traverse(visit),
            Some(Pending::Intercept { .. }) | None => Ok(()),
        }
    }
}
