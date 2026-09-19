use litellm_host::event::{MachineEvent, RequestContext, Timing, WireRequest};
use litellm_host_python::{
    LifecycleEvent, LifecycleStep, PythonLifecycle, from_py, missing_state, to_py,
};
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use serde_json::{Map, Value};

use crate::call::NextSurface;
use crate::envelope::{CallFacts, Envelope, ErrorFacts, Event, RequestFacts, Sequencer};
use crate::next_python::NextPython;
use crate::patch::{WirePatch, apply};
use crate::registry::{Handler, Subscriber};

enum Then {
    Done,
    Arguments(Py<PyDict>),
    Wire(Box<WireRequest>),
}

enum Pending {
    Observe {
        envelope: Envelope,
        next: usize,
        then: Then,
        awaiting: String,
    },
    Intercept {
        wire: Box<WireRequest>,
        context: RequestContext,
        next: usize,
        awaiting: String,
    },
}

pub struct NextLifecycle {
    surface: NextSurface,
    subscribers: Vec<Subscriber>,
    asynchronous: bool,
    start_time: f64,
    sequencer: Option<Sequencer>,
    streamed: bool,
    pending: Option<Pending>,
    closed: bool,
}

fn is_cancellation(py: Python<'_>, error: &PyErr) -> bool {
    !error.is_instance_of::<PyException>(py)
}

impl Then {
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

impl NextLifecycle {
    pub fn new(surface: NextSurface, subscribers: Vec<Subscriber>, asynchronous: bool) -> Self {
        Self {
            surface,
            subscribers,
            asynchronous,
            start_time: 0.0,
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
        name: &str,
        envelope: &Envelope,
        error: PyErr,
    ) -> PyResult<()> {
        let event: &'static str = envelope.event.kind().into();
        match NextPython::Report.call(py, (name, event, error.value(py))) {
            Err(report_error) if is_cancellation(py, &report_error) => Err(report_error),
            _ => Ok(()),
        }
    }

    fn observer_at(
        &self,
        py: Python<'_>,
        index: usize,
        envelope: &Envelope,
    ) -> Option<(String, bool, Py<PyAny>)> {
        let subscriber = &self.subscribers[index];
        if subscriber.schema != envelope.schema
            || !subscriber.events.contains(&envelope.event.kind())
        {
            return None;
        }
        subscriber.observe.as_ref().map(|handler| {
            let asynchronous = matches!(handler, Handler::Async(_));
            (
                subscriber.name.clone(),
                asynchronous,
                handler.object().clone_ref(py),
            )
        })
    }

    fn observe(
        &mut self,
        py: Python<'_>,
        envelope: Envelope,
        from: usize,
        then: Then,
    ) -> PyResult<LifecycleStep> {
        for index in from..self.subscribers.len() {
            let Some((name, asynchronous, handler)) = self.observer_at(py, index, &envelope) else {
                continue;
            };
            let argument = match to_py(py, &envelope) {
                Ok(argument) => argument,
                Err(error) => {
                    self.report(py, &name, &envelope, error)?;
                    continue;
                }
            };
            match handler.bind(py).call1((argument,)) {
                Ok(awaitable) if asynchronous => {
                    self.pending = Some(Pending::Observe {
                        envelope,
                        next: index + 1,
                        then,
                        awaiting: name,
                    });
                    return Ok(LifecycleStep::Await(awaitable.unbind()));
                }
                Ok(_) => {}
                Err(error) if is_cancellation(py, &error) => return Err(error),
                Err(error) => self.report(py, &name, &envelope, error)?,
            }
        }
        Ok(then.into_step())
    }

    fn apply_patch(
        &self,
        py: Python<'_>,
        name: &str,
        wire: WireRequest,
        result: Py<PyAny>,
    ) -> PyResult<WireRequest> {
        let patch = if result.bind(py).is_none() {
            WirePatch::default()
        } else {
            from_py::<WirePatch>(result.bind(py))?
        };
        apply(wire, patch)
            .map_err(|error| PyValueError::new_err(format!("callback {name}: {error}")))
    }

    fn interceptor_at(&self, py: Python<'_>, index: usize) -> Option<(String, bool, Py<PyAny>)> {
        let subscriber = &self.subscribers[index];
        if subscriber.schema != crate::envelope::SCHEMA_V1 {
            return None;
        }
        subscriber.intercept.as_ref().map(|handler| {
            let asynchronous = matches!(handler, Handler::Async(_));
            (
                subscriber.name.clone(),
                asynchronous,
                handler.object().clone_ref(py),
            )
        })
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
            let Some((name, asynchronous, handler)) = self.interceptor_at(py, index) else {
                continue;
            };
            let request = to_py(py, &RequestFacts::new(&current, &context))?;
            match handler.bind(py).call1((request,)) {
                Ok(awaitable) if asynchronous => {
                    self.pending = Some(Pending::Intercept {
                        wire: current,
                        context,
                        next: index + 1,
                        awaiting: name,
                    });
                    return Ok(LifecycleStep::Await(awaitable.unbind()));
                }
                Ok(result) => {
                    current = Box::new(self.apply_patch(py, &name, *current, result.unbind())?);
                }
                Err(error) => return Err(error),
            }
        }
        let envelope =
            self.envelope(Event::RequestSending(RequestFacts::new(&current, &context)))?;
        self.observe(py, envelope, 0, Then::Wire(current))
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
        NextPython::NewCallId.call(py, ())?.extract()
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

impl PythonLifecycle for NextLifecycle {
    fn begin(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        _started_at: f64,
    ) -> PyResult<LifecycleStep> {
        let call_id = Self::call_id(py, arguments.bind(py))?;
        let (metadata, metadata_dropped) = Self::metadata(arguments.bind(py))?;
        self.sequencer = Some(Sequencer::new(call_id, self.surface.call_type));
        let envelope = self.envelope(Event::CallStarted(CallFacts {
            start_time: self.start_time,
            asynchronous: self.asynchronous,
            metadata,
            metadata_dropped,
        }))?;
        self.observe(py, envelope, 0, Then::Arguments(arguments))
    }

    fn before_send(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<LifecycleStep> {
        self.intercept(py, wire, context.clone(), 0)
    }

    fn after_success(
        &mut self,
        _py: Python<'_>,
        response: Py<PyAny>,
        _timing: Timing,
    ) -> PyResult<LifecycleStep> {
        Ok(LifecycleStep::Response(response))
    }

    fn emit(&mut self, py: Python<'_>, event: LifecycleEvent<'_>) -> PyResult<LifecycleStep> {
        match event {
            LifecycleEvent::Started { start_time } => {
                self.start_time = start_time;
                Ok(LifecycleStep::Done)
            }
            LifecycleEvent::Machine(MachineEvent::ResponseReceived { raw }) => {
                let envelope = self.envelope(Event::ResponseReceived {
                    body: raw.body.clone(),
                })?;
                self.observe(py, envelope, 0, Then::Done)
            }
            LifecycleEvent::Succeeded { timing, response } => {
                let projection = NextPython::ProjectResponse.call(py, (response,));
                let (response, response_error) = match projection {
                    Ok(value) => match from_py::<Value>(&value) {
                        Ok(response) => (response, None),
                        Err(error) if is_cancellation(py, &error) => return Err(error),
                        Err(error) => (Value::Null, Some(error.to_string())),
                    },
                    Err(error) if is_cancellation(py, &error) => return Err(error),
                    Err(error) => (Value::Null, Some(error.to_string())),
                };
                let envelope = self.envelope(Event::CallSucceeded {
                    timing: timing.into(),
                    streamed: self.streamed,
                    response,
                    response_error,
                })?;
                self.observe(py, envelope, 0, Then::Done)
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
                self.observe(py, envelope, 0, Then::Done)
            }
        }
    }

    fn opened(&mut self, _py: Python<'_>) -> PyResult<()> {
        self.streamed = true;
        Ok(())
    }

    fn delivered(&mut self, _py: Python<'_>, _chunk: &Py<PyAny>) -> PyResult<()> {
        Ok(())
    }

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep> {
        match self.pending.take().ok_or_else(missing_state)? {
            Pending::Observe {
                envelope,
                next,
                then,
                awaiting,
            } => {
                if let Err(error) = result {
                    if is_cancellation(py, &error) {
                        return Err(error);
                    }
                    self.report(py, &awaiting, &envelope, error)?;
                }
                self.observe(py, envelope, next, then)
            }
            Pending::Intercept {
                wire,
                context,
                next,
                awaiting,
            } => {
                let value = result?;
                let patched = self.apply_patch(py, &awaiting, *wire, value)?;
                self.intercept(py, Box::new(patched), context, next)
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
            if let Some(handler) = &subscriber.observe {
                handler.traverse(visit)?;
            }
            if let Some(handler) = &subscriber.intercept {
                handler.traverse(visit)?;
            }
        }
        match &self.pending {
            Some(Pending::Observe { then, .. }) => then.traverse(visit),
            Some(Pending::Intercept { .. }) | None => Ok(()),
        }
    }
}
