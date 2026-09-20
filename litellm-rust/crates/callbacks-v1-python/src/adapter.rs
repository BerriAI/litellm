//! The v1 lifecycle in CPython. The contract's [`CallSession`] decides which envelope a
//! step produces, who receives it and how patches fold into the wire; this adapter only
//! projects Python values into facts and runs the Python handlers: every observer of an
//! emission gets a fresh conversion of its envelope, inline, and every interceptor's
//! answer goes back to the session.

use litellm_callbacks_v1::{
    CallFacts, CallSession, Emission, Envelope, ErrorFacts, Interception, Subscription, WirePatch,
};
use litellm_host::event::{MachineEvent, RequestContext, WireRequest};
use litellm_host_python::{
    LifecycleEvent, LifecycleStep, PythonLifecycle, from_py, is_cancellation, missing_state, to_py,
};
use pyo3::{
    exceptions::PyValueError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyString},
};
use serde_json::{Map, Value};

use crate::{call::V1PythonSurface, python::V1Python, subscribers::Handlers};

/// What a step hands back to the driver once every observer of its envelope has run.
enum Continuation {
    Done,
    Arguments(Py<PyDict>),
    Wire(Box<WireRequest>),
}

/// The suspended step, while the driver awaits one handler's awaitable.
enum Pending {
    /// `position` is the place in `emission.observers` of the observer being awaited.
    Observe {
        emission: Emission,
        position: usize,
        then: Continuation,
    },
    /// `subscriber` is the interceptor whose turn it is.
    Intercept {
        interception: Interception,
        subscriber: usize,
    },
}

pub struct V1PythonLifecycle {
    surface: V1PythonSurface,
    /// Held until `begin` learns the call's identity and hands them to the session.
    subscriptions: Vec<Subscription>,
    /// The Python side of each subscription, at the same index.
    handlers: Vec<Handlers>,
    asynchronous: bool,
    session: Option<CallSession>,
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
        subscriptions: Vec<Subscription>,
        handlers: Vec<Handlers>,
        asynchronous: bool,
    ) -> Self {
        Self {
            surface,
            subscriptions,
            handlers,
            asynchronous,
            session: None,
            pending: None,
            closed: false,
        }
    }

    fn report(
        &self,
        py: Python<'_>,
        index: usize,
        envelope: &Envelope,
        error: PyErr,
    ) -> PyResult<()> {
        let name = &self.handlers[index].name;
        let event: &'static str = envelope.event.kind().into();
        match V1Python::Report.call(py, (name, event, error.value(py))) {
            Err(report_error) if is_cancellation(py, &report_error) => Err(report_error),
            _ => Ok(()),
        }
    }

    /// Runs the emission's observers inline from `from`, which is one legal schedule of
    /// the contract and nothing a callback may rely on.
    fn observe(
        &mut self,
        py: Python<'_>,
        emission: Emission,
        from: usize,
        then: Continuation,
    ) -> PyResult<LifecycleStep> {
        for position in from..emission.observers.len() {
            let index = emission.observers[position];
            let Some(handler) = self.handlers[index]
                .observe
                .as_ref()
                .map(|handler| handler.clone_ref(py))
            else {
                continue;
            };
            let outcome =
                to_py(py, &emission.envelope).and_then(|argument| handler.call(py, argument));
            match outcome {
                Ok(awaitable) if handler.is_async() => {
                    self.pending = Some(Pending::Observe {
                        emission,
                        position,
                        then,
                    });
                    return Ok(LifecycleStep::Await(awaitable.unbind()));
                }
                Ok(_) => {}
                Err(error) if is_cancellation(py, &error) => return Err(error),
                Err(error) => self.report(py, index, &emission.envelope, error)?,
            }
        }
        Ok(then.into_step())
    }

    fn patched(
        &self,
        py: Python<'_>,
        interception: Interception,
        subscriber: usize,
        result: Py<PyAny>,
    ) -> PyResult<Interception> {
        let patch = if result.bind(py).is_none() {
            WirePatch::default()
        } else {
            from_py::<WirePatch>(result.bind(py))?
        };
        let name = &self.handlers[subscriber].name;
        interception
            .patched(patch)
            .map_err(|error| PyValueError::new_err(format!("callback {name}: {error}")))
    }

    fn intercept(&mut self, py: Python<'_>, interception: Interception) -> PyResult<LifecycleStep> {
        let mut current = interception;
        while let Some(turn) = current.turn() {
            let Some(handler) = self.handlers[turn.subscriber]
                .intercept
                .as_ref()
                .map(|handler| handler.clone_ref(py))
            else {
                return Err(missing_state());
            };
            let result = handler.call(py, to_py(py, &turn.request)?)?;
            if handler.is_async() {
                self.pending = Some(Pending::Intercept {
                    interception: current,
                    subscriber: turn.subscriber,
                });
                return Ok(LifecycleStep::Await(result.unbind()));
            }
            current = self.patched(py, current, turn.subscriber, result.unbind())?;
        }
        let (emission, wire) = self
            .session
            .as_mut()
            .ok_or_else(missing_state)?
            .request_sending(current);
        self.observe(py, emission, 0, Continuation::Wire(Box::new(wire)))
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
        let (session, emission) = CallSession::start(
            call_id,
            self.surface.call_type,
            std::mem::take(&mut self.subscriptions),
            CallFacts {
                start_time,
                asynchronous: self.asynchronous,
                metadata,
                metadata_dropped,
            },
        );
        self.session = Some(session);
        self.observe(py, emission, 0, Continuation::Arguments(arguments))
    }

    fn before_send(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<LifecycleStep> {
        let interception = self
            .session
            .as_ref()
            .ok_or_else(missing_state)?
            .interception(*wire, context.clone());
        self.intercept(py, interception)
    }

    fn emit(&mut self, py: Python<'_>, event: LifecycleEvent<'_>) -> PyResult<LifecycleStep> {
        match event {
            LifecycleEvent::Machine(MachineEvent::ResponseReceived { raw }) => {
                let emission = self
                    .session
                    .as_mut()
                    .ok_or_else(missing_state)?
                    .response_received(raw.body.clone());
                self.observe(py, emission, 0, Continuation::Done)
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
                let session = self.session.take().ok_or_else(missing_state)?;
                let emission = session.succeeded(timing, response, response_error);
                self.observe(py, emission, 0, Continuation::Done)
            }
            LifecycleEvent::Failed {
                timing,
                origin,
                error,
            } => {
                let facts = Self::error_facts(py, error)?;
                let session = self.session.take().ok_or_else(missing_state)?;
                let emission = session.failed(timing, origin, facts);
                self.observe(py, emission, 0, Continuation::Done)
            }
        }
    }

    fn opened(&mut self, _py: Python<'_>) -> PyResult<()> {
        if let Some(session) = self.session.as_mut() {
            session.opened();
        }
        Ok(())
    }

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep> {
        match self.pending.take().ok_or_else(missing_state)? {
            Pending::Observe {
                emission,
                position,
                then,
            } => {
                if let Err(error) = result {
                    if is_cancellation(py, &error) {
                        return Err(error);
                    }
                    self.report(py, emission.observers[position], &emission.envelope, error)?;
                }
                self.observe(py, emission, position + 1, then)
            }
            Pending::Intercept {
                interception,
                subscriber,
            } => {
                let patched = self.patched(py, interception, subscriber, result?)?;
                self.intercept(py, patched)
            }
        }
    }

    fn close(&mut self, _py: Python<'_>) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.pending = None;
        self.session = None;
        self.subscriptions.clear();
        self.handlers.clear();
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for handlers in &self.handlers {
            handlers.traverse(visit)?;
        }
        match &self.pending {
            Some(Pending::Observe { then, .. }) => then.traverse(visit),
            Some(Pending::Intercept { .. }) | None => Ok(()),
        }
    }
}
