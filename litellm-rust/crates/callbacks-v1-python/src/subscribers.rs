//! The per-call subscriber snapshot: what Python's `register` recorded, read once, with
//! one handler chosen per role for the call's execution mode. `register` has already
//! rejected everything rejectable, so this only parses. Each subscriber splits in two: the
//! [`Subscription`] the contract's session routes by, and the [`Handlers`] only this host
//! can call.

use std::collections::BTreeSet;

use litellm_callbacks_v1::{
    EventKind, ExecutionMode, HandlerSelection, Subscription, select_handler,
};
use pyo3::{
    exceptions::PyValueError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
};

use crate::python::V1Python;

/// One callable of a subscriber, tagged with whether calling it returns an awaitable.
pub enum Handler {
    Sync(Py<PyAny>),
    Async(Py<PyAny>),
}

/// The Python callables of one subscriber, at the index of its [`Subscription`].
pub struct Handlers {
    pub name: String,
    /// `on_event` or `async_on_event`: cannot affect the call.
    pub observe: Option<Handler>,
    /// `before_send` or `async_before_send`: returns a patch; its failure fails the call.
    pub intercept: Option<Handler>,
}

fn selected_handler(
    subscriber: &Bound<'_, PyAny>,
    mode: ExecutionMode,
    sync_name: &str,
    async_name: &str,
) -> PyResult<Option<Handler>> {
    let sync = subscriber.getattr(sync_name)?;
    let asynchronous = subscriber.getattr(async_name)?;
    Ok(
        match select_handler(mode, !sync.is_none(), !asynchronous.is_none()) {
            HandlerSelection::Sync => Some(Handler::Sync(sync.unbind())),
            HandlerSelection::Async => Some(Handler::Async(asynchronous.unbind())),
            HandlerSelection::Skip => None,
        },
    )
}

pub fn snapshot(py: Python<'_>, asynchronous: bool) -> PyResult<Vec<(Subscription, Handlers)>> {
    let mode = if asynchronous {
        ExecutionMode::Async
    } else {
        ExecutionMode::Sync
    };
    V1Python::Snapshot
        .call(py, ())?
        .try_iter()?
        .map(|item| {
            let subscriber = item?;
            let events = subscriber
                .getattr("events")?
                .extract::<BTreeSet<String>>()?
                .into_iter()
                .map(|name| {
                    name.parse::<EventKind>().map_err(|_| {
                        PyValueError::new_err(format!("unknown callback event: {name}"))
                    })
                })
                .collect::<PyResult<BTreeSet<_>>>()?;
            let name: String = subscriber.getattr("name")?.extract()?;
            let observe = selected_handler(&subscriber, mode, "on_event", "async_on_event")?;
            let intercept =
                selected_handler(&subscriber, mode, "before_send", "async_before_send")?;
            Ok((
                Subscription {
                    name: name.clone(),
                    events,
                    observes: observe.is_some(),
                    intercepts: intercept.is_some(),
                },
                Handlers {
                    name,
                    observe,
                    intercept,
                },
            ))
        })
        .collect()
}

impl Handlers {
    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for handler in self.observe.iter().chain(&self.intercept) {
            visit.call(handler.object())?;
        }
        Ok(())
    }
}

impl Handler {
    fn object(&self) -> &Py<PyAny> {
        match self {
            Self::Sync(value) | Self::Async(value) => value,
        }
    }

    pub fn is_async(&self) -> bool {
        matches!(self, Self::Async(_))
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Sync(value) => Self::Sync(value.clone_ref(py)),
            Self::Async(value) => Self::Async(value.clone_ref(py)),
        }
    }

    pub fn call<'py>(&self, py: Python<'py>, argument: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.object().bind(py).call1((argument,))
    }
}
