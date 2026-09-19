use std::collections::BTreeSet;

use litellm_callbacks_v1::{EventKind, ExecutionMode, HandlerSelection, SCHEMA_V1, select_handler};
use pyo3::exceptions::PyValueError;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;

use crate::python::V1Python;

pub enum Handler {
    Sync(Py<PyAny>),
    Async(Py<PyAny>),
}

pub struct Subscriber {
    pub name: String,
    pub schema: u32,
    pub events: BTreeSet<EventKind>,
    pub observe: Option<Handler>,
    pub intercept: Option<Handler>,
}

fn selected_handler(
    subscriber: &Bound<'_, PyAny>,
    asynchronous: bool,
    sync_name: &str,
    async_name: &str,
) -> PyResult<Option<Handler>> {
    let sync = subscriber.getattr(sync_name)?;
    let asynchronous_handler = subscriber.getattr(async_name)?;
    let selection = select_handler(
        if asynchronous {
            ExecutionMode::Async
        } else {
            ExecutionMode::Sync
        },
        !sync.is_none(),
        !asynchronous_handler.is_none(),
    );
    Ok(match selection {
        HandlerSelection::Sync => Some(Handler::Sync(sync.unbind())),
        HandlerSelection::Async => Some(Handler::Async(asynchronous_handler.unbind())),
        HandlerSelection::Skip => None,
    })
}

pub fn snapshot(py: Python<'_>, asynchronous: bool) -> PyResult<Vec<Subscriber>> {
    V1Python::Snapshot
        .call(py, ())?
        .try_iter()?
        .map(|item| {
            let subscriber = item?;
            let schema = subscriber.getattr("schema")?.extract::<u32>()?;
            if schema != SCHEMA_V1 {
                return Err(PyValueError::new_err(format!(
                    "unsupported callback schema: {schema}"
                )));
            }
            let event_names = subscriber
                .getattr("events")?
                .extract::<BTreeSet<String>>()?;
            let events = event_names
                .into_iter()
                .map(|name| {
                    name.parse::<EventKind>().map_err(|_| {
                        PyValueError::new_err(format!("unknown callback event: {name}"))
                    })
                })
                .collect::<PyResult<BTreeSet<_>>>()?;
            Ok(Subscriber {
                name: subscriber.getattr("name")?.extract()?,
                schema,
                events,
                observe: selected_handler(&subscriber, asynchronous, "on_event", "async_on_event")?,
                intercept: selected_handler(
                    &subscriber,
                    asynchronous,
                    "before_send",
                    "async_before_send",
                )?,
            })
        })
        .collect()
}

impl Handler {
    pub fn object(&self) -> &Py<PyAny> {
        match self {
            Self::Sync(value) | Self::Async(value) => value,
        }
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.object())
    }
}
