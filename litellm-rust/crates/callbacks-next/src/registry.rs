use std::collections::BTreeSet;

use pyo3::exceptions::PyValueError;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;

use crate::envelope::{EventKind, SCHEMA_V1};
use crate::next_python::NextPython;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Sync,
    Async,
}

pub(crate) fn select(asynchronous: bool, has_sync: bool, has_async: bool) -> Option<Kind> {
    match (asynchronous, has_sync, has_async) {
        (true, _, true) => Some(Kind::Async),
        (_, true, _) => Some(Kind::Sync),
        _ => None,
    }
}

fn selected_handler(
    subscriber: &Bound<'_, PyAny>,
    asynchronous: bool,
    sync_name: &str,
    async_name: &str,
) -> PyResult<Option<Handler>> {
    let sync = subscriber.getattr(sync_name)?;
    let asynchronous_handler = subscriber.getattr(async_name)?;
    let kind = select(
        asynchronous,
        !sync.is_none(),
        !asynchronous_handler.is_none(),
    );
    Ok(match kind {
        Some(Kind::Sync) => Some(Handler::Sync(sync.unbind())),
        Some(Kind::Async) => Some(Handler::Async(asynchronous_handler.unbind())),
        None => None,
    })
}

pub fn snapshot(py: Python<'_>, asynchronous: bool) -> PyResult<Vec<Subscriber>> {
    NextPython::Snapshot
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(true, true, true, Some(Kind::Async))]
    #[case(true, false, true, Some(Kind::Async))]
    #[case(true, true, false, Some(Kind::Sync))]
    #[case(false, true, true, Some(Kind::Sync))]
    #[case(false, true, false, Some(Kind::Sync))]
    #[case(false, false, true, None)]
    #[case(false, false, false, None)]
    fn handler_selection_never_selects_async_for_sync_calls(
        #[case] asynchronous: bool,
        #[case] has_sync: bool,
        #[case] has_async: bool,
        #[case] expected: Option<Kind>,
    ) {
        assert_eq!(select(asynchronous, has_sync, has_async), expected);
    }
}
