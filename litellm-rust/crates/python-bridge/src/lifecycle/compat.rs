use litellm_core::call_lifecycle::{
    CallbackFamily, Delivery, ReleaseGate, SuccessFacts, plan_success,
};
use pyo3::prelude::*;

use super::bindings::PythonLogger;
use super::{PythonCallState, dispatch, missing_state};

pub(super) fn dispatch_success(
    py: Python<'_>,
    state: &PythonCallState,
    logger: &PythonLogger,
) -> PyResult<()> {
    let facts = SuccessFacts {
        asynchronous: state.asynchronous,
        internal: state.internal,
        fallbacks: !state
            .kwargs
            .bind(py)
            .get_item("fallbacks")?
            .is_none_or(|value| value.is_none()),
        deferred: logger.defers_async_logging(py),
        sync_target_kinds: sync_kinds(py, logger)?,
    };
    let leaves = dispatch::leaves(py)?;
    let object = logger.object(py);
    for selected in plan_success(&facts) {
        match (selected.family, selected.delivery, selected.gate) {
            (CallbackFamily::SyncSuccess, Delivery::Worker, _) => {
                let handler = object.getattr("success_handler")?;
                let bound = py.import("functools")?.getattr("partial")?.call1((
                    handler,
                    &state.response,
                    &state.start,
                    &state.end,
                ))?;
                leaves.getattr("submit_worker")?.call1((bound,))?;
            }
            (CallbackFamily::AsyncSuccess, Delivery::Background, ReleaseGate::Immediate) => {
                let coroutine = object.call_method1(
                    "async_success_handler",
                    (&state.response, &state.start, &state.end),
                )?;
                let enqueue = leaves.getattr("enqueue_background")?.call1((&coroutine,));
                if enqueue.is_err()
                    && let Err(error) = coroutine.call_method0("close")
                {
                    error.write_unraisable(py, Some(&coroutine));
                }
                enqueue?;
            }
            (CallbackFamily::AsyncSuccess, Delivery::Background, ReleaseGate::Deferred) => {
                let pending = Py::new(
                    py,
                    SuppliedDeferred {
                        logger: Some(logger.clone_ref(py)),
                        response: state.response.as_ref().map(|value| value.clone_ref(py)),
                        start: state.start.clone_ref(py),
                        end: state.end.as_ref().map(|value| value.clone_ref(py)),
                    },
                )?;
                object.setattr("_native_pending_logging", pending)?;
            }
            _ => return Err(missing_state()),
        }
    }
    Ok(())
}

fn sync_kinds(
    py: Python<'_>,
    logger: &PythonLogger,
) -> PyResult<Vec<litellm_core::call_lifecycle::CallbackKind>> {
    let (targets, ids) = dispatch::family_targets(py, logger, CallbackFamily::SyncSuccess)?;
    Ok(targets.kinds(&ids))
}

pub(super) fn dispatch_failure(
    py: Python<'_>,
    state: &PythonCallState,
    family: CallbackFamily,
) -> PyResult<Option<Py<PyAny>>> {
    let logger = state.logger()?.object(py);
    let error = state.error.as_ref().ok_or_else(missing_state)?;
    let trace = py
        .import("traceback")?
        .getattr("format_exception")?
        .call1((error,))?;
    let trace = pyo3::types::PyString::new(py, "").call_method1("join", (trace,))?;
    match family {
        CallbackFamily::SyncFailure => {
            logger.call_method1("failure_handler", (error, trace, &state.start, &state.end))?;
            Ok(None)
        }
        CallbackFamily::AsyncFailure => Ok(Some(
            logger
                .call_method1(
                    "async_failure_handler",
                    (error, trace, &state.start, &state.end),
                )?
                .unbind(),
        )),
        _ => Err(missing_state()),
    }
}

#[pyclass]
struct SuppliedDeferred {
    logger: Option<PythonLogger>,
    response: Option<Py<PyAny>>,
    start: Py<PyAny>,
    end: Option<Py<PyAny>>,
}

#[pymethods]
impl SuppliedDeferred {
    fn __call__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let logger = slf.borrow_mut().logger.take();
        let Some(logger) = logger else {
            return Ok(());
        };
        let (response, start, end) = {
            let this = slf.borrow();
            (
                this.response.as_ref().map(|value| value.clone_ref(py)),
                this.start.clone_ref(py),
                this.end.as_ref().map(|value| value.clone_ref(py)),
            )
        };
        let object = logger.object(py);
        let coroutine = object.call_method1("async_success_handler", (response, start, end))?;
        let enqueue = dispatch::leaves(py)?
            .getattr("enqueue_background")?
            .call1((&coroutine,));
        if enqueue.is_err()
            && let Err(error) = coroutine.call_method0("close")
        {
            error.write_unraisable(py, Some(&coroutine));
        }
        match enqueue {
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                error.write_unraisable(py, Some(object));
                Ok(())
            }
            result => result.map(|_| ()),
        }
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        if let Some(logger) = &self.logger {
            logger.traverse(&visit)?;
        }
        visit.call(&self.response)?;
        visit.call(&self.start)?;
        visit.call(&self.end)
    }

    fn close(slf: &Bound<'_, Self>) {
        let mut this = slf.borrow_mut();
        this.logger = None;
        this.response = None;
        this.end = None;
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        Self::close(slf);
    }
}
