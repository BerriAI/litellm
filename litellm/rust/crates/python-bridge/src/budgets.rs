use std::sync::Arc;

use litellm_callbacks_budgets::{
    BudgetReservation, Budgets, Check, EngineError, ErrorKind, Shared, decide,
};
use litellm_callbacks_python::RustCallback;
use litellm_host_python::{Pythonized, from_py_argument, json_from_py, run_async_value, to_py};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use serde::de::DeserializeOwned;

use crate::redis::store::spend_counters::NativeSpendCounters;

fn from_model<T: DeserializeOwned>(model: &Bound<'_, PyAny>) -> PyResult<T> {
    from_py_argument(&model.call_method0("model_dump")?)
}

fn from_record<T: DeserializeOwned>(record: &Bound<'_, PyAny>) -> PyResult<T> {
    serde_json::from_value(json_from_py(&record.call_method0("model_dump")?)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn settle_error(error: EngineError) -> PyErr {
    match error {
        EngineError::Reservation(error) => match error.kind {
            ErrorKind::Type => PyTypeError::new_err(error.message),
            ErrorKind::Value => PyValueError::new_err(error.message),
        },
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

#[pyclass(frozen, name = "NativeBudgets")]
pub(crate) struct NativeBudgets(Arc<Budgets>);

#[pymethods]
impl NativeBudgets {
    #[new]
    fn new(spend_counters: PyRef<'_, NativeSpendCounters>) -> Self {
        Self(Arc::new(Budgets::new(spend_counters.store())))
    }

    fn callback(&self) -> PyResult<RustCallback> {
        Ok(RustCallback::new(Arc::new(Shared(Arc::clone(&self.0))))?)
    }

    fn settle<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = from_record)] reservation: BudgetReservation,
        window_resets: Vec<Option<String>>,
        actual: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let budgets = Arc::clone(&self.0);
        run_async_value(py, async move {
            budgets
                .settle(&reservation, &window_resets, actual)
                .await
                .map(Pythonized)
                .map_err(settle_error)
        })
    }
}

#[pyfunction]
pub(crate) fn budget_check_decision(
    py: Python<'_>,
    #[pyo3(from_py_with = from_model)] check: Check,
) -> PyResult<Py<PyAny>> {
    to_py(py, &decide(&check))
}
