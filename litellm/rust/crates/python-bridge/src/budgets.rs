use std::sync::Arc;

use litellm_callbacks_budgets::{
    BudgetCounterSpend, BudgetPlan, BudgetReservation, Budgets, Check, EngineError, ErrorKind,
    Settle, decide,
};
use litellm_host_python::{Pythonized, from_py_argument, json_from_py, run_async_value, to_py};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use serde::de::DeserializeOwned;

use crate::redis::store::{spend_counters::NativeSpendCounters, ttl};

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
    fn new(py: Python<'_>, spend_counters: PyRef<'_, NativeSpendCounters>) -> Self {
        Self(Arc::new(
            Budgets::new(spend_counters.store()).with_logger(crate::logger::capture(py)),
        ))
    }

    #[pyo3(signature = (plan, observed, ttl_seconds = None))]
    fn reserve<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = from_record)] plan: BudgetPlan,
        #[pyo3(from_py_with = from_py_argument)] observed: Vec<BudgetCounterSpend>,
        ttl_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let budgets = Arc::clone(&self.0);
        let ttl = ttl(ttl_seconds)?;
        run_async_value(py, async move {
            budgets
                .admit(plan, &observed, ttl)
                .await
                .map(Pythonized)
                .map_err(PyRuntimeError::new_err)
        })
    }

    #[pyo3(signature = (reservation, window_resets, observed, actual, apply_consistent, ttl_seconds = None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "keeps the Python settle call's positional inputs and the spend counter write TTL"
    )]
    fn settle<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = from_record)] reservation: BudgetReservation,
        window_resets: Vec<Option<String>>,
        observed: Vec<Option<f64>>,
        actual: f64,
        apply_consistent: bool,
        ttl_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let budgets = Arc::clone(&self.0);
        let ttl = ttl(ttl_seconds)?;
        run_async_value(py, async move {
            budgets
                .settle(&Settle {
                    reservation: &reservation,
                    window_resets: &window_resets,
                    observed: &observed,
                    actual,
                    apply_consistent,
                    ttl,
                })
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
