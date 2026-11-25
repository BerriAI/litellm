use std::sync::Arc;

use litellm_callbacks_budgets::{
    BudgetCounterSpend, BudgetPlan, BudgetReservation, Budgets, Check, CounterStep, EngineError,
    ErrorKind, Settle, decide,
};
use litellm_host_python::{
    Execution, Pythonized, from_py_argument, json_from_py, run_async_value, to_py,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde::de::DeserializeOwned;

use crate::redis::{
    events::RedisCall,
    store::{request::NativeRequestRedis, spend_counters::NativeSpendCounters},
};

fn from_model<T: DeserializeOwned>(model: &Bound<'_, PyAny>) -> PyResult<T> {
    from_py_argument(&model.call_method0("model_dump")?)
}

fn from_optional_record<T: DeserializeOwned>(record: &Bound<'_, PyAny>) -> PyResult<Option<T>> {
    if record.is_none() {
        return Ok(None);
    }
    from_record(record).map(Some)
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

    fn reserve<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = from_record)] plan: BudgetPlan,
        #[pyo3(from_py_with = from_py_argument)] observed: Vec<BudgetCounterSpend>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let budgets = Arc::clone(&self.0);
        let call = RedisCall::caller(py, None)?;
        run_async_value(py, async move {
            call.scope(budgets.admit(plan, &observed))
                .await
                .map(Pythonized)
                .map_err(PyRuntimeError::new_err)
        })
    }

    /// The post-call counter step for one charged request, as an execution the proxy drives with
    /// `host` answering its host calls. `record` is `reservation` as its typed record; the step
    /// stamps the outcome on `reservation` itself.
    #[pyo3(signature = (step, record, reservation, request, host, /))]
    fn charge_counters(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = from_py_argument)] step: CounterStep,
        #[pyo3(from_py_with = from_optional_record)] record: Option<BudgetReservation>,
        reservation: Option<Py<PyDict>>,
        request: Option<PyRef<'_, NativeRequestRedis>>,
        host: Py<PyAny>,
    ) -> PyResult<Execution> {
        Ok(crate::budget_counters::start(
            Arc::clone(&self.0),
            RedisCall::caller(py, None)?,
            step,
            record,
            reservation,
            request.map(|request| request.0.clone()),
            host,
        ))
    }

    fn settle<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = from_record)] reservation: BudgetReservation,
        window_resets: Vec<Option<String>>,
        observed: Vec<Option<f64>>,
        actual: f64,
        apply_consistent: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let budgets = Arc::clone(&self.0);
        let call = RedisCall::caller(py, None)?;
        run_async_value(py, async move {
            call.scope(budgets.settle(&Settle {
                reservation: &reservation,
                window_resets: &window_resets,
                observed: &observed,
                actual,
                apply_consistent,
            }))
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
