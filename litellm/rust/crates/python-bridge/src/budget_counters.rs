//! The post-call spend counter step from `litellm-callbacks-budgets`, run as a Python execution.
//!
//! The step's host calls go to the Python host object, awaited in the caller's task. When the step
//! finishes, its stamps land on the caller's shared reservation dict and the step's first failure
//! is raised. The request's post-call batch tracks its committed twins itself.

use std::sync::Arc;

use litellm_callbacks_budgets::{
    BudgetReservation, Budgets, CounterError, CounterOutcome, CounterStep, Failure,
};
use litellm_callbacks_python::{host_call_error, run_hosted};
use litellm_host_python::Execution;
use litellm_redis_store::RequestRedis;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use crate::redis::store::store_error;

fn failure_error(failure: &Failure) -> PyErr {
    match &**failure {
        CounterError::Host(error) => host_call_error(error),
        CounterError::Store(error) => store_error(error),
    }
}

/// Writes each entry's `applied_adjustment` and the `finalized` flag the way the proxy's
/// `stamp_budget_reservation_actual_cost` and the fan-out's last step do.
fn stamp(reservation: &Bound<'_, PyDict>, outcome: &CounterOutcome) -> PyResult<()> {
    if let Some(entries) = reservation.get_item("entries")?
        && let Ok(entries) = entries.cast::<PyList>()
    {
        for (entry, stamp) in entries.iter().zip(&outcome.stamps) {
            if let (Some(stamp), Ok(entry)) = (stamp, entry.cast::<PyDict>()) {
                entry.set_item("applied_adjustment", *stamp)?;
            }
        }
    }
    if outcome.finalized {
        reservation.set_item("finalized", true)?;
    }
    Ok(())
}

/// Starts the counter step for `step` as an execution the proxy drives.
pub(crate) fn start(
    budgets: Arc<Budgets>,
    step: CounterStep,
    record: Option<BudgetReservation>,
    reservation: Option<Py<PyDict>>,
    request: Option<RequestRedis>,
    host: Py<PyAny>,
) -> Execution {
    run_hosted(
        move |host| async move {
            budgets
                .charge_counters(&host, request.as_ref(), &step, record.as_ref())
                .await
        },
        host,
        move |py, outcome: CounterOutcome| {
            if let Some(reservation) = &reservation {
                stamp(reservation.bind(py), &outcome)?;
            }
            match &outcome.error {
                Some(failure) => Err(failure_error(failure)),
                None => Ok(py.None()),
            }
        },
    )
}
