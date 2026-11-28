//! The spend log row builder for Python: `litellm/proxy/spend_tracking/rust_spend_log_rows.py`
//! builds a [`RustSpendLogRowBuilder`] per settings change from the typed settings Python
//! resolves, hands it each request's input as UTF-8 JSON, and gets the `LiteLLM_SpendLogs` row
//! back as a dict. The ledger columns are computed from the row dict and typed ledger arguments,
//! read straight off the Python objects. The GIL is released while a row or its ledger columns
//! are built.

use litellm_callbacks_spend_log::{
    AccountingSavings, AccountingWindow, LedgerInput, RowConfig, RowError, RowInput, SpendLogRow,
    SpendLogRowBuilder,
};
use litellm_host_python::{from_py_argument, release_gil, to_py};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::redaction::NativeTokenRedactor;

fn row_error(error: RowError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// One [`SpendLogRowBuilder`] under the proxy settings it was built with.
#[pyclass(frozen, name = "RustSpendLogRowBuilder")]
pub(crate) struct RustSpendLogRowBuilder(SpendLogRowBuilder);

#[pymethods]
impl RustSpendLogRowBuilder {
    /// Each setting is the value `get_logging_payload` reads through its own Python helper; the
    /// token redactor is the process's `secret_redaction._token_redactor()`.
    #[new]
    #[pyo3(signature = (
        *,
        token_redactor,
        latest_turn_only,
        redact_tokens,
        drop_requester_ip_address,
        omit_missing_session_id,
        max_string_length_prompt_in_db,
        disable_end_user_cost_tracking,
        request_correlation_in_logs,
        turn_off_message_logging,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        token_redactor: PyRef<'_, NativeTokenRedactor>,
        latest_turn_only: bool,
        redact_tokens: bool,
        drop_requester_ip_address: bool,
        omit_missing_session_id: bool,
        max_string_length_prompt_in_db: i64,
        disable_end_user_cost_tracking: bool,
        request_correlation_in_logs: bool,
        turn_off_message_logging: bool,
    ) -> Self {
        Self(SpendLogRowBuilder::new(
            RowConfig {
                latest_turn_only,
                redact_tokens,
                drop_requester_ip_address,
                omit_missing_session_id,
                max_string_length_prompt_in_db,
                disable_end_user_cost_tracking,
                request_correlation_in_logs,
                turn_off_message_logging,
            },
            token_redactor.shared(),
        ))
    }

    /// The row for one request, from the glue's input JSON (UTF-8 bytes), as a dict whose
    /// timestamps are ISO-8601 text.
    fn build_row(&self, py: Python<'_>, input_json: &[u8]) -> PyResult<Py<PyAny>> {
        let row = release_gil(py, || -> Result<SpendLogRow, RowError> {
            let text = std::str::from_utf8(input_json)
                .map_err(|_| RowError::Input("the input JSON is not UTF-8"))?;
            self.0.build_row(&RowInput::from_json(text)?)
        })
        .map_err(row_error)?;
        to_py(py, &row)
    }

    /// The ledger columns (`spend`, `accounting_delivery_id`, `accounting_payload`) of a row the
    /// builder made, as a dict. `row` is that row with ISO-8601 timestamps.
    #[staticmethod]
    #[pyo3(signature = (
        row,
        *,
        billable_cost,
        org_id,
        team_id,
        delivery_id,
        entities,
        model_access_groups,
        windows,
        savings,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn ledger_columns(
        py: Python<'_>,
        #[pyo3(from_py_with = from_py_argument)] row: SpendLogRow,
        billable_cost: Option<f64>,
        org_id: Option<String>,
        team_id: Option<String>,
        delivery_id: String,
        entities: Vec<(String, Option<String>)>,
        model_access_groups: Vec<String>,
        #[pyo3(from_py_with = from_py_argument)] windows: Vec<AccountingWindow>,
        #[pyo3(from_py_with = from_py_argument)] savings: AccountingSavings,
    ) -> PyResult<Py<PyAny>> {
        let ledger = LedgerInput {
            billable_cost,
            org_id,
            team_id,
            delivery_id,
            entities,
            model_access_groups,
            windows,
            savings,
        };
        let columns = release_gil(py, || row.ledger_columns(&ledger)).map_err(row_error)?;
        to_py(py, &columns)
    }
}
