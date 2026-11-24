//! Python binding for the Rust spend log callback (`litellm-callbacks-spend-logger`). The proxy
//! builds one [`RustSpendLog`] over the installed Rust spend log writer, the shared spend counters
//! and the process's token redactor, and registers its `.callback()` as a Rust `CustomLoggerV2`.

use std::sync::Arc;

use litellm_callbacks::{Callback, ErasedCallback};
use litellm_callbacks_budgets::Budgets;
use litellm_callbacks_python::{RustCallback, hook_input, run_hosted_hook};
use litellm_callbacks_spend_logger::{SpendLogCallback, SpendLogParts};
use litellm_host_python::Execution;
use litellm_spend_log_writer::PgLedgerStore;
use pyo3::{prelude::*, types::PyDict};

use crate::{
    redaction::NativeTokenRedactor,
    redis::store::{request::NativeRequestRedis, spend_counters::NativeSpendCounters},
    spend_log_writer::RustSpendLogWriter,
};

/// The spend log callback over the writer the proxy's monitor flushes and the spend counters
/// admission reads.
#[pyclass(frozen, name = "RustSpendLog")]
pub(crate) struct RustSpendLog(Arc<SpendLogCallback<PgLedgerStore>>);

#[pymethods]
impl RustSpendLog {
    #[new]
    fn new(
        py: Python<'_>,
        writer: PyRef<'_, RustSpendLogWriter>,
        spend_counters: PyRef<'_, NativeSpendCounters>,
        token_redactor: PyRef<'_, NativeTokenRedactor>,
    ) -> Self {
        Self(Arc::new(SpendLogCallback::new(SpendLogParts {
            writer: writer.shared(),
            budgets: Arc::new(
                Budgets::new(spend_counters.store()).with_logger(crate::logger::capture(py)),
            ),
            tokens: token_redactor.shared(),
        })))
    }

    fn callback(&self) -> PyResult<RustCallback> {
        Ok(RustCallback::new(
            Arc::clone(&self.0) as Arc<dyn ErasedCallback>
        )?)
    }

    /// One hook run as an execution the proxy drives, `host` answering its host calls. The
    /// counter increments ride `request`'s post-call batch, which tracks the request's own
    /// committed charge until it lands.
    #[pyo3(signature = (hook, fields, call, host, request, /))]
    fn start<'py>(
        &self,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        host: Py<PyAny>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Execution> {
        let input = hook_input(Callback::manifest(&*self.0), hook, fields, call)?;
        let callback = Arc::clone(&self.0);
        let redis = request.map(|request| request.0.clone());
        Ok(run_hosted_hook(
            move |host| async move {
                Box::pin(callback.handle_with(&input, host, redis.as_ref())).await
            },
            host,
        ))
    }
}
