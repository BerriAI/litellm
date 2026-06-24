//! Local guardrail engines exposed to Python.
//!
//! Both classes compile their patterns once at construction and are reused
//! across requests, so the per-request cost is the scan itself. Each scan runs
//! with the GIL released (via the bridge's [`crate::gil`] chokepoint), keeping
//! the proxy event loop responsive while Rust does the regex work.

use litellm_ai_gateway::io::guardrails::{config_supported, run_guardrail, Unsupported};
use litellm_core::guardrails::{
    GuardrailInput, GuardrailStatus, InputType, LiteralTerm, LocalPiiConfig, LocalPiiEngine,
    RegexTerm, RequestContext, Scanner, Verdict,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::gil;

pyo3::create_exception!(
    litellm_python_bridge,
    GuardrailUnsupported,
    pyo3::exceptions::PyException,
    "Raised when the Rust engine cannot handle a guardrail config; the caller \
     should fall back to the Python implementation."
);

#[derive(Deserialize)]
struct ApplyRequest {
    guardrail_type: String,
    params: serde_json::Value,
    input: GuardrailInput,
    input_type: InputType,
    #[serde(default)]
    context: RequestContext,
}

#[derive(Serialize)]
struct ApplyResponse {
    verdict: Verdict,
    guardrail_status: GuardrailStatus,
    provider_response: serde_json::Value,
    duration_ms: u64,
}

/// Build the provider config, run it, and return the verdict as JSON, all in one
/// call with the GIL released. Raises [`GuardrailUnsupported`] when the engine
/// cannot handle this config so the caller can fall back to Python.
#[pyfunction]
fn apply_guardrail(py: Python<'_>, request_json: String) -> PyResult<String> {
    let req: ApplyRequest = serde_json::from_str(&request_json)
        .map_err(|e| PyValueError::new_err(format!("invalid guardrail request JSON: {e}")))?;

    let outcome = gil::release_gil(py, move || {
        run_guardrail(
            &req.guardrail_type,
            &req.params,
            &req.input,
            req.input_type,
            &req.context,
        )
    });

    match outcome {
        Ok(outcome) => {
            let response = ApplyResponse {
                guardrail_status: outcome.status(),
                verdict: outcome.verdict,
                provider_response: outcome.provider_response,
                duration_ms: outcome.duration_ms,
            };
            serde_json::to_string(&response)
                .map_err(|e| PyRuntimeError::new_err(format!("failed to serialize verdict: {e}")))
        }
        Err(Unsupported(reason)) => Err(GuardrailUnsupported::new_err(reason)),
    }
}

/// Whether the Rust engine can handle this guardrail type and params. Called at
/// init time so Python can register the Rust path only when it applies and fall
/// back to the Python guardrail otherwise.
#[pyfunction]
fn guardrail_config_supported(guardrail_type: &str, params_json: &str) -> PyResult<bool> {
    let params: serde_json::Value = serde_json::from_str(params_json)
        .map_err(|e| PyValueError::new_err(format!("invalid guardrail params JSON: {e}")))?;
    Ok(config_supported(guardrail_type, &params))
}

#[derive(Deserialize)]
struct ScannerConfig {
    #[serde(default)]
    literals: Vec<LiteralTerm>,
    #[serde(default)]
    regexes: Vec<RegexTerm>,
}

/// Compiled multi-pattern scanner for the content filter. Built once from a
/// keyword/pattern config; `scan` releases the GIL while matching.
#[pyclass]
pub struct ContentScanner {
    inner: Scanner,
    compile_errors_json: String,
}

#[pymethods]
impl ContentScanner {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let cfg: ScannerConfig = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid scanner config JSON: {e}")))?;
        let (inner, errors) = Scanner::build(&cfg.literals, &cfg.regexes);
        let compile_errors_json =
            serde_json::to_string(&errors).unwrap_or_else(|_| "[]".to_owned());
        Ok(Self {
            inner,
            compile_errors_json,
        })
    }

    /// JSON array of terms that failed to compile (`{id, message}`), so the
    /// caller can fall back to its own implementation for just those terms.
    #[getter]
    fn compile_errors(&self) -> &str {
        &self.compile_errors_json
    }

    /// Return every match as `(id, start, end)`. Byte offsets index into the
    /// UTF-8 text. Runs with the GIL released.
    fn scan(&self, py: Python<'_>, text: String) -> Vec<(u32, usize, usize)> {
        gil::release_gil(py, || {
            self.inner
                .scan(&text)
                .into_iter()
                .map(|m| (m.id, m.start, m.end))
                .collect()
        })
    }
}

/// Compiled in-process PII engine. Built once from a [`LocalPiiConfig`]; each
/// `scan` releases the GIL and returns the verdict as a JSON string (the
/// `{action, ...}` shape the Python guardrail consumes).
#[pyclass]
pub struct PiiEngine {
    inner: LocalPiiEngine,
}

#[pymethods]
impl PiiEngine {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let cfg: LocalPiiConfig = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid local_pii config JSON: {e}")))?;
        Ok(Self {
            inner: LocalPiiEngine::new(cfg),
        })
    }

    /// Screen and mask a batch of texts, returning the verdict as JSON. Runs
    /// the scan with the GIL released.
    fn scan(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<String> {
        let verdict = gil::release_gil(py, || self.inner.scan_texts(texts));
        serde_json::to_string(&verdict)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to serialize verdict: {e}")))
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ContentScanner>()?;
    module.add_class::<PiiEngine>()?;
    module.add_function(wrap_pyfunction!(apply_guardrail, module)?)?;
    module.add_function(wrap_pyfunction!(guardrail_config_supported, module)?)?;
    module.add(
        "GuardrailUnsupported",
        module.py().get_type::<GuardrailUnsupported>(),
    )?;
    Ok(())
}
