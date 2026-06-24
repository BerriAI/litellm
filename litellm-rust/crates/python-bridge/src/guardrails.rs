//! Local guardrail engines exposed to Python.
//!
//! Both classes compile their patterns once at construction and are reused
//! across requests, so the per-request cost is the scan itself. Each scan runs
//! with the GIL released (via the bridge's [`crate::gil`] chokepoint), keeping
//! the proxy event loop responsive while Rust does the regex work.

use litellm_core::guardrails::{LiteralTerm, LocalPiiConfig, LocalPiiEngine, RegexTerm, Scanner};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::Deserialize;

use crate::gil;

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
    Ok(())
}
