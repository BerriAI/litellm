use std::sync::Arc;

use litellm_core::token_counter::types::CountableRequest;
use litellm_core::token_counter::{
    InputTokenCount, TokenCountError, TokenCounter as CoreTokenCounter,
};
use litellm_python_interop::release_gil;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::errors::RustBridgeDeclined;
use crate::execution::run_async;

/// Counts the input tokens of a raw request body off the Python event loop with
/// the GIL released. Python owns which requests get here and what to do with
/// the count.
#[pyclass(frozen)]
struct TokenCounter {
    inner: Arc<CoreTokenCounter>,
}

#[pymethods]
impl TokenCounter {
    #[new]
    fn new(py: Python<'_>, tokenizer_json: &str) -> PyResult<Self> {
        let inner = release_gil(py, || CoreTokenCounter::from_json(tokenizer_json))
            .map_err(token_count_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn acount_request<'py>(&self, py: Python<'py>, body: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let counter = Arc::clone(&self.inner);
        let body = body.to_vec();
        run_async(
            py,
            async move {
                tokio::task::spawn_blocking(move || count_body(&counter, &body))
                    .await
                    .map_err(|error| TokenCountError::Encode(error.to_string()))?
            },
            token_count_error_to_pyerr,
        )
    }
}

fn count_body(counter: &CoreTokenCounter, body: &[u8]) -> Result<InputTokenCount, TokenCountError> {
    let request = CountableRequest::parse(body)?;
    counter.count_request(&request)
}

fn token_count_error_to_pyerr(error: TokenCountError) -> PyErr {
    match error {
        TokenCountError::Load(message) => PyValueError::new_err(message),
        TokenCountError::Unsupported(message) => RustBridgeDeclined::new_err(message),
        TokenCountError::Encode(message) => PyRuntimeError::new_err(message),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TokenCounter>()
}
