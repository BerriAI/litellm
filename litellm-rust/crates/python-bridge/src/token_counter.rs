use std::num::NonZero;
use std::sync::Arc;
use std::thread::available_parallelism;

use litellm_python_interop::release_gil;
use litellm_token_counter::{
    CountableRequest, Error, InputTokenCount, TokenCounter as CoreTokenCounter,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tokio::sync::Semaphore;

use crate::constants::TOKEN_COUNT_FALLBACK_PARALLELISM;
use crate::errors::RustBridgeDeclined;
use crate::execution::run_async;

/// Counts the input tokens of a raw request body off the Python event loop with
/// the GIL released. Python owns which requests get here and what to do with
/// the count. At most one encode per core runs at a time; the rest wait in the
/// async task, where a cancelled Python awaiter drops them before any blocking
/// work is scheduled.
#[pyclass(frozen)]
struct TokenCounter {
    inner: Arc<CoreTokenCounter>,
    encode_slots: Arc<Semaphore>,
}

#[pymethods]
impl TokenCounter {
    #[new]
    fn new(py: Python<'_>, tokenizer_json: &str) -> PyResult<Self> {
        Self::load(py, || CoreTokenCounter::from_json(tokenizer_json))
    }

    #[staticmethod]
    fn from_cl100k_ranks(py: Python<'_>, rank_file: &str) -> PyResult<Self> {
        Self::load(py, || CoreTokenCounter::from_cl100k_ranks(rank_file))
    }

    #[staticmethod]
    fn from_o200k_ranks(py: Python<'_>, rank_file: &str) -> PyResult<Self> {
        Self::load(py, || CoreTokenCounter::from_o200k_ranks(rank_file))
    }

    fn acount_request<'py>(&self, py: Python<'py>, body: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let counter = Arc::clone(&self.inner);
        let encode_slots = Arc::clone(&self.encode_slots);
        let body = body.to_vec();
        run_async(
            py,
            async move {
                let _slot = encode_slots
                    .acquire_owned()
                    .await
                    .map_err(|error| Error::Task(error.to_string()))?;
                tokio::task::spawn_blocking(move || count_body(&counter, &body))
                    .await
                    .map_err(|error| Error::Task(error.to_string()))?
            },
            token_count_error_to_pyerr,
        )
    }
}

impl TokenCounter {
    fn load(
        py: Python<'_>,
        load: impl FnOnce() -> Result<CoreTokenCounter, Error> + Send,
    ) -> PyResult<Self> {
        let inner = release_gil(py, load).map_err(token_count_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
            encode_slots: Arc::new(Semaphore::new(encode_parallelism())),
        })
    }
}

fn encode_parallelism() -> usize {
    available_parallelism().map_or(TOKEN_COUNT_FALLBACK_PARALLELISM, NonZero::get)
}

fn count_body(counter: &CoreTokenCounter, body: &[u8]) -> Result<InputTokenCount, Error> {
    let request = CountableRequest::parse(body)?;
    counter.count_request(&request)
}

fn token_count_error_to_pyerr(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::Load(_) | Error::Ranks(_) | Error::UnicodeClasses => PyValueError::new_err(message),
        Error::RequestParse(_)
        | Error::MissingInput
        | Error::FloatText
        | Error::ContentBlock
        | Error::ArrayItems
        | Error::JsonSerialization(_)
        | Error::JsonUtf8(_) => RustBridgeDeclined::new_err(message),
        Error::Encode(_) | Error::Task(_) => PyRuntimeError::new_err(message),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TokenCounter>()
}
