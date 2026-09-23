use crate::logger::run_async;
use std::sync::Arc;
use std::{num::NonZero, thread::available_parallelism};

use litellm_host_python::enter_native;
use litellm_token_counter::{
    CountableRequest, Error, InputTokenCount, TokenCounter as CoreTokenCounter,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyAny,
};
use tokio::sync::Semaphore;

use crate::errors::RustBridgeDeclined;
use crate::tokenizer::Tokenizer;

/// Counts the input tokens of a raw request body off the Python event loop with
/// the GIL released. Python owns which requests get here and what to do with
/// the count. At most one encode per core runs at a time; the rest wait in the
/// async task, where a cancelled Python awaiter drops them before any blocking
/// work is scheduled.
#[pyclass(frozen)]
pub(crate) struct TokenCounter {
    inner: Arc<CoreTokenCounter>,
    encode_slots: Arc<Semaphore>,
}

#[pymethods]
impl TokenCounter {
    #[staticmethod]
    #[pyo3(signature = (tokenizer, fast = false))]
    fn from_tokenizer(py: Python<'_>, tokenizer: &Tokenizer, fast: bool) -> PyResult<Self> {
        enter_native()?;
        let inner = CoreTokenCounter::new(tokenizer.counter(py, fast));
        Ok(Self {
            inner: Arc::new(inner),
            encode_slots: Arc::new(Semaphore::new(encode_parallelism())),
        })
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

fn encode_parallelism() -> usize {
    available_parallelism().map_or(1, NonZero::get)
}

fn count_body(counter: &CoreTokenCounter, body: &[u8]) -> Result<InputTokenCount, Error> {
    let request = CountableRequest::parse(body)?;
    counter.count_request(&request)
}

pub(crate) fn token_count_error_to_pyerr(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::Load(_)
        | Error::Ranks(_)
        | Error::UnicodeClasses
        | Error::UnsupportedTokenizer(_) => PyValueError::new_err(message),
        Error::RequestParse(_)
        | Error::MissingInput
        | Error::FloatText
        | Error::ContentBlock
        | Error::ArrayItems
        | Error::JsonSerialization(_)
        | Error::JsonUtf8(_) => RustBridgeDeclined::new_err(message),
        Error::Encode(_) | Error::Decode(_) | Error::Task(_) => PyRuntimeError::new_err(message),
    }
}
