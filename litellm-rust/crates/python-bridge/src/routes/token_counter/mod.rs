use std::collections::HashMap;
use std::num::NonZero;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::available_parallelism;

use litellm_python_interop::release_gil;
use litellm_token_counter::{
    CountableRequest, Error, InputTokenCount, TokenCounter as CoreTokenCounter,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tokio::sync::Semaphore;

use crate::constants::TOKEN_COUNT_FALLBACK_PARALLELISM;
use crate::errors::{RustBridgeDeclined, RustBridgeUnavailable};
use crate::execution::run_async;

struct CachedCounter {
    counter: Arc<CoreTokenCounter>,
    encode_slots: Arc<Semaphore>,
}

static COUNTERS: OnceLock<Mutex<HashMap<&'static str, Arc<CachedCounter>>>> = OnceLock::new();

#[pyfunction]
#[pyo3(signature = (body, kind, encoding, disabled, legacy_accounting, resource_loader))]
fn count_input_tokens<'py>(
    py: Python<'py>,
    body: &[u8],
    kind: Option<&str>,
    encoding: &str,
    disabled: bool,
    legacy_accounting: bool,
    resource_loader: Py<PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let tokenizer =
        litellm_token_counter::admit_tokenizer(kind, encoding, disabled, legacy_accounting)
            .map_err(admission_error_to_pyerr)?;
    CoreTokenCounter::admit_request(body).map_err(admission_error_to_pyerr)?;
    let cached = cached_counter(py, tokenizer, resource_loader)?;
    let body = body.to_vec();
    run_async(
        py,
        async move {
            let _slot = Arc::clone(&cached.encode_slots)
                .acquire_owned()
                .await
                .map_err(|error| Error::Task(error.to_string()))?;
            tokio::task::spawn_blocking(move || count_body(&cached.counter, &body))
                .await
                .map_err(|error| Error::Task(error.to_string()))?
        },
        token_count_error_to_pyerr,
    )
}

fn cached_counter(
    py: Python<'_>,
    tokenizer: &'static str,
    resource_loader: Py<PyAny>,
) -> PyResult<Arc<CachedCounter>> {
    let counters = COUNTERS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(counter) = counters
        .lock()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .get(tokenizer)
        .cloned()
    {
        return Ok(counter);
    }
    let resource: String = resource_loader
        .call1(py, (tokenizer,))
        .and_then(|value| value.extract(py))
        .map_err(|error| RustBridgeUnavailable::new_err(error.to_string()))?;
    let counter = release_gil(py, move || load_counter(tokenizer, &resource))
        .map_err(token_count_error_to_pyerr)?;
    let cached = Arc::new(CachedCounter {
        counter: Arc::new(counter),
        encode_slots: Arc::new(Semaphore::new(encode_parallelism())),
    });
    counters
        .lock()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .insert(tokenizer, Arc::clone(&cached));
    Ok(cached)
}

fn load_counter(tokenizer: &str, resource: &str) -> Result<CoreTokenCounter, Error> {
    match tokenizer {
        "anthropic" => CoreTokenCounter::from_json(resource),
        "cl100k_base" => CoreTokenCounter::from_cl100k_ranks(resource),
        "o200k_base" => CoreTokenCounter::from_o200k_ranks(resource),
        _ => Err(Error::UnsupportedTokenizer),
    }
}

fn encode_parallelism() -> usize {
    available_parallelism().map_or(TOKEN_COUNT_FALLBACK_PARALLELISM, NonZero::get)
}

fn count_body(counter: &CoreTokenCounter, body: &[u8]) -> Result<InputTokenCount, Error> {
    let request = CountableRequest::parse(body)?;
    counter.count_request(&request)
}

fn admission_error_to_pyerr(error: Error) -> PyErr {
    RustBridgeDeclined::new_err(error.to_string())
}

fn token_count_error_to_pyerr(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::Load(_) | Error::Ranks(_) | Error::UnicodeClasses => {
            RustBridgeUnavailable::new_err(message)
        }
        Error::UnsupportedTokenizer
        | Error::RequestParse(_)
        | Error::MissingInput
        | Error::FloatText
        | Error::ContentBlock
        | Error::ArrayItems
        | Error::JsonSerialization(_)
        | Error::JsonUtf8(_) => RustBridgeDeclined::new_err(message),
        Error::Encode(_) | Error::Task(_) => PyRuntimeError::new_err(message),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(count_input_tokens, module)?)
}
