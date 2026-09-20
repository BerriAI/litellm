use std::sync::Arc;

#[cfg(any(feature = "fast", feature = "huggingface", feature = "tiktoken"))]
use std::{num::NonZero, thread::available_parallelism};

use litellm_host_python::release_gil;
use litellm_host_python::run_async;
use litellm_token_counter::{
    CountableRequest, Error, InputTokenCount, TextCodec, TokenCounter as CoreTokenCounter,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyAny,
};
use tokio::sync::Semaphore;

use crate::errors::RustBridgeDeclined;

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

#[pyclass(frozen, name = "Tokenizer")]
pub(crate) struct Tokenizer {
    inner: Arc<dyn TextCodec>,
}

#[pymethods]
impl TokenCounter {
    #[new]
    fn new(py: Python<'_>, tokenizer_json: &str) -> PyResult<Self> {
        #[cfg(feature = "fast")]
        {
            Self::load(py, || CoreTokenCounter::from_json_fast(tokenizer_json))
        }
        #[cfg(all(not(feature = "fast"), feature = "huggingface"))]
        {
            Self::load(py, || CoreTokenCounter::from_json(tokenizer_json))
        }
        #[cfg(not(any(feature = "fast", feature = "huggingface")))]
        {
            let _ = (py, tokenizer_json);
            Err(RustBridgeDeclined::new_err(
                "tokenizer backend requires the fast or huggingface feature",
            ))
        }
    }

    #[staticmethod]
    fn from_cl100k_ranks(py: Python<'_>, rank_file: &str) -> PyResult<Self> {
        #[cfg(feature = "fast")]
        {
            Self::load(py, || CoreTokenCounter::from_cl100k_ranks(rank_file))
        }
        #[cfg(not(feature = "fast"))]
        {
            let _ = (py, rank_file);
            Err(RustBridgeDeclined::new_err(
                "tokenizer backend requires the fast feature",
            ))
        }
    }

    #[staticmethod]
    fn from_o200k_ranks(py: Python<'_>, rank_file: &str) -> PyResult<Self> {
        #[cfg(feature = "fast")]
        {
            Self::load(py, || CoreTokenCounter::from_o200k_ranks(rank_file))
        }
        #[cfg(not(feature = "fast"))]
        {
            let _ = (py, rank_file);
            Err(RustBridgeDeclined::new_err(
                "tokenizer backend requires the fast feature",
            ))
        }
    }

    #[staticmethod]
    fn from_tiktoken(py: Python<'_>, encoding: &str) -> PyResult<Self> {
        #[cfg(feature = "tiktoken")]
        {
            Self::load(py, || CoreTokenCounter::from_tiktoken(encoding))
        }
        #[cfg(not(feature = "tiktoken"))]
        {
            let _ = (py, encoding);
            Err(RustBridgeDeclined::new_err(
                "tokenizer backend requires the tiktoken feature",
            ))
        }
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

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    fn from_tiktoken(py: Python<'_>, encoding: &str) -> PyResult<Self> {
        #[cfg(feature = "tiktoken")]
        {
            let encoding = encoding.to_owned();
            Self::load(py, move || {
                litellm_token_counter::tiktoken::TiktokenTokenizer::from_name(&encoding)
                    .map_err(Error::from)
            })
        }
        #[cfg(not(feature = "tiktoken"))]
        {
            let _ = (py, encoding);
            Err(RustBridgeDeclined::new_err(
                "tokenizer backend requires the tiktoken feature",
            ))
        }
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, tokenizer_json: &str) -> PyResult<Self> {
        #[cfg(feature = "huggingface")]
        {
            let tokenizer_json = tokenizer_json.to_owned();
            Self::load(py, move || {
                litellm_token_counter::huggingface::HuggingFaceTokenizer::from_json(&tokenizer_json)
                    .map_err(Error::from)
            })
        }
        #[cfg(not(feature = "huggingface"))]
        {
            let _ = (py, tokenizer_json);
            Err(RustBridgeDeclined::new_err(
                "tokenizer backend requires the huggingface feature",
            ))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (identifier, revision = "main", token = None))]
    fn from_pretrained(
        py: Python<'_>,
        identifier: &str,
        revision: &str,
        token: Option<&str>,
    ) -> PyResult<Self> {
        #[cfg(feature = "huggingface")]
        {
            let identifier = identifier.to_owned();
            let revision = revision.to_owned();
            let token = token.map(str::to_owned);
            Self::load(py, move || {
                litellm_token_counter::huggingface::HuggingFaceTokenizer::from_pretrained(
                    &identifier,
                    &revision,
                    token.as_deref(),
                )
                .map_err(Error::from)
            })
        }
        #[cfg(not(feature = "huggingface"))]
        {
            let _ = (py, identifier, revision, token);
            Err(RustBridgeDeclined::new_err(
                "tokenizer backend requires the huggingface feature",
            ))
        }
    }

    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        let inner = Arc::clone(&self.inner);
        let text = text.to_owned();
        release_gil(py, move || inner.encode(&text)).map_err(token_count_error_to_pyerr)
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        release_gil(py, move || inner.decode(&ids, skip_special_tokens))
            .map_err(token_count_error_to_pyerr)
    }

    fn count(&self, py: Python<'_>, text: &str) -> PyResult<usize> {
        let inner = Arc::clone(&self.inner);
        let text = text.to_owned();
        release_gil(py, move || inner.count_tokens(&text)).map_err(token_count_error_to_pyerr)
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }
}

impl Tokenizer {
    #[cfg(any(feature = "huggingface", feature = "tiktoken"))]
    fn load<T>(py: Python<'_>, load: impl FnOnce() -> Result<T, Error> + Send) -> PyResult<Self>
    where
        T: TextCodec + 'static,
    {
        let inner = release_gil(py, load).map_err(token_count_error_to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

#[pyfunction]
pub(crate) fn tiktoken_encoding_for_model(model: &str) -> Option<String> {
    #[cfg(feature = "tiktoken")]
    {
        litellm_token_counter::tiktoken::encoding_for_model(model).map(str::to_owned)
    }
    #[cfg(not(feature = "tiktoken"))]
    {
        let _ = model;
        None
    }
}

impl TokenCounter {
    #[cfg(any(feature = "fast", feature = "huggingface", feature = "tiktoken"))]
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

#[cfg(any(feature = "fast", feature = "huggingface", feature = "tiktoken"))]
fn encode_parallelism() -> usize {
    available_parallelism().map_or(1, NonZero::get)
}

fn count_body(counter: &CoreTokenCounter, body: &[u8]) -> Result<InputTokenCount, Error> {
    let request = CountableRequest::parse(body)?;
    counter.count_request(&request)
}

fn token_count_error_to_pyerr(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::Load(_)
        | Error::Download(_)
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
