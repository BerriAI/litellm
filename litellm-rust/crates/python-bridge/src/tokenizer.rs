#[cfg(feature = "tiktoken")]
use std::collections::HashSet;
use std::sync::Arc;

use litellm_host_python::release_gil;
#[cfg(feature = "huggingface")]
use litellm_token_counter::Error;
use litellm_token_counter::TextCodec;
use pyo3::{exceptions::PyUnicodeEncodeError, prelude::*, types::PyString};

#[cfg(any(feature = "tiktoken", feature = "huggingface"))]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "huggingface")]
use pyo3::{exceptions::PyIOError, types::PyDict};
#[cfg(feature = "tiktoken")]
use pyo3::{
    exceptions::{PyKeyError, PyRuntimeError},
    types::PyBytes,
};

#[cfg(not(all(feature = "tiktoken", feature = "huggingface")))]
use crate::errors::RustBridgeDeclined;
use crate::token_counter::token_count_error_to_pyerr;

#[cfg(feature = "huggingface")]
use litellm_token_counter::huggingface::{
    EncodeInput, Encoding, HuggingFaceTokenizer, InputSequence,
};
#[cfg(feature = "tiktoken")]
use litellm_token_counter::tiktoken::TiktokenTokenizer;

enum Codec {
    #[cfg(feature = "tiktoken")]
    Tiktoken(TiktokenTokenizer),
    #[cfg(feature = "huggingface")]
    HuggingFace(HuggingFaceTokenizer),
}

impl Codec {
    fn codec(&self) -> &dyn TextCodec {
        match *self {
            #[cfg(feature = "tiktoken")]
            Self::Tiktoken(ref tokenizer) => tokenizer,
            #[cfg(feature = "huggingface")]
            Self::HuggingFace(ref tokenizer) => tokenizer,
        }
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Tokenizer {
    inner: Arc<Codec>,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    fn from_tiktoken(py: Python<'_>, encoding: &str) -> PyResult<Self> {
        #[cfg(feature = "tiktoken")]
        {
            let tokenizer = release_gil(py, || TiktokenTokenizer::from_name(encoding))
                .map_err(|error| token_count_error_to_pyerr(error.into()))?;
            Ok(Self {
                inner: Arc::new(Codec::Tiktoken(tokenizer)),
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
            let tokenizer = release_gil(py, || HuggingFaceTokenizer::from_json(tokenizer_json))
                .map_err(|error| token_count_error_to_pyerr(error.into()))?;
            Ok(Self {
                inner: Arc::new(Codec::HuggingFace(tokenizer)),
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
            let kwargs = PyDict::new(py);
            kwargs.set_item("repo_id", identifier)?;
            kwargs.set_item("filename", "tokenizer.json")?;
            kwargs.set_item("revision", revision)?;
            kwargs.set_item("token", token)?;
            let path: String = PyModule::import(py, "huggingface_hub")?
                .getattr("hf_hub_download")?
                .call((), Some(&kwargs))?
                .extract()?;
            let json =
                release_gil(py, || std::fs::read_to_string(path)).map_err(PyIOError::new_err)?;
            Self::from_json(py, &json)
        }
        #[cfg(not(feature = "huggingface"))]
        {
            let _ = (py, identifier, revision, token);
            Err(RustBridgeDeclined::new_err(
                "tokenizer backend requires the huggingface feature",
            ))
        }
    }

    fn encode(&self, py: Python<'_>, text: &Bound<'_, PyString>) -> PyResult<Vec<u32>> {
        let text = self.text(text)?;
        release_gil(py, || self.inner.codec().encode(&text)).map_err(token_count_error_to_pyerr)
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        release_gil(py, || self.inner.codec().decode(&ids, skip_special_tokens))
            .map_err(token_count_error_to_pyerr)
    }

    fn count(&self, py: Python<'_>, text: &Bound<'_, PyString>) -> PyResult<usize> {
        let text = self.text(text)?;
        release_gil(py, || self.inner.codec().count_tokens(&text))
            .map_err(token_count_error_to_pyerr)
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.codec().name()
    }

    #[cfg(feature = "tiktoken")]
    fn encode_special(
        &self,
        py: Python<'_>,
        text: &Bound<'_, PyString>,
        allowed: Vec<String>,
    ) -> PyResult<Vec<u32>> {
        let text = self.text(text)?;
        match *self.inner {
            #[cfg(feature = "tiktoken")]
            Codec::Tiktoken(ref tokenizer) => {
                release_gil(py, || tokenizer.encode_special(&text, &allowed))
                    .map_err(PyRuntimeError::new_err)
            }
            #[cfg(feature = "huggingface")]
            Codec::HuggingFace(_) => Err(PyValueError::new_err("requires a tiktoken encoding")),
        }
    }

    #[cfg(feature = "tiktoken")]
    fn special_tokens(&self, py: Python<'_>) -> PyResult<HashSet<String>> {
        match *self.inner {
            #[cfg(feature = "tiktoken")]
            Codec::Tiktoken(ref tokenizer) => Ok(release_gil(py, || tokenizer.special_tokens())),
            #[cfg(feature = "huggingface")]
            Codec::HuggingFace(_) => Err(PyValueError::new_err("requires a tiktoken encoding")),
        }
    }

    #[cfg(feature = "tiktoken")]
    fn decode_bytes<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        match *self.inner {
            #[cfg(feature = "tiktoken")]
            Codec::Tiktoken(ref tokenizer) => {
                let bytes = release_gil(py, || tokenizer.decode_bytes(&ids))
                    .map_err(PyKeyError::new_err)?;
                Ok(PyBytes::new(py, &bytes))
            }
            #[cfg(feature = "huggingface")]
            Codec::HuggingFace(_) => Err(PyValueError::new_err("requires a tiktoken encoding")),
        }
    }

    #[cfg(feature = "huggingface")]
    #[pyo3(signature = (sequence, pair = None, is_pretokenized = false, add_special_tokens = true, fast = false))]
    fn encode_huggingface(
        &self,
        py: Python<'_>,
        sequence: Sequence,
        pair: Option<Sequence>,
        is_pretokenized: bool,
        add_special_tokens: bool,
        fast: bool,
    ) -> PyResult<HuggingFaceEncoding> {
        let sequence = sequence.input(is_pretokenized)?;
        let input = match pair {
            Some(pair) => EncodeInput::Dual(sequence, pair.input(is_pretokenized)?),
            None => EncodeInput::Single(sequence),
        };
        match *self.inner {
            Codec::HuggingFace(ref tokenizer) => release_gil(py, || {
                tokenizer.encode_result(input, add_special_tokens, fast)
            })
            .map(|inner| HuggingFaceEncoding { inner })
            .map_err(|error| token_count_error_to_pyerr(Error::from(error))),
            #[cfg(feature = "tiktoken")]
            Codec::Tiktoken(_) => Err(PyValueError::new_err("requires a Hugging Face tokenizer")),
        }
    }
    #[cfg(feature = "huggingface")]
    #[pyo3(signature = (inputs, is_pretokenized = false, add_special_tokens = true, fast = false))]
    fn encode_batch_huggingface(
        &self,
        py: Python<'_>,
        inputs: Vec<(Sequence, Option<Sequence>)>,
        is_pretokenized: bool,
        add_special_tokens: bool,
        fast: bool,
    ) -> PyResult<Vec<HuggingFaceEncoding>> {
        let inputs = inputs
            .into_iter()
            .map(|(sequence, pair)| {
                let sequence = sequence.input(is_pretokenized)?;
                match pair {
                    Some(pair) => Ok(EncodeInput::Dual(sequence, pair.input(is_pretokenized)?)),
                    None => Ok(EncodeInput::Single(sequence)),
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        match *self.inner {
            Codec::HuggingFace(ref tokenizer) => release_gil(py, || {
                tokenizer.encode_batch_result(inputs, add_special_tokens, fast)
            })
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|inner| HuggingFaceEncoding { inner })
                    .collect()
            })
            .map_err(|error| token_count_error_to_pyerr(Error::from(error))),
            #[cfg(feature = "tiktoken")]
            Codec::Tiktoken(_) => Err(PyValueError::new_err("requires a Hugging Face tokenizer")),
        }
    }

    #[cfg(feature = "huggingface")]
    #[pyo3(signature = (pretty = false))]
    fn to_json(&self, py: Python<'_>, pretty: bool) -> PyResult<String> {
        match *self.inner {
            #[cfg(feature = "huggingface")]
            Codec::HuggingFace(ref tokenizer) => release_gil(py, || tokenizer.to_json(pretty))
                .map_err(|error| token_count_error_to_pyerr(Error::from(error))),
            #[cfg(feature = "tiktoken")]
            Codec::Tiktoken(_) => Err(PyValueError::new_err("requires a Hugging Face tokenizer")),
        }
    }
}

impl Tokenizer {
    fn text(&self, text: &Bound<'_, PyString>) -> PyResult<String> {
        match text.to_cow() {
            Ok(text) => Ok(text.into_owned()),
            Err(error)
                if self.inner.codec().name() == "huggingface"
                    || !error.is_instance_of::<PyUnicodeEncodeError>(text.py()) =>
            {
                Err(error)
            }
            Err(_) => text
                .call_method1("encode", ("utf-16", "surrogatepass"))?
                .call_method1("decode", ("utf-16", "replace"))?
                .extract(),
        }
    }
}

#[cfg(feature = "huggingface")]
#[derive(FromPyObject)]
pub(crate) enum Sequence {
    Text(String),
    Words(Vec<String>),
}

#[cfg(feature = "huggingface")]
impl Sequence {
    fn input(self, is_pretokenized: bool) -> PyResult<InputSequence<'static>> {
        match (self, is_pretokenized) {
            (Self::Text(text), false) => Ok(text.into()),
            (Self::Words(words), true) => Ok(words.into()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "input must match is_pretokenized",
            )),
        }
    }
}

#[cfg(feature = "huggingface")]
#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct HuggingFaceEncoding {
    inner: Encoding,
}

#[cfg(feature = "huggingface")]
#[pymethods]
impl HuggingFaceEncoding {
    #[new]
    #[pyo3(signature = (json = None))]
    fn new(json: Option<&str>) -> PyResult<Self> {
        let inner = match json {
            Some(json) => serde_json::from_str(json)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            None => Encoding::default(),
        };
        Ok(Self { inner })
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, pyo3::types::PyType>, (String,))> {
        let json = serde_json::to_string(&self.inner)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((py.get_type::<Self>(), (json,)))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.inner.get_ids().to_vec()
    }
    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.get_tokens().to_vec()
    }
    #[getter]
    fn offsets(&self) -> Vec<(usize, usize)> {
        self.inner.get_offsets().to_vec()
    }
    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        self.inner.get_type_ids().to_vec()
    }
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        self.inner.get_attention_mask().to_vec()
    }
    #[getter]
    fn special_tokens_mask(&self) -> Vec<u32> {
        self.inner.get_special_tokens_mask().to_vec()
    }
    #[getter]
    fn word_ids(&self) -> Vec<Option<u32>> {
        self.inner.get_word_ids().to_vec()
    }
    #[getter]
    fn sequence_ids(&self) -> Vec<Option<usize>> {
        self.inner.get_sequence_ids()
    }
    #[getter]
    fn overflowing(&self) -> Vec<Self> {
        self.inner
            .get_overflowing()
            .iter()
            .cloned()
            .map(|inner| Self { inner })
            .collect()
    }
    #[getter]
    fn n_sequences(&self) -> usize {
        self.inner.n_sequences()
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
