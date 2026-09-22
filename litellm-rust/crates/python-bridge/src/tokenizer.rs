//! The Python face of the text codecs: one `Tokenizer` class over the tiktoken and Hugging
//! Face backends, carrying the read-only surface of `tiktoken.Encoding` and
//! `tokenizers.Tokenizer` that `litellm/litellm_core_utils/tokenizer.py` wraps.
use std::borrow::Cow;
#[cfg(any(feature = "tiktoken", feature = "huggingface"))]
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "fast")]
use std::sync::OnceLock;

use litellm_host_python::{enter_native, release_gil};
#[cfg(feature = "fast")]
use litellm_token_counter::fast::{FastCounter, FastTokenizer};
use litellm_token_counter::{Error, TextCodec};
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
use crate::routes::token_counter::token_count_error_to_pyerr;

#[cfg(feature = "huggingface")]
use litellm_token_counter::huggingface::{
    EncodeInput, Encoding, HuggingFaceTokenizer, InputSequence, PaddingDirection, PaddingStrategy,
    TruncationDirection, encoding_from_json, encoding_to_json,
};
#[cfg(feature = "tiktoken")]
use litellm_token_counter::tiktoken::{TiktokenTokenizer, Vocabulary};

#[cfg(feature = "tiktoken")]
pub(crate) fn load_tiktoken(py: Python<'_>, encoding: &str) -> PyResult<TiktokenTokenizer> {
    enter_native()?;
    let resource: std::path::PathBuf =
        PyModule::import(py, "litellm.litellm_core_utils.tokenizers")?
            .getattr("__file__")?
            .extract()?;
    release_gil(py, || {
        TiktokenTokenizer::from_cached_ranks(encoding, |file| {
            std::fs::read_to_string(resource.with_file_name(file))
        })
    })
    .map_err(|error| token_count_error_to_pyerr(error.into()))
}

pub(crate) enum Codec {
    #[cfg(feature = "tiktoken")]
    Tiktoken(TiktokenTokenizer),
    #[cfg(feature = "huggingface")]
    HuggingFace(HuggingFaceTokenizer),
}

impl Codec {
    pub(crate) fn codec(&self) -> &dyn TextCodec {
        match *self {
            #[cfg(feature = "tiktoken")]
            Self::Tiktoken(ref tokenizer) => tokenizer,
            #[cfg(feature = "huggingface")]
            Self::HuggingFace(ref tokenizer) => tokenizer,
        }
    }

    #[cfg(feature = "fast")]
    fn fast_counter(&self) -> Option<FastTokenizer> {
        match *self {
            #[cfg(feature = "tiktoken")]
            Self::Tiktoken(ref tokenizer) => tokenizer.fast_counter(),
            #[cfg(feature = "huggingface")]
            Self::HuggingFace(ref tokenizer) => tokenizer.fast_counter(),
        }
    }
}

/// The loaded model is shared: `TokenCounter::from_tokenizer` counts with the same parse,
/// and the opt-in count-only counter is derived from it once, on first use.
#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Tokenizer {
    inner: Arc<Codec>,
    #[cfg(feature = "fast")]
    fast: OnceLock<Option<Arc<FastTokenizer>>>,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    fn from_tiktoken(py: Python<'_>, encoding: &str) -> PyResult<Self> {
        #[cfg(feature = "tiktoken")]
        {
            let tokenizer = load_tiktoken(py, encoding)?;
            Ok(Self::new(Codec::Tiktoken(tokenizer)))
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
            enter_native()?;
            let tokenizer = release_gil(py, || HuggingFaceTokenizer::from_json(tokenizer_json))
                .map_err(|error| token_count_error_to_pyerr(error.into()))?;
            Ok(Self::new(Codec::HuggingFace(tokenizer)))
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
            enter_native()?;
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
        enter_native()?;
        let text = self.text(text)?;
        release_gil(py, || self.inner.codec().encode(&text)).map_err(token_count_error_to_pyerr)
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        enter_native()?;
        release_gil(py, || self.inner.codec().decode(&ids, skip_special_tokens))
            .map_err(token_count_error_to_pyerr)
    }

    #[pyo3(signature = (text, fast = false))]
    fn count(&self, py: Python<'_>, text: &Bound<'_, PyString>, fast: bool) -> PyResult<usize> {
        enter_native()?;
        let text = self.text(text)?;
        let counter = self.counter(py, fast);
        release_gil(py, || {
            litellm_token_counter::Tokenizer::count_tokens(&counter, &text)
        })
        .map_err(token_count_error_to_pyerr)
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.codec().name()
    }

    // ---- tiktoken: the `tiktoken.Encoding` surface ------------------------------------------

    #[cfg(feature = "tiktoken")]
    fn encode_special(
        &self,
        py: Python<'_>,
        text: &Bound<'_, PyString>,
        allowed: Vec<String>,
    ) -> PyResult<Vec<u32>> {
        enter_native()?;
        let tokenizer = self.tiktoken()?;
        let text = self.text(text)?;
        release_gil(py, || tokenizer.encode_special(&text, &allowed))
            .map_err(PyRuntimeError::new_err)
    }

    /// tiktoken's `encode_with_unstable`: `(stable_tokens, completions)`.
    #[cfg(feature = "tiktoken")]
    fn encode_with_unstable(
        &self,
        py: Python<'_>,
        text: &Bound<'_, PyString>,
        allowed: Vec<String>,
    ) -> PyResult<(Vec<u32>, Vec<Vec<u32>>)> {
        enter_native()?;
        let tokenizer = self.tiktoken()?;
        let text = self.text(text)?;
        Ok(release_gil(py, || {
            tokenizer.encode_with_unstable(&text, &allowed)
        }))
    }

    /// The special tokens by text: tiktoken's `_special_tokens`.
    #[cfg(feature = "tiktoken")]
    fn special_tokens(&self) -> PyResult<HashMap<String, u32>> {
        Ok(self
            .vocabulary()?
            .special_tokens()
            .map(|(token, rank)| (token.to_owned(), rank))
            .collect())
    }

    #[cfg(feature = "tiktoken")]
    fn max_token_value(&self) -> PyResult<u32> {
        Ok(self.vocabulary()?.max_token_value())
    }

    #[cfg(feature = "tiktoken")]
    fn is_special_token(&self, token: u32) -> PyResult<bool> {
        Ok(self.vocabulary()?.is_special_token(token))
    }

    /// Every mergeable token's bytes, sorted bytewise like tiktoken's `token_byte_values`.
    #[cfg(feature = "tiktoken")]
    fn token_byte_values<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let vocabulary = self.vocabulary()?;
        let values = release_gil(py, || vocabulary.token_byte_values());
        Ok(values.iter().map(|value| PyBytes::new(py, value)).collect())
    }

    /// The token of one whole piece; `KeyError` when it is not in the vocabulary.
    #[cfg(feature = "tiktoken")]
    fn encode_single_token(&self, py: Python<'_>, piece: Vec<u8>) -> PyResult<u32> {
        self.vocabulary()?
            .encode_single_token(&piece)
            .ok_or_else(|| PyKeyError::new_err(PyBytes::new(py, &piece).unbind()))
    }

    #[cfg(feature = "tiktoken")]
    fn decode_bytes<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        enter_native()?;
        let tokenizer = self.tiktoken()?;
        let bytes =
            release_gil(py, || tokenizer.decode_bytes(&ids)).map_err(PyKeyError::new_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    // ---- Hugging Face: the `tokenizers.Tokenizer` surface -----------------------------------

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
        enter_native()?;
        let tokenizer = self.huggingface()?;
        let sequence = sequence.input(is_pretokenized)?;
        let input = match pair {
            Some(pair) => EncodeInput::Dual(sequence, pair.input(is_pretokenized)?),
            None => EncodeInput::Single(sequence),
        };
        release_gil(py, || {
            tokenizer.encode_result(input, add_special_tokens, fast)
        })
        .map(|inner| HuggingFaceEncoding { inner })
        .map_err(|error| token_count_error_to_pyerr(Error::from(error)))
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
        enter_native()?;
        let tokenizer = self.huggingface()?;
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
        release_gil(py, || {
            tokenizer.encode_batch_result(inputs, add_special_tokens, fast)
        })
        .map(|encodings| {
            encodings
                .into_iter()
                .map(|inner| HuggingFaceEncoding { inner })
                .collect()
        })
        .map_err(|error| token_count_error_to_pyerr(Error::from(error)))
    }

    #[cfg(feature = "huggingface")]
    #[pyo3(signature = (pretty = false))]
    fn to_json(&self, py: Python<'_>, pretty: bool) -> PyResult<String> {
        enter_native()?;
        let tokenizer = self.huggingface()?;
        release_gil(py, || tokenizer.to_json(pretty))
            .map_err(|error| token_count_error_to_pyerr(Error::from(error)))
    }

    #[cfg(feature = "huggingface")]
    fn token_to_id(&self, token: &str) -> PyResult<Option<u32>> {
        Ok(self.huggingface()?.token_to_id(token))
    }

    #[cfg(feature = "huggingface")]
    fn id_to_token(&self, id: u32) -> PyResult<Option<String>> {
        Ok(self.huggingface()?.id_to_token(id))
    }

    #[cfg(feature = "huggingface")]
    #[pyo3(signature = (with_added_tokens = true))]
    fn get_vocab(&self, py: Python<'_>, with_added_tokens: bool) -> PyResult<HashMap<String, u32>> {
        let tokenizer = self.huggingface()?;
        Ok(release_gil(py, || tokenizer.vocab(with_added_tokens)))
    }

    #[cfg(feature = "huggingface")]
    #[pyo3(signature = (with_added_tokens = true))]
    fn get_vocab_size(&self, with_added_tokens: bool) -> PyResult<usize> {
        Ok(self.huggingface()?.vocab_size(with_added_tokens))
    }

    /// The added tokens by id as `(id, (content, single_word, lstrip, rstrip, normalized,
    /// special))`, for Python to rebuild as `tokenizers.AddedToken`.
    #[cfg(feature = "huggingface")]
    fn added_tokens_decoder(&self) -> PyResult<Vec<(u32, AddedTokenFields)>> {
        Ok(self
            .huggingface()?
            .added_tokens_decoder()
            .into_iter()
            .map(|(id, token)| {
                (
                    id,
                    (
                        token.content,
                        token.single_word,
                        token.lstrip,
                        token.rstrip,
                        token.normalized,
                        token.special,
                    ),
                )
            })
            .collect())
    }

    /// The padding parameters as `tokenizers.Tokenizer.padding` reports them.
    #[cfg(feature = "huggingface")]
    fn padding<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(params) = self.huggingface()?.padding() else {
            return Ok(None);
        };
        let padding = PyDict::new(py);
        padding.set_item(
            "length",
            match params.strategy {
                PaddingStrategy::BatchLongest => None,
                PaddingStrategy::Fixed(length) => Some(length),
            },
        )?;
        padding.set_item("pad_to_multiple_of", params.pad_to_multiple_of)?;
        padding.set_item("pad_id", params.pad_id)?;
        padding.set_item("pad_type_id", params.pad_type_id)?;
        padding.set_item("pad_token", &params.pad_token)?;
        padding.set_item("direction", params.direction.as_ref())?;
        Ok(Some(padding))
    }

    /// The truncation parameters as `tokenizers.Tokenizer.truncation` reports them.
    #[cfg(feature = "huggingface")]
    fn truncation<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(params) = self.huggingface()?.truncation() else {
            return Ok(None);
        };
        let truncation = PyDict::new(py);
        truncation.set_item("max_length", params.max_length)?;
        truncation.set_item("stride", params.stride)?;
        truncation.set_item("strategy", params.strategy.as_ref())?;
        truncation.set_item("direction", params.direction.as_ref())?;
        Ok(Some(truncation))
    }

    #[cfg(feature = "huggingface")]
    fn num_special_tokens_to_add(&self, is_pair: bool) -> PyResult<usize> {
        Ok(self.huggingface()?.num_special_tokens_to_add(is_pair))
    }

    #[cfg(feature = "huggingface")]
    fn encode_special_tokens(&self) -> PyResult<bool> {
        Ok(self.huggingface()?.encode_special_tokens())
    }
}

#[cfg(feature = "huggingface")]
type AddedTokenFields = (String, bool, bool, bool, bool, bool);

impl Tokenizer {
    fn new(inner: Codec) -> Self {
        Self {
            inner: Arc::new(inner),
            #[cfg(feature = "fast")]
            fast: OnceLock::new(),
        }
    }

    pub(crate) fn counter(&self, py: Python<'_>, fast: bool) -> SharedCounter {
        #[cfg(feature = "fast")]
        if fast {
            let counter = self.fast.get().unwrap_or_else(|| {
                release_gil(py, || {
                    self.fast
                        .get_or_init(|| self.inner.fast_counter().map(Arc::new))
                })
            });
            if let Some(counter) = counter {
                return SharedCounter::Fast(Arc::clone(counter));
            }
        }
        #[cfg(not(feature = "fast"))]
        let _ = (py, fast);
        SharedCounter::Codec(Arc::clone(&self.inner))
    }

    /// A Python `str` as UTF-8. tiktoken replaces lone surrogates the way its Python `encode`
    /// does; `tokenizers` rejects them, so that backend keeps the encode error.
    fn text<'a>(&self, text: &'a Bound<'_, PyString>) -> PyResult<Cow<'a, str>> {
        match text.to_cow() {
            Ok(text) => Ok(text),
            Err(error) => match *self.inner {
                #[cfg(feature = "tiktoken")]
                Codec::Tiktoken(_) if error.is_instance_of::<PyUnicodeEncodeError>(text.py()) => {
                    text.call_method1("encode", ("utf-16", "surrogatepass"))?
                        .call_method1("decode", ("utf-16", "replace"))?
                        .extract::<String>()
                        .map(Cow::Owned)
                }
                _ => Err(error),
            },
        }
    }

    #[cfg(feature = "tiktoken")]
    fn tiktoken(&self) -> PyResult<&TiktokenTokenizer> {
        match *self.inner {
            Codec::Tiktoken(ref tokenizer) => Ok(tokenizer),
            #[cfg(feature = "huggingface")]
            Codec::HuggingFace(_) => Err(PyValueError::new_err("requires a tiktoken encoding")),
        }
    }

    #[cfg(feature = "tiktoken")]
    fn vocabulary(&self) -> PyResult<&Vocabulary> {
        self.tiktoken()?.vocabulary().ok_or_else(|| {
            PyRuntimeError::new_err("this encoding was built without its vocabulary")
        })
    }

    #[cfg(feature = "huggingface")]
    fn huggingface(&self) -> PyResult<&HuggingFaceTokenizer> {
        match *self.inner {
            Codec::HuggingFace(ref tokenizer) => Ok(tokenizer),
            #[cfg(feature = "tiktoken")]
            Codec::Tiktoken(_) => Err(PyValueError::new_err("requires a Hugging Face tokenizer")),
        }
    }
}

pub(crate) enum SharedCounter {
    Codec(Arc<Codec>),
    #[cfg(feature = "fast")]
    Fast(Arc<FastTokenizer>),
}

impl litellm_token_counter::Tokenizer for SharedCounter {
    fn count_tokens(&self, text: &str) -> Result<usize, Error> {
        match self {
            Self::Codec(codec) => codec.codec().count_tokens(text),
            #[cfg(feature = "fast")]
            Self::Fast(counter) => counter.count_tokens(text).map_err(Error::from),
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
fn direction<T>(value: &str, left: T, right: T, what: &str) -> PyResult<T> {
    match value {
        "left" => Ok(left),
        "right" => Ok(right),
        other => Err(PyValueError::new_err(format!(
            "invalid {what} direction {other:?}: expected 'left' or 'right'"
        ))),
    }
}

/// `tokenizers.Encoding`, mutable like the original: `pad`, `truncate` and `set_sequence_id`
/// change it in place.
#[cfg(feature = "huggingface")]
#[pyclass(module = "litellm.rust_bridge._native")]
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
            Some(json) => encoding_from_json(json)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            None => Encoding::default(),
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (encodings, growing_offsets = true))]
    fn merge(encodings: Vec<PyRef<'_, Self>>, growing_offsets: bool) -> Self {
        Self {
            inner: Encoding::merge(
                encodings.iter().map(|encoding| encoding.inner.clone()),
                growing_offsets,
            ),
        }
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, pyo3::types::PyType>, (String,))> {
        let json = encoding_to_json(&self.inner)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((py.get_type::<Self>(), (json,)))
    }

    fn __repr__(&self) -> String {
        format!(
            "Encoding(num_tokens={}, attributes=[ids, type_ids, tokens, offsets, \
             attention_mask, special_tokens_mask, overflowing])",
            self.inner.len()
        )
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

    #[pyo3(signature = (word_index, sequence_index = 0))]
    fn word_to_tokens(&self, word_index: u32, sequence_index: usize) -> Option<(usize, usize)> {
        self.inner.word_to_tokens(word_index, sequence_index)
    }
    #[pyo3(signature = (word_index, sequence_index = 0))]
    fn word_to_chars(&self, word_index: u32, sequence_index: usize) -> Option<(usize, usize)> {
        self.inner.word_to_chars(word_index, sequence_index)
    }
    fn token_to_sequence(&self, token_index: usize) -> Option<usize> {
        self.inner.token_to_sequence(token_index)
    }
    fn token_to_chars(&self, token_index: usize) -> Option<(usize, usize)> {
        self.inner
            .token_to_chars(token_index)
            .map(|(_, offsets)| offsets)
    }
    fn token_to_word(&self, token_index: usize) -> Option<u32> {
        self.inner.token_to_word(token_index).map(|(_, word)| word)
    }
    #[pyo3(signature = (char_pos, sequence_index = 0))]
    fn char_to_token(&self, char_pos: usize, sequence_index: usize) -> Option<usize> {
        self.inner.char_to_token(char_pos, sequence_index)
    }
    #[pyo3(signature = (char_pos, sequence_index = 0))]
    fn char_to_word(&self, char_pos: usize, sequence_index: usize) -> Option<u32> {
        self.inner.char_to_word(char_pos, sequence_index)
    }

    fn set_sequence_id(&mut self, sequence_id: usize) {
        self.inner.set_sequence_id(sequence_id);
    }

    #[pyo3(signature = (length, direction = "right", pad_id = 0, pad_type_id = 0, pad_token = "[PAD]"))]
    fn pad(
        &mut self,
        length: usize,
        direction: &str,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
    ) -> PyResult<()> {
        let direction = self::direction(
            direction,
            PaddingDirection::Left,
            PaddingDirection::Right,
            "padding",
        )?;
        self.inner
            .pad(length, pad_id, pad_type_id, pad_token, direction);
        Ok(())
    }

    #[pyo3(signature = (max_length, stride = 0, direction = "right"))]
    fn truncate(&mut self, max_length: usize, stride: usize, direction: &str) -> PyResult<()> {
        let direction = self::direction(
            direction,
            TruncationDirection::Left,
            TruncationDirection::Right,
            "truncation",
        )?;
        self.inner.truncate(max_length, stride, direction);
        Ok(())
    }
}
