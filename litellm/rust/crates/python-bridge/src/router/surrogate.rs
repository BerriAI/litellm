//! Lossless UTF-8 escaping for Python `str` values containing lone surrogates at the router
//! Python/Rust boundary. Valid scalar text without the escape sentinel passes through untouched
//! so prefix checks, wildcard matching and exact comparisons inside `router-core` see ordinary
//! UTF-8 characters (`openai/`, `*`, `/`, `\n`, ...).

use std::fmt::Write as _;

use litellm_host_python::{from_py_argument, to_py};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};
use serde_json::{Map, Number, Value};

const SENTINEL: char = '\u{F8FE}';
const SENTINEL_ESCAPE: &str = "\u{F8FE}F8FE";

fn escape_sentinel_only(text: &str) -> String {
    if !text.contains(SENTINEL) {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len() + 4);
    for ch in text.chars() {
        if ch == SENTINEL {
            out.push_str(SENTINEL_ESCAPE);
        } else {
            out.push(ch);
        }
    }
    out
}

fn encode_wtf8_bytes(bytes: &[u8]) -> Option<String> {
    let mut out = String::with_capacity(bytes.len());
    let mut rest = bytes;
    while !rest.is_empty() {
        match std::str::from_utf8(rest) {
            Ok(valid) => {
                out.push_str(&escape_sentinel_only(valid));
                break;
            }
            Err(error) => {
                let valid_len = error.valid_up_to();
                if valid_len > 0 {
                    let valid = std::str::from_utf8(&rest[..valid_len]).ok()?;
                    out.push_str(&escape_sentinel_only(valid));
                    rest = &rest[valid_len..];
                    continue;
                }
                let [0xED, b1 @ 0xA0..=0xBF, b2 @ 0x80..=0xBF, tail @ ..] = rest else {
                    return None;
                };
                let cp: u32 =
                    (0x0Du32 << 12) | ((u32::from(*b1) & 0x3F) << 6) | (u32::from(*b2) & 0x3F);
                let _ = write!(out, "{SENTINEL}{cp:04X}");
                rest = tail;
            }
        }
    }
    Some(out)
}

fn is_upper_hex(byte: u8) -> bool {
    byte.is_ascii_digit() || matches!(byte, b'A'..=b'F')
}

fn decode_to_wtf8_bytes(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let mut rest = text;
    let mut sentinel_utf8 = [0u8; 4];
    let sentinel_bytes = SENTINEL.encode_utf8(&mut sentinel_utf8).as_bytes();
    while let Some(pos) = rest.find(SENTINEL) {
        out.extend_from_slice(&rest.as_bytes()[..pos]);
        let after = &rest[pos + SENTINEL.len_utf8()..];
        if let Some(hex) = after.get(..4)
            && hex.bytes().all(is_upper_hex)
            && let Ok(cp) = u32::from_str_radix(hex, 16)
        {
            if (0xD800..=0xDFFF).contains(&cp) {
                out.push(0xED);
                out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
                out.push(0x80 | (cp & 0x3F) as u8);
                rest = &after[4..];
                continue;
            }
            if cp == u32::from(SENTINEL) {
                out.extend_from_slice(sentinel_bytes);
                rest = &after[4..];
                continue;
            }
        }
        out.extend_from_slice(sentinel_bytes);
        rest = after;
    }
    out.extend_from_slice(rest.as_bytes());
    out
}

pub(crate) fn encode_str(text: &Bound<'_, PyString>) -> PyResult<String> {
    if let Ok(valid) = text.to_str() {
        return Ok(escape_sentinel_only(valid));
    }
    let encoded = text.call_method1("encode", ("utf-8", "surrogatepass"))?;
    let bytes = encoded.cast::<PyBytes>()?.as_bytes();
    encode_wtf8_bytes(bytes)
        .ok_or_else(|| PyValueError::new_err("unrecognized WTF-8 byte sequence from Python str"))
}

pub(crate) fn encode_opt_str(text: Option<&Bound<'_, PyString>>) -> PyResult<Option<String>> {
    text.map(encode_str).transpose()
}

pub(crate) fn encode_vec_str(items: Vec<Bound<'_, PyString>>) -> PyResult<Vec<String>> {
    items.iter().map(encode_str).collect()
}

pub(crate) fn encode_opt_vec_str(
    items: Option<Vec<Bound<'_, PyString>>>,
) -> PyResult<Option<Vec<String>>> {
    items.map(encode_vec_str).transpose()
}

pub(crate) fn encode_py_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(Value::Bool(flag.is_true()));
    }
    if let Ok(int) = value.cast::<PyInt>() {
        if let Ok(number) = int.extract::<i64>() {
            return Ok(Value::from(number));
        }
        if let Ok(number) = int.extract::<u64>() {
            return Ok(Value::from(number));
        }
        return from_py_argument(value);
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        return match Number::from_f64(float.value()) {
            Some(number) => Ok(Value::Number(number)),
            None => from_py_argument(value),
        };
    }
    if let Ok(text) = value.cast::<PyString>() {
        return encode_str(text).map(Value::String);
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = Map::with_capacity(dict.len());
        for (key, item) in dict.iter() {
            let key_str = match key.cast::<PyString>() {
                Ok(s) => encode_str(s)?,
                Err(_) => from_py_argument(&key)?,
            };
            map.insert(key_str, encode_py_to_value(&item)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = value.cast::<PyList>() {
        return list
            .iter()
            .map(|item| encode_py_to_value(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return tuple
            .iter()
            .map(|item| encode_py_to_value(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    from_py_argument(value)
}

pub(crate) fn decode_str<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyString>> {
    if !text.contains(SENTINEL) {
        return Ok(PyString::new(py, text));
    }
    let wtf8 = decode_to_wtf8_bytes(text);
    PyBytes::new(py, &wtf8)
        .call_method1("decode", ("utf-8", "surrogatepass"))?
        .cast_into::<PyString>()
        .map_err(Into::into)
}

pub(crate) fn decode_opt_str<'py>(
    py: Python<'py>,
    text: Option<&str>,
) -> PyResult<Option<Bound<'py, PyString>>> {
    text.map(|s| decode_str(py, s)).transpose()
}

pub(crate) fn decode_vec_str<'py>(
    py: Python<'py>,
    items: &[String],
) -> PyResult<Vec<Bound<'py, PyString>>> {
    items.iter().map(|s| decode_str(py, s)).collect()
}

pub(crate) fn decode_list_str<'py>(
    py: Python<'py>,
    items: &[String],
) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, decode_vec_str(py, items)?)
}

fn value_has_sentinel(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
        Value::String(s) => s.contains(SENTINEL),
        Value::Array(items) => items.iter().any(value_has_sentinel),
        Value::Object(map) => map
            .iter()
            .any(|(key, item)| key.contains(SENTINEL) || value_has_sentinel(item)),
    }
}

pub(crate) fn decode_value<'py>(py: Python<'py>, value: &Value) -> PyResult<Py<PyAny>> {
    if !value_has_sentinel(value) {
        return to_py(py, value);
    }
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => to_py(py, value),
        Value::String(s) => decode_str(py, s).map(|s| s.into_any().unbind()),
        Value::Array(items) => {
            let decoded = items
                .iter()
                .map(|item| decode_value(py, item))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, decoded).map(|list| list.into_any().unbind())
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(decode_str(py, key)?, decode_value(py, item)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{ffi::c_str, types::PyDict};
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::plain_ascii("r'openai/gpt-4o'", "openai/gpt-4o")]
    #[case::high_surrogate(r"'openai/\ud800'", "openai/\u{F8FE}D800")]
    #[case::low_surrogate(r"'openai/\udc00'", "openai/\u{F8FE}DC00")]
    #[case::reversed_pair(r"'openai/\udc00\ud800'", "openai/\u{F8FE}DC00\u{F8FE}D800")]
    #[case::adjacent_pair(r"'openai/\ud800\udc00'", "openai/\u{F8FE}D800\u{F8FE}DC00")]
    #[case::literal_sentinel(r"'openai/\uf8fe\ud800'", "openai/\u{F8FE}F8FE\u{F8FE}D800")]
    #[case::mixed_multibyte(r"'日本語/\ud83d/テスト'", "日本語/\u{F8FE}D83D/テスト")]
    fn python_strings_round_trip_losslessly_through_utf8_bridge(
        #[case] py_expr: &str,
        #[case] expected_encoded: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let expr = std::ffi::CString::new(py_expr).unwrap();
            let raw = py.eval(&expr, None, None).unwrap();
            let py_str = raw.cast::<PyString>().unwrap();
            let encoded = encode_str(py_str).unwrap();
            assert_eq!(encoded, expected_encoded);
            let decoded = decode_str(py, &encoded).unwrap();
            assert!(decoded.as_any().eq(py_str).unwrap());
        });
    }

    #[test]
    fn nested_document_round_trips_surrogate_keys_and_values() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c_str!(
                    r#"
payload = {
    "model": "openai/\ud800",
    "meta\udfff": ["a\udc00b", "\uf8fe\ud800", 1, True, None],
}
"#
                ),
                None,
                Some(&locals),
            )
            .unwrap();
            let payload = locals.get_item("payload").unwrap().unwrap();
            let encoded = encode_py_to_value(&payload).unwrap();
            assert_eq!(
                encoded,
                json!({
                    "model": "openai/\u{F8FE}D800",
                    "meta\u{F8FE}DFFF": ["a\u{F8FE}DC00b", "\u{F8FE}F8FE\u{F8FE}D800", 1, true, null],
                })
            );
            let decoded = decode_value(py, &encoded).unwrap();
            assert!(decoded.bind(py).eq(&payload).unwrap());
        });
    }
}
