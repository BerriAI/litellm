use litellm_python_compat::{Value, json, literal, repr::to_str};

use crate::{Error, FromValue, IntoValue, Number};

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("Redis value is not UTF-8")]
    NotUtf8,
    #[error("Redis value is neither JSON nor a Python literal: {0}")]
    Unparsed(#[source] litellm_python_compat::Error),
    #[error("Redis value has the wrong shape: {0}")]
    Shape(&'static str),
}

/// `RedisCache._get_cache_logic`: `json.loads`, falling back to `ast.literal_eval`.
pub(crate) fn python_value(raw: &[u8]) -> Result<Value, DecodeError> {
    let text = std::str::from_utf8(raw).map_err(|_| DecodeError::NotUtf8)?;
    json::loads(text).or_else(|_| literal::literal_eval(text).map_err(DecodeError::Unparsed))
}

/// A cached value as `RedisCache._get_cache_logic` reads it.
pub(crate) fn decode<V: FromValue>(raw: &[u8]) -> Result<V, DecodeError> {
    V::from_value(&python_value(raw)?)
}

/// `json.dumps(value)`, what `RedisCache.async_set_cache` writes.
pub(crate) fn dumps<V: IntoValue>(value: V) -> Result<String, Error> {
    Ok(json::dumps(&value.into_value())?)
}

/// `str(value)`, what the sync `RedisCache.set_cache` writes.
pub(crate) fn str<V: IntoValue>(value: V) -> String {
    to_str(&value.into_value())
}

/// A number written by Python, `shape` naming what it was meant to be when it is not one.
pub(crate) fn number(raw: &[u8], shape: &'static str) -> Result<Number, DecodeError> {
    Number::from_value(&python_value(raw)?).ok_or(DecodeError::Shape(shape))
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::{Value, json, repr::repr};
    use rstest::rstest;

    use super::{number, python_value};
    use crate::Number;

    // Expected values are the repr of main's RedisCache._get_cache_logic on these bytes
    #[rstest]
    #[case::nan(b"NaN", "nan")]
    #[case::infinity(b"Infinity", "inf")]
    #[case::negative_infinity(b"-Infinity", "-inf")]
    #[case::nan_inside_a_document(br#"{"spend": NaN}"#, "{'spend': nan}")]
    #[case::python_literal_fallback(b"{'a': (1, 2)}", "{'a': (1, 2)}")]
    fn reads_what_python_reads(#[case] raw: &[u8], #[case] expected: &str) {
        assert_eq!(repr(&python_value(raw).unwrap()), expected);
    }

    #[rstest]
    #[case::nan(f64::NAN)]
    #[case::infinity(f64::INFINITY)]
    #[case::negative_infinity(f64::NEG_INFINITY)]
    fn a_non_finite_number_reads_back_as_written(#[case] value: f64) {
        let written = json::dumps(&Value::Float(value)).unwrap();

        let read = number(written.as_bytes(), "a number").unwrap();

        assert_eq!(
            repr(&read.to_value()),
            repr(&Number::Float(value).to_value())
        );
    }
}
