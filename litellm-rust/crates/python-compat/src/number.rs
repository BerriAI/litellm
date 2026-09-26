//! `int(value)` and `float(value)` for plain data.

use num_bigint::BigInt;
use num_traits::{FromPrimitive, ToPrimitive};

use crate::repr::repr;
use crate::{Error, PythonException, Value};

/// Digits with single underscores between them, as Python's numeric literals allow.
pub(crate) fn without_digit_separators(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let separators_ok = bytes.iter().enumerate().all(|(index, byte)| {
        *byte != b'_'
            || (index > 0
                && bytes[index - 1].is_ascii_digit()
                && bytes.get(index + 1).is_some_and(u8::is_ascii_digit))
    });
    separators_ok.then(|| text.replace('_', ""))
}

pub(crate) fn parse_int_literal(text: &str) -> Option<BigInt> {
    let trimmed = text.trim();
    let digits = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    if digits.is_empty()
        || !digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'_')
    {
        return None;
    }
    without_digit_separators(trimmed)?.parse().ok()
}

fn int_from_text(text: &str, shown: &Value) -> Result<BigInt, Error> {
    parse_int_literal(text).ok_or_else(|| {
        Error::conversion(
            PythonException::ValueError,
            format!("invalid literal for int() with base 10: {}", repr(shown)),
        )
    })
}

pub(crate) fn int_from_float(value: f64) -> Result<BigInt, Error> {
    if value.is_nan() {
        return Err(Error::conversion(
            PythonException::ValueError,
            "cannot convert float NaN to integer",
        ));
    }
    BigInt::from_f64(value.trunc()).ok_or_else(|| {
        Error::conversion(
            PythonException::OverflowError,
            "cannot convert float infinity to integer",
        )
    })
}

/// `int(value)`.
pub fn int(value: &Value) -> Result<BigInt, Error> {
    match value {
        Value::Bool(flag) => Ok(BigInt::from(u8::from(*flag))),
        Value::Int(number) => Ok(number.clone()),
        Value::Float(number) => int_from_float(*number),
        Value::Str(text) => int_from_text(text, value),
        Value::Bytes(bytes) => int_from_text(&String::from_utf8_lossy(bytes), value),
        other => Err(Error::conversion(
            PythonException::TypeError,
            format!(
                "int() argument must be a string, a bytes-like object or a real number, not '{}'",
                other.type_name()
            ),
        )),
    }
}

fn float_from_text(text: &str, shown: &Value) -> Result<f64, Error> {
    let trimmed = text.trim();
    let well_formed = !trimmed.is_empty()
        && trimmed
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'+' | b'-'));
    well_formed
        .then(|| without_digit_separators(trimmed))
        .flatten()
        .and_then(|number| number.parse().ok())
        .ok_or_else(|| {
            Error::conversion(
                PythonException::ValueError,
                format!("could not convert string to float: {}", repr(shown)),
            )
        })
}

/// `float(value)`.
pub fn float(value: &Value) -> Result<f64, Error> {
    match value {
        Value::Bool(flag) => Ok(f64::from(u8::from(*flag))),
        Value::Int(number) => number
            .to_f64()
            .filter(|float| float.is_finite())
            .ok_or_else(|| {
                Error::conversion(
                    PythonException::OverflowError,
                    "int too large to convert to float",
                )
            }),
        Value::Float(number) => Ok(*number),
        Value::Str(text) => float_from_text(text, value),
        Value::Bytes(bytes) => float_from_text(&String::from_utf8_lossy(bytes), value),
        other => Err(Error::conversion(
            PythonException::TypeError,
            format!(
                "float() argument must be a string or a real number, not '{}'",
                other.type_name()
            ),
        )),
    }
}
