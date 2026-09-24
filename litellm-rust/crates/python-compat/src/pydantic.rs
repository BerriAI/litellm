//! pydantic's lax-mode validation, which LiteLLM's usage and response models apply to the
//! numbers providers send.

use num_bigint::BigInt;
use num_traits::Zero;

use crate::number::{int_from_float, parse_int_literal};
use crate::{Error, PythonException, Value};

/// pydantic-core converts floats through `i64`, so larger magnitudes fail validation.
const I64_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// pydantic raises `ValidationError`, a `ValueError` subclass, for every rejected input.
fn invalid(message: &str) -> Error {
    Error::conversion(PythonException::ValueError, message)
}

fn int_from_text(text: &str) -> Result<BigInt, Error> {
    let trimmed = text.trim();
    let without_zero_fraction = trimmed
        .split_once('.')
        .filter(|(_, fraction)| !fraction.is_empty() && fraction.bytes().all(|byte| byte == b'0'))
        .map_or(trimmed, |(whole, _)| whole);
    parse_int_literal(without_zero_fraction).ok_or_else(|| {
        invalid("Input should be a valid integer, unable to parse string as an integer")
    })
}

/// `TypeAdapter(int).validate_python(value)` in lax mode.
pub fn lax_int(value: &Value) -> Result<BigInt, Error> {
    match value {
        Value::Bool(flag) => Ok(BigInt::from(u8::from(*flag))),
        Value::Int(number) => Ok(number.clone()),
        Value::Float(number) if !number.is_finite() => {
            Err(invalid("Input should be a finite number"))
        }
        Value::Float(number) if number.abs() >= I64_LIMIT => Err(invalid(
            "Input should be a valid integer, unable to parse number as an integer",
        )),
        Value::Float(number) => {
            let whole = int_from_float(*number)?;
            if (number - number.trunc()).is_zero() {
                Ok(whole)
            } else {
                Err(invalid(
                    "Input should be a valid integer, got a number with a fractional part",
                ))
            }
        }
        Value::Str(text) => int_from_text(text),
        Value::Bytes(bytes) => int_from_text(&String::from_utf8_lossy(bytes)),
        _ => Err(invalid("Input should be a valid integer")),
    }
}
