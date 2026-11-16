//! Pydantic's lax `bool` and `int` validation: `TypeAdapter(bool)` and `TypeAdapter(int)`.

use num_bigint::BigInt;
use serde::{Deserialize, Deserializer, de::Error};
use serde_with::DeserializeAs;

use crate::Value;

const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;
/// pydantic refuses an integer longer than this once cleaned, a `-` included. Checking first also
/// keeps decimal parsing, quadratic in the digit count, from running on a huge string.
const MAX_INT_TEXT_LEN: usize = 4300;

/// `TypeAdapter(bool).validate_python(value)`, or `None` where pydantic raises `ValidationError`.
pub fn lax_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::Int(number) => match i8::try_from(number) {
            Ok(0) => Some(false),
            Ok(1) => Some(true),
            _ => None,
        },
        Value::Float(number) if *number == 0.0 => Some(false),
        Value::Float(number) if *number == 1.0 => Some(true),
        Value::Str(text) => match text.to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
            _ => None,
        },
        _ => None,
    }
}

/// `TypeAdapter(int).validate_python(value)`, or `None` where pydantic raises `ValidationError`.
pub fn lax_int(value: &Value) -> Option<BigInt> {
    match value {
        Value::Bool(flag) => Some(BigInt::from(u8::from(*flag))),
        Value::Int(number) => Some(number.clone()),
        Value::Float(number)
            if number.fract() == 0.0 && (-TWO_POW_63 < *number && *number < TWO_POW_63) =>
        {
            Some(BigInt::from(*number as i64))
        }
        Value::Str(text) => int_text(text.trim()),
        _ => None,
    }
}

fn int_text(text: &str) -> Option<BigInt> {
    let whole = match text.split_once('.') {
        Some((whole, fraction))
            if !fraction.is_empty() && fraction.bytes().all(|byte| byte == b'0') =>
        {
            whole
        }
        Some(_) => return None,
        None => text,
    };
    let (negative, digits) = match whole.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, whole.strip_prefix('+').unwrap_or(whole)),
    };
    let well_formed = !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__")
        && digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'_');
    if !well_formed {
        return None;
    }
    let digits = digits.replace('_', "");
    let significant = digits.trim_start_matches('0').len();
    if usize::from(negative) + significant > MAX_INT_TEXT_LEN {
        return None;
    }
    let magnitude = BigInt::parse_bytes(digits.as_bytes(), 10)?;
    Some(if negative { -magnitude } else { magnitude })
}

/// [`lax_int`] as a `serde_with` adapter on an `i64`, failing where pydantic raises or the integer
/// does not fit.
pub struct LaxI64;

impl<'de> DeserializeAs<'de, i64> for LaxI64 {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
        lax_int(&Value::deserialize(deserializer)?)
            .and_then(|number| i64::try_from(number).ok())
            .ok_or_else(|| D::Error::custom("expected an integer in the i64 range"))
    }
}
