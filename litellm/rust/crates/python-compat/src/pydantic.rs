//! Pydantic's lax `bool` and `int` validation: `TypeAdapter(bool)` and `TypeAdapter(int)`.

use num_bigint::BigInt;

use crate::Value;

const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

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
    let magnitude = BigInt::parse_bytes(digits.replace('_', "").as_bytes(), 10)?;
    Some(if negative { -magnitude } else { magnitude })
}
