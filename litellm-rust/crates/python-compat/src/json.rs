//! `json.dumps` with CPython's default options, and the JSON value `json.loads` returns.
//!
//! Defaults are `ensure_ascii=True`, `allow_nan=True`, separators `(", ", ": ")`, and no key
//! sorting. Python's `json.loads` is mapped by [`from_json`]; a `serde_json` number that does
//! not fit `i64` or `u64` arrives as a float, where Python would keep an `int`.

use std::fmt::Write;

use serde_json::{Map, Number};

use crate::{Error, Value, repr::float_repr};

/// `json.dumps(value)`.
pub fn dumps(value: &Value) -> Result<String, Error> {
    let mut out = String::new();
    write_value(&mut out, value)?;
    Ok(out)
}

/// The Python value `json.loads` returns for a JSON document.
pub fn from_json(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::None,
        serde_json::Value::Bool(value) => Value::Bool(value),
        serde_json::Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                Value::Int(value.into())
            } else if let Some(value) = number.as_u64() {
                Value::Int(value.into())
            } else {
                Value::Float(number.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(text) => Value::Str(text),
        serde_json::Value::Array(values) => {
            Value::List(values.into_iter().map(from_json).collect())
        }
        serde_json::Value::Object(entries) => Value::Dict(
            entries
                .into_iter()
                .map(|(key, value)| (Value::Str(key), from_json(value)))
                .collect(),
        ),
    }
}

/// `json.loads(json.dumps(value))` as a `serde_json` value: tuples become arrays and dict
/// keys are coerced to strings as `json.dumps` does. Non-finite floats, which Python writes
/// as `NaN` and `Infinity`, have no `serde_json` form and fail with [`Error::NonFiniteFloat`].
pub fn to_json(value: &Value) -> Result<serde_json::Value, Error> {
    Ok(match value {
        Value::None => serde_json::Value::Null,
        Value::Bool(value) => serde_json::Value::Bool(*value),
        Value::Int(value) => {
            let text = value.to_string();
            serde_json::Value::Number(
                text.parse::<i64>()
                    .map(Number::from)
                    .or_else(|_| text.parse::<u64>().map(Number::from))
                    .map_err(|_| Error::IntegerOutOfRange)?,
            )
        }
        Value::Float(value) => {
            serde_json::Value::Number(Number::from_f64(*value).ok_or(Error::NonFiniteFloat)?)
        }
        Value::Str(text) => serde_json::Value::String(text.clone()),
        Value::List(values) | Value::Tuple(values) => {
            serde_json::Value::Array(values.iter().map(to_json).collect::<Result<Vec<_>, _>>()?)
        }
        Value::Dict(entries) => serde_json::Value::Object(
            entries
                .iter()
                .map(|(key, value)| Ok((json_key(key)?, to_json(value)?)))
                .collect::<Result<Map<_, _>, Error>>()?,
        ),
        value @ (Value::Bytes(_) | Value::Set(_) | Value::Complex { .. }) => {
            return Err(Error::NotJsonSerializable(value.type_name()));
        }
    })
}

fn write_value(out: &mut String, value: &Value) -> Result<(), Error> {
    match value {
        Value::None => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(value) => {
            let _ = write!(out, "{value}");
        }
        Value::Float(value) => out.push_str(&float_text(*value)),
        Value::Str(text) => write_string(out, text),
        Value::List(values) | Value::Tuple(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_value(out, value)?;
            }
            out.push(']');
        }
        Value::Dict(entries) => {
            out.push('{');
            for (index, (key, value)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(out, &json_key(key)?);
                out.push_str(": ");
                write_value(out, value)?;
            }
            out.push('}');
        }
        value @ (Value::Bytes(_) | Value::Set(_) | Value::Complex { .. }) => {
            return Err(Error::NotJsonSerializable(value.type_name()));
        }
    }
    Ok(())
}

/// `json.encoder.JSONEncoder.iterencode`'s `floatstr` with `allow_nan=True`.
fn float_text(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_owned()
    } else if value.is_infinite() {
        if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
    } else {
        float_repr(value)
    }
}

/// Dict key coercion in `json.dumps`: scalars become their JSON text, other keys fail.
fn json_key(key: &Value) -> Result<String, Error> {
    Ok(match key {
        Value::Str(text) => text.clone(),
        Value::Int(value) => value.to_string(),
        Value::Float(value) => float_text(*value),
        Value::Bool(true) => "true".to_owned(),
        Value::Bool(false) => "false".to_owned(),
        Value::None => "null".to_owned(),
        key => return Err(Error::InvalidJsonKey(key.type_name())),
    })
}

/// `py_encode_basestring_ascii`: escape `"`, `\`, control characters, and everything outside
/// printable ASCII as `\uXXXX`, with surrogate pairs above the BMP.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            ch => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}
