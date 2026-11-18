use litellm_python_compat::number;
pub use litellm_python_compat::repr::float_repr;
use serde_json::{Map, Value};

use super::{Out, PythonServed, printable::NOT_PRINTABLE};
use crate::python::raw_scalar;

#[derive(Clone, Copy, Debug)]
pub enum Py<'a> {
    None,
    Bool(bool),
    Int(i128),
    BigInt(&'a str),
    Float(f64),
    Str(&'a str),
    List(&'a [Value]),
    Dict(&'a Map<String, Value>),
}

pub fn py(value: &Value) -> Out<Py<'_>> {
    Ok(match value {
        Value::Null => Py::None,
        Value::Bool(flag) => Py::Bool(*flag),
        Value::Number(number) => match (number.as_i64(), number.as_u64(), number.as_f64()) {
            (Some(int), _, _) => Py::Int(i128::from(int)),
            (None, Some(int), _) => Py::Int(i128::from(int)),
            (None, None, Some(float)) => Py::Float(float),
            _ => return Err(PythonServed),
        },
        Value::String(text) => match raw_scalar(value) {
            None => Py::Str(text),
            Some("NaN") => Py::Float(f64::NAN),
            Some("Infinity") => Py::Float(f64::INFINITY),
            Some("-Infinity") => Py::Float(f64::NEG_INFINITY),
            Some(raw) if is_int_literal(raw) => match raw.parse::<i128>() {
                Ok(int) => Py::Int(int),
                Err(_) => Py::BigInt(raw),
            },
            Some(_) => return Err(PythonServed),
        },
        Value::Array(items) => Py::List(items),
        Value::Object(fields) => Py::Dict(fields),
    })
}

fn is_int_literal(raw: &str) -> bool {
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

pub fn text(value: &Value) -> Out<Option<&str>> {
    Ok(match py(value)? {
        Py::Str(text) => Some(text),
        _ => None,
    })
}

pub fn truthy(value: &Value) -> Out<bool> {
    Ok(match py(value)? {
        Py::None => false,
        Py::Bool(flag) => flag,
        Py::Int(int) => int != 0,
        Py::BigInt(_) => true,
        Py::Float(float) => float != 0.0,
        Py::Str(text) => !text.is_empty(),
        Py::List(items) => !items.is_empty(),
        Py::Dict(fields) => !fields.is_empty(),
    })
}

pub fn length(text: &str) -> i128 {
    text.chars().count() as i128
}

fn printable(code: u32) -> bool {
    NOT_PRINTABLE
        .binary_search_by(|(start, end)| {
            if *end < code {
                std::cmp::Ordering::Less
            } else if *start > code {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_err()
}

fn str_repr_length(text: &str) -> i128 {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    2 + text
        .chars()
        .map(|character| match character {
            '\\' | '\t' | '\n' | '\r' => 2,
            _ if character == quote => 2,
            _ if printable(character as u32) => 1,
            _ if (character as u32) < 0x100 => 4,
            _ if (character as u32) < 0x10000 => 6,
            _ => 10,
        })
        .sum::<i128>()
}

fn int_length(int: i128) -> i128 {
    int.to_string().len() as i128
}

pub fn repr_length(value: &Value) -> Out<i128> {
    Ok(match py(value)? {
        Py::None => 4,
        Py::Bool(true) => 4,
        Py::Bool(false) => 5,
        Py::Int(int) => int_length(int),
        Py::BigInt(raw) => raw.len() as i128,
        Py::Float(float) => float_repr(float).len() as i128,
        Py::Str(text) => str_repr_length(text),
        Py::List(items) => {
            let inner = items.iter().map(repr_length).sum::<Out<i128>>()?;
            2 + inner + separators(items.len(), 2)
        }
        Py::Dict(fields) => {
            let inner = fields
                .iter()
                .map(|(key, item)| Ok(str_repr_length(key) + 2 + repr_length(item)?))
                .sum::<Out<i128>>()?;
            2 + inner + separators(fields.len(), 2)
        }
    })
}

pub fn str_length(value: &Value) -> Out<i128> {
    match py(value)? {
        Py::Str(text) => Ok(length(text)),
        _ => repr_length(value),
    }
}

fn separators(count: usize, width: i128) -> i128 {
    count.saturating_sub(1) as i128 * width
}

fn json_str_length(text: &str) -> i128 {
    2 + text
        .chars()
        .map(|character| match character {
            '"' | '\\' | '\n' | '\r' | '\t' | '\u{8}' | '\u{c}' => 2,
            ' '..='~' => 1,
            _ if (character as u32) < 0x10000 => 6,
            _ => 12,
        })
        .sum::<i128>()
}

pub fn json_length(value: &Value, compact: bool) -> Out<i128> {
    let item_separator = if compact { 1 } else { 2 };
    Ok(match py(value)? {
        Py::None => 4,
        Py::Bool(true) => 4,
        Py::Bool(false) => 5,
        Py::Int(int) => int_length(int),
        Py::BigInt(raw) => raw.len() as i128,
        Py::Float(float) if float.is_nan() => 3,
        Py::Float(float) if float.is_infinite() => {
            if float > 0.0 {
                8
            } else {
                9
            }
        }
        Py::Float(float) => float_repr(float).len() as i128,
        Py::Str(text) => json_str_length(text),
        Py::List(items) => {
            let inner = items
                .iter()
                .map(|item| json_length(item, compact))
                .sum::<Out<i128>>()?;
            2 + inner + separators(items.len(), item_separator)
        }
        Py::Dict(fields) => {
            let inner = fields
                .iter()
                .map(|(key, item)| {
                    Ok(json_str_length(key) + item_separator + json_length(item, compact)?)
                })
                .sum::<Out<i128>>()?;
            2 + inner + separators(fields.len(), item_separator)
        }
    })
}

/// Non-ASCII text stays with Python: [`number`] reads ASCII digits only, where Python's `int`
/// and `float` take any Unicode decimal digit.
fn ascii(text: &str) -> Out<&str> {
    if text.is_ascii() {
        Ok(text)
    } else {
        Err(PythonServed)
    }
}

pub fn python_float_text(text: &str) -> Out<Option<f64>> {
    Ok(number::float(ascii(text)?))
}

pub fn python_int_text(text: &str) -> Out<Option<i128>> {
    number::int(ascii(text)?)
        .map(|value| i128::try_from(&value).map_err(|_| PythonServed))
        .transpose()
}

pub fn truncated(float: f64) -> Out<Option<i128>> {
    if !float.is_finite() {
        return Ok(None);
    }
    let whole = float.trunc();
    if whole.abs() >= 2f64.powi(126) {
        return Err(PythonServed);
    }
    Ok(Some(whole as i128))
}

pub fn big_int_as_float(raw: &str) -> Option<f64> {
    raw.parse::<f64>().ok().filter(|float| float.is_finite())
}
