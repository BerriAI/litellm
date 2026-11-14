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

pub fn float_repr(float: f64) -> String {
    if float.is_nan() {
        return "nan".to_owned();
    }
    if float.is_infinite() {
        return if float > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let scientific = format!("{:e}", float.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let point = exponent + 1;
    let sign = if float.is_sign_negative() { "-" } else { "" };
    let body = if (-3..=16).contains(&point) {
        if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        }
    } else {
        let fraction = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits.clone()
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{fraction}e{exponent_sign}{:02}", exponent.unsigned_abs())
    };
    format!("{sign}{body}")
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

fn python_digits(text: &str) -> Option<String> {
    let mut cleaned = String::with_capacity(text.len());
    let mut previous_digit = false;
    let bytes = text.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'_' {
            let next_digit = bytes.get(index + 1).is_some_and(u8::is_ascii_digit);
            if !previous_digit || !next_digit {
                return None;
            }
            previous_digit = false;
            continue;
        }
        previous_digit = byte.is_ascii_digit();
        cleaned.push(*byte as char);
    }
    Some(cleaned)
}

const PYTHON_ASCII_SPACE: [char; 6] = [' ', '\t', '\n', '\u{b}', '\u{c}', '\r'];

fn ascii_stripped(text: &str) -> Out<&str> {
    if !text.is_ascii() {
        return Err(PythonServed);
    }
    Ok(text.trim_matches(&PYTHON_ASCII_SPACE[..]))
}

pub fn python_float_text(text: &str) -> Out<Option<f64>> {
    let stripped = ascii_stripped(text)?;
    let lowered = stripped.to_ascii_lowercase();
    let unsigned = lowered.trim_start_matches(['+', '-']);
    if lowered.len() - unsigned.len() > 1 {
        return Ok(None);
    }
    if matches!(unsigned, "inf" | "infinity" | "nan") {
        return Ok(lowered.parse::<f64>().ok());
    }
    let Some(cleaned) = python_digits(&lowered) else {
        return Ok(None);
    };
    let grammar = cleaned.trim_start_matches(['+', '-']);
    let (number, exponent) = match grammar.split_once('e') {
        Some((number, exponent)) => (number, Some(exponent)),
        None => (grammar, None),
    };
    let (whole, fraction) = number.split_once('.').unwrap_or((number, ""));
    let valid_number = (!whole.is_empty() || !fraction.is_empty())
        && whole.bytes().all(|byte| byte.is_ascii_digit())
        && fraction.bytes().all(|byte| byte.is_ascii_digit());
    let valid_exponent = exponent.is_none_or(|exponent| {
        let digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
    });
    if !valid_number || !valid_exponent {
        return Ok(None);
    }
    Ok(cleaned.parse::<f64>().ok())
}

pub fn python_int_text(text: &str) -> Out<Option<i128>> {
    let stripped = ascii_stripped(text)?;
    let unsigned = stripped.strip_prefix(['+', '-']).unwrap_or(stripped);
    if unsigned.is_empty()
        || !unsigned
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_digit())
    {
        return Ok(None);
    }
    let Some(cleaned) = python_digits(stripped) else {
        return Ok(None);
    };
    if !cleaned
        .trim_start_matches(['+', '-'])
        .bytes()
        .all(|byte| byte.is_ascii_digit())
    {
        return Ok(None);
    }
    cleaned.parse::<i128>().map(Some).map_err(|_| PythonServed)
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
