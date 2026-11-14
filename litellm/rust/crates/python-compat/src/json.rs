//! `json.dumps` with CPython's default options, and `json.loads` with its default decoder.
//!
//! Defaults are `ensure_ascii=True`, `allow_nan=True`, separators `(", ", ": ")`, and no key
//! sorting on the way out, and `strict=True` with `NaN`, `Infinity` and `-Infinity` read as floats
//! on the way in.

use std::{
    collections::{HashMap, hash_map::Entry},
    fmt::Write,
    str::FromStr,
};

use num_bigint::BigInt;
use serde_json::{Map, Number};

use crate::{Error, MAX_DEPTH, Value, repr::float_repr};

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

/// `json.loads(text)`. Integers keep full precision, and a repeated key keeps its first position
/// with its last value, as `dict(pairs)` does. An escape that leaves a lone surrogate fails, since
/// a Rust `String` cannot hold one.
pub fn loads(text: &str) -> Result<Value, Error> {
    let mut decoder = Decoder {
        text,
        bytes: text.as_bytes(),
        pos: 0,
        depth: 0,
    };
    decoder.skip_whitespace();
    let value = decoder.value()?;
    decoder.skip_whitespace();
    if decoder.pos != decoder.bytes.len() {
        return Err(decoder.error());
    }
    Ok(value)
}

/// CPython's `json.scanner` and `scanstring`, one pass over the bytes.
struct Decoder<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl Decoder<'_> {
    fn error(&self) -> Error {
        Error::InvalidJson(self.pos)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn eat(&mut self, token: &str) -> bool {
        let found = self.bytes[self.pos..].starts_with(token.as_bytes());
        if found {
            self.pos += token.len();
        }
        found
    }

    fn expect(&mut self, token: &str) -> Result<(), Error> {
        if self.eat(token) {
            Ok(())
        } else {
            Err(self.error())
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn value(&mut self) -> Result<Value, Error> {
        match self.peek() {
            Some(b'"') => self.string().map(Value::Str),
            Some(b'[') => self.nested(Self::array),
            Some(b'{') => self.nested(Self::object),
            _ if self.eat("null") => Ok(Value::None),
            _ if self.eat("true") => Ok(Value::Bool(true)),
            _ if self.eat("false") => Ok(Value::Bool(false)),
            _ if self.eat("NaN") => Ok(Value::Float(f64::NAN)),
            _ if self.eat("Infinity") => Ok(Value::Float(f64::INFINITY)),
            _ if self.eat("-Infinity") => Ok(Value::Float(f64::NEG_INFINITY)),
            _ => self.number(),
        }
    }

    fn nested(&mut self, parse: fn(&mut Self) -> Result<Value, Error>) -> Result<Value, Error> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(Error::TooDeep);
        }
        let value = parse(self)?;
        self.depth -= 1;
        Ok(value)
    }

    fn array(&mut self) -> Result<Value, Error> {
        self.expect("[")?;
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.eat("]") {
            return Ok(Value::List(values));
        }
        loop {
            values.push(self.value()?);
            self.skip_whitespace();
            if self.eat("]") {
                return Ok(Value::List(values));
            }
            self.expect(",")?;
            self.skip_whitespace();
        }
    }

    fn object(&mut self) -> Result<Value, Error> {
        self.expect("{")?;
        self.skip_whitespace();
        let mut entries = Vec::new();
        let mut positions = HashMap::<_, usize>::new();
        if self.eat("}") {
            return Ok(Value::Dict(entries));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(self.error());
            }
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(":")?;
            self.skip_whitespace();
            let value = self.value()?;
            match positions.entry(key) {
                Entry::Occupied(position) => entries[*position.get()].1 = value,
                Entry::Vacant(position) => {
                    entries.push((Value::Str(position.key().clone()), value));
                    position.insert(entries.len() - 1);
                }
            }
            self.skip_whitespace();
            if self.eat("}") {
                return Ok(Value::Dict(entries));
            }
            self.expect(",")?;
            self.skip_whitespace();
        }
    }

    /// `scanstring` with `strict=True`: control characters must be escaped.
    fn string(&mut self) -> Result<String, Error> {
        self.expect("\"")?;
        let mut out = String::new();
        loop {
            let start = self.pos;
            while self
                .peek()
                .is_some_and(|byte| byte != b'"' && byte != b'\\' && byte >= 0x20)
            {
                self.pos += 1;
            }
            out.push_str(&self.text[start..self.pos]);
            match self.peek() {
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    out.push(self.escape()?);
                }
                _ => return Err(self.error()),
            }
        }
    }

    fn escape(&mut self) -> Result<char, Error> {
        let escaped = match self.peek() {
            Some(b'"') => '"',
            Some(b'\\') => '\\',
            Some(b'/') => '/',
            Some(b'b') => '\u{8}',
            Some(b'f') => '\u{c}',
            Some(b'n') => '\n',
            Some(b'r') => '\r',
            Some(b't') => '\t',
            Some(b'u') => {
                self.pos += 1;
                return self.unicode_escape();
            }
            _ => return Err(self.error()),
        };
        self.pos += 1;
        Ok(escaped)
    }

    /// `\uXXXX`, joining a high surrogate with the low one escaped right after it.
    fn unicode_escape(&mut self) -> Result<char, Error> {
        let unit = self.hex4()?;
        let code = if (0xd800..0xdc00).contains(&unit) {
            self.expect("\\u")?;
            let low = self.hex4()?;
            if !(0xdc00..0xe000).contains(&low) {
                return Err(self.error());
            }
            0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00)
        } else {
            unit
        };
        char::from_u32(code).ok_or_else(|| self.error())
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        let digits = self
            .text
            .get(self.pos..self.pos + 4)
            .filter(|digits| digits.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .ok_or_else(|| self.error())?;
        self.pos += 4;
        u32::from_str_radix(digits, 16).map_err(|_| self.error())
    }

    /// `NUMBER_RE`: `(-?(?:0|[1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?`, an `int` unless a
    /// fraction or exponent matched.
    fn number(&mut self) -> Result<Value, Error> {
        let start = self.pos;
        self.eat("-");
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(Error::InvalidJson(start)),
        }
        let integer_end = self.pos;
        if self.peek() == Some(b'.') && self.bytes.get(self.pos + 1).is_some_and(u8::is_ascii_digit)
        {
            self.pos += 1;
            self.digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let exponent = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                self.digits();
            } else {
                self.pos = exponent;
            }
        }
        let text = &self.text[start..self.pos];
        if self.pos == integer_end {
            BigInt::from_str(text)
                .map(Value::Int)
                .map_err(|_| Error::InvalidJson(start))
        } else {
            f64::from_str(text)
                .map(Value::Float)
                .map_err(|_| Error::InvalidJson(start))
        }
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.pos += 1;
        }
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{dumps, loads};
    use crate::{Error, Value, repr::repr};

    // Expected values are the repr of CPython 3.14 json.loads(text)
    #[rstest]
    #[case::nan("NaN", "nan")]
    #[case::infinity("Infinity", "inf")]
    #[case::negative_infinity("-Infinity", "-inf")]
    #[case::non_finite_in_a_list("[NaN, Infinity, -Infinity]", "[nan, inf, -inf]")]
    #[case::beyond_u64("18446744073709551616", "18446744073709551616")]
    #[case::exponent_overflow("1e400", "inf")]
    #[case::upper_exponent("1E+2", "100.0")]
    #[case::negative_zero_int("-0", "0")]
    #[case::negative_zero_float("-0.0", "-0.0")]
    #[case::repeated_key(r#"{"a": 1, "b": 2, "a": 3}"#, "{'a': 3, 'b': 2}")]
    #[case::surrogate_pair(r#""😀""#, "'😀'")]
    #[case::upper_hex_surrogate_pair(r#""😀""#, "'😀'")]
    #[case::escapes(
        r#"{"k": [true, false, null, 1.5, "é\/"]}"#,
        "{'k': [True, False, None, 1.5, 'é/']}"
    )]
    #[case::delete_is_not_a_control_character("\"\x7f\"", "'\\x7f'")]
    #[case::surrounding_whitespace(" \n[1]\t", "[1]")]
    #[case::empty_containers("[{}, []]", "[{}, []]")]
    fn loads_reads_like_python(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(repr(&loads(text).unwrap()), expected);
    }

    // CPython 3.14 json.loads raises JSONDecodeError on each
    #[rstest]
    #[case::lowercase_nan("nan")]
    #[case::signed_nan("-NaN")]
    #[case::plus_sign("+1")]
    #[case::bare_minus("-")]
    #[case::leading_zero("01")]
    #[case::bare_point("1.")]
    #[case::bare_exponent("1e")]
    #[case::trailing_data("Infinity1")]
    #[case::trailing_comma("[1,]")]
    #[case::missing_comma("[1 2]")]
    #[case::missing_colon(r#"{"a" 1}"#)]
    #[case::unquoted_key("{1: 2}")]
    #[case::raw_newline_in_string("\"a\nb\"")]
    #[case::unknown_escape(r#""\x41""#)]
    #[case::empty("")]
    #[case::byte_order_mark("\u{feff}1")]
    fn loads_rejects_what_python_rejects(#[case] text: &str) {
        assert!(matches!(loads(text), Err(Error::InvalidJson(_))));
    }

    #[rstest]
    fn a_lone_surrogate_is_a_known_limit() {
        assert!(matches!(loads(r#""\ud800""#), Err(Error::InvalidJson(_))));
    }

    #[rstest]
    fn loads_reads_back_the_non_finite_floats_dumps_writes() {
        let written = dumps(&Value::List(vec![
            Value::Float(f64::NAN),
            Value::Float(f64::INFINITY),
            Value::Float(f64::NEG_INFINITY),
        ]))
        .unwrap();

        assert_eq!(repr(&loads(&written).unwrap()), "[nan, inf, -inf]");
    }

    #[rstest]
    fn loads_rejects_nesting_past_max_depth() {
        let text = format!(
            "{}{}",
            "[".repeat(crate::MAX_DEPTH + 1),
            "]".repeat(crate::MAX_DEPTH + 1)
        );

        assert!(matches!(loads(&text), Err(Error::TooDeep)));
    }
}
