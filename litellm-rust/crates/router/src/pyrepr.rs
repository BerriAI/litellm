//! Python literal text for the values Python's sync `RedisCache.set_cache` writes with
//! `str(value)`, and the reverse for reading them back (`ast.literal_eval`).

use std::fmt::Write;

use serde::Deserialize;

/// A number as Python typed it, since `repr` of `5` and `5.0` differ.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PyNumber {
    Int(i64),
    Float(f64),
}

impl Default for PyNumber {
    fn default() -> Self {
        Self::Int(0)
    }
}

impl PyNumber {
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    pub fn repr(self) -> String {
        match self {
            Self::Int(value) => value.to_string(),
            Self::Float(value) => float_repr(value),
        }
    }
}

/// `repr(float)`: the shortest round-trip digits, in exponent form below 1e-4 or from 1e16.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into();
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if (-4..16).contains(&exponent) {
        let fixed = format!("{value}");
        return if fixed.contains('.') {
            fixed
        } else {
            format!("{fixed}.0")
        };
    }
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.abs())
}

/// `repr(str)`: single quotes unless the text has a single quote and no double quote.
pub fn str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ if character == quote => {
                out.push('\\');
                out.push(character);
            }
            _ if is_printable(character) => out.push(character),
            _ => {
                let code = u32::from(character);
                let _ = match code {
                    0..=0xff => write!(out, "\\x{code:02x}"),
                    0x100..=0xffff => write!(out, "\\u{code:04x}"),
                    _ => write!(out, "\\U{code:08x}"),
                };
            }
        }
    }
    out.push(quote);
    out
}

/// `str.isprintable()` for the characters a masked exception message realistically holds:
/// controls, separators other than the ASCII space, and unassigned or private-use code points
/// are escaped.
fn is_printable(character: char) -> bool {
    !(character.is_control()
        || (character.is_whitespace() && character != ' ')
        || matches!(
            u32::from(character),
            0xad | 0x2028 | 0x2029 | 0xfeff | 0xe000..=0xf8ff
        ))
}

/// A parsed Python literal, limited to what router state holds.
#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Str(String),
    Number(PyNumber),
    Dict(Vec<(String, Literal)>),
}

/// Reads a value the way `RedisCache._get_cache_logic` does: JSON first, then a Python literal.
pub fn parse(text: &str) -> Option<Literal> {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| from_json(&value))
        .or_else(|| {
            let mut parser = Parser { rest: text.trim() };
            let literal = parser.literal()?;
            parser.rest.trim().is_empty().then_some(literal)
        })
}

fn from_json(value: &serde_json::Value) -> Option<Literal> {
    match value {
        serde_json::Value::String(text) => Some(Literal::Str(text.clone())),
        serde_json::Value::Number(number) => Some(Literal::Number(
            number
                .as_i64()
                .map(PyNumber::Int)
                .or_else(|| number.as_f64().map(PyNumber::Float))?,
        )),
        serde_json::Value::Object(entries) => entries
            .iter()
            .map(|(key, value)| from_json(value).map(|literal| (key.clone(), literal)))
            .collect::<Option<Vec<_>>>()
            .map(Literal::Dict),
        _ => None,
    }
}

struct Parser<'a> {
    rest: &'a str,
}

impl Parser<'_> {
    fn literal(&mut self) -> Option<Literal> {
        self.skip_space();
        match self.rest.chars().next()? {
            '{' => self.dict(),
            '\'' | '"' => self.string().map(Literal::Str),
            _ => self.number().map(Literal::Number),
        }
    }

    fn dict(&mut self) -> Option<Literal> {
        self.expect('{')?;
        let mut entries = Vec::new();
        loop {
            self.skip_space();
            if self.eat('}') {
                return Some(Literal::Dict(entries));
            }
            let key = self.string()?;
            self.skip_space();
            self.expect(':')?;
            let value = self.literal()?;
            entries.push((key, value));
            self.skip_space();
            if !self.eat(',') {
                self.skip_space();
                self.expect('}')?;
                return Some(Literal::Dict(entries));
            }
        }
    }

    fn string(&mut self) -> Option<String> {
        self.skip_space();
        let quote = self.rest.chars().next()?;
        let mut chars = self.rest[quote.len_utf8()..].char_indices();
        let mut out = String::new();
        while let Some((index, character)) = chars.next() {
            match character {
                '\\' => {
                    let (_, escaped) = chars.next()?;
                    match escaped {
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'x' | 'u' | 'U' => {
                            let width = match escaped {
                                'x' => 2,
                                'u' => 4,
                                _ => 8,
                            };
                            let digits: String = (0..width)
                                .map(|_| chars.next().map(|(_, digit)| digit))
                                .collect::<Option<String>>()?;
                            out.push(char::from_u32(u32::from_str_radix(&digits, 16).ok()?)?);
                        }
                        other => out.push(other),
                    }
                }
                _ if character == quote => {
                    self.rest = &self.rest[quote.len_utf8() + index + character.len_utf8()..];
                    return Some(out);
                }
                _ => out.push(character),
            }
        }
        None
    }

    fn number(&mut self) -> Option<PyNumber> {
        let end = self
            .rest
            .find(|character: char| {
                !(character.is_ascii_alphanumeric() || "+-.".contains(character))
            })
            .unwrap_or(self.rest.len());
        let (text, rest) = self.rest.split_at(end);
        self.rest = rest;
        text.parse::<i64>()
            .map(PyNumber::Int)
            .ok()
            .or_else(|| text.parse::<f64>().ok().map(PyNumber::Float))
    }

    fn skip_space(&mut self) {
        self.rest = self.rest.trim_start();
    }

    fn eat(&mut self, expected: char) -> bool {
        match self.rest.strip_prefix(expected) {
            Some(rest) => {
                self.rest = rest;
                true
            }
            None => false,
        }
    }

    fn expect(&mut self, expected: char) -> Option<()> {
        self.eat(expected).then_some(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Literal, PyNumber, float_repr, parse, str_repr};

    // Expected strings are CPython's repr() output; repr's float and str formats are part of
    // the language reference and stable since 3.1.
    #[rstest]
    #[case::integral(5.0, "5.0")]
    #[case::timestamp(1728345600.123456, "1728345600.123456")]
    #[case::small(0.0001, "0.0001")]
    #[case::smaller(0.00001, "1e-05")]
    #[case::large(1e16, "1e+16")]
    #[case::below_large(9999999999999998.0, "9999999999999998.0")]
    #[case::mantissa(1.5e-7, "1.5e-07")]
    #[case::negative(-2.5, "-2.5")]
    fn float_repr_matches_python(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(float_repr(value), expected);
    }

    #[rstest]
    #[case::plain("abc", "'abc'")]
    #[case::single_quote("it's", "\"it's\"")]
    #[case::both_quotes("it's \"x\"", "'it\\'s \"x\"'")]
    #[case::escapes("a\\b\nc\td\r", "'a\\\\b\\nc\\td\\r'")]
    #[case::control("\u{1}", "'\\x01'")]
    #[case::unicode("caf\u{e9} \u{1f600}", "'caf\u{e9} \u{1f600}'")]
    fn str_repr_matches_python(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(str_repr(text), expected);
    }

    #[rstest]
    #[case::python_literal("{'a': 'x\\'y', 'b': 5, 'c': 1.5e-05}")]
    #[case::json("{\"a\": \"x'y\", \"b\": 5, \"c\": 1.5e-05}")]
    fn parse_reads_both_encodings(#[case] text: &str) {
        assert_eq!(
            parse(text),
            Some(Literal::Dict(vec![
                ("a".into(), Literal::Str("x'y".into())),
                ("b".into(), Literal::Number(PyNumber::Int(5))),
                ("c".into(), Literal::Number(PyNumber::Float(1.5e-5))),
            ]))
        );
    }

    #[rstest]
    fn repr_round_trips_through_parse() {
        let text = format!("{{{}: {}}}", str_repr("k"), str_repr("it's\n\u{1}"));
        assert_eq!(
            parse(&text),
            Some(Literal::Dict(vec![(
                "k".into(),
                Literal::Str("it's\n\u{1}".into())
            )]))
        );
    }
}
