//! `json.dumps(value)` with Python's default arguments: `", "` and `": "`
//! separators, `ensure_ascii=True`, and keys in insertion order.

use std::io::{self, Write};

use serde::Serialize;
use serde_json::ser::{Formatter, Serializer};

use super::Error;
use super::types::TextValue;

pub(super) fn dumps(value: &TextValue) -> Result<String, Error> {
    let mut output = Vec::with_capacity(serialized_len(value)?);
    value
        .serialize(&mut Serializer::with_formatter(
            &mut output,
            PythonFormatter,
        ))
        .map_err(Error::JsonSerialization)?;
    debug_assert_eq!(output.len(), output.capacity());
    String::from_utf8(output).map_err(Error::JsonUtf8)
}

fn serialized_len(value: &TextValue) -> Result<usize, Error> {
    match value {
        TextValue::Null => Ok(4),
        TextValue::Bool(true) => Ok(4),
        TextValue::Bool(false) => Ok(5),
        TextValue::Number(number) => match (number.as_i64(), number.as_u64()) {
            (Some(number), _) => Ok(unsigned_len(number.unsigned_abs()) + usize::from(number < 0)),
            (_, Some(number)) => Ok(unsigned_len(number)),
            _ => Err(Error::FloatText),
        },
        TextValue::Text(text) => Ok(quoted_len(text)),
        TextValue::List(items) => items
            .iter()
            .try_fold(2 + items.len().saturating_sub(1) * 2, |len, item| {
                Ok(len + serialized_len(item)?)
            }),
        TextValue::Object(entries) => entries.iter().try_fold(
            2 + entries.len().saturating_sub(1) * 2,
            |len, (key, value)| Ok(len + quoted_len(key) + 2 + serialized_len(value)?),
        ),
    }
}

fn unsigned_len(number: u64) -> usize {
    if number == 0 {
        1
    } else {
        number.ilog10() as usize + 1
    }
}

fn quoted_len(value: &str) -> usize {
    value.chars().fold(2, |len, character| {
        len + match character {
            '"' | '\\' | '\u{0008}' | '\u{000c}' | '\n' | '\r' | '\t' => 2,
            '\u{0000}'..='\u{001f}' => 6,
            ' '..='~' => 1,
            _ => character.len_utf16() * 6,
        }
    })
}

struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        for character in fragment.chars() {
            if (' '..='~').contains(&character) {
                write!(writer, "{character}")?;
                continue;
            }
            let mut units = [0u16; 2];
            for unit in character.encode_utf16(&mut units) {
                write!(writer, "\\u{unit:04x}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::dumps;

    #[rstest]
    #[case::null("null", "null")]
    #[case::boolean("true", "true")]
    #[case::signed_integer("-3", "-3")]
    #[case::unsigned_integer("18446744073709551615", "18446744073709551615")]
    #[case::empty_array("[]", "[]")]
    #[case::empty_object("{}", "{}")]
    #[case::nested(
        r#"{"first":1,"second":{"ok":true,"none":null},"third":[false,2]}"#,
        r#"{"first": 1, "second": {"ok": true, "none": null}, "third": [false, 2]}"#
    )]
    #[case::string_escaping(
        r#""caf\u00e9 \u2014 \ud83d\ude00 \"q\" \\ \n\t\u0001\u007f ~ /""#,
        r#""caf\u00e9 \u2014 \ud83d\ude00 \"q\" \\ \n\t\u0001\u007f ~ /""#
    )]
    #[case::short_control_escapes(r#""\b\f\r""#, r#""\b\f\r""#)]
    fn matches_python_json_dumps(#[case] input: &str, #[case] expected: &str) {
        let value = serde_json::from_str(input).expect("fixture parses");

        assert_eq!(dumps(&value).expect("fixture dumps"), expected);
    }

    #[rstest]
    #[case::top_level("1.5")]
    #[case::array("[1,2.5]")]
    #[case::object(r#"{"nested":{"value":-0.25}}"#)]
    fn rejects_floats(#[case] input: &str) {
        let value = serde_json::from_str(input).expect("fixture parses");
        let error = dumps(&value).expect_err("floats are declined");

        assert_eq!(
            error.to_string(),
            "unsupported by the rust token counter: float text values are counted by the python path"
        );
    }
}
