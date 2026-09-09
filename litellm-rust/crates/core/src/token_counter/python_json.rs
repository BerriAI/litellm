//! `json.dumps(value)` with Python's default arguments: `", "` and `": "`
//! separators, `ensure_ascii=True`, and keys in insertion order.

use std::fmt::Write;

use super::TokenCountError;
use super::types::TextValue;

pub(super) fn dumps(value: &TextValue) -> Result<String, TokenCountError> {
    let mut out = String::new();
    write_value(&mut out, value)?;
    Ok(out)
}

fn write_value(out: &mut String, value: &TextValue) -> Result<(), TokenCountError> {
    match value {
        TextValue::Null => out.push_str("null"),
        TextValue::Bool(true) => out.push_str("true"),
        TextValue::Bool(false) => out.push_str("false"),
        TextValue::Integer(number) => write_number(out, number),
        TextValue::Float(_) => {
            return Err(TokenCountError::Unsupported(
                "float repr is formatted by the python path".to_string(),
            ));
        }
        TextValue::Text(text) => write_string(out, text),
        TextValue::List(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_value(out, item)?;
            }
            out.push(']');
        }
        TextValue::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn write_number(out: &mut String, number: &i64) {
    // Writing an integer into a String cannot fail.
    let _ = write!(out, "{number}");
}

fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(character),
            _ => {
                let mut units = [0u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}
