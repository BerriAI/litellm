use std::io;

use serde::Serialize;
use serde_json::ser::{CharEscape, CompactFormatter, Formatter};

const MARK: char = '\u{F8FF}';
const PREFIX: &str = "\u{F8FF}litellm-raw-json:";

pub fn encode(raw: &str) -> String {
    format!("{PREFIX}{raw}")
}

pub fn decode(text: &str) -> Option<&str> {
    text.strip_prefix(PREFIX).filter(|raw| is_scalar(raw))
}

pub fn is_marked(text: &str) -> bool {
    text.starts_with(PREFIX)
}

pub fn to_vec<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(128);
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, RawLeaves::default());
    value.serialize(&mut serializer)?;
    Ok(out)
}

fn is_scalar(raw: &str) -> bool {
    matches!(raw, "NaN" | "Infinity" | "-Infinity") || is_integer(raw) || is_string_literal(raw)
}

fn is_integer(raw: &str) -> bool {
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    match digits.as_bytes() {
        [b'0'] => true,
        [b'1'..=b'9', rest @ ..] => rest.iter().all(u8::is_ascii_digit),
        _ => false,
    }
}

fn is_string_literal(raw: &str) -> bool {
    let Some(inner) = raw
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    else {
        return false;
    };
    let mut bytes = inner.bytes();
    while let Some(byte) = bytes.next() {
        match byte {
            b'\\' => match bytes.next() {
                Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => {}
                Some(b'u') => {
                    if !(0..4).all(|_| bytes.next().is_some_and(|hex| hex.is_ascii_hexdigit())) {
                        return false;
                    }
                }
                _ => return false,
            },
            b'"' => return false,
            0x20..=0x7e => {}
            _ => return false,
        }
    }
    true
}

fn unescaped(escape: CharEscape) -> char {
    match escape {
        CharEscape::Quote => '"',
        CharEscape::ReverseSolidus => '\\',
        CharEscape::Solidus => '/',
        CharEscape::Backspace => '\u{8}',
        CharEscape::FormFeed => '\u{c}',
        CharEscape::LineFeed => '\n',
        CharEscape::CarriageReturn => '\r',
        CharEscape::Tab => '\t',
        CharEscape::AsciiControl(byte) => char::from(byte),
    }
}

#[derive(Default)]
enum Text {
    #[default]
    Closed,
    Opened,
    Plain,
    Marked(String),
}

#[derive(Default)]
struct RawLeaves {
    key: bool,
    text: Text,
}

impl Formatter for RawLeaves {
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.key = true;
        CompactFormatter.begin_object_key(writer, first)
    }

    fn end_object_key<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.key = false;
        CompactFormatter.end_object_key(writer)
    }

    fn begin_string<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.text = Text::Opened;
        Ok(())
    }

    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        match &mut self.text {
            Text::Opened if fragment.starts_with(MARK) => {
                self.text = Text::Marked(fragment.to_owned());
                Ok(())
            }
            Text::Opened => {
                self.text = Text::Plain;
                writer.write_all(b"\"")?;
                writer.write_all(fragment.as_bytes())
            }
            Text::Marked(held) => {
                held.push_str(fragment);
                Ok(())
            }
            Text::Plain | Text::Closed => writer.write_all(fragment.as_bytes()),
        }
    }

    fn write_char_escape<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        escape: CharEscape,
    ) -> io::Result<()> {
        match &mut self.text {
            Text::Opened => {
                self.text = Text::Plain;
                writer.write_all(b"\"")?;
                CompactFormatter.write_char_escape(writer, escape)
            }
            Text::Marked(held) => {
                held.push(unescaped(escape));
                Ok(())
            }
            Text::Plain | Text::Closed => CompactFormatter.write_char_escape(writer, escape),
        }
    }

    fn end_string<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        match std::mem::take(&mut self.text) {
            Text::Opened => writer.write_all(b"\"\""),
            Text::Plain | Text::Closed => writer.write_all(b"\""),
            Text::Marked(held) => match decode(&held) {
                Some(raw) if !self.key || raw.starts_with('"') => writer.write_all(raw.as_bytes()),
                _ => serde_json::to_writer(writer, &held).map_err(io::Error::other),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::nan("NaN")]
    #[case::infinity("Infinity")]
    #[case::negative_infinity("-Infinity")]
    #[case::wide_integer("18446744073709551616")]
    #[case::negative_wide_integer("-1180591620717411303424")]
    #[case::zero("0")]
    #[case::surrogate_string(r#""sys \ud800 lone""#)]
    #[case::escaped_string(r#""a\"b\\c\n\u00e9""#)]
    fn scalar_texts_are_written_back_verbatim(#[case] raw: &str) {
        let body = json!({"leaf": encode(raw), "list": [encode(raw)]});
        assert_eq!(
            String::from_utf8(to_vec(&body).unwrap()).unwrap(),
            format!(r#"{{"leaf":{raw},"list":[{raw}]}}"#)
        );
    }

    #[rstest]
    #[case::structure("1, \"injected\": true")]
    #[case::object("{}")]
    #[case::leading_zero("012")]
    #[case::float("1.5")]
    #[case::unquoted_quote(r#""a"b""#)]
    #[case::non_ascii("\"\u{e9}\"")]
    #[case::bad_escape(r#""\x""#)]
    #[case::short_unicode_escape(r#""\u12""#)]
    #[case::empty("")]
    fn other_marked_texts_stay_strings(#[case] raw: &str) {
        let marked = encode(raw);
        assert_eq!(decode(&marked), None);
        assert_eq!(
            to_vec(&json!([marked])).unwrap(),
            serde_json::to_vec(&json!([marked])).unwrap()
        );
    }

    #[test]
    fn keys_take_only_string_texts() {
        let body = Value::Object(
            [
                (encode(r#""k\udc00""#), json!(1)),
                (encode("NaN"), json!(2)),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(
            String::from_utf8(to_vec(&body).unwrap()).unwrap(),
            format!(
                r#"{{"k\udc00":1,{}:2}}"#,
                serde_json::to_string(&encode("NaN")).unwrap()
            )
        );
    }

    #[rstest]
    #[case::plain(json!({"a": "text", "b": ["", "\"quoted\"\n", "\u{F8FF}"], "c": 1.5, "d": null}))]
    #[case::mark_inside(json!(["x\u{F8FF}litellm-raw-json:NaN"]))]
    #[case::control(json!(["\u{1}\t"]))]
    fn unmarked_documents_serialize_as_serde_json_does(#[case] body: Value) {
        assert_eq!(to_vec(&body).unwrap(), serde_json::to_vec(&body).unwrap());
    }
}
