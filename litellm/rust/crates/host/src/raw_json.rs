use std::{borrow::Cow, io};

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

pub fn mark_wide_numbers(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out = String::new();
    let mut copied = 0;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let end = string_end(bytes, index + 1);
                if let Some(escaped) = escaped_marked_string(&text[index..end]) {
                    out.push_str(&text[copied..index]);
                    out.push_str(&escaped);
                    copied = end;
                }
                index = end;
            }
            b'-' | b'0'..=b'9' => {
                let end = bytes[index..]
                    .iter()
                    .position(|byte| {
                        !matches!(byte, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                    })
                    .map_or(bytes.len(), |length| index + length);
                if let Some(raw) = wide_number(&text[index..end]) {
                    out.push_str(&text[copied..index]);
                    out.push_str(&serde_json::Value::String(encode(raw)).to_string());
                    copied = end;
                }
                index = end;
            }
            _ => index += 1,
        }
    }
    if copied == 0 {
        return Cow::Borrowed(text);
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

fn escaped_marked_string(token: &str) -> Option<String> {
    let inner = token.strip_prefix('"')?;
    let escaped_mark = inner
        .get(..6)
        .is_some_and(|head| head.eq_ignore_ascii_case("\\uf8ff"));
    if !inner.starts_with(MARK) && !escaped_mark {
        return None;
    }
    let text = serde_json::from_str::<String>(token).ok()?;
    is_marked(&text)
        .then(|| serde_json::Value::String(encode(&ascii_string_literal(&text))).to_string())
}

fn ascii_string_literal(text: &str) -> String {
    let mut literal = String::with_capacity(text.len() + 2);
    literal.push('"');
    for character in text.chars() {
        match character {
            '"' => literal.push_str("\\\""),
            '\\' => literal.push_str("\\\\"),
            '\n' => literal.push_str("\\n"),
            '\r' => literal.push_str("\\r"),
            '\t' => literal.push_str("\\t"),
            ' '..='~' => literal.push(character),
            _ => {
                let mut units = [0; 2];
                for unit in character.encode_utf16(&mut units) {
                    literal.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    literal.push('"');
    literal
}

fn string_end(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'"' => return index + 1,
            _ => index += 1,
        }
    }
    index
}

fn wide_number(raw: &str) -> Option<&str> {
    if is_integer(raw) {
        return (raw.parse::<i64>().is_err() && raw.parse::<u64>().is_err()).then_some(raw);
    }
    match raw.parse::<f64>() {
        Ok(number) if number == f64::INFINITY => Some("Infinity"),
        Ok(number) if number == f64::NEG_INFINITY => Some("-Infinity"),
        _ => None,
    }
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
    #[case::wide_integer(r#"{"a":18446744073709551616}"#, json!({"a": encode("18446744073709551616")}))]
    #[case::negative_wide_integer(r#"[-12345678901234567890123]"#, json!([encode("-12345678901234567890123")]))]
    #[case::overflowing_float(r#"{"a":[1e400,-1E+400]}"#, json!({"a": [encode("Infinity"), encode("-Infinity")]}))]
    #[case::numbers_that_fit(r#"{"a":[18446744073709551615,-9223372036854775808,-0.0,5e-324,1E2,0.30000000000000004]}"#, json!({"a": [18_446_744_073_709_551_615_u64, i64::MIN, -0.0, 5e-324, 100.0, 0.300_000_000_000_000_04]}))]
    #[case::provider_marked_text("{\"a\":\"\u{F8FF}litellm-raw-json:42\"}", json!({"a": encode(r#""\uf8fflitellm-raw-json:42""#)}))]
    #[case::provider_escaped_mark(r#"{"\uF8FFlitellm-raw-json:1":["\uf8fflitellm-raw-json:\"x\"\n\u00e9"]}"#, json!({encode(r#""\uf8fflitellm-raw-json:1""#): [encode(r#""\uf8fflitellm-raw-json:\"x\"\n\u00e9""#)]}))]
    #[case::mark_not_leading(r#"["x\uf8fflitellm-raw-json:NaN"]"#, json!(["x\u{F8FF}litellm-raw-json:NaN"]))]
    #[case::numbers_inside_strings(r#"{"18446744073709551616":"1e400 \" 18446744073709551616"}"#, json!({"18446744073709551616": "1e400 \" 18446744073709551616"}))]
    fn wide_numbers_are_read_as_marked_texts(#[case] text: &str, #[case] read: Value) {
        assert_eq!(
            serde_json::from_str::<Value>(&mark_wide_numbers(text)).unwrap(),
            read
        );
    }

    #[test]
    fn a_text_without_wide_numbers_is_borrowed() {
        assert!(matches!(
            mark_wide_numbers(r#"{"a":[1,2.5,"18446744073709551616"]}"#),
            Cow::Borrowed(_)
        ));
    }

    #[rstest]
    #[case::plain(json!({"a": "text", "b": ["", "\"quoted\"\n", "\u{F8FF}"], "c": 1.5, "d": null}))]
    #[case::mark_inside(json!(["x\u{F8FF}litellm-raw-json:NaN"]))]
    #[case::control(json!(["\u{1}\t"]))]
    fn unmarked_documents_serialize_as_serde_json_does(#[case] body: Value) {
        assert_eq!(to_vec(&body).unwrap(), serde_json::to_vec(&body).unwrap());
    }
}
