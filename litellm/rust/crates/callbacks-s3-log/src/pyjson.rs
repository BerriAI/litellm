use serde_json::Value;

const HEX: &[u8; 16] = b"0123456789abcdef";

pub fn write_str(out: &mut Vec<u8>, text: &str) {
    out.push(b'"');
    let mut units = [0u16; 2];
    for character in text.chars() {
        match character {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            ' '..='~' => out.push(character as u8),
            _ => {
                for unit in character.encode_utf16(&mut units) {
                    write_unit(out, *unit);
                }
            }
        }
    }
    out.push(b'"');
}

fn write_unit(out: &mut Vec<u8>, unit: u16) {
    out.extend_from_slice(b"\\u");
    for shift in [12, 8, 4, 0] {
        out.push(HEX[usize::from((unit >> shift) & 0xf)]);
    }
}

pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    let shortest = format!("{value:e}");
    let length = shortest
        .split('e')
        .next()
        .unwrap_or_default()
        .chars()
        .filter(char::is_ascii_digit)
        .count();
    let scientific = format!("{value:.precision$e}", precision = length.saturating_sub(1));
    let (negative, scientific) = match scientific.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, scientific.as_str()),
    };
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let mut text = String::with_capacity(digits.len() + 8);
    if negative {
        text.push('-');
    }
    let point = exponent + 1;
    if (-3..=16).contains(&point) {
        let count = digits.len() as i32;
        if point <= 0 {
            text.push_str("0.");
            text.extend(std::iter::repeat_n('0', (-point) as usize));
            text.push_str(&digits);
        } else if point >= count {
            text.push_str(&digits);
            text.extend(std::iter::repeat_n('0', (point - count) as usize));
            text.push_str(".0");
        } else {
            let (whole, fraction) = digits.split_at(point as usize);
            text.push_str(whole);
            text.push('.');
            text.push_str(fraction);
        }
        return text;
    }
    let (first, rest) = digits.split_at(1);
    text.push_str(first);
    if !rest.is_empty() {
        text.push('.');
        text.push_str(rest);
    }
    text.push('e');
    text.push(if exponent < 0 { '-' } else { '+' });
    text.push_str(&format!("{:02}", exponent.unsigned_abs()));
    text
}

pub fn write_value(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(number) => match (number.as_i64(), number.as_u64(), number.as_f64()) {
            (Some(integer), _, _) => out.extend_from_slice(integer.to_string().as_bytes()),
            (None, Some(integer), _) => out.extend_from_slice(integer.to_string().as_bytes()),
            (None, None, Some(float)) => out.extend_from_slice(float_repr(float).as_bytes()),
            (None, None, None) => out.extend_from_slice(number.to_string().as_bytes()),
        },
        Value::String(text) => write_str(out, text),
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.extend_from_slice(b", ");
                }
                write_value(out, item);
            }
            out.push(b']');
        }
        Value::Object(entries) => {
            out.push(b'{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.extend_from_slice(b", ");
                }
                write_str(out, key);
                out.extend_from_slice(b": ");
                write_value(out, item);
            }
            out.push(b'}');
        }
    }
}

pub fn dumps(value: &Value) -> String {
    let mut out = Vec::new();
    write_value(&mut out, value);
    String::from_utf8(out).unwrap_or_default()
}

pub fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(text) => text.clone(),
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => float_repr(float),
            _ => number.to_string(),
        },
        Value::Array(_) | Value::Object(_) => dumps(value),
    }
}

pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|float| float != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case(0.0, "0.0")]
    #[case(-0.0, "-0.0")]
    #[case(1.0, "1.0")]
    #[case(0.1, "0.1")]
    #[case(1.5e-5, "1.5e-05")]
    #[case(1e-4, "0.0001")]
    #[case(1e-5, "1e-05")]
    #[case(1e15, "1000000000000000.0")]
    #[case(1e16, "1e+16")]
    #[case(1.2345678901234568e17, "1.2345678901234568e+17")]
    #[case(123.456, "123.456")]
    #[case(1e100, "1e+100")]
    #[case(-2.5e-300, "-2.5e-300")]
    #[case(f64::NAN, "NaN")]
    #[case(f64::INFINITY, "Infinity")]
    #[case(f64::NEG_INFINITY, "-Infinity")]
    #[case(0.30000000000000004, "0.30000000000000004")]
    #[case(0.07872390747070312, "0.07872390747070312")]
    #[case(1794700894.4414062, "1794700894.4414062")]
    fn floats_print_as_python_repr_prints_them(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(float_repr(value), expected);
    }

    #[rstest]
    #[case("plain", "\"plain\"")]
    #[case("quote\" back\\", "\"quote\\\" back\\\\\"")]
    #[case("\n\r\t\u{8}\u{c}\u{1}\u{7f}", "\"\\n\\r\\t\\b\\f\\u0001\\u007f\"")]
    #[case("café", "\"caf\\u00e9\"")]
    #[case("😀", "\"\\ud83d\\ude00\"")]
    fn strings_are_escaped_as_ensure_ascii_escapes_them(
        #[case] text: &str,
        #[case] expected: &str,
    ) {
        let mut out = Vec::new();
        write_str(&mut out, text);
        assert_eq!(String::from_utf8(out).unwrap(), expected);
    }

    #[rstest]
    fn containers_use_the_default_separators_and_keep_key_order() {
        assert_eq!(
            dumps(&json!({"b": [1, 2.5, null, true], "a": {"x": "y"}})),
            "{\"b\": [1, 2.5, null, true], \"a\": {\"x\": \"y\"}}"
        );
    }
}
