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
    let scientific = format!("{value:e}");
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::whole(1.0, "1.0")]
    #[case::negative_fraction(-0.25, "-0.25")]
    #[case::small(0.0001, "0.0001")]
    #[case::tiny(0.00001, "1e-05")]
    #[case::large(1e16, "1e+16")]
    #[case::below_large(1234567890123456.0, "1234567890123456.0")]
    #[case::shortest(0.1, "0.1")]
    #[case::nan(f64::NAN, "NaN")]
    #[case::negative_infinity(f64::NEG_INFINITY, "-Infinity")]
    fn a_float_renders_as_python_repr_does(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(float_repr(value), expected);
    }

    #[rstest]
    #[case::quote_and_controls("a\"b\\\n\t", r#""a\"b\\\n\t""#)]
    #[case::non_ascii("é", r#""\u00e9""#)]
    #[case::astral("😀", r#""\ud83d\ude00""#)]
    #[case::control("\u{1}", "\"\\u0001\"")]
    fn a_string_is_escaped_as_json_dumps_does_with_ensure_ascii(
        #[case] text: &str,
        #[case] expected: &str,
    ) {
        let mut out = Vec::new();
        write_str(&mut out, text);
        assert_eq!(String::from_utf8(out).unwrap(), expected);
    }
}
