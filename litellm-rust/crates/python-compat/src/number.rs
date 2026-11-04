use num_bigint::BigInt;

/// `float(text)` for a `str`, `None` where CPython raises `ValueError`. Non-ASCII digits, which CPython reads as their
/// decimal value, are refused
pub fn float(text: &str) -> Option<f64> {
    let digits = without_underscores(text.trim_matches(char::is_whitespace))?;
    digits.parse().ok()
}

/// `int(text)` for a `str` in base 10, `None` where CPython raises `ValueError`. Non-ASCII digits, which CPython reads
/// as their decimal value, are refused
pub fn int(text: &str) -> Option<BigInt> {
    let digits = without_underscores(text.trim_matches(char::is_whitespace))?;
    let unsigned = digits.strip_prefix(['+', '-']).unwrap_or(&digits);
    if unsigned.is_empty() || !unsigned.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// CPython accepts a single `_` only between two digits
fn without_underscores(text: &str) -> Option<String> {
    if !text.is_ascii() {
        return None;
    }
    let bytes = text.as_bytes();
    let misplaced = bytes.iter().enumerate().any(|(at, byte)| {
        *byte == b'_'
            && !(at > 0
                && bytes[at - 1].is_ascii_digit()
                && bytes.get(at + 1).is_some_and(u8::is_ascii_digit))
    });
    (!misplaced).then(|| text.replace('_', ""))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Each expected value is what CPython 3.14.7 returns for `float(text)` and `int(text)`
    #[rstest]
    #[case::scientific("3e-07", Some(3e-07), None)]
    #[case::upper_exponent("3E-07", Some(3e-07), None)]
    #[case::padded(" 8192 ", Some(8192.0), Some(8192))]
    #[case::unicode_whitespace("\u{a0}12\u{3000}", Some(12.0), Some(12))]
    #[case::next_line("\u{85}1\u{85}", Some(1.0), Some(1))]
    #[case::file_separator_is_not_whitespace("\u{1c}12\u{1f}", None, None)]
    #[case::zero_width_space_is_not_whitespace("\u{200b}12", None, None)]
    #[case::byte_order_mark_is_not_whitespace("\u{feff}1", None, None)]
    #[case::underscores("1_000", Some(1000.0), Some(1000))]
    #[case::underscore_in_exponent("1.5e3_0", Some(1.5e30), None)]
    #[case::underscore_before_point("1_0.5", Some(10.5), None)]
    #[case::double_underscore("1__0", None, None)]
    #[case::leading_underscore("_1", None, None)]
    #[case::trailing_underscore("1_", None, None)]
    #[case::plus("+5", Some(5.0), Some(5))]
    #[case::negative_zero("-0", Some(-0.0), Some(0))]
    #[case::leading_zeros("0010", Some(10.0), Some(10))]
    #[case::overflow("1e400", Some(f64::INFINITY), None)]
    #[case::negative_overflow("-1e400", Some(f64::NEG_INFINITY), None)]
    #[case::infinity("Infinity", Some(f64::INFINITY), None)]
    #[case::signed_inf("+inf", Some(f64::INFINITY), None)]
    #[case::trailing_point("1.", Some(1.0), None)]
    #[case::point_before_exponent("5.e3", Some(5000.0), None)]
    #[case::leading_point(".5", Some(0.5), None)]
    #[case::bare_point(".", None, None)]
    #[case::bare_exponent("e5", None, None)]
    #[case::open_exponent("1e", None, None)]
    #[case::empty("", None, None)]
    #[case::blank("  ", None, None)]
    #[case::hex("0x10", None, None)]
    #[case::complex("1.5j", None, None)]
    #[case::suffix("12abc", None, None)]
    #[case::non_ascii_digits_are_a_known_limit("\u{661}\u{662}", None, None)]
    fn matches_cpython(
        #[case] text: &str,
        #[case] float_value: Option<f64>,
        #[case] int_value: Option<i64>,
    ) {
        assert_eq!(float(text).map(f64::to_bits), float_value.map(f64::to_bits));
        assert_eq!(int(text), int_value.map(BigInt::from));
    }

    #[rstest]
    #[case::nan("nan")]
    #[case::signed_nan("-NaN")]
    fn nan_parses(#[case] text: &str) {
        assert!(float(text).is_some_and(f64::is_nan));
        assert_eq!(int(text), None);
    }

    #[rstest]
    fn int_has_no_size_limit() {
        let digits = "9".repeat(40);
        assert_eq!(int(&digits).map(|value| value.to_string()), Some(digits));
    }
}
