use std::{str::FromStr, time::Duration};

use num_bigint::BigInt;

/// `int(text)` in base 10, or `None` where Python raises `ValueError`.
pub fn int(text: &str) -> Option<BigInt> {
    parse(text)
}

/// `float(text)`, or `None` where Python raises `ValueError`.
pub fn float(text: &str) -> Option<f64> {
    parse(text)
}

/// Why [`seconds`] refused a value, as `int(value)` would. Messages are CPython's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SecondsError {
    /// `OverflowError`. Finite values past `Duration::MAX` land here too, where Python builds a big int.
    #[error("cannot convert float infinity to integer")]
    Overflow,
    /// `ValueError`.
    #[error("cannot convert float NaN to integer")]
    NotANumber,
}

/// A float of seconds as a `Duration`, refusing what `int(value)` refuses. Finite negative values clamp to zero.
pub fn seconds(value: f64) -> Result<Duration, SecondsError> {
    match value {
        value if value.is_nan() => Err(SecondsError::NotANumber),
        value if value.is_infinite() => Err(SecondsError::Overflow),
        value => Duration::try_from_secs_f64(value.max(0.0)).map_err(|_| SecondsError::Overflow),
    }
}

fn parse<N: FromStr>(text: &str) -> Option<N> {
    let text = text.trim();
    let digit_at = |index: Option<usize>| {
        index
            .and_then(|index| text.as_bytes().get(index))
            .is_some_and(u8::is_ascii_digit)
    };
    let underscores_between_digits = text
        .match_indices('_')
        .all(|(index, _)| digit_at(index.checked_sub(1)) && digit_at(Some(index + 1)));
    if !underscores_between_digits {
        return None;
    }
    text.replace('_', "").parse().ok()
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use rstest::rstest;

    use std::time::Duration;

    use super::{SecondsError, float, int, seconds};

    // Expected values are CPython 3.14 int(text)
    #[rstest]
    #[case::trailing_newline("2\n", Some(2))]
    #[case::surrounding_spaces_and_underscore(" 6_380 ", Some(6380))]
    #[case::plus_sign("+5", Some(5))]
    #[case::negative_zero("-0", Some(0))]
    #[case::leading_zeros("007", Some(7))]
    #[case::unicode_spaces("\u{a0}7\u{2003}", Some(7))]
    #[case::beyond_i64("99999999999999999999", Some(99_999_999_999_999_999_999_i128))]
    #[case::double_underscore("1__0", None)]
    #[case::leading_underscore("_1", None)]
    #[case::trailing_underscore("1_", None)]
    #[case::decimal_point("5.0", None)]
    #[case::information_separator("\x1c7", None)]
    #[case::empty("", None)]
    #[case::sign_only("+", None)]
    #[case::non_ascii_digit_is_a_known_limit("\u{667}", None)]
    fn int_parses_like_python(#[case] text: &str, #[case] expected: Option<i128>) {
        assert_eq!(int(text), expected.map(BigInt::from));
    }

    // Expected values are CPython 3.14 float(text)
    #[rstest]
    #[case::underscore_and_newline(" 1_0.5\n", Some(10.5))]
    #[case::exponent("1e3", Some(1000.0))]
    #[case::leading_point(".5", Some(0.5))]
    #[case::trailing_point("5.", Some(5.0))]
    #[case::plus_sign("+1.5", Some(1.5))]
    #[case::infinity_word("Infinity", Some(f64::INFINITY))]
    #[case::infinity_with_space("infinity ", Some(f64::INFINITY))]
    #[case::underscore_before_exponent("1_e3", None)]
    #[case::underscore_after_exponent("1e_3", None)]
    #[case::underscore_after_point("1._5", None)]
    #[case::underscore_after_word("inf_", None)]
    #[case::hexadecimal("0x10", None)]
    #[case::bare_exponent("1.5e", None)]
    fn float_parses_like_python(#[case] text: &str, #[case] expected: Option<f64>) {
        assert_eq!(float(text), expected);
    }

    #[rstest]
    fn float_reads_a_signed_nan_like_python() {
        assert!(float("-NaN").is_some_and(f64::is_nan));
    }

    // CPython 3.14: int(inf) and int(-inf) raise OverflowError, int(nan) raises ValueError
    #[rstest]
    #[case::fraction(1.5, Ok(Duration::from_millis(1500)))]
    #[case::zero(0.0, Ok(Duration::ZERO))]
    #[case::negative_clamps_to_zero(-3.0, Ok(Duration::ZERO))]
    #[case::infinity(f64::INFINITY, Err(SecondsError::Overflow))]
    #[case::negative_infinity(f64::NEG_INFINITY, Err(SecondsError::Overflow))]
    #[case::past_duration_max(1e20, Err(SecondsError::Overflow))]
    #[case::nan(f64::NAN, Err(SecondsError::NotANumber))]
    #[case::negative_nan(-f64::NAN, Err(SecondsError::NotANumber))]
    fn seconds_refuses_what_int_refuses(
        #[case] value: f64,
        #[case] expected: Result<Duration, SecondsError>,
    ) {
        assert_eq!(seconds(value), expected);
    }
}
