//! Python number semantics: `int()` and `float()` parsing of text, Neumaier-compensated float
//! sums, and the float and duration forms the proxy reads from settings.

use std::{str::FromStr, time::Duration};

use num_bigint::BigInt;
use serde::{Deserialize, Serialize, Serializer};

use crate::{
    format::fixed,
    repr::{float_repr, str_repr},
};

/// `round(value, ndigits)` for a finite `float`: the exact binary value rounded half to even at
/// `ndigits` decimals, read back as the nearest float.
pub fn round(value: f64, ndigits: usize) -> f64 {
    fixed(value, ndigits).parse().unwrap_or(value)
}

/// `int(text)` in base 10, or `None` where Python raises `ValueError`.
pub fn int(text: &str) -> Option<BigInt> {
    parse(text)
}

/// `int(text, 16)`, or `None` where Python raises `ValueError`.
pub fn hex_int(text: &str) -> Option<BigInt> {
    let text = text.trim();
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let digits = match unsigned
        .strip_prefix("0x")
        .or_else(|| unsigned.strip_prefix("0X"))
    {
        Some(rest) => rest.strip_prefix('_').unwrap_or(rest),
        None => unsigned,
    };
    let hex_at = |index: Option<usize>| {
        index
            .and_then(|index| digits.as_bytes().get(index))
            .is_some_and(u8::is_ascii_hexdigit)
    };
    let well_formed = digits.char_indices().all(|(index, c)| {
        c.is_ascii_hexdigit()
            || (c == '_' && hex_at(index.checked_sub(1)) && hex_at(Some(index + 1)))
    });
    if digits.is_empty() || !well_formed {
        return None;
    }
    let magnitude = BigInt::parse_bytes(digits.replace('_', "").as_bytes(), 16)?;
    Some(if negative { -magnitude } else { magnitude })
}

/// `float(text)`, or `None` where Python raises `ValueError`.
pub fn float(text: &str) -> Option<f64> {
    parse(text)
}

/// `sum(values)` over floats: CPython 3.12+ adds them with Neumaier compensation, so the
/// result can differ from a plain left fold in the last bit.
pub fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// Why [`int_of`] or [`float_of`] refused a value. Messages are CPython's, except
/// [`ConversionError::TooBig`], where Python would build a big int instead.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConversionError {
    /// `ValueError` from `int(text)`; holds the text's `repr`.
    #[error("invalid literal for int() with base 10: {0}")]
    IntLiteral(String),
    /// `ValueError` from `float(text)`; holds the text's `repr`.
    #[error("could not convert string to float: {0}")]
    FloatLiteral(String),
    /// `ValueError` from `int(nan)`.
    #[error("cannot convert float NaN to integer")]
    NotANumber,
    /// `OverflowError` from `int(inf)`.
    #[error("cannot convert float infinity to integer")]
    Infinite,
    /// An integer outside `i64`; holds its `str`.
    #[error("{0} does not fit in a 64-bit int")]
    TooBig(String),
    /// `TypeError` from `int(value)`; holds the value's type name.
    #[error("int() argument must be a string, a bytes-like object or a real number, not '{0}'")]
    IntType(&'static str),
    /// `TypeError` from `float(value)`; holds the value's type name.
    #[error("float() argument must be a string or a real number, not '{0}'")]
    FloatType(&'static str),
}

/// `int(value)` of a float as an `i64`: truncated toward zero, refusing NaN and infinity.
pub fn float_to_int(value: f64) -> Result<i64, ConversionError> {
    if value.is_nan() {
        return Err(ConversionError::NotANumber);
    }
    if value.is_infinite() {
        return Err(ConversionError::Infinite);
    }
    let truncated = value.trunc();
    if !(-9.223_372_036_854_776e18..9.223_372_036_854_776e18).contains(&truncated) {
        return Err(ConversionError::TooBig(float_repr(value)));
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(truncated as i64)
}

/// `int(value)` of what `json.loads` returns, as an `i64`.
pub fn int_of(value: &serde_json::Value) -> Result<i64, ConversionError> {
    match value {
        serde_json::Value::Number(number) => match number.as_i64() {
            Some(int) => Ok(int),
            None => match number.as_u64() {
                Some(big) => Err(ConversionError::TooBig(big.to_string())),
                None => float_to_int(number.as_f64().unwrap_or(f64::NAN)),
            },
        },
        serde_json::Value::Bool(flag) => Ok(i64::from(*flag)),
        serde_json::Value::String(text) => {
            let parsed = int(text).ok_or_else(|| ConversionError::IntLiteral(str_repr(text)))?;
            i64::try_from(&parsed).map_err(|_| ConversionError::TooBig(parsed.to_string()))
        }
        other => Err(ConversionError::IntType(crate::json::type_name(other))),
    }
}

/// `float(value)` of what `json.loads` returns.
pub fn float_of(value: &serde_json::Value) -> Result<f64, ConversionError> {
    match value {
        serde_json::Value::Number(number) => Ok(number.as_f64().unwrap_or(f64::NAN)),
        serde_json::Value::Bool(flag) => Ok(f64::from(u8::from(*flag))),
        serde_json::Value::String(text) => {
            float(text).ok_or_else(|| ConversionError::FloatLiteral(str_repr(text)))
        }
        other => Err(ConversionError::FloatType(crate::json::type_name(other))),
    }
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
/// A Python `int` or `float`, kept apart so it prints as `str(value)` does.
pub enum PyNumber {
    Int(i64),
    Float(f64),
}

impl PyNumber {
    /// `float(value)`.
    pub fn value(self) -> f64 {
        match self {
            #[allow(clippy::cast_precision_loss)]
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    /// `str(value)`.
    pub fn text(self) -> String {
        match self {
            Self::Int(value) => value.to_string(),
            Self::Float(value) => float_repr(value),
        }
    }

    /// `format(value, f".{precision}f")`.
    pub fn fixed(self, precision: usize) -> String {
        fixed(self.value(), precision)
    }
}

impl Serialize for PyNumber {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Int(value) => serializer.serialize_i64(*value),
            Self::Float(value) => serializer.serialize_f64(*value),
        }
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

    use super::{SecondsError, float, hex_int, int, round, seconds};

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

    // Expected values are CPython 3.14 int(text, 16)
    #[rstest]
    #[case::lower("ff", Some(255))]
    #[case::upper("FF", Some(255))]
    #[case::surrounding_whitespace(" ff\n", Some(255))]
    #[case::unicode_spaces("\u{a0}ff\u{2003}", Some(255))]
    #[case::plus_sign("+ff", Some(255))]
    #[case::minus_sign("-ff", Some(-255))]
    #[case::prefix("0xff", Some(255))]
    #[case::upper_prefix("0XFF", Some(255))]
    #[case::underscore_after_prefix("0x_ff", Some(255))]
    #[case::underscore_between_digits("f_f", Some(255))]
    #[case::binary_looking("0b1", Some(177))]
    #[case::double_underscore("f__f", None)]
    #[case::leading_underscore("_ff", None)]
    #[case::trailing_underscore("ff_", None)]
    #[case::bare_prefix("0x", None)]
    #[case::prefix_and_underscore("0x_", None)]
    #[case::empty("", None)]
    #[case::not_hex("g", None)]
    #[case::information_separator("\x1cff", None)]
    #[case::decimal_point("1.5", None)]
    #[case::non_ascii_digit_is_a_known_limit("\u{667}", None)]
    fn hex_int_parses_like_python(#[case] text: &str, #[case] expected: Option<i128>) {
        assert_eq!(hex_int(text), expected.map(BigInt::from));
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

    // Expected values are CPython 3.14 round(value, 4)
    #[rstest]
    #[case::fewer_digits_than_kept(1.5, 1.5)]
    #[case::rounded_down(12.345_678, 12.3457)]
    #[case::a_decimal_tie_stored_above_it(2.675_05, 2.6751)]
    #[case::negative(-0.000_04, -0.0)]
    fn round_keeps_ndigits_as_python_does(#[case] value: f64, #[case] expected: f64) {
        assert_eq!(round(value, 4).to_bits(), expected.to_bits());
    }
}
