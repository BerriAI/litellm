use serde::{
    Deserializer,
    de::{Error, Visitor},
};
use serde_with::DeserializeAs;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

pub struct LaxI64;
pub struct FiniteF64;

pub const PYTHON_DEFAULT_INT_MAX_STR_DIGITS: usize = 4_300;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParsedInteger {
    Value(i64),
    OutOfRange,
    Invalid,
}

pub fn parse_python_integer(value: &str, max_digits: usize) -> ParsedInteger {
    let Some(normalized) = value
        .trim()
        .chars()
        .map(|character| match character {
            '+' | '-' | '_' => Some(character),
            _ => decimal_digit(character),
        })
        .collect::<Option<String>>()
    else {
        return ParsedInteger::Invalid;
    };
    let digits = normalized.strip_prefix(['+', '-']).unwrap_or(&normalized);
    if !digits
        .split('_')
        .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        || (max_digits > 0 && digits.bytes().filter(u8::is_ascii_digit).count() > max_digits)
    {
        return ParsedInteger::Invalid;
    }
    normalized
        .replace('_', "")
        .parse::<i64>()
        .map(ParsedInteger::Value)
        .unwrap_or(ParsedInteger::OutOfRange)
}

fn decimal_digit(character: char) -> Option<char> {
    if character.is_ascii_digit() {
        return Some(character);
    }
    if character.general_category() != GeneralCategory::DecimalNumber {
        return None;
    }
    // Unicode decimal digits form consecutive groups of ten
    let preceding_digits = (0..u32::from(character))
        .rev()
        .take_while(|code| {
            char::from_u32(*code).is_some_and(|character| {
                character.general_category() == GeneralCategory::DecimalNumber
            })
        })
        .count();
    char::from_digit(u32::try_from(preceding_digits % 10).ok()?, 10)
}

impl<'de> DeserializeAs<'de, i64> for LaxI64 {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
        deserializer.deserialize_any(Self)
    }
}

impl<'de> Visitor<'de> for LaxI64 {
    type Value = i64;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an integer in the i64 range")
    }

    fn visit_i64<E: Error>(self, value: i64) -> Result<i64, E> {
        Ok(value)
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<i64, E> {
        i64::try_from(value).map_err(E::custom)
    }

    fn visit_f64<E: Error>(self, value: f64) -> Result<i64, E> {
        integral_float(value).ok_or_else(|| E::custom("expected an integer in the i64 range"))
    }

    fn visit_str<E: Error>(self, value: &str) -> Result<i64, E> {
        integer_string(value.trim())
            .ok_or_else(|| E::custom("expected an integer in the i64 range"))
    }

    fn visit_bool<E: Error>(self, value: bool) -> Result<i64, E> {
        Ok(i64::from(value))
    }
}

impl<'de> DeserializeAs<'de, f64> for FiniteF64 {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        deserializer.deserialize_any(Self)
    }
}

impl<'de> Visitor<'de> for FiniteF64 {
    type Value = f64;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a finite number")
    }

    fn visit_i64<E: Error>(self, value: i64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_f64<E: Error>(self, value: f64) -> Result<f64, E> {
        value
            .is_finite()
            .then_some(value)
            .ok_or_else(|| E::custom("expected a finite number"))
    }

    fn visit_str<E: Error>(self, value: &str) -> Result<f64, E> {
        self.visit_f64(value.trim().parse::<f64>().map_err(E::custom)?)
    }

    fn visit_bool<E: Error>(self, value: bool) -> Result<f64, E> {
        Ok(f64::from(value))
    }
}

fn integer_string(value: &str) -> Option<i64> {
    let integer = match value.split_once('.') {
        Some((integer, fraction)) => {
            if fraction.is_empty() || !fraction.bytes().all(|byte| byte == b'0') {
                return None;
            }
            integer
        }
        None => value,
    };
    if !integer.is_ascii() || integer.trim() != integer {
        return None;
    }
    match parse_python_integer(integer, 0) {
        ParsedInteger::Value(value) => Some(value),
        ParsedInteger::OutOfRange | ParsedInteger::Invalid => None,
    }
}

fn integral_float(value: f64) -> Option<i64> {
    (value.is_finite()
        && value.fract() == 0.0
        && value >= i64::MIN as f64
        && value < -(i64::MIN as f64))
        .then_some(value as i64)
}

#[cfg(test)]
mod tests {
    use super::{
        FiniteF64, LaxI64, PYTHON_DEFAULT_INT_MAX_STR_DIGITS, ParsedInteger, parse_python_integer,
    };

    use rstest::rstest;
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};
    use serde_with::serde_as;

    #[serde_as]
    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    struct Numbers {
        #[serde_as(deserialize_as = "Option<Vec<LaxI64>>")]
        integers: Option<Vec<i64>>,
        #[serde_as(deserialize_as = "Option<FiniteF64>")]
        float: Option<f64>,
    }

    #[rstest]
    fn adapters_compose_and_serialize_as_numbers() {
        let numbers: Numbers = serde_json::from_value(json!({
            "integers": ["9007199254740993.0", "1_000", " +2.000 ", 3.0, true],
            "float": " 1.5 "
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(numbers).unwrap(),
            json!({"integers": [9_007_199_254_740_993_i64, 1000, 2, 3, 1], "float": 1.5})
        );
    }

    #[rstest]
    #[case::missing(json!({}))]
    #[case::null(json!({"integers": null, "float": null}))]
    fn optional_adapters_accept_missing_and_null_fields(#[case] input: Value) {
        assert_eq!(
            serde_json::from_value::<Numbers>(input).unwrap(),
            Numbers {
                integers: None,
                float: None
            }
        );
    }

    #[rstest]
    #[case::minimum(json!(i64::MIN), i64::MIN)]
    #[case::maximum(json!(i64::MAX), i64::MAX)]
    #[case::maximum_string(json!(i64::MAX.to_string()), i64::MAX)]
    fn integers_preserve_bounds(#[case] input: Value, #[case] expected: i64) {
        let numbers: Numbers = serde_json::from_value(json!({"integers": [input]})).unwrap();
        assert_eq!(numbers.integers, Some(vec![expected]));
    }

    #[rstest]
    #[case::unsigned_maximum(json!(u64::MAX))]
    #[case::above_maximum(json!(9_223_372_036_854_775_808_u64))]
    #[case::float_above_maximum(json!(9_223_372_036_854_775_808.0))]
    #[case::below_minimum(json!("-9223372036854775809"))]
    #[case::precise_fraction(json!("1.0000000000000001"))]
    #[case::exponent(json!("1e3"))]
    #[case::missing_fraction(json!("2."))]
    #[case::missing_integer(json!(".0"))]
    #[case::space_before_fraction(json!("2 .0"))]
    #[case::unicode_digit(json!("١"))]
    #[case::leading_separator(json!("_2"))]
    #[case::repeated_separator(json!("2__0"))]
    #[case::fraction(json!(2.5))]
    #[case::null(json!(null))]
    #[case::object(json!({}))]
    fn integers_reject_invalid_values(#[case] input: Value) {
        assert!(serde_json::from_value::<Numbers>(json!({"integers": [input]})).is_err());
    }

    #[rstest]
    #[case::nan(json!("NaN"))]
    #[case::positive_infinity(json!("inf"))]
    #[case::negative_infinity(json!("-inf"))]
    #[case::overflow(json!("1e999"))]
    #[case::array(json!([]))]
    fn floats_reject_nonfinite_and_invalid_values(#[case] input: Value) {
        assert!(serde_json::from_value::<Numbers>(json!({"float": input})).is_err());
    }

    #[rstest]
    #[case::integer(json!(2), 2.0)]
    #[case::float(json!(2.5), 2.5)]
    #[case::boolean(json!(true), 1.0)]
    fn floats_accept_finite_numbers(#[case] input: Value, #[case] expected: f64) {
        let numbers: Numbers = serde_json::from_value(json!({"float": input})).unwrap();
        assert_eq!(numbers.float, Some(expected));
    }

    #[rstest]
    #[case::padded(" \t+0_1\n", ParsedInteger::Value(1))]
    #[case::negative("-1", ParsedInteger::Value(-1))]
    #[case::mixed_scripts("٠_１", ParsedInteger::Value(1))]
    #[case::mathematical_digit("𝟙", ParsedInteger::Value(1))]
    #[case::minimum("-9223372036854775808", ParsedInteger::Value(i64::MIN))]
    #[case::positive_overflow("9223372036854775808", ParsedInteger::OutOfRange)]
    #[case::negative_overflow("-9223372036854775809", ParsedInteger::OutOfRange)]
    #[case::empty("", ParsedInteger::Invalid)]
    #[case::sign_only("+", ParsedInteger::Invalid)]
    #[case::leading_separator("_1", ParsedInteger::Invalid)]
    #[case::trailing_separator("1_", ParsedInteger::Invalid)]
    #[case::repeated_separator("1__0", ParsedInteger::Invalid)]
    #[case::embedded_sign("1+0", ParsedInteger::Invalid)]
    #[case::superscript("²", ParsedInteger::Invalid)]
    #[case::fraction("1.0", ParsedInteger::Invalid)]
    fn python_integer_strings(#[case] input: &str, #[case] expected: ParsedInteger) {
        assert_eq!(
            parse_python_integer(input, PYTHON_DEFAULT_INT_MAX_STR_DIGITS),
            expected
        );
    }

    #[rstest]
    #[case::at_limit("0".repeat(PYTHON_DEFAULT_INT_MAX_STR_DIGITS), PYTHON_DEFAULT_INT_MAX_STR_DIGITS, ParsedInteger::Value(0))]
    #[case::over_limit("0".repeat(PYTHON_DEFAULT_INT_MAX_STR_DIGITS + 1), PYTHON_DEFAULT_INT_MAX_STR_DIGITS, ParsedInteger::Invalid)]
    #[case::custom_limit("0".repeat(4), 3, ParsedInteger::Invalid)]
    #[case::unlimited("0".repeat(PYTHON_DEFAULT_INT_MAX_STR_DIGITS + 1), 0, ParsedInteger::Value(0))]
    #[case::sign_and_separators(" +٠_０ ".into(), 2, ParsedInteger::Value(0))]
    fn python_integer_digit_limits(
        #[case] input: String,
        #[case] max_digits: usize,
        #[case] expected: ParsedInteger,
    ) {
        assert_eq!(parse_python_integer(&input, max_digits), expected);
    }
}
