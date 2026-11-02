use std::fmt;

use serde::Deserializer;
use serde::de::{self, Visitor};
use serde_with::DeserializeAs;

pub(crate) struct LaxI64;

pub(crate) struct Count;

impl<'de> DeserializeAs<'de, i64> for LaxI64 {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
        deserializer.deserialize_any(Self)
    }
}

impl Visitor<'_> for LaxI64 {
    type Value = i64;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an integer in the i64 range")
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<i64, E> {
        Ok(value)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<i64, E> {
        i64::try_from(value).map_err(E::custom)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<i64, E> {
        integral_float(value).ok_or_else(|| E::custom("expected an integer in the i64 range"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<i64, E> {
        integer_string(value.trim())
            .ok_or_else(|| E::custom("expected an integer in the i64 range"))
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<i64, E> {
        Ok(i64::from(value))
    }
}

impl<'de> DeserializeAs<'de, u64> for Count {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let count = LaxI64::deserialize_as(deserializer)?;
        u64::try_from(count).map_err(|_| {
            de::Error::invalid_value(de::Unexpected::Signed(count), &"a count of zero or more")
        })
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
    if integer.starts_with('_') || integer.ends_with('_') || integer.contains("__") {
        return None;
    }
    let digits = integer.strip_prefix(['+', '-']).unwrap_or(integer);
    if digits.is_empty()
        || digits.starts_with('_')
        || !digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'_')
    {
        return None;
    }
    integer.replace('_', "").parse().ok()
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
    use rstest::rstest;
    use serde_json::{Value, json};
    use serde_with::de::DeserializeAsWrap;

    use super::*;

    fn count(value: Value) -> Option<u64> {
        DeserializeAsWrap::<u64, Count>::deserialize(value)
            .ok()
            .map(DeserializeAsWrap::into_inner)
    }

    use serde::Deserialize;

    #[rstest]
    #[case::integer(json!(128000), Some(128_000))]
    #[case::whole_float(json!(128000.0), Some(128_000))]
    #[case::string(json!(" 8192 "), Some(8192))]
    #[case::separated(json!("1_000"), Some(1000))]
    #[case::trailing_zeros(json!("2.000"), Some(2))]
    #[case::boolean(json!(true), Some(1))]
    #[case::maximum(json!(i64::MAX.to_string()), Some(i64::MAX as u64))]
    #[case::negative(json!(-5), None)]
    #[case::fraction(json!(2.5), None)]
    #[case::precise_fraction(json!("1.0000000000000001"), None)]
    #[case::exponent(json!("1e3"), None)]
    #[case::missing_fraction(json!("2."), None)]
    #[case::repeated_separator(json!("2__0"), None)]
    #[case::above_i64(json!(u64::MAX), None)]
    #[case::float_above_i64(json!(9_223_372_036_854_775_808.0), None)]
    #[case::word(json!("8k"), None)]
    #[case::object(json!({}), None)]
    fn counts_accept_whole_numbers_however_they_are_written(
        #[case] written: Value,
        #[case] expected: Option<u64>,
    ) {
        assert_eq!(count(written), expected);
    }
}
