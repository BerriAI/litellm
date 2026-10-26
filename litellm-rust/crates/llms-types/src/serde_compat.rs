use serde::{
    Deserializer,
    de::{Error, Visitor},
};
use serde_with::DeserializeAs;

pub struct LaxI64;
pub struct FiniteF64;

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
    use crate::serde_compat::{FiniteF64, LaxI64};

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
}
