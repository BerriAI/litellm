use serde::{Deserialize, Deserializer, de::Error};
use serde_json::Value;
use serde_with::DeserializeAs;

pub(crate) struct LaxI64;
pub(crate) struct FiniteF64;

impl<'de> DeserializeAs<'de, i64> for LaxI64 {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::Number(number) if number.is_f64() => number.as_f64().and_then(integral_float),
            Value::Number(number) => number.as_i64(),
            Value::String(value) => integer_string(value.trim()),
            Value::Bool(value) => Some(i64::from(value)),
            _ => None,
        }
        .ok_or_else(|| D::Error::custom("expected an integer in the i64 range"))
    }
}

impl<'de> DeserializeAs<'de, f64> for FiniteF64 {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::Number(number) => number.as_f64(),
            Value::String(value) => value.trim().parse::<f64>().ok(),
            Value::Bool(value) => Some(f64::from(value)),
            _ => None,
        }
        .filter(|value| value.is_finite())
        .ok_or_else(|| D::Error::custom("expected a finite number"))
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
    use super::*;
    use serde::Serialize;
    use serde_json::json;
    use serde_with::serde_as;

    #[serde_as]
    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    struct Numbers {
        #[serde_as(deserialize_as = "Option<Vec<LaxI64>>")]
        integers: Option<Vec<i64>>,
        #[serde_as(deserialize_as = "Option<FiniteF64>")]
        float: Option<f64>,
    }

    #[test]
    fn adapters_compose_and_serialize_as_numbers() {
        let numbers: Numbers = serde_json::from_value(json!({
            "integers": ["9007199254740993.0", "1_000", " +2.000 ", 3.0, true],
            "float": " 1.5 "
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(numbers).unwrap(),
            json!({
                "integers": [9_007_199_254_740_993_i64, 1000, 2, 3, 1], "float": 1.5
            })
        );
        for input in [json!({}), json!({"integers": null, "float": null})] {
            assert_eq!(
                serde_json::from_value::<Numbers>(input).unwrap(),
                Numbers {
                    integers: None,
                    float: None,
                }
            );
        }
    }

    #[test]
    fn integer_bounds_and_invalid_values_are_checked() {
        for input in [
            json!(i64::MIN),
            json!(i64::MAX),
            json!(i64::MAX.to_string()),
        ] {
            assert!(serde_json::from_value::<Numbers>(json!({"integers": [input]})).is_ok());
        }
        for input in [
            json!(u64::MAX),
            json!(9_223_372_036_854_775_808_u64),
            json!(9_223_372_036_854_775_808.0),
            json!("-9223372036854775809"),
            json!("1.0000000000000001"),
            json!("1e3"),
            json!("2."),
            json!(".0"),
            json!("_2"),
            json!("2__0"),
            json!(2.5),
            json!(null),
            json!({}),
        ] {
            assert!(serde_json::from_value::<Numbers>(json!({"integers": [input]})).is_err());
        }
    }

    #[test]
    fn floats_reject_nonfinite_and_invalid_values() {
        for input in [
            json!("NaN"),
            json!("inf"),
            json!("-inf"),
            json!("1e999"),
            json!([]),
        ] {
            assert!(serde_json::from_value::<Numbers>(json!({"float": input})).is_err());
        }
        for (input, expected) in [(json!(2), 2.0), (json!(2.5), 2.5), (json!(true), 1.0)] {
            let numbers: Numbers = serde_json::from_value(json!({"float": input})).unwrap();
            assert_eq!(numbers.float, Some(expected));
        }
    }
}
