use serde::{
    Deserializer,
    de::{Error, Visitor},
};
use serde_with::DeserializeAs;

pub struct LaxI64;
pub struct FiniteF64;

pub fn parse_str_bool(value: &str) -> Option<bool> {
    let token = value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    });
    if token.eq_ignore_ascii_case("true") {
        return Some(true);
    }
    token.eq_ignore_ascii_case("false").then_some(false)
}

/// `redis-py` string Booleans: only `1`, `true`, and `yes` (case-insensitive) are true.
pub fn parse_redis_bool(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
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
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use serde_with::serde_as;

    use super::*;

    #[serde_as]
    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    struct Numbers {
        #[serde_as(deserialize_as = "Option<Vec<LaxI64>>")]
        integers: Option<Vec<i64>>,
        #[serde_as(deserialize_as = "Option<FiniteF64>")]
        float: Option<f64>,
    }

    #[test]
    fn boolean_tokens_follow_python_string_trimming_without_redis_tokens() {
        for (input, expected) in [
            (" True ", Some(true)),
            ("\u{1c}TRUE\u{1f}", Some(true)),
            ("\u{a0}False\u{2003}", Some(false)),
            ("true\u{200b}", None),
            ("yes", None),
            ("1", None),
            ("", None),
            ("unknown", None),
        ] {
            assert_eq!(parse_str_bool(input), expected, "{input:?}");
        }
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
