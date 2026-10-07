use serde::{
    Deserializer,
    de::{Error, Visitor},
};
use serde_with::DeserializeAs;

pub fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

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

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(untagged)]
pub enum Nullable<T> {
    Value(T),
    Null,
}

impl<T> Nullable<T> {
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Value(value) => Some(value),
            Self::Null => None,
        }
    }

    pub fn into_value(self) -> Option<T> {
        match self {
            Self::Value(value) => Some(value),
            Self::Null => None,
        }
    }
}

impl<T: std::ops::Deref> Nullable<T> {
    pub fn as_deref(&self) -> Option<&T::Target> {
        self.value().map(std::ops::Deref::deref)
    }
}
