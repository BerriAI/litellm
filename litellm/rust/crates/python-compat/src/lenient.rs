//! How Python LiteLLM reads a config or metadata value it never validates: each rule is the Python
//! expression it mirrors, and none of them fails, since the Python read never raises either.
//!
//! | Rule | Python expression |
//! |---|---|
//! | [`IsTrue`] | `value is True` |
//! | [`Truthy`] | `bool(value)` |
//! | [`StrBool`] | a bool as is, or `str_to_bool(text) is True` for text |
//! | [`Lenient`] | the value when it has the expected shape, otherwise ignored |
//! | [`LaxBoolOr`] | `TypeAdapter(bool).validate_python(value)`, or the fallback where pydantic raises |
//! | [`LaxBool`] | `core_helpers.lax_bool(value)` |
//! | [`UintOrUnset`] | `get_env_int_or_none(name)`, unset for a negative value too |
//!
//! The bool rules are `serde_with` adapters, used as `#[serde_as(as = "IsTrue")]` on a `bool`, and
//! [`LaxBool`] on an `Option<bool>`.

use std::fmt;

use serde::{
    Deserialize, Deserializer,
    de::{Error, IgnoredAny, MapAccess, SeqAccess, Visitor},
};
use serde_with::DeserializeAs;

use crate::{Value, flags, flags::str_to_bool, number::int, pydantic::lax_bool};

/// `value is True`: only a boolean `true` is true. `"true"`, `1` and anything else are false.
pub struct IsTrue;

impl<'de> DeserializeAs<'de, bool> for IsTrue {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bool(bool),
            Other(IgnoredAny),
        }

        Ok(matches!(Raw::deserialize(deserializer)?, Raw::Bool(true)))
    }
}

/// Pydantic's lax `bool`, with `FALLBACK` for a value pydantic refuses, `null` included.
pub struct LaxBoolOr<const FALLBACK: bool>;

impl<'de, const FALLBACK: bool> DeserializeAs<'de, bool> for LaxBoolOr<FALLBACK> {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
        Ok(Value::deserialize(deserializer)
            .ok()
            .and_then(|value| lax_bool(&value))
            .unwrap_or(FALLBACK))
    }
}

/// [`flags::lax_bool`]: `None` for `None` and for a value pydantic refuses.
pub struct LaxBool;

impl<'de> DeserializeAs<'de, Option<bool>> for LaxBool {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<Option<bool>, D::Error> {
        Ok(Value::deserialize(deserializer)
            .ok()
            .and_then(|value| flags::lax_bool(&value)))
    }
}

/// `get_env_int_or_none`, which reads text that `int(text)` refuses as unset. A negative value is
/// unset too, where Python keeps it.
pub struct UintOrUnset;

impl<'de> DeserializeAs<'de, Option<u64>> for UintOrUnset {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<Option<u64>, D::Error> {
        let text = String::deserialize(deserializer)?;
        Ok(int(&text).and_then(|value| u64::try_from(value).ok()))
    }
}

/// `bool(value)` for plain data: `None`, `False`, zero, and empty strings and containers are false,
/// as [`crate::truthy::truthy`] reads a [`crate::Value`].
pub struct Truthy;

impl<'de> DeserializeAs<'de, bool> for Truthy {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
        deserializer.deserialize_any(TruthyVisitor)
    }
}

struct TruthyVisitor;

impl<'de> Visitor<'de> for TruthyVisitor {
    type Value = bool;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any value")
    }

    fn visit_bool<E: Error>(self, value: bool) -> Result<bool, E> {
        Ok(value)
    }

    fn visit_i64<E: Error>(self, value: i64) -> Result<bool, E> {
        Ok(value != 0)
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<bool, E> {
        Ok(value != 0)
    }

    fn visit_i128<E: Error>(self, value: i128) -> Result<bool, E> {
        Ok(value != 0)
    }

    fn visit_u128<E: Error>(self, value: u128) -> Result<bool, E> {
        Ok(value != 0)
    }

    fn visit_f64<E: Error>(self, value: f64) -> Result<bool, E> {
        Ok(value != 0.0)
    }

    fn visit_str<E: Error>(self, value: &str) -> Result<bool, E> {
        Ok(!value.is_empty())
    }

    fn visit_bytes<E: Error>(self, value: &[u8]) -> Result<bool, E> {
        Ok(!value.is_empty())
    }

    fn visit_none<E: Error>(self) -> Result<bool, E> {
        Ok(false)
    }

    fn visit_unit<E: Error>(self) -> Result<bool, E> {
        Ok(false)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<bool, D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<bool, A::Error> {
        let non_empty = seq.next_element::<IgnoredAny>()?.is_some();
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(non_empty)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<bool, A::Error> {
        let non_empty = map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some();
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(non_empty)
    }
}

/// A flag the proxy also accepts as text: a bool as is, text that [`str_to_bool`] reads as true,
/// and anything else false.
pub struct StrBool;

impl<'de> DeserializeAs<'de, bool> for StrBool {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bool(bool),
            Text(String),
            Other(IgnoredAny),
        }

        Ok(match Raw::deserialize(deserializer)? {
            Raw::Bool(flag) => flag,
            Raw::Text(text) => str_to_bool(&text) == Some(true),
            Raw::Other(_) => false,
        })
    }
}

/// A value Python only uses when it has the shape `T`, such as a list of route strings. Any
/// other value is `Malformed` rather than an error, and the caller decides what that grants.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(from = "RawLenient<T>", bound(deserialize = "T: Deserialize<'de>"))]
pub enum Lenient<T> {
    Valid(T),
    Malformed,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawLenient<T> {
    Valid(T),
    Malformed(IgnoredAny),
}

impl<T> From<RawLenient<T>> for Lenient<T> {
    fn from(raw: RawLenient<T>) -> Self {
        match raw {
            RawLenient::Valid(value) => Self::Valid(value),
            RawLenient::Malformed(_) => Self::Malformed,
        }
    }
}

impl<T> Lenient<T> {
    pub fn valid(&self) -> Option<&T> {
        match self {
            Self::Valid(value) => Some(value),
            Self::Malformed => None,
        }
    }
}

impl<T> Lenient<Box<[T]>> {
    /// The list's items, none when malformed.
    pub fn items(&self) -> &[T] {
        self.valid().map(|items| &**items).unwrap_or_default()
    }
}
