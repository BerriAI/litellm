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
//! | [`FloatOrUnset`] | `get_env_float_or_none(name)` |
//! | [`IsOn`] | `core_helpers.lax_bool(value) is True` |
//! | [`ParamNames`] | `core_helpers.allowed_openai_param_names(value)` |
//! | [`TextEntries`] | `[entry for entry in value if isinstance(entry, str)]` when `value` is a list |
//! | [`Present`] | `mapping[key]` read by another rule when `key in mapping`, `None` included |
//!
//! The bool rules are `serde_with` adapters, used as `#[serde_as(as = "IsTrue")]` on a `bool`, and
//! [`LaxBool`] on an `Option<bool>`.

use std::{fmt, marker::PhantomData};

use serde::{
    Deserialize, Deserializer,
    de::{Error, IgnoredAny, MapAccess, SeqAccess, Visitor},
};
use serde_with::DeserializeAs;

use crate::{
    Value, flags,
    flags::str_to_bool,
    number::{float, int},
    pydantic::lax_bool,
};

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

/// `lax_bool(value) is True`: on only for a value [`flags::lax_bool`] reads as true.
pub struct IsOn;

impl<'de> DeserializeAs<'de, bool> for IsOn {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
        Ok(LaxBool::deserialize_as(deserializer)? == Some(true))
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

/// `get_env_float_or_none`, which reads text that `float(text)` refuses as unset.
pub struct FloatOrUnset;

impl<'de> DeserializeAs<'de, Option<f64>> for FloatOrUnset {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
        Ok(float(&String::deserialize(deserializer)?))
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

/// `allowed_openai_param_names`: the `\w+` words of text, a mapping's text keys, or a list's
/// text entries. `None` and any other value name nothing.
pub struct ParamNames;

impl<'de> DeserializeAs<'de, Box<[String]>> for ParamNames {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<Box<[String]>, D::Error> {
        Ok(match Value::deserialize(deserializer)? {
            Value::Str(text) => text
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .filter(|word| !word.is_empty())
                .map(str::to_owned)
                .collect(),
            Value::Dict(entries) => text_items(entries.into_iter().map(|(key, _)| key)),
            Value::List(items) => text_items(items),
            _ => Box::default(),
        })
    }
}

/// A list's text entries, skipping the rest. Any value but a list holds none.
pub struct TextEntries;

impl<'de> DeserializeAs<'de, Box<[String]>> for TextEntries {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<Box<[String]>, D::Error> {
        Ok(match Value::deserialize(deserializer)? {
            Value::List(items) => text_items(items),
            _ => Box::default(),
        })
    }
}

fn text_items(items: impl IntoIterator<Item = Value>) -> Box<[String]> {
    items
        .into_iter()
        .filter_map(|item| match item {
            Value::Str(text) => Some(text),
            _ => None,
        })
        .collect()
}

/// A key that is there, `None` included, read by `Rule`. On a `#[serde(default)]` field a key
/// that is missing stays `None`, so `key in mapping` survives the read.
pub struct Present<Rule>(PhantomData<Rule>);

impl<'de, T, Rule: DeserializeAs<'de, T>> DeserializeAs<'de, Option<T>> for Present<Rule> {
    fn deserialize_as<D: Deserializer<'de>>(deserializer: D) -> Result<Option<T>, D::Error> {
        Rule::deserialize_as(deserializer).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde::Deserialize;
    use serde_json::{Value, json};
    use serde_with::serde_as;

    use super::{IsOn, ParamNames, Present, TextEntries};

    #[serde_as]
    #[derive(Deserialize)]
    struct Names(#[serde_as(as = "ParamNames")] Box<[String]>);

    /// Expected values from CPython: `allowed_openai_param_names(json.loads(value))`.
    #[rstest]
    #[case::text(json!("response_format, tools;x-y"), &["response_format", "tools", "x", "y"][..])]
    #[case::mapping(json!({"response_format": null, "tools": 1}), &["response_format", "tools"][..])]
    #[case::list(json!(["response_format", 5, null, "tools"]), &["response_format", "tools"][..])]
    #[case::null(json!(null), &[][..])]
    #[case::number(json!(5), &[][..])]
    fn param_names_read_text_mapping_keys_and_list_entries(
        #[case] value: Value,
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            &*serde_json::from_value::<Names>(value).unwrap().0,
            expected
        );
    }

    #[serde_as]
    #[derive(Deserialize)]
    struct Entries(#[serde_as(as = "TextEntries")] Box<[String]>);

    #[rstest]
    #[case::list(json!(["tools", 5, "response_format"]), &["tools", "response_format"][..])]
    #[case::text(json!("tools"), &[][..])]
    #[case::mapping(json!({"tools": 1}), &[][..])]
    fn text_entries_are_a_lists_text_only(#[case] value: Value, #[case] expected: &[&str]) {
        assert_eq!(
            &*serde_json::from_value::<Entries>(value).unwrap().0,
            expected
        );
    }

    #[serde_as]
    #[derive(Deserialize)]
    struct Flag {
        #[serde_as(as = "Present<IsOn>")]
        #[serde(default)]
        on: Option<bool>,
    }

    /// Expected values from CPython: `lax_bool(mapping["on"]) is True` when `"on" in mapping`.
    #[rstest]
    #[case::missing(json!({}), None)]
    #[case::null(json!({"on": null}), Some(false))]
    #[case::on(json!({"on": true}), Some(true))]
    #[case::off(json!({"on": false}), Some(false))]
    #[case::on_word(json!({"on": " True "}), Some(true))]
    #[case::no_flag_word(json!({"on": "maybe"}), Some(false))]
    #[case::list(json!({"on": [true]}), Some(false))]
    fn a_present_key_is_read_and_a_missing_one_is_not(
        #[case] mapping: Value,
        #[case] expected: Option<bool>,
    ) {
        assert_eq!(
            serde_json::from_value::<Flag>(mapping).unwrap().on,
            expected
        );
    }
}
