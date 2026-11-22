use std::fmt;

use num_bigint::BigInt;
use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, SeqAccess, Visitor},
};

/// A Python value built only from literals: what `ast.literal_eval` can return.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(BigInt),
    Float(f64),
    Complex {
        re: f64,
        im: f64,
    },
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    /// Insertion-ordered, with Python's key equality already applied.
    Dict(Vec<(Value, Value)>),
    /// Literal order, with Python's member equality already applied.
    Set(Vec<Value>),
}

impl Value {
    /// Python's type name, as it appears in `TypeError` messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Complex { .. } => "complex",
            Value::Str(_) => "str",
            Value::Bytes(_) => "bytes",
            Value::Tuple(_) => "tuple",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Set(_) => "set",
        }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int(value.into())
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::Str(value.to_owned())
    }
}

/// The Python value a self-describing format (JSON, YAML) hands to Python code: integers keep full
/// precision, `null` is `None`, and mappings keep their order.
impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ValueVisitor)
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("plain data")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Int(value.into()))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Int(value.into()))
    }

    fn visit_i128<E: de::Error>(self, value: i128) -> Result<Value, E> {
        Ok(Value::Int(value.into()))
    }

    fn visit_u128<E: de::Error>(self, value: u128) -> Result<Value, E> {
        Ok(Value::Int(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        Ok(Value::Float(value))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Value, E> {
        Ok(Value::Str(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Value, E> {
        Ok(Value::Str(value))
    }

    fn visit_bytes<E: de::Error>(self, value: &[u8]) -> Result<Value, E> {
        Ok(Value::Bytes(value.to_vec()))
    }

    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::None)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::None)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        Value::deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::with_capacity(seq.size_hint().unwrap_or_default());
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(Value::List(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut entries = Vec::with_capacity(map.size_hint().unwrap_or_default());
        while let Some(entry) = map.next_entry()? {
            entries.push(entry);
        }
        Ok(Value::Dict(entries))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde::Deserialize;
    use serde_json::json;

    use crate::{Value, json::from_json};

    #[rstest]
    #[case::null(json!(null))]
    #[case::boolean(json!(true))]
    #[case::negative(json!(-3))]
    #[case::beyond_i64(json!(u64::MAX))]
    #[case::float(json!(0.5))]
    #[case::text(json!("a"))]
    #[case::nested(json!({"b": [1, {"a": null}], "a": "x"}))]
    fn deserializing_json_matches_json_loads(#[case] value: serde_json::Value) {
        assert_eq!(Value::deserialize(&value).unwrap(), from_json(value));
    }
}
