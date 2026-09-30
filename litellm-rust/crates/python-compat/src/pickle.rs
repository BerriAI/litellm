//! `pickle.loads` and `pickle.dumps` for plain data, as diskcache stores LiteLLM values.
//!
//! Both directions go through `serde-pickle`'s serde interface rather than
//! `serde_pickle::Value`, because that value type keeps dicts in a `BTreeMap` and would
//! reorder keys. The serde interface keeps insertion order, at the cost of reporting
//! `tuple`, `set`, and `frozenset` as sequences: [`loads`] decodes all three as lists.
//! Python objects that need a class (`GLOBAL`/`REDUCE`) and recursive structures fail.

use std::fmt;

use serde::{
    Deserializer, Serialize, Serializer,
    de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor},
    ser::{SerializeMap, SerializeSeq, SerializeTuple},
};
use serde_pickle::{DeOptions, SerOptions};

use crate::{Error, MAX_DEPTH, Value};

/// `pickle.loads(data)` for any protocol from 0 to 5.
pub fn loads(data: &[u8]) -> Result<Value, Error> {
    let mut deserializer = serde_pickle::Deserializer::new(data, DeOptions::new());
    let value = Seed { depth: 0 }
        .deserialize(&mut deserializer)
        .map_err(|error| Error::InvalidPickle(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| Error::InvalidPickle(error.to_string()))?;
    Ok(value)
}

/// `pickle.dumps(value, protocol=3)`. Every Python 3 reads protocol 3, whatever its own
/// default. Sets and complex numbers are rejected rather than silently changing type.
pub fn dumps(value: &Value) -> Result<Vec<u8>, Error> {
    check_picklable(value, 0)?;
    serde_pickle::to_vec(&Pickled(value), SerOptions::new())
        .map_err(|error| Error::InvalidPickle(error.to_string()))
}

fn check_picklable(value: &Value, depth: usize) -> Result<(), Error> {
    if depth > MAX_DEPTH {
        return Err(Error::TooDeep);
    }
    match value {
        Value::Int(value) if i64::try_from(value).is_err() => Err(Error::IntegerOutOfRange),
        value @ (Value::Set(_) | Value::Complex { .. }) => {
            Err(Error::NotPicklable(value.type_name()))
        }
        Value::List(values) | Value::Tuple(values) => values
            .iter()
            .try_for_each(|value| check_picklable(value, depth + 1)),
        Value::Dict(entries) => entries.iter().try_for_each(|(key, value)| {
            check_picklable(key, depth + 1)?;
            check_picklable(value, depth + 1)
        }),
        _ => Ok(()),
    }
}

struct Pickled<'a>(&'a Value);

impl Serialize for Pickled<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Value::None => serializer.serialize_unit(),
            Value::Bool(value) => serializer.serialize_bool(*value),
            Value::Int(value) => {
                let value = i64::try_from(value)
                    .map_err(|_| serde::ser::Error::custom("integer out of i64 range"))?;
                serializer.serialize_i64(value)
            }
            Value::Float(value) => serializer.serialize_f64(*value),
            Value::Str(text) => serializer.serialize_str(text),
            Value::Bytes(bytes) => serializer.serialize_bytes(bytes),
            Value::List(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    seq.serialize_element(&Pickled(value))?;
                }
                seq.end()
            }
            Value::Tuple(values) => {
                let mut tuple = serializer.serialize_tuple(values.len())?;
                for value in values {
                    tuple.serialize_element(&Pickled(value))?;
                }
                tuple.end()
            }
            Value::Dict(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(&Pickled(key), &Pickled(value))?;
                }
                map.end()
            }
            value @ (Value::Set(_) | Value::Complex { .. }) => Err(serde::ser::Error::custom(
                format!("{} cannot be pickled as plain data", value.type_name()),
            )),
        }
    }
}

#[derive(Clone, Copy)]
struct Seed {
    depth: usize,
}

impl Seed {
    fn child<E: de::Error>(self) -> Result<Self, E> {
        if self.depth >= MAX_DEPTH {
            return Err(E::custom(Error::TooDeep));
        }
        Ok(Self {
            depth: self.depth + 1,
        })
    }
}

impl<'de> DeserializeSeed<'de> for Seed {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Seed {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a plain Python data value")
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::None)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Int(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Int(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Value, E> {
        Ok(Value::Float(value))
    }

    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(Value::Str(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::Str(value))
    }

    fn visit_bytes<E>(self, value: &[u8]) -> Result<Value, E> {
        Ok(Value::Bytes(value.to_vec()))
    }

    fn visit_byte_buf<E>(self, value: Vec<u8>) -> Result<Value, E> {
        Ok(Value::Bytes(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let child = self.child()?;
        let mut values = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
        while let Some(value) = seq.next_element_seed(child)? {
            values.push(value);
        }
        Ok(Value::List(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let child = self.child()?;
        let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0).min(4096));
        while let Some(key) = map.next_key_seed(child)? {
            entries.push((key, map.next_value_seed(child)?));
        }
        Ok(Value::Dict(entries))
    }
}
