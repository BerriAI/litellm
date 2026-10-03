use std::{collections::BTreeMap, io::Write};

use opentelemetry_proto::tonic::common::v1::{
    AnyValue, KeyValue, any_value::Value as AttributeValue,
};
use serde::{
    Serialize, Serializer,
    ser::{SerializeMap, SerializeSeq},
};

use super::limits::{Budget, MAX_ATTRIBUTES};
use crate::Error;

struct AttributeWriter<'a> {
    body: Vec<u8>,
    budget: &'a mut Budget,
}

impl Write for AttributeWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.budget
            .consume(bytes.len())
            .map_err(std::io::Error::other)?;
        self.body.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn attributes(
    values: Vec<KeyValue>,
    budget: &mut Budget,
) -> Result<BTreeMap<String, String>, Error> {
    if values.len() > MAX_ATTRIBUTES {
        return Err(Error::TooLarge);
    }
    values
        .into_iter()
        .map(|entry| {
            budget.consume(entry.key.len() + 96)?;
            let text = match entry.value {
                Some(AnyValue {
                    value: Some(AttributeValue::StringValue(value)),
                }) => {
                    budget.consume(value.len())?;
                    value
                }
                Some(AnyValue {
                    value: Some(AttributeValue::BytesValue(value)),
                }) => {
                    budget.consume(value.len().saturating_mul(3))?;
                    String::from_utf8_lossy(&value).into_owned()
                }
                value => {
                    let mut writer = AttributeWriter {
                        body: Vec::new(),
                        budget,
                    };
                    serde_json::to_writer(&mut writer, &AttributeJson(value.as_ref()))
                        .map_err(|_| Error::TooLarge)?;
                    String::from_utf8(writer.body).map_err(|_| Error::InvalidPayload)?
                }
            };
            Ok((entry.key, text))
        })
        .collect()
}

struct AttributeJson<'a>(Option<&'a AnyValue>);

impl Serialize for AttributeJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0.and_then(|value| value.value.as_ref()) {
            Some(AttributeValue::StringValue(value)) => serializer.serialize_str(value),
            Some(AttributeValue::BoolValue(value)) => serializer.serialize_bool(*value),
            Some(AttributeValue::IntValue(value)) => serializer.serialize_i64(*value),
            Some(AttributeValue::DoubleValue(value)) => serializer.serialize_f64(*value),
            Some(AttributeValue::BytesValue(value)) => {
                serializer.serialize_str(&String::from_utf8_lossy(value))
            }
            Some(AttributeValue::ArrayValue(value)) => {
                let mut sequence = serializer.serialize_seq(Some(value.values.len()))?;
                for entry in &value.values {
                    sequence.serialize_element(&AttributeJson(Some(entry)))?;
                }
                sequence.end()
            }
            Some(AttributeValue::KvlistValue(value)) => {
                let mut map = serializer.serialize_map(Some(value.values.len()))?;
                for entry in &value.values {
                    map.serialize_entry(&entry.key, &AttributeJson(entry.value.as_ref()))?;
                }
                map.end()
            }
            Some(AttributeValue::StringValueStrindex(value)) => serializer.serialize_i32(*value),
            None => serializer.serialize_unit(),
        }
    }
}
