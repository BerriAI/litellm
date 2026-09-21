mod value;

use litellm_cache::Error;
use py_literal::Value;

use crate::{StoredValue, ValueAdapter};

#[derive(Clone, Copy, Debug, Default)]
pub struct PythonDiskCacheAdapter;

impl PythonDiskCacheAdapter {
    fn python_get_cache(value: StoredValue) -> Result<Option<Value>, Error> {
        let value = match value {
            StoredValue::Bytes(value) => Value::Bytes(value),
            StoredValue::Text(value) => Value::String(value),
            StoredValue::Integer(value) => {
                value::from_json(serde_json::Value::Number(value.into()))
            }
            StoredValue::Float(value) => Value::Float(value),
            StoredValue::Pickle(value) => value::from_pickle(&value)?,
        };
        if !value::is_truthy(&value) {
            return Ok(None);
        }
        match value {
            Value::String(text) => Ok(Some(
                value::from_json_text(&text).unwrap_or(Value::String(text)),
            )),
            Value::Bytes(bytes) => match std::str::from_utf8(&bytes) {
                Ok(text) => Ok(Some(
                    value::from_json_text(text).unwrap_or(Value::Bytes(bytes)),
                )),
                Err(_) => Ok(Some(Value::Bytes(bytes))),
            },
            value => Ok(Some(value)),
        }
    }
}

impl ValueAdapter for PythonDiskCacheAdapter {
    fn read(&self, value: StoredValue) -> Result<Option<Vec<u8>>, Error> {
        let raw = match &value {
            StoredValue::Text(value) => Some(value.as_bytes().to_vec()),
            StoredValue::Bytes(value) => Some(value.clone()),
            StoredValue::Integer(_) | StoredValue::Float(_) | StoredValue::Pickle(_) => None,
        };
        let Some(value) = Self::python_get_cache(value)? else {
            return Ok(None);
        };
        if let Some(raw) = raw {
            return Ok(Some(raw));
        }
        value::to_json(&value).map(Some)
    }

    fn write(&self, payload: Vec<u8>) -> StoredValue {
        StoredValue::Bytes(payload)
    }

    fn counter_seed(&self, value: Option<StoredValue>) -> Result<f64, Error> {
        let Some(value) = value else {
            return Ok(0.0);
        };
        let Some(value) = Self::python_get_cache(value)? else {
            return Ok(0.0);
        };
        Ok(if value::is_int(&value) {
            value::to_f64(&value).unwrap_or(0.0)
        } else {
            0.0
        })
    }

    fn counter_value(&self, value: f64) -> StoredValue {
        if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
            StoredValue::Integer(value as i64)
        } else {
            StoredValue::Float(value)
        }
    }
}
