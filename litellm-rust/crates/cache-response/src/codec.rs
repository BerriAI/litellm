use litellm_cache::{CacheCodec, Error};
use serde_json::Value;

use crate::CacheEntry;

#[derive(Clone, Copy, Debug, Default)]
pub struct ResponseCacheCodec;

impl CacheCodec for ResponseCacheCodec {
    type Value = CacheEntry;

    fn encode(&self, value: &CacheEntry) -> Result<Vec<u8>, Error> {
        if value
            .timestamp
            .is_some_and(|timestamp| !timestamp.is_finite())
        {
            return Err(Error::InvalidEntry);
        }
        // Python reads a `response` that is either a dict or a serialized string, so every
        // other shape is written serialized. A string on the wire is therefore always a
        // serialized response, which keeps string-valued responses unambiguous.
        if value.timestamp.is_none() || value.response.is_object() {
            return serde_json::to_vec(value).map_err(|_| Error::InvalidEntry);
        }
        let response = serde_json::to_string(&value.response).map_err(|_| Error::InvalidEntry)?;
        serde_json::to_vec(&CacheEntry {
            timestamp: value.timestamp,
            response: Value::String(response),
        })
        .map_err(|_| Error::InvalidEntry)
    }

    fn decode(&self, bytes: &[u8]) -> Result<CacheEntry, Error> {
        let text = std::str::from_utf8(bytes).map_err(|_| Error::InvalidEntry)?;
        let value = decode_value(text)?;
        let Some(timestamp) = value.get("timestamp") else {
            return Ok(CacheEntry {
                timestamp: None,
                response: value,
            });
        };
        let Some(timestamp) = timestamp.as_f64().filter(|timestamp| timestamp.is_finite()) else {
            return Err(Error::InvalidEntry);
        };
        let response = match value.get("response").ok_or(Error::InvalidEntry)? {
            Value::String(text) => decode_value(text)?,
            response => response.clone(),
        };
        Ok(CacheEntry {
            timestamp: Some(timestamp),
            response,
        })
    }
}

fn decode_value(text: &str) -> Result<Value, Error> {
    if let Ok(value) = serde_json::from_str(text) {
        return Ok(value);
    }
    check_literal_depth(text)?;
    let literal: py_literal::Value = text.parse().map_err(|_| Error::InvalidEntry)?;
    literal_value(literal, 0)
}

fn literal_value(value: py_literal::Value, depth: usize) -> Result<Value, Error> {
    use py_literal::Value as Literal;
    if depth > 128 {
        return Err(Error::InvalidEntry);
    }
    match value {
        Literal::String(text) => Ok(Value::String(text)),
        Literal::Boolean(value) => Ok(Value::Bool(value)),
        Literal::None => Ok(Value::Null),
        Literal::Integer(value) => {
            serde_json::from_str(&value.to_string()).map_err(|_| Error::InvalidEntry)
        }
        Literal::Float(value) => serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or(Error::InvalidEntry),
        Literal::List(values) | Literal::Tuple(values) => values
            .into_iter()
            .map(|value| literal_value(value, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Literal::Dict(entries) => entries
            .into_iter()
            .map(|(key, value)| {
                let Literal::String(key) = key else {
                    return Err(Error::InvalidEntry);
                };
                Ok((key, literal_value(value, depth + 1)?))
            })
            .collect::<Result<serde_json::Map<_, _>, _>>()
            .map(Value::Object),
        _ => Err(Error::InvalidEntry),
    }
}

fn check_literal_depth(text: &str) -> Result<(), Error> {
    let mut quote = None;
    let mut escaped = false;
    let mut depth = 0usize;
    for ch in text.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if let Some(delimiter) = quote {
            if ch == '\\' {
                escaped = true;
            } else if ch == delimiter {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => quote = Some(ch),
            '[' | '{' | '(' => {
                depth += 1;
                if depth > 128 {
                    return Err(Error::InvalidEntry);
                }
            }
            ']' | '}' | ')' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}
