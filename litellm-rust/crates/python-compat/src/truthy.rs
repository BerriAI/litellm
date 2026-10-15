use num_bigint::Sign;

use crate::Value;

/// `bool(value)` for plain data: `None`, `False`, zero, and empty containers are false.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::None => false,
        Value::Bool(value) => *value,
        Value::Int(value) => value.sign() != Sign::NoSign,
        Value::Float(value) => *value != 0.0,
        Value::Complex { re, im } => *re != 0.0 || *im != 0.0,
        Value::Str(value) => !value.is_empty(),
        Value::Bytes(value) => !value.is_empty(),
        Value::Tuple(values) | Value::List(values) | Value::Set(values) => !values.is_empty(),
        Value::Dict(entries) => !entries.is_empty(),
    }
}

/// [`truthy`] for the value `json.loads` returns, without converting it
pub fn json_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(number) => number.as_f64() != Some(0.0),
        serde_json::Value::String(value) => !value.is_empty(),
        serde_json::Value::Array(values) => !values.is_empty(),
        serde_json::Value::Object(values) => !values.is_empty(),
    }
}
