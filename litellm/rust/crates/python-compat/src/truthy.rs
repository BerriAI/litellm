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
