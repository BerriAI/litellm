use litellm_cache::Error;
use py_literal::Value;
use serde_json::{Map, Number};

pub(crate) fn from_pickle(bytes: &[u8]) -> Result<Value, Error> {
    let value = serde_pickle::value_from_slice(bytes, Default::default())
        .map_err(|_| Error::InvalidEntry)?;
    from_pickle_value(value)
}

fn from_pickle_value(value: serde_pickle::Value) -> Result<Value, Error> {
    match value {
        serde_pickle::Value::None => Ok(Value::None),
        serde_pickle::Value::Bool(value) => Ok(Value::Boolean(value)),
        serde_pickle::Value::I64(value) => integer(value.to_string()),
        serde_pickle::Value::Int(value) => integer(value.to_string()),
        serde_pickle::Value::F64(value) => Ok(Value::Float(value)),
        serde_pickle::Value::String(value) => Ok(Value::String(value)),
        serde_pickle::Value::Bytes(value) => Ok(Value::Bytes(value)),
        serde_pickle::Value::List(values) => values
            .into_iter()
            .map(from_pickle_value)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::List),
        serde_pickle::Value::Tuple(values) => values
            .into_iter()
            .map(from_pickle_value)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Tuple),
        serde_pickle::Value::Set(values) => values
            .into_iter()
            .map(from_pickle_hashable)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Set),
        serde_pickle::Value::FrozenSet(values) => values
            .into_iter()
            .map(from_pickle_hashable)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Set),
        serde_pickle::Value::Dict(values) => values
            .into_iter()
            .map(|(key, value)| Ok((from_pickle_hashable(key)?, from_pickle_value(value)?)))
            .collect::<Result<Vec<_>, Error>>()
            .map(Value::Dict),
    }
}

fn from_pickle_hashable(value: serde_pickle::HashableValue) -> Result<Value, Error> {
    Ok(match value {
        serde_pickle::HashableValue::None => Value::None,
        serde_pickle::HashableValue::Bool(value) => Value::Boolean(value),
        serde_pickle::HashableValue::I64(value) => integer(value.to_string())?,
        serde_pickle::HashableValue::Int(value) => integer(value.to_string())?,
        serde_pickle::HashableValue::F64(value) => Value::Float(value),
        serde_pickle::HashableValue::Bytes(value) => Value::Bytes(value),
        serde_pickle::HashableValue::String(value) => Value::String(value),
        serde_pickle::HashableValue::Tuple(values) => Value::Tuple(
            values
                .into_iter()
                .map(from_pickle_hashable)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        serde_pickle::HashableValue::FrozenSet(values) => Value::Set(
            values
                .into_iter()
                .map(from_pickle_hashable)
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

fn integer(value: String) -> Result<Value, Error> {
    value.parse().map_err(|_| Error::InvalidEntry)
}

pub(crate) fn from_json(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::None,
        serde_json::Value::Bool(value) => Value::Boolean(value),
        serde_json::Value::Number(value) => {
            if value.is_i64() || value.is_u64() {
                integer(value.to_string())
                    .unwrap_or(Value::Float(value.as_f64().unwrap_or(f64::NAN)))
            } else {
                Value::Float(value.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(value) => Value::String(value),
        serde_json::Value::Array(values) => {
            Value::List(values.into_iter().map(from_json).collect())
        }
        serde_json::Value::Object(values) => Value::Dict(
            values
                .into_iter()
                .map(|(key, value)| (Value::String(key), from_json(value)))
                .collect(),
        ),
    }
}

pub(crate) fn from_json_text(value: &str) -> Result<Value, Error> {
    serde_json::from_str(value)
        .map(from_json)
        .map_err(|_| Error::InvalidEntry)
}

pub(crate) fn is_truthy(value: &Value) -> bool {
    match value {
        Value::None => false,
        Value::Boolean(value) => *value,
        Value::Integer(value) => value.to_string() != "0",
        Value::Float(value) => *value != 0.0,
        Value::Complex(value) => value.re != 0.0 || value.im != 0.0,
        Value::String(value) => !value.is_empty(),
        Value::Bytes(value) => !value.is_empty(),
        Value::Tuple(value) | Value::List(value) | Value::Set(value) => !value.is_empty(),
        Value::Dict(value) => !value.is_empty(),
    }
}

pub(crate) fn is_int(value: &Value) -> bool {
    matches!(value, Value::Integer(_) | Value::Boolean(_))
}

pub(crate) fn to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(value) => value.to_string().parse().ok(),
        Value::Boolean(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

pub(crate) fn to_json(value: &Value) -> Result<Vec<u8>, Error> {
    serde_json::to_vec(&to_json_value(value)?).map_err(|_| Error::InvalidEntry)
}

fn to_json_value(value: &Value) -> Result<serde_json::Value, Error> {
    Ok(match value {
        Value::None => serde_json::Value::Null,
        Value::Boolean(value) => serde_json::Value::Bool(*value),
        Value::Integer(value) => serde_json::Value::Number(
            value
                .to_string()
                .parse::<Number>()
                .map_err(|_| Error::InvalidEntry)?,
        ),
        Value::Float(value) => {
            serde_json::Value::Number(Number::from_f64(*value).ok_or(Error::InvalidEntry)?)
        }
        Value::Complex(_) | Value::Bytes(_) => return Err(Error::InvalidEntry),
        Value::String(value) => serde_json::Value::String(value.clone()),
        Value::Tuple(values) | Value::List(values) | Value::Set(values) => {
            serde_json::Value::Array(
                values
                    .iter()
                    .map(to_json_value)
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        Value::Dict(values) => {
            let values = values
                .iter()
                .map(|(key, value)| {
                    let Value::String(key) = key else {
                        return Err(Error::InvalidEntry);
                    };
                    Ok((key.clone(), to_json_value(value)?))
                })
                .collect::<Result<Map<String, serde_json::Value>, _>>()?;
            serde_json::Value::Object(values)
        }
    })
}
