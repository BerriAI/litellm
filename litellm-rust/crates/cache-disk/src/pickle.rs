use litellm_cache::Error;
use serde_json::{Map, Number, Value};

pub(crate) fn decode(bytes: &[u8]) -> Result<Value, Error> {
    let value = serde_pickle::value_from_slice(bytes, Default::default())
        .map_err(|_| Error::InvalidEntry)?;
    convert(value)
}

fn convert(value: serde_pickle::Value) -> Result<Value, Error> {
    match value {
        serde_pickle::Value::None => Ok(Value::Null),
        serde_pickle::Value::Bool(value) => Ok(Value::Bool(value)),
        serde_pickle::Value::I64(value) => Ok(Value::Number(value.into())),
        serde_pickle::Value::Int(value) => {
            if let Ok(value) = value.to_string().parse::<i64>() {
                Ok(Value::Number(value.into()))
            } else if let Ok(value) = value.to_string().parse::<u64>() {
                Ok(Value::Number(value.into()))
            } else {
                Err(Error::InvalidEntry)
            }
        }
        serde_pickle::Value::F64(value) => Number::from_f64(value)
            .map(Value::Number)
            .ok_or(Error::InvalidEntry),
        serde_pickle::Value::String(value) => Ok(Value::String(value)),
        serde_pickle::Value::List(values) | serde_pickle::Value::Tuple(values) => values
            .into_iter()
            .map(convert)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        serde_pickle::Value::Set(values) | serde_pickle::Value::FrozenSet(values) => values
            .into_iter()
            .map(|value| convert(value.into_value()))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        serde_pickle::Value::Dict(values) => values
            .into_iter()
            .map(|(key, value)| match key {
                serde_pickle::HashableValue::String(key) => Ok((key, convert(value)?)),
                _ => Err(Error::InvalidEntry),
            })
            .collect::<Result<Map<_, _>, _>>()
            .map(Value::Object),
        serde_pickle::Value::Bytes(_) => Err(Error::InvalidEntry),
    }
}
