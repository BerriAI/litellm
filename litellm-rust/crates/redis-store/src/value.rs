use litellm_python_compat::Value;

use crate::{DecodeError, Number};

/// A store type read from the plain Python data its host sends, so the host never names a field.
pub trait FromValue: Sized {
    fn from_value(value: &Value) -> Result<Self, DecodeError>;
}

/// A store type written as the plain Python data its host hands back.
pub trait IntoValue {
    fn into_value(self) -> Value;
}

impl FromValue for Value {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        Ok(value.clone())
    }
}

impl IntoValue for Value {
    fn into_value(self) -> Value {
        self
    }
}

impl FromValue for Number {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        number(value, "a number")
    }
}

impl IntoValue for Number {
    fn into_value(self) -> Value {
        self.to_value()
    }
}

/// The fields of a Python tuple of exactly `N` items.
pub(crate) fn fields<'a, const N: usize>(
    value: &'a Value,
    shape: &'static str,
) -> Result<&'a [Value; N], DecodeError> {
    match value {
        Value::Tuple(items) | Value::List(items) => items
            .as_slice()
            .try_into()
            .map_err(|_| DecodeError::Shape(shape)),
        _ => Err(DecodeError::Shape(shape)),
    }
}

pub(crate) fn text(value: &Value, shape: &'static str) -> Result<String, DecodeError> {
    match value {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(DecodeError::Shape(shape)),
    }
}

pub(crate) fn optional_text(
    value: &Value,
    shape: &'static str,
) -> Result<Option<String>, DecodeError> {
    match value {
        Value::None => Ok(None),
        value => text(value, shape).map(Some),
    }
}

pub(crate) fn integer(value: &Value, shape: &'static str) -> Result<i64, DecodeError> {
    match value {
        Value::Int(value) => i64::try_from(value).map_err(|_| DecodeError::Shape(shape)),
        _ => Err(DecodeError::Shape(shape)),
    }
}

pub(crate) fn number(value: &Value, shape: &'static str) -> Result<Number, DecodeError> {
    Number::from_value(value).ok_or(DecodeError::Shape(shape))
}
