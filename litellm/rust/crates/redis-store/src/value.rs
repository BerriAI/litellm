use litellm_python_compat::Value;

use crate::{DecodeError, Number};

/// A store value read from the plain Python data it is held as.
pub trait FromValue: Sized {
    fn from_value(value: &Value) -> Result<Self, DecodeError>;
}

/// A store value written as the plain Python data it is held as.
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

pub(crate) fn text(value: &Value, shape: &'static str) -> Result<String, DecodeError> {
    match value {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(DecodeError::Shape(shape)),
    }
}

pub(crate) fn number(value: &Value, shape: &'static str) -> Result<Number, DecodeError> {
    Number::from_value(value).ok_or(DecodeError::Shape(shape))
}
