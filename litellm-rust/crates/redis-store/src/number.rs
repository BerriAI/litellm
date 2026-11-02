use std::{cmp::Ordering, ops::Add};

use litellm_python_compat::{BigInt, Value};

/// A Python `int` or `float`, kept apart because `repr` and `json.dumps` write them differently.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    /// Python's `int(value)`, which truncates toward zero.
    pub fn truncate(self) -> i64 {
        match self {
            Self::Int(value) => value,
            Self::Float(value) => value.trunc() as i64,
        }
    }

    pub fn to_value(self) -> Value {
        match self {
            Self::Int(value) => Value::Int(BigInt::from(value)),
            Self::Float(value) => Value::Float(value),
        }
    }

    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int(value) => i64::try_from(value).ok().map(Self::Int),
            Value::Float(value) => Some(Self::Float(*value)),
            _ => None,
        }
    }

    /// Python's `min` over numbers: the first of the smallest, keeping its type.
    pub fn min(values: impl IntoIterator<Item = Self>) -> Option<Self> {
        values
            .into_iter()
            .fold(None, |smallest, value| match smallest {
                Some(smallest)
                    if value.as_f64().partial_cmp(&smallest.as_f64()) != Some(Ordering::Less) =>
                {
                    Some(smallest)
                }
                _ => Some(value),
            })
    }
}

/// Python's `+`: two ints stay an int, anything with a float is a float.
impl Add for Number {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Self::Int(left), Self::Int(right)) => Self::Int(left + right),
            (left, right) => Self::Float(left.as_f64() + right.as_f64()),
        }
    }
}
