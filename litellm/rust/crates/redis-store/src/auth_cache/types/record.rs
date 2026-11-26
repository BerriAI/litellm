use std::collections::HashMap;

use jiff::Timestamp;
use litellm_python_compat::{
    Value,
    datetime::{IsoDateTime, UtcDateTime, fromisoformat},
};

use crate::{DecodeError, Number, codec::python_value};

pub(in crate::auth_cache) fn decode(raw: Option<Vec<u8>>) -> Result<Option<Value>, DecodeError> {
    raw.as_deref().map(python_value).transpose()
}

/// One field of a cached record, read from and written as the plain data pydantic's JSON mode
/// dumps for it.
pub(super) trait Field: Sized {
    fn read(value: &Value) -> Option<Self>;
    fn write(self) -> Value;
}

impl Field for Value {
    fn read(value: &Value) -> Option<Self> {
        Some(value.clone())
    }

    fn write(self) -> Value {
        self
    }
}

impl Field for String {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Str(text) => Some(text.clone()),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::Str(self)
    }
}

impl Field for bool {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Bool(flag) => Some(*flag),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::Bool(self)
    }
}

impl Field for i64 {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Int(number) => i64::try_from(number).ok(),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::from(self)
    }
}

/// A `float` field also takes the JSON integer pydantic accepts for it.
impl Field for f64 {
    fn read(value: &Value) -> Option<Self> {
        match Number::from_value(value)? {
            Number::Float(number) => Some(number),
            Number::Int(number) => Some(number as f64),
        }
    }

    fn write(self) -> Value {
        Value::Float(self)
    }
}

/// A `datetime` field. Pydantic writes an aware UTC one as `2026-11-19T10:00:00.123000Z`, with
/// no fraction when it is zero. One written without an offset is read as UTC.
impl Field for Timestamp {
    fn read(value: &Value) -> Option<Self> {
        let Value::Str(text) = value else {
            return None;
        };
        let (IsoDateTime::Aware(microseconds) | IsoDateTime::Naive(microseconds)) =
            fromisoformat(text)?;
        Timestamp::from_microsecond(microseconds).ok()
    }

    fn write(self) -> Value {
        let microseconds = self.as_microsecond();
        let at = UtcDateTime::from_timestamp(
            microseconds.div_euclid(1_000_000),
            microseconds.rem_euclid(1_000_000) as u32,
        );
        let fraction = match at.microsecond {
            0 => String::new(),
            microsecond => format!(".{microsecond:06}"),
        };
        Value::Str(format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{fraction}Z",
            at.year, at.month, at.day, at.hour, at.minute, at.second
        ))
    }
}

impl<T: Field> Field for Option<T> {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::None => Some(None),
            value => T::read(value).map(Some),
        }
    }

    fn write(self) -> Value {
        self.map_or(Value::None, T::write)
    }
}

impl<T: Field> Field for Vec<T> {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::List(items) => items.iter().map(T::read).collect(),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::List(self.into_iter().map(T::write).collect())
    }
}

/// The string-keyed entries of a dumped model, by field name.
pub(super) fn fields(entries: &[(Value, Value)]) -> HashMap<&str, &Value> {
    entries
        .iter()
        .filter_map(|(key, value)| match key {
            Value::Str(key) => Some((key.as_str(), value)),
            _ => None,
        })
        .collect()
}

/// The default of a field the model requires, which an entry without it does not decode to.
macro_rules! required {
    ($field:literal) => {
        return Err($crate::DecodeError::Shape($field))
    };
}

pub(super) use required;

/// A record the auth cache holds as a pydantic model's JSON dump. The fields are that model's,
/// in its order, which is the order they are written in, and each takes only the shapes the
/// model takes, so nothing the model would refuse decodes or is written. A field the entry lacks
/// takes the model's default, and one this build does not name is kept in `unread` and written
/// after the named ones, so an entry a newer pod wrote loses nothing when it is read. A field
/// held as plain data may hold credentials, so `Debug` redacts it.
macro_rules! record {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $($(#[$field_meta:meta])* $field:ident: $kind:ty $(= $default:expr)?,)+
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, PartialEq, veil::Redact)]
        pub struct $name {
            $($(#[$field_meta])* pub $field: $kind,)+
            #[redact(with = "[REDACTED]")]
            pub unread: Vec<(String, ::litellm_python_compat::Value)>,
        }

        impl $crate::FromValue for $name {
            fn from_value(
                value: &::litellm_python_compat::Value,
            ) -> Result<Self, $crate::DecodeError> {
                let ::litellm_python_compat::Value::Dict(entries) = value else {
                    return Err($crate::DecodeError::Shape(stringify!($name)));
                };
                let mut unnamed = $crate::auth_cache::types::record::fields(entries);
                Ok(Self {
                    $($field: match unnamed.remove(stringify!($field)) {
                        None => record!(@default $($default)?),
                        Some(value) => <$kind as $crate::auth_cache::types::record::Field>::read(value)
                            .ok_or($crate::DecodeError::Shape(stringify!($field)))?,
                    },)+
                    unread: entries
                        .iter()
                        .filter_map(|(key, value)| match key {
                            ::litellm_python_compat::Value::Str(key) if unnamed.contains_key(key.as_str()) => {
                                Some((key.clone(), value.clone()))
                            }
                            _ => None,
                        })
                        .collect(),
                })
            }
        }

        impl $crate::IntoValue for $name {
            fn into_value(self) -> ::litellm_python_compat::Value {
                ::litellm_python_compat::Value::Dict(
                    [$((stringify!($field).to_owned(), $crate::auth_cache::types::record::Field::write(self.$field)),)+]
                        .into_iter()
                        .chain(self.unread)
                        .map(|(name, value)| (::litellm_python_compat::Value::Str(name), value))
                        .collect(),
                )
            }
        }

        impl $crate::auth_cache::types::record::Field for $name {
            fn read(value: &::litellm_python_compat::Value) -> Option<Self> {
                <Self as $crate::FromValue>::from_value(value).ok()
            }

            fn write(self) -> ::litellm_python_compat::Value {
                $crate::IntoValue::into_value(self)
            }
        }
    };
    (@default) => {
        Default::default()
    };
    (@default $default:expr) => {
        $default
    };
}

pub(super) use record;
