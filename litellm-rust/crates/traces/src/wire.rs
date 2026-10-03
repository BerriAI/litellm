use serde::{Deserialize, Deserializer, de::Error};

pub(crate) fn flag<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u8, D::Error> {
    match u8::deserialize(deserializer)? {
        value @ 0..=1 => Ok(value),
        _ => Err(D::Error::custom("expected 0 or 1")),
    }
}

pub fn span_type<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let value = String::deserialize(deserializer)?;
    match crate::ObservationType::try_from(value.as_str()) {
        Ok(_) => Ok(value),
        Err(_) => Err(D::Error::custom("unknown trace span type")),
    }
}
