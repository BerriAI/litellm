use serde::{Deserialize, Deserializer, Serializer, de::Error};

pub fn flag<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    match u8::deserialize(deserializer)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(D::Error::custom("expected 0 or 1")),
    }
}

pub fn serialize_flag<S: Serializer>(value: &bool, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_u8(u8::from(*value))
}

pub fn evidence<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<crate::CallEvidenceKind>, D::Error> {
    let value = String::deserialize(deserializer)?;
    if value.is_empty() {
        return Ok(None);
    }
    serde_json::from_value(serde_json::Value::String(value))
        .map(Some)
        .map_err(D::Error::custom)
}

pub fn serialize_evidence<S: Serializer>(
    value: &Option<crate::CallEvidenceKind>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match value {
        Some(kind) => serde::Serialize::serialize(kind, serializer),
        None => serializer.serialize_str(""),
    }
}

pub fn serialize_status<S: Serializer>(
    value: &crate::SpanStatus,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(match value {
        crate::SpanStatus::Ok => "STATUS_CODE_OK",
        crate::SpanStatus::Error => "STATUS_CODE_ERROR",
        crate::SpanStatus::Unset => "STATUS_CODE_UNSET",
    })
}
