//! JSON columns read as raw text, so a store can hand them to its caller without parsing them.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;
use sqlx::{
    Postgres,
    error::BoxDynError,
    postgres::{PgTypeInfo, PgValueRef},
    types::Json,
};

/// A JSON column read as its raw text, passed through to the caller unparsed.
///
/// These columns are settings bags (key metadata, router settings, permissions) that may hold
/// credentials, so `Debug` redacts the whole value.
#[derive(Clone, veil::Redact)]
pub struct RawJson(#[redact(with = "[REDACTED]")] pub Box<RawValue>);

impl sqlx::Type<Postgres> for RawJson {
    fn type_info() -> PgTypeInfo {
        <Json<Box<RawValue>> as sqlx::Type<Postgres>>::type_info()
    }

    fn compatible(ty: &PgTypeInfo) -> bool {
        <Json<Box<RawValue>> as sqlx::Type<Postgres>>::compatible(ty)
    }
}

impl<'r> sqlx::Decode<'r, Postgres> for RawJson {
    fn decode(value: PgValueRef<'r>) -> Result<Self, BoxDynError> {
        <Json<Box<RawValue>> as sqlx::Decode<'r, Postgres>>::decode(value).map(|json| Self(json.0))
    }
}

impl PartialEq for RawJson {
    fn eq(&self, other: &Self) -> bool {
        self.0.get() == other.0.get()
    }
}

impl RawJson {
    /// The column's JSON text exactly as Postgres returned it.
    pub fn get(&self) -> &str {
        self.0.get()
    }
}

impl<'de> Deserialize<'de> for RawJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Box::<RawValue>::deserialize(deserializer).map(Self)
    }
}

impl Serialize for RawJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::RawJson;
    use serde_json::value::RawValue;

    const JSON: &str = r#"{"api_key":"credential-value","nested":{"token":"credential-value"}}"#;

    fn raw_json() -> RawJson {
        RawJson(RawValue::from_string(JSON.to_owned()).expect("valid JSON"))
    }

    #[test]
    fn debug_redacts_the_whole_value() {
        let value = raw_json();
        let debug = format!("{value:?}");
        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    #[test]
    fn serialize_passes_the_raw_json_through() {
        let value = raw_json();
        assert_eq!(serde_json::to_string(&value).expect("serializes"), JSON);
        assert_eq!(value.get(), JSON);
    }
}
