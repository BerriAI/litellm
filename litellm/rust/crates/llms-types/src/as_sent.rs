use serde::{Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned};
use serde_json::Value;

/// A field Python forwards as the provider sent it, read once as `T`. It serializes back to what
/// was sent, explicit nulls and unknown keys included, and [`Self::read`] is `None` when that is
/// not a `T`, so typing the field accepts and writes exactly what the raw value did.
#[derive(Clone, Debug, PartialEq)]
pub struct AsSent<T> {
    sent: Value,
    read: Option<T>,
}

impl<T: DeserializeOwned> AsSent<T> {
    pub fn new(sent: Value) -> Self {
        let read = T::deserialize(&sent).ok();
        Self { sent, read }
    }
}

impl<T> AsSent<T> {
    pub fn read(&self) -> Option<&T> {
        self.read.as_ref()
    }

    pub fn sent(&self) -> &Value {
        &self.sent
    }

    pub fn into_sent(self) -> Value {
        self.sent
    }
}

/// `deserialize_with` for an optional field whose explicit `null` Python forwards: `null` becomes
/// `Some` of a sent `null`, which reads as nothing.
pub fn present<'de, D, T>(deserializer: D) -> Result<Option<AsSent<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    AsSent::deserialize(deserializer).map(Some)
}

impl<T> Serialize for AsSent<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.sent.serialize(serializer)
    }
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for AsSent<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Value::deserialize(deserializer).map(Self::new)
    }
}

#[cfg(feature = "schema")]
impl<T> schemars::JsonSchema for AsSent<T> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        Value::schema_name()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Value::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::formats::messages::MessagesUsage;

    #[rstest]
    #[case::a_usage(
        json!({"input_tokens": 7, "server_tool_use": null, "unknown": [1]}),
        Some(MessagesUsage {
            input_tokens: Some(7),
            extra: serde_json::Map::from_iter([("unknown".to_owned(), json!([1]))]),
            ..MessagesUsage::default()
        })
    )]
    #[case::an_explicit_null(json!(null), None)]
    #[case::not_an_object(json!("7 tokens"), None)]
    #[case::a_count_of_the_wrong_type(json!({"input_tokens": "7"}), None)]
    fn reads_what_it_can_and_writes_back_what_was_sent(
        #[case] sent: Value,
        #[case] read: Option<MessagesUsage>,
    ) {
        let parsed: AsSent<MessagesUsage> = serde_json::from_value(sent.clone()).unwrap();
        assert_eq!(parsed.read(), read.as_ref());
        assert_eq!(serde_json::to_value(&parsed).unwrap(), sent);
    }
}
