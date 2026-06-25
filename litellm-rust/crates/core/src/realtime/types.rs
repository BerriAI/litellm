use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RealtimeFieldValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<RealtimeFieldValue>),
    Object(BTreeMap<String, RealtimeFieldValue>),
}

impl RealtimeFieldValue {
    pub fn as_object(&self) -> Option<&BTreeMap<String, RealtimeFieldValue>> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(value) => value.as_u64(),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&RealtimeFieldValue> {
        self.as_object()?.get(key)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealtimeEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(flatten)]
    pub data: BTreeMap<String, RealtimeFieldValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealtimeTransformResult {
    pub events: Vec<RealtimeEvent>,
}

impl RealtimeTransformResult {
    pub fn passthrough(event: RealtimeEvent) -> Self {
        Self {
            events: vec![event],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(raw: &str) -> RealtimeEvent {
        serde_json::from_str(raw).expect("valid event json")
    }

    #[test]
    fn realtime_event_round_trips_type_and_extra_fields() {
        let raw = r#"{"type":"response.output_text.delta","delta":"hi","response_id":"r1"}"#;
        let parsed = event(raw);
        assert_eq!(parsed.event_type, "response.output_text.delta");
        assert_eq!(
            parsed.data.get("delta"),
            Some(&RealtimeFieldValue::String("hi".into()))
        );
        let reparsed: RealtimeEvent =
            serde_json::from_str(&serde_json::to_string(&parsed).unwrap()).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn passthrough_produces_single_element_vec() {
        let parsed = event(r#"{"type":"session.update"}"#);
        let result = RealtimeTransformResult::passthrough(parsed.clone());
        assert_eq!(result.events, vec![parsed]);
    }
}
