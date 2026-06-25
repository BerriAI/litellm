use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeSession {
    pub id: Option<String>,
    pub model: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeResponse {
    pub usage: Option<RealtimeUsage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RealtimeEventType(String);

impl RealtimeEventType {
    pub fn new(event_type: impl Into<String>) -> Self {
        Self(event_type.into())
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<String> for RealtimeEventType {
    fn from(event_type: String) -> Self {
        Self::new(event_type)
    }
}

impl From<&str> for RealtimeEventType {
    fn from(event_type: &str) -> Self {
        Self::new(event_type)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawRealtimeEventJson(String);

impl RawRealtimeEventJson {
    pub fn new(raw_json: impl Into<String>) -> Self {
        Self(raw_json.into())
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<String> for RawRealtimeEventJson {
    fn from(raw_json: String) -> Self {
        Self::new(raw_json)
    }
}

impl From<&str> for RawRealtimeEventJson {
    fn from(raw_json: &str) -> Self {
        Self::new(raw_json)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RealtimeEvent {
    pub event_type: RealtimeEventType,
    pub raw_json: RawRealtimeEventJson,
    pub session: Option<RealtimeSession>,
    pub response: Option<RealtimeResponse>,
    pub delta: Option<String>,
}

impl RealtimeEvent {
    pub fn passthrough(
        event_type: impl Into<RealtimeEventType>,
        raw_json: impl Into<RawRealtimeEventJson>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            raw_json: raw_json.into(),
            session: None,
            response: None,
            delta: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
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

    #[test]
    fn passthrough_produces_single_element_vec() {
        let parsed = RealtimeEvent::passthrough("session.update", r#"{"type":"session.update"}"#);
        let result = RealtimeTransformResult::passthrough(parsed.clone());
        assert_eq!(result.events, vec![parsed]);
    }
}
