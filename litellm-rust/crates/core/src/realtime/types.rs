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

#[derive(Clone, Debug, PartialEq)]
pub struct RealtimeEvent {
    pub event_type: String,
    pub raw_json: String,
    pub session: Option<RealtimeSession>,
    pub response: Option<RealtimeResponse>,
    pub delta: Option<String>,
}

impl RealtimeEvent {
    pub fn passthrough(event_type: impl Into<String>, raw_json: impl Into<String>) -> Self {
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
