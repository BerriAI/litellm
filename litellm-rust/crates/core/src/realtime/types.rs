use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A single realtime event exchanged over the WebSocket.
///
/// The `type` discriminator is a typed field; the remaining fields are
/// preserved losslessly in `data` so a transform can pass an event through, or
/// inspect/modify specific fields, without enumerating every event variant.
/// Wire (de)serialization happens at the host edge — `core`/`providers` operate
/// only on this typed form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealtimeEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(flatten)]
    pub data: Map<String, Value>,
}

/// One or more typed events produced by a realtime transform.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealtimeTransformResult {
    pub events: Vec<RealtimeEvent>,
}

impl RealtimeTransformResult {
    /// Forward a single event unchanged (the OpenAI baseline).
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
        assert_eq!(parsed.data.get("delta"), Some(&Value::String("hi".into())));
        // Re-serializing yields a semantically-equal event (key order may differ).
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
