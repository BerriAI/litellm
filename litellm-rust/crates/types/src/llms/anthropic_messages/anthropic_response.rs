use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<Value>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    fn response(
        stop_reason: Option<&str>,
        stop_sequence: Option<&str>,
        usage: Option<Value>,
        container: Option<Value>,
    ) -> AnthropicMessagesResponse {
        AnthropicMessagesResponse {
            id: "msg_1".to_string(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude".to_string(),
            content: vec![],
            stop_reason: stop_reason.map(str::to_string),
            stop_sequence: stop_sequence.map(str::to_string),
            usage,
            container,
            extra: Map::new(),
        }
    }

    #[rstest]
    #[case::turn_in_progress(None, None, json!(null), json!(null))]
    #[case::ended_on_end_turn(Some("end_turn"), None, json!("end_turn"), json!(null))]
    #[case::ended_on_stop_sequence(Some("stop_sequence"), Some("###"), json!("stop_sequence"), json!("###"))]
    fn stop_fields_are_always_serialized(
        #[case] stop_reason: Option<&str>,
        #[case] stop_sequence: Option<&str>,
        #[case] expected_reason: Value,
        #[case] expected_sequence: Value,
    ) {
        let body: Value = serde_json::to_value(response(stop_reason, stop_sequence, None, None))
            .expect("serializable");
        assert_eq!(body.get("stop_reason"), Some(&expected_reason));
        assert_eq!(body.get("stop_sequence"), Some(&expected_sequence));
    }

    #[rstest]
    #[case::absent(None, None)]
    #[case::present(Some(json!({"input_tokens": 1})), Some(json!({"id": "c_1"})))]
    fn usage_and_container_are_omitted_only_when_none(
        #[case] usage: Option<Value>,
        #[case] container: Option<Value>,
    ) {
        let body: Value =
            serde_json::to_value(response(None, None, usage.clone(), container.clone()))
                .expect("serializable");
        assert_eq!(body.get("usage").cloned(), usage);
        assert_eq!(body.get("container").cloned(), container);
    }
}
