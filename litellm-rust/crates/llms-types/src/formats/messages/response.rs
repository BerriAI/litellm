use serde_json::{Map, Value};

use super::{
    ContentBlock, ContextManagementResponse, MessagesContainer, MessagesUsage, StopDetails,
};
use crate::recognized::Recognized;

#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: super::MessageType,
    pub role: super::MessageRole,
    pub model: String,
    pub content: Vec<Recognized<ContentBlock>>,
    pub stop_reason: Option<super::StopReason>,
    pub stop_sequence: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub usage: Option<Recognized<MessagesUsage>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub container: Option<Recognized<MessagesContainer>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub stop_details: Option<Recognized<StopDetails>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub context_management: Option<Recognized<ContextManagementResponse>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub safeguard_results: Option<Recognized<Vec<Map<String, Value>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn response(
        stop_reason: Option<&str>,
        stop_sequence: Option<&str>,
        usage: Option<Value>,
        container: Option<Value>,
    ) -> MessagesResponse {
        MessagesResponse {
            id: "msg_1".to_string(),
            message_type: super::super::MessageType::Message,
            role: super::super::MessageRole::Assistant,
            model: "claude".to_string(),
            content: vec![],
            stop_reason: stop_reason.map(|value| value.to_string().into()),
            stop_sequence: stop_sequence.map(str::to_string),
            usage: usage.map(|value| serde_json::from_value(value).unwrap()),
            container: container.map(|value| serde_json::from_value(value).unwrap()),
            stop_details: None,
            context_management: None,
            safeguard_results: None,
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
