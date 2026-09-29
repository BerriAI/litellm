use serde_json::{Map, Value};

use crate::recognized::{Recognized, deserialize_present};

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MessagesUsage {
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_tokens: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_tokens: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_creation_input_tokens: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_read_input_tokens: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<Value>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub usage: Option<Recognized<MessagesUsage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
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
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude".to_string(),
            content: vec![],
            stop_reason: stop_reason.map(str::to_string),
            stop_sequence: stop_sequence.map(str::to_string),
            usage: usage.map(|value| serde_json::from_value(value).unwrap()),
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

    #[rstest]
    #[case::known(json!(7), Some(7))]
    #[case::zero(json!(0), Some(0))]
    #[case::maximum(json!(u64::MAX), Some(u64::MAX))]
    #[case::null(json!(null), None)]
    #[case::negative(json!(-1), None)]
    #[case::fractional(json!(1.5), None)]
    #[case::string(json!("7"), None)]
    #[case::object(json!({"count": 7}), None)]
    fn usage_counts_are_typed_without_changing_the_wire_data(
        #[case] value: Value,
        #[case] expected: Option<u64>,
    ) {
        let wire = json!({
            "input_tokens": value,
            "output_tokens": value,
            "cache_creation_input_tokens": value,
            "cache_read_input_tokens": value,
            "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 4},
            "future_usage": {"tokens": 9}
        });
        let parsed = response(None, None, Some(wire.clone()), None);
        let usage = parsed.usage.as_ref().and_then(Recognized::known).unwrap();
        assert_eq!(
            usage
                .input_tokens
                .as_ref()
                .and_then(Recognized::known)
                .copied(),
            expected
        );
        assert_eq!(
            usage
                .output_tokens
                .as_ref()
                .and_then(Recognized::known)
                .copied(),
            expected
        );
        assert_eq!(
            usage
                .cache_creation_input_tokens
                .as_ref()
                .and_then(Recognized::known)
                .copied(),
            expected
        );
        assert_eq!(
            usage
                .cache_read_input_tokens
                .as_ref()
                .and_then(Recognized::known)
                .copied(),
            expected
        );
        assert_eq!(serde_json::to_value(usage).unwrap(), wire);
    }

    #[rstest]
    #[case::absent(None)]
    #[case::empty(Some(json!({})))]
    #[case::typed(Some(json!({"input_tokens": 1, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4})))]
    #[case::null(Some(json!(null)))]
    #[case::unrecognized(Some(json!([1, 2])))]
    fn response_usage_preserves_presence_and_unknown_shapes(#[case] usage: Option<Value>) {
        let wire = serde_json::to_value(response(None, None, usage, None)).unwrap();
        let parsed: MessagesResponse = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
    }
}
