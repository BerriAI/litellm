use serde_json::{Map, Value};

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum MistralFimStop {
    String(String),
    Strings(Vec<String>),
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralFimRequest {
    pub model: String,
    pub prompt: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub suffix: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub temperature: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub top_p: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub max_tokens: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub min_tokens: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub stop: Option<Option<MistralFimStop>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub random_seed: Option<Option<i64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub stream: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub metadata: Option<Option<Map<String, Value>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub prompt_cache_key: Option<Option<String>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::MistralFimRequest;
    use crate::providers::mistral::chat::MistralChatResponse;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::single_stop(json!("end"))]
    #[case::multiple_stops(json!(["end", "stop"]))]
    fn fim_request_preserves_suffix_and_generation_options(#[case] stop: Value) {
        let wire = json!({"model":"fim-model","prompt":"def function():","suffix":"return value","stop":stop,"max_tokens":12,"random_seed":42,"min_tokens":null,"future_option":true});
        let request: MistralFimRequest = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(request.suffix, Some(Some("return value".into())));
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }

    #[rstest]
    fn fim_response_uses_the_native_completion_contract() {
        let wire = json!({"id":"response-id","object":"chat.completion","model":"fim-model","created":1,"choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","content":"generated code"}}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}});
        let response: MistralChatResponse = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(response).unwrap(), wire);
    }

    #[rstest]
    fn fim_request_rejects_non_string_prompts() {
        assert!(
            serde_json::from_value::<MistralFimRequest>(
                json!({"model":"fim-model","prompt":false})
            )
            .is_err()
        );
    }
}
