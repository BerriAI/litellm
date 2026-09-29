use serde_json::{Map, Value};

use super::MistralUsage;
use crate::{formats::chat_completions::ChatContentPart, recognized::Recognized};

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MistralChatOptions {
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
    pub safe_prompt: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub prompt_mode: Option<Option<String>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum MistralContentPart {
    Common(ChatContentPart),
    Thinking(MistralThinkingPart),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type")]
pub enum MistralThinkingPart {
    #[serde(rename = "thinking")]
    Thinking {
        thinking: Vec<Recognized<ChatContentPart>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "serde_with::rust::double_option::deserialize"
        )]
        signature: Option<Option<String>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "serde_with::rust::double_option::deserialize"
        )]
        closed: Option<Option<bool>>,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum MistralChatContent {
    Text(String),
    Parts(Vec<Recognized<MistralContentPart>>),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum MistralFunctionArguments {
    Json(Map<String, Value>),
    String(String),
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralFunctionCall {
    pub name: String,
    pub arguments: MistralFunctionArguments,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MistralToolCallType {
    Function,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralToolCall {
    pub function: MistralFunctionCall,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub id: Option<Option<String>>,
    #[serde(rename = "type")]
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub kind: Option<Option<Recognized<MistralToolCallType>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub index: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MistralChatMessage {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub role: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub content: Option<Option<MistralChatContent>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub tool_calls: Option<Option<Vec<MistralToolCall>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub tool_call_id: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub index: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub prefix: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub metadata: Option<Option<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MistralFunctionDelta {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub name: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub arguments: Option<Option<MistralFunctionArguments>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralToolCallDelta {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub function: Option<Option<MistralFunctionDelta>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub id: Option<Option<String>>,
    #[serde(rename = "type")]
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub kind: Option<Option<Recognized<MistralToolCallType>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub index: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MistralChatDelta {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub role: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub content: Option<Option<MistralChatContent>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub tool_calls: Option<Option<Vec<MistralToolCallDelta>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub tool_call_id: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub index: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub metadata: Option<Option<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MistralFinishReason {
    Stop,
    Length,
    ModelLength,
    Error,
    ToolCalls,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralChatChoice {
    pub index: u64,
    pub finish_reason: Recognized<MistralFinishReason>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub message: Option<Option<MistralChatMessage>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub messages: Option<Option<Vec<MistralChatDelta>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralChatResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub created: u64,
    pub usage: MistralUsage,
    pub choices: Vec<MistralChatChoice>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralChatStreamChoice {
    pub index: u64,
    pub delta: MistralChatDelta,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub finish_reason: Option<Option<Recognized<MistralFinishReason>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralChatChunk {
    pub id: String,
    pub model: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub object: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub created: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub usage: Option<Option<MistralUsage>>,
    pub choices: Vec<MistralChatStreamChoice>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{
        MistralChatChunk, MistralChatContent, MistralChatOptions, MistralChatResponse,
        MistralContentPart, MistralFinishReason, MistralFunctionArguments,
    };
    use crate::{formats::chat_completions::ChatCompletionsRequest, recognized::Recognized};
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    fn shared_chat_request_preserves_native_options_and_prefix() {
        let wire = json!({
            "model":"chat-model", "messages":[{"role":"assistant","content":"prefix","prefix":true}],
            "random_seed":42, "safe_prompt":false, "prompt_mode":"reasoning", "future_option":null
        });
        let request: ChatCompletionsRequest = serde_json::from_value(wire.clone()).unwrap();
        let options: MistralChatOptions =
            serde_json::from_value(Value::Object(request.extra.clone())).unwrap();
        assert_eq!(options.random_seed, Some(Some(42)));
        assert_eq!(options.safe_prompt, Some(Some(false)));
        assert!(!options.extra_fields.contains_key("random_seed"));
        assert_eq!(
            serde_json::to_value(options).unwrap(),
            Value::Object(request.extra.clone())
        );
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }

    #[rstest]
    #[case::json(json!({"city":"Paris"}), true)]
    #[case::string(json!("{\"city\":\"Paris\"}"), false)]
    fn native_chat_response_exposes_thinking_and_tool_arguments(
        #[case] arguments: Value,
        #[case] object: bool,
    ) {
        let wire = json!({
            "id":"response-id", "object":"chat.completion", "model":"chat-model", "created":1,
            "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5},
            "choices":[{"index":0,"finish_reason":"tool_calls","message":{
                "role":"assistant", "content":[
                    {"type":"thinking","thinking":[{"type":"text","text":"reason"}],"signature":"sig","closed":true},
                    {"type":"text","text":"answer"}, {"type":"future-part","payload":null}
                ], "tool_calls":[{"id":"call-id","type":"function","index":0,"function":{"name":"weather","arguments":arguments}}]
            }}], "future_field":{"nested":null}
        });
        let response: MistralChatResponse = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(
            response.choices[0].finish_reason,
            Recognized::Known(MistralFinishReason::ToolCalls)
        );
        let message = response.choices[0]
            .message
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap();
        let Some(MistralChatContent::Parts(parts)) = message.content.as_ref().unwrap() else {
            panic!("expected content parts")
        };
        assert!(matches!(
            parts[0].known(),
            Some(MistralContentPart::Thinking(_))
        ));
        assert!(parts[2].known().is_none());
        let function = &message.tool_calls.as_ref().unwrap().as_ref().unwrap()[0].function;
        assert_eq!(
            matches!(function.arguments, MistralFunctionArguments::Json(_)),
            object
        );
        let encoded = serde_json::to_value(response).unwrap();
        assert_eq!(
            encoded["choices"][0]["message"]["content"],
            wire["choices"][0]["message"]["content"]
        );
        assert_eq!(
            encoded["choices"][0]["message"]["tool_calls"],
            wire["choices"][0]["message"]["tool_calls"]
        );
        assert_eq!(encoded["future_field"], wire["future_field"]);
    }

    #[rstest]
    fn streaming_keeps_partial_arguments_and_unknown_finish_reasons() {
        let wire = json!({
            "id":"response-id", "model":"chat-model", "choices":[{
                "index":0, "finish_reason":"future-reason", "delta":{
                    "tool_calls":[{"id":"call-id","type":"function","index":0,"function":{"arguments":"{\"city\":"}}]
                }
            }]
        });
        let chunk: MistralChatChunk = serde_json::from_value(wire.clone()).unwrap();
        assert!(
            chunk.choices[0]
                .finish_reason
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap()
                .known()
                .is_none()
        );
        let encoded = serde_json::to_value(chunk).unwrap();
        assert_eq!(
            encoded["choices"][0]["delta"]["tool_calls"],
            wire["choices"][0]["delta"]["tool_calls"]
        );
        assert_eq!(
            encoded["choices"][0]["finish_reason"],
            wire["choices"][0]["finish_reason"]
        );
    }

    #[rstest]
    fn tool_arguments_reject_non_object_json_values() {
        assert!(serde_json::from_value::<MistralFunctionArguments>(json!([1, 2])).is_err());
    }
}
