use serde_json::{Map, Value};

use super::ResponsesItem;
use crate::recognized::Recognized;

#[macro_rules_attribute::apply(wire_type)]
pub struct ResponsesApiResponse {
    pub id: String,
    pub model: String,
    pub output: Vec<Recognized<ResponsesItem>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::ResponsesApiResponse;
    use crate::formats::responses::{ResponsesContent, ResponsesContentPart, ResponsesItem};
    use crate::recognized::Recognized;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    fn response_exposes_message_reasoning_and_calls_without_losing_extensions() {
        let wire = json!({
            "id": "response-1", "model": "test-model", "status": "completed",
            "output": [
                {"type": "reasoning", "id": "reasoning-1", "summary": [], "content": [{"type": "reasoning_text", "text": "thinking", "future": null}]},
                {"type": "message", "id": "message-1", "role": "assistant", "content": [{"type": "output_text", "text": "hello", "annotations": [], "future": null}]},
                {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{\"key\":null}", "status": null},
                {"type": "future_item", "payload": {"nested": [null, 1]}}
            ],
            "usage": {"input_tokens": 3, "provider_counter": null},
            "provider_field": [null, true]
        });
        let response: ResponsesApiResponse = serde_json::from_value(wire.clone()).unwrap();
        assert!(
            matches!(response.output[0].known(), Some(ResponsesItem::Reasoning { summary, .. }) if summary.is_empty())
        );
        let Some(ResponsesItem::Message {
            content: ResponsesContent::Parts(parts),
            ..
        }) = response.output[1].known()
        else {
            panic!("expected typed message content");
        };
        assert!(
            matches!(parts[0].known(), Some(ResponsesContentPart::OutputText { text, .. }) if text == "hello")
        );
        assert!(
            matches!(response.output[2].known(), Some(ResponsesItem::FunctionCall { call_id, name, arguments, .. }) if call_id == "call-1" && name == "lookup" && arguments == "{\"key\":null}")
        );
        assert!(matches!(response.output[3], Recognized::Unrecognized(_)));
        assert_eq!(serde_json::to_value(response).unwrap(), wire);
    }
}
