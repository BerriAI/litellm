use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::recognized::Recognized;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesContent {
    Text(String),
    Parts(Vec<Recognized<ResponsesContentPart>>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    InputText {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    OutputText {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    InputImage {
        image_url: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    SummaryText {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ReasoningText {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesItem {
    Message {
        role: String,
        content: ResponsesContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    FunctionCallOutput {
        call_id: String,
        output: ResponsesContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Reasoning {
        summary: Vec<Recognized<ResponsesContentPart>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[cfg(test)]
mod tests {
    use super::{ResponsesContentPart, ResponsesItem};
    use crate::recognized::Recognized;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::message(json!({"type": "message", "role": "user", "content": "hi", "id": null}), true)]
    #[case::message_parts(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi", "future": null}, {"type": "future_part", "data": null}]}), true)]
    #[case::function_call(json!({"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "incomplete {"}), true)]
    #[case::function_output_text(json!({"type": "function_call_output", "call_id": "call-1", "output": "ok", "status": null}), true)]
    #[case::function_output_parts(json!({"type": "function_call_output", "call_id": "call-1", "output": [{"type": "input_image", "image_url": "https://example.test/image", "detail": null}]}), true)]
    #[case::reasoning(json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary", "extra": null}], "encrypted_content": null}), true)]
    #[case::shorthand_message(json!({"role": "user", "content": "hi"}), false)]
    #[case::unknown(json!({"type": "future_item", "role": "assistant", "content": "hi"}), false)]
    #[case::null_content(json!({"type": "message", "role": "assistant", "content": null}), false)]
    #[case::missing_call_id(json!({"type": "function_call", "name": "lookup", "arguments": "{}"}), false)]
    #[case::wrong_summary_shape(json!({"type": "reasoning", "summary": "summary"}), false)]
    #[case::null(json!(null), false)]
    fn items_recognize_supported_shapes_and_preserve_every_value(
        #[case] wire: Value,
        #[case] known: bool,
    ) {
        let item: Recognized<ResponsesItem> = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(item.known().is_some(), known);
        assert_eq!(serde_json::to_value(item).unwrap(), wire);
    }

    #[rstest]
    #[case::input_text(json!({"type": "input_text", "text": "hi", "extra": null}), true)]
    #[case::output_text(json!({"type": "output_text", "text": "hi", "annotations": []}), true)]
    #[case::image(json!({"type": "input_image", "image_url": "https://example.test/image", "detail": "auto"}), true)]
    #[case::summary(json!({"type": "summary_text", "text": "summary"}), true)]
    #[case::reasoning(json!({"type": "reasoning_text", "text": "reasoning"}), true)]
    #[case::unknown(json!({"type": "future_content", "text": "hi"}), false)]
    #[case::wrong_text_type(json!({"type": "output_text", "text": null}), false)]
    #[case::image_by_file(json!({"type": "input_image", "file_id": "file-1"}), false)]
    fn content_parts_keep_discriminators_and_nested_extensions(
        #[case] wire: Value,
        #[case] known: bool,
    ) {
        let part: Recognized<ResponsesContentPart> = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(part.known().is_some(), known);
        assert_eq!(serde_json::to_value(part).unwrap(), wire);
    }
}
