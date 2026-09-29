use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::ResponsesItem;
use crate::recognized::Recognized;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<Recognized<ResponsesItem>>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{ResponsesInput, ResponsesRequest};
    use crate::formats::responses::ResponsesItem;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::text(json!("hi"))]
    #[case::items(json!([
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"role": "user", "content": "shorthand"},
        {"type": "future_item", "value": null},
        null
    ]))]
    fn request_preserves_input_and_unmodeled_parameters(#[case] input: Value) {
        let wire = json!({"model": "test-model", "input": input,
            "reasoning": {"effort": "future-effort", "extension": null},
            "previous_response_id": null, "store": false, "provider_field": [1, null]});
        let request: ResponsesRequest = serde_json::from_value(wire.clone()).unwrap();
        match &request.input {
            ResponsesInput::Text(text) => assert_eq!(text, "hi"),
            ResponsesInput::Items(items) => assert!(matches!(
                items[0].known(),
                Some(ResponsesItem::Message { .. })
            )),
        }
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }

    #[rstest]
    #[case::null(json!(null))]
    #[case::number(json!(7))]
    #[case::object(json!({"role": "user", "content": "hi"}))]
    fn input_rejects_shapes_outside_the_existing_request_contract(#[case] input: Value) {
        assert!(serde_json::from_value::<ResponsesInput>(input).is_err());
    }
}
