use litellm_llms::{
    Error, base_llm::responses::transformation::BaseResponsesApiConfig,
    openai::responses::transformation::OpenAiResponsesApiConfig,
};
use litellm_llms_types::formats::responses::ResponsesItem;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::text(json!("hi"))]
#[case::items(json!([
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi", "extension": null}]},
    {"role": "user", "content": "shorthand"},
    {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
    {"type": "future_item", "data": [null, 1]},
    null
]))]
fn request_preserves_items_and_extensions_while_enforcing_model_and_input(#[case] input: Value) {
    let params = json!({
        "model": "wrong-model", "input": "wrong-input",
        "stream": false, "reasoning": {"effort": "future-effort"}, "provider_option": null
    });
    let body = OpenAiResponsesApiConfig
        .transform_responses_api_request(
            "test-model",
            input.clone(),
            params.as_object().unwrap().clone(),
        )
        .unwrap();
    assert_eq!(
        body,
        json!({
            "model": "test-model", "input": input,
            "stream": false, "reasoning": {"effort": "future-effort"}, "provider_option": null
        })
    );
}

#[rstest]
#[case::invalid_input(json!({}), json!({}), "responses input must be a string or an array")]
#[case::null_input(json!(null), json!({}), "responses input must be a string or an array")]
#[case::invalid_stream(json!("hi"), json!({"stream": "true"}), "stream must be a boolean")]
#[case::input_validation_first(json!(7), json!({"stream": null}), "responses input must be a string or an array")]
fn request_retains_validation_and_error_precedence(
    #[case] input: Value,
    #[case] params: Value,
    #[case] expected: &str,
) {
    assert_eq!(
        OpenAiResponsesApiConfig.transform_responses_api_request(
            "test-model",
            input,
            params.as_object().unwrap().clone(),
        ),
        Err(Error::InvalidRequest(expected.into()))
    );
}

#[rstest]
fn response_decoder_preserves_known_and_unknown_output_items() {
    let wire = json!({"id": "response-1", "model": "test-model", "output": [
        {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}", "extension": null},
        {"type": "future_item", "payload": {"nested": null}}
    ], "usage": {"provider_counter": null}});
    let response = OpenAiResponsesApiConfig
        .transform_response_api_response(wire.clone())
        .unwrap();
    assert!(
        matches!(response.output[0].known(), Some(ResponsesItem::FunctionCall { call_id, .. }) if call_id == "call-1")
    );
    assert!(response.output[1].known().is_none());
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}
