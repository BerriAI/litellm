use litellm_llms::{Error, openai::responses::transformation::OpenAiResponsesApiConfig};
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};

#[fixture]
fn config() -> OpenAiResponsesApiConfig {
    OpenAiResponsesApiConfig
}

#[rstest]
#[case::text(json!({"input": "hello"}))]
#[case::items(json!({"input": [{"type": "function_call_output", "call_id": "call_test", "output": "42"}]}))]
#[case::prompt(json!({"prompt": {"id": "pmpt_test"}}))]
fn request_preserves_fields_and_replaces_model(
    config: OpenAiResponsesApiConfig,
    #[case] input: Value,
) {
    let body: Map<String, Value> = input
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .chain([
            ("model".into(), json!("public-alias")),
            ("previous_response_id".into(), json!("resp_previous")),
            (
                "tools".into(),
                json!([{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]),
            ),
            ("reasoning".into(), json!({"effort": "low"})),
            ("text".into(), json!({"format": {"type": "json_object"}})),
            ("store".into(), json!(false)),
            ("stream".into(), json!(true)),
            ("future_option".into(), json!({"enabled": true})),
        ])
        .collect();
    let expected: Map<String, Value> = body
        .clone()
        .into_iter()
        .map(|(name, value)| {
            if name == "model" {
                (name, json!("test-model"))
            } else {
                (name, value)
            }
        })
        .collect();
    assert_eq!(
        config.transform_request("test-model", body).unwrap(),
        expected
    );
}

#[rstest]
#[case::empty_model("", json!({"input": "hello"}))]
#[case::object_input("test-model", json!({"input": {"role": "user"}}))]
#[case::numeric_input("test-model", json!({"input": 42}))]
#[case::string_stream("test-model", json!({"stream": "true"}))]
fn invalid_request_is_rejected(
    config: OpenAiResponsesApiConfig,
    #[case] model: &str,
    #[case] body: Value,
) {
    assert!(matches!(
        config.transform_request(model, body.as_object().unwrap().clone()),
        Err(Error::InvalidRequest(_))
    ));
}

#[rstest]
#[case::base("https://example.test/v1", "https://example.test/v1/responses")]
#[case::trailing_slash("https://example.test/v1/", "https://example.test/v1/responses")]
#[case::full_url(
    "https://example.test/v1/responses/",
    "https://example.test/v1/responses"
)]
#[case::query(
    "https://example.test/v1?revision=test",
    "https://example.test/v1/responses?revision=test"
)]
fn url_preserves_base_path_and_query(
    config: OpenAiResponsesApiConfig,
    #[case] base: &str,
    #[case] expected: &str,
) {
    assert_eq!(config.get_complete_url(Some(base)).unwrap(), expected);
}

#[rstest]
#[case::invalid_json(b"invalid")]
#[case::array(b"[]")]
#[case::missing_output(br#"{"id":"resp_test","object":"response"}"#)]
#[case::wrong_object(br#"{"id":"resp_test","object":"chat.completion","output":[]}"#)]
fn malformed_response_is_rejected(config: OpenAiResponsesApiConfig, #[case] body: &[u8]) {
    assert!(matches!(
        config.transform_response(body),
        Err(Error::InvalidResponse(_))
    ));
}

#[rstest]
fn response_preserves_tools_usage_and_unknown_fields(config: OpenAiResponsesApiConfig) {
    let body = json!({
        "id": "resp_test", "object": "response", "status": "completed", "model": "test-model",
        "output": [{"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "call_test"}],
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        "error": null, "future_field": {"nested": [true]}
    });
    let response = config
        .transform_response(&serde_json::to_vec(&body).unwrap())
        .unwrap();
    assert_eq!(serde_json::to_value(response).unwrap(), body);
}
