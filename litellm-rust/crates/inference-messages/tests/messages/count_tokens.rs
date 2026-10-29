use rstest::rstest;

use super::*;

#[rstest]
#[tokio::test]
async fn vertex_ai_counts_at_the_publisher_count_endpoint_with_the_unversioned_model(
    call: MessagesCall,
) {
    let upstream = upstream([json_response(json!({"input_tokens": 14}))]).await;
    let messages = json!([{"role": "user", "content": "hi"}]);

    let counted = messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            body: body(json!({"model": "claude-sonnet-4-5@20250929", "messages": messages})),
            custom_llm_provider: Some("vertex_ai".into()),
            litellm_params: serde_json::from_value(json!({
                "model": "vertex_ai/claude-sonnet-4-5@20250929",
                "vertex_project": "proj",
                "vertex_location": "europe-west1",
            }))
            .unwrap(),
            api_base: Some(upstream.uri()),
            extra_headers: headers([
                ("Authorization", "Bearer caller-token"),
                ("anthropic-version", "2023-06-01"),
            ]),
            ..call
        })
        .await
        .expect("count succeeds");

    assert_eq!(counted.input_tokens, 14);
    let request = only_request(&upstream).await;
    assert_eq!(
        request.url.path(),
        "/v1/projects/proj/locations/europe-west1/publishers/anthropic/models/count-tokens:rawPredict"
    );
    assert_eq!(request.header("authorization"), Some("Bearer caller-token"));
    assert_eq!(request.header("anthropic-version"), None);
    assert_eq!(
        request.json(),
        json!({"model": "claude-sonnet-4-5", "messages": messages})
    );
}

#[rstest]
#[tokio::test]
async fn a_count_without_input_tokens_is_an_invalid_response(call: MessagesCall) {
    let upstream = upstream([json_response(json!({"tokens": 14}))]).await;

    let error = messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            api_key: Some("sk-ant".into()),
            api_base: Some(upstream.uri()),
            ..call
        })
        .await
        .unwrap_err();

    assert!(error.is_response(), "{error:?}");
}
