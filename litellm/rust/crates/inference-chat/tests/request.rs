use std::time::Duration;

use litellm_core_utils::decline::Decline;
use litellm_inference_chat::{
    Error,
    types::{ChatCompletionsRequest, ClientSideCredentials, DropParams, OptionalParams},
};
use rstest::rstest;
use serde_json::{Map, Value, json};
use wiremock::ResponseTemplate;

mod support;
use support::*;

const ANTHROPIC_MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":4}}"#;

fn request<'a>(
    model: &'a str,
    provider: Option<&'a str>,
    messages: Value,
    optional_params: Value,
) -> ChatCompletionsRequest<'a> {
    let Value::Object(optional_params) = optional_params else {
        panic!("params must be an object");
    };
    ChatCompletionsRequest {
        model,
        messages,
        optional_params: OptionalParams::ProviderMapped(optional_params),
        api_key: Some("sk-test"),
        api_base: None,
        custom_llm_provider: provider,
        extra_headers: None,
        timeout: Some(Duration::from_secs(10)),
    }
}

async fn complete(
    request: ChatCompletionsRequest<'_>,
) -> Result<litellm_llms_types::formats::chat_completions::ChatCompletionsResponse, Error> {
    chat_completions_route().execute(request, &(), None).await
}

fn object(value: Value) -> Map<String, Value> {
    let Value::Object(map) = value else {
        panic!("expected an object");
    };
    map
}

fn openai(params: Value) -> OptionalParams {
    OptionalParams::OpenAi {
        params: object(params),
        client_side: ClientSideCredentials::Only(Box::default()),
        drop_params: DropParams::Off,
    }
}

fn anthropic_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(ANTHROPIC_MESSAGE, "application/json")
}

fn bedrock_response() -> ResponseTemplate {
    json_response(json!({
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}
    }))
}

#[rstest]
#[tokio::test]
async fn resolves_the_provider_from_the_model_prefix() {
    let upstream = upstream([anthropic_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        ..request(
            "anthropic/claude-sonnet-4-5",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 16}),
        )
    })
    .await
    .expect("call succeeds");

    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/v1/messages");
    assert_eq!(sent.json()["model"], "claude-sonnet-4-5");
}

#[rstest]
#[tokio::test]
async fn strips_an_explicit_provider_prefix_from_the_model() {
    let upstream = upstream([anthropic_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        ..request(
            "anthropic/claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .expect("call succeeds");

    assert_eq!(
        only_request(&upstream).await.json()["model"],
        "claude-sonnet-4-5"
    );
}

#[rstest]
#[tokio::test]
async fn adds_the_auth_and_default_headers() {
    let upstream = upstream([anthropic_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        ..request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .expect("call succeeds");

    let sent = only_request(&upstream).await;
    assert_eq!(sent.header_values("x-api-key"), ["sk-test"]);
    assert_eq!(sent.header("anthropic-version"), Some("2023-06-01"));
}

#[rstest]
#[tokio::test]
async fn the_deployment_credential_replaces_a_caller_supplied_auth_header() {
    let upstream = upstream([anthropic_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        extra_headers: Some(object(json!({"X-Api-Key": "sk-caller"}))),
        ..request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .expect("call succeeds");

    assert_eq!(
        only_request(&upstream).await.header_values("x-api-key"),
        ["sk-test"]
    );
}

#[rstest]
#[tokio::test]
async fn a_forwarded_authorization_header_suppresses_the_resolved_api_key_header() {
    let upstream = upstream([anthropic_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        extra_headers: Some(object(json!({
            "Authorization": "Bearer sk-ant-oat01-token",
            "X-Api-Key": "sk-caller"
        }))),
        ..request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .expect("call succeeds");

    let sent = only_request(&upstream).await;
    assert_eq!(sent.header_values("x-api-key"), ["sk-caller"]);
    assert_eq!(
        sent.header_values("authorization"),
        ["Bearer sk-ant-oat01-token"]
    );
}

#[rstest]
#[tokio::test]
async fn an_unrelated_forwarded_authorization_does_not_defer_the_resolved_key() {
    let upstream = upstream([anthropic_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        extra_headers: Some(object(json!({
            "Authorization": "Bearer unrelated",
            "X-Api-Key": "sk-caller"
        }))),
        ..request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .expect("call succeeds");

    let sent = only_request(&upstream).await;
    assert_eq!(sent.header_values("x-api-key"), ["sk-test"]);
    assert_eq!(sent.header_values("authorization"), ["Bearer unrelated"]);
}

#[rstest]
#[tokio::test]
async fn rejects_empty_messages_before_resolving_credentials() {
    let error = complete(ChatCompletionsRequest {
        api_key: None,
        ..request("claude-sonnet-4-5", Some("anthropic"), json!([]), json!({}))
    })
    .await
    .expect_err("empty messages are rejected");
    assert!(error.is_request(), "{error:?}");
}

#[rstest]
#[tokio::test]
async fn rejects_an_unknown_provider() {
    assert_eq!(
        complete(request(
            "openai/gpt-4o",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        ))
        .await
        .unwrap_err(),
        Error::InvalidProvider("openai".into())
    );
}

#[rstest]
#[tokio::test]
async fn rejects_a_model_with_no_resolvable_provider() {
    let error = complete(request(
        "claude-sonnet-4-5",
        None,
        json!([{"role": "user", "content": "hi"}]),
        json!({}),
    ))
    .await
    .unwrap_err();
    assert!(error.is_request(), "{error:?}");
    let report = litellm_error::Report::new(&error).to_string();
    assert!(
        report.contains("unable to resolve custom_llm_provider"),
        "{report}"
    );
}

#[rstest]
#[case::empty(json!([]), "chat completions requires at least one message")]
#[case::malformed(json!("not a list"), "invalid chat completions messages")]
#[tokio::test]
async fn rejects_an_empty_or_malformed_message_list(#[case] messages: Value, #[case] reason: &str) {
    let error = complete(request(
        "anthropic/claude-sonnet-4-5",
        None,
        messages,
        json!({}),
    ))
    .await
    .unwrap_err();
    assert!(error.is_request(), "{error:?}");
    let report = litellm_error::Report::new(&error).to_string();
    assert!(report.contains(reason), "{report}");
}

#[rstest]
#[tokio::test]
async fn rejects_non_string_extra_headers() {
    assert_eq!(
        complete(ChatCompletionsRequest {
            extra_headers: Some(Map::from_iter([("x-trace".to_string(), json!(7))])),
            ..request(
                "anthropic/claude-sonnet-4-5",
                None,
                json!([{"role": "user", "content": "hi"}]),
                json!({}),
            )
        })
        .await
        .unwrap_err(),
        Error::Headers(litellm_http::request::HeaderError {
            context: "chat completions",
            name: "x-trace".to_string(),
            actual: "number",
        })
    );
}

#[rstest]
#[tokio::test]
async fn a_forwarded_client_header_does_not_enter_the_bedrock_signature() {
    let upstream = upstream([bedrock_response()]).await;
    let base = upstream.uri();
    let mut call = request(
        "bedrock/us-east-1/anthropic.claude-v2",
        None,
        json!([{"role": "user", "content": "hi"}]),
        json!({
            "maxTokens": 16,
            "aws_access_key_id": "AKIDEXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
        }),
    );
    call.api_key = None;
    call.api_base = Some(&base);
    call.extra_headers = Some(Map::from_iter([(
        "x-request-id".to_string(),
        json!("abc-123"),
    )]));
    complete(call).await.expect("call succeeds");

    let sent = only_request(&upstream).await;
    let authorization = sent.header("authorization").expect("signed request");
    assert!(authorization.starts_with("AWS4-HMAC-SHA256"));
    assert!(!authorization.contains("x-request-id"));
    assert_eq!(sent.header("x-request-id"), Some("abc-123"));
    assert_eq!(sent.url.path(), "/model/anthropic.claude-v2/converse");
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["inferenceConfig"], json!({"maxTokens": 16}));
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": [{"text": "hi"}]}])
    );
}

#[rstest]
#[case::authorization("Authorization", Some("forged"), false, None)]
#[case::amz_date("x-amz-date", None, false, Some("x-amz-date"))]
#[case::security_token(
    "x-amz-security-token",
    None,
    true,
    Some("x-amz-date;x-amz-security-token")
)]
#[case::date("Date", None, false, Some("content-type;date;host"))]
#[tokio::test]
async fn a_forwarded_header_the_signer_computes_is_sent_as_python_sends_it(
    #[case] forwarded: &str,
    #[case] authorization: Option<&str>,
    #[case] kept: bool,
    #[case] signed_headers: Option<&str>,
) {
    let upstream = upstream([bedrock_response()]).await;
    let base = upstream.uri();
    let call = ChatCompletionsRequest {
        api_key: None,
        api_base: Some(&base),
        extra_headers: Some(Map::from_iter([(forwarded.to_string(), json!("forged"))])),
        ..request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({
                "maxTokens": 16,
                "aws_access_key_id": "AKIDEXAMPLE",
                "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
            }),
        )
    };
    complete(call).await.expect("call succeeds");

    let sent = only_request(&upstream).await;
    let signature = sent.header("authorization").expect("an authorization");
    match authorization {
        Some(expected) => assert_eq!(signature, expected),
        None => {
            assert!(signature.starts_with("AWS4-HMAC-SHA256"));
            assert_eq!(sent.header(forwarded) == Some("forged"), kept);
            assert!(signed_headers.is_some_and(|names| signature.contains(names)));
        }
    }
    let dated_by_date = forwarded == "Date";
    assert_eq!(
        (
            sent.header("x-amz-date")
                .is_some_and(|date| date != "forged"),
            sent.header("date").is_some_and(|date| date != "forged"),
        ),
        (!dated_by_date, dated_by_date)
    );
}

#[rstest]
#[tokio::test]
async fn a_bedrock_deployment_bearer_outranks_a_forwarded_authorization() {
    let upstream = upstream([bedrock_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        extra_headers: Some(Map::from_iter([(
            "Authorization".to_string(),
            json!("Bearer caller-supplied"),
        )])),
        ..request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"maxTokens": 16}),
        )
    })
    .await
    .expect("call succeeds");

    assert_eq!(
        only_request(&upstream).await.header_values("authorization"),
        ["Bearer sk-test"]
    );
}

#[rstest]
#[tokio::test]
async fn an_anthropic_forwarded_oauth_bearer_still_outranks_the_resolved_key() {
    let upstream = upstream([anthropic_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        extra_headers: Some(Map::from_iter([(
            "authorization".to_string(),
            json!("Bearer sk-ant-oat01-forwarded"),
        )])),
        ..request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .expect("call succeeds");

    let sent = only_request(&upstream).await;
    assert!(sent.header_values("x-api-key").is_empty());
    assert_eq!(
        sent.header_values("authorization"),
        ["Bearer sk-ant-oat01-forwarded"]
    );
}

#[rstest]
#[tokio::test]
async fn a_bedrock_api_key_is_sent_as_a_bearer_token_instead_of_being_signed() {
    let upstream = upstream([bedrock_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        ..request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"maxTokens": 16}),
        )
    })
    .await
    .expect("call succeeds");

    assert_eq!(
        only_request(&upstream).await.header_values("authorization"),
        ["Bearer sk-test"]
    );
}

#[rstest]
#[case::openai_names(json!({"max_tokens": 16, "top_p": 0.5, "stop": "5"}))]
#[case::converse_names(json!({"maxTokens": 16, "topP": 0.5, "stopSequences": ["5"]}))]
#[tokio::test]
async fn an_openai_shaped_bedrock_call_sends_converse_inference_config(#[case] params: Value) {
    let upstream = upstream([bedrock_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        optional_params: openai(params),
        ..request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .expect("call succeeds");

    assert_eq!(
        only_request(&upstream).await.json()["inferenceConfig"],
        json!({"maxTokens": 16, "topP": 0.5, "stopSequences": ["5"]})
    );
}

#[rstest]
#[case::unknown_openai_param(openai(json!({"top_k": 40})))]
#[case::openai_name_in_the_provider_shape(OptionalParams::ProviderMapped(object(
    json!({"max_tokens": 16})
)))]
#[tokio::test]
async fn a_param_named_outside_its_shape_declines_before_the_call(
    #[case] optional_params: OptionalParams,
) {
    let error = complete(ChatCompletionsRequest {
        api_base: Some(UNREACHABLE_BASE),
        optional_params,
        ..request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .unwrap_err();

    assert_eq!(
        error.declined(),
        Some(&Decline::new("unrecognized request parameter"))
    );
}

#[rstest]
#[case::without_an_opt_in(&[])]
#[case::with_an_opt_in_for_another_param(&["aws_bedrock_runtime_endpoint"])]
#[tokio::test]
async fn a_client_side_credential_outside_the_opt_in_declines_before_the_call(
    #[case] allowed: &[&str],
) {
    let error = complete(ChatCompletionsRequest {
        api_base: Some(UNREACHABLE_BASE),
        optional_params: OptionalParams::OpenAi {
            params: object(json!({"aws_role_name": "arn:aws:iam::123456789012:role/caller"})),
            client_side: ClientSideCredentials::Only(
                allowed.iter().map(|name| name.to_string()).collect(),
            ),
            drop_params: DropParams::Off,
        },
        ..request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .unwrap_err();

    assert_eq!(
        error.declined(),
        Some(&Decline::new(
            "client-side credential needs general_settings.allow_client_side_credentials or the deployment's configurable_clientside_auth_params"
        ))
    );
}

#[rstest]
#[tokio::test]
async fn an_openai_shaped_anthropic_call_sends_stop_sequences() {
    let upstream = upstream([anthropic_response()]).await;
    let base = upstream.uri();
    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        optional_params: openai(json!({"max_tokens": 16, "stop": "5"})),
        ..request(
            "anthropic/claude-sonnet-4-5",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        )
    })
    .await
    .expect("call succeeds");

    assert_eq!(
        only_request(&upstream).await.json()["stop_sequences"],
        json!(["5"])
    );
}
