mod support;

use std::time::Duration;

use litellm_core::{
    Phase, RouteError,
    responses::{
        responses,
        types::{ResponsesBody, ResponsesRequest},
    },
};
use litellm_http::transport::Error as TransportError;
use rstest::{fixture, rstest};
use serde_json::json;
use support::{ReceivedRequest, RecordingSecrets};
use wiremock::ResponseTemplate;

#[fixture]
fn request() -> ResponsesRequest<'static> {
    ResponsesRequest {
        model: "openai/test-model",
        body: json!({"input": "hello"}).as_object().unwrap().clone(),
        api_key: Some("configured-key"),
        api_base: Some(support::UNREACHABLE_BASE),
        custom_llm_provider: None,
        extra_headers: None,
        timeout: Some(Duration::from_secs(5)),
    }
}

#[rstest]
#[case::prefixed("openai/test-model", None)]
#[case::bare("test-model", None)]
#[case::explicit("test-model", Some("openai"))]
#[tokio::test]
async fn sends_resolved_model_and_deployment_credentials(
    request: ResponsesRequest<'_>,
    #[case] model: &str,
    #[case] provider: Option<&str>,
) {
    let expected = json!({"id": "resp_test", "object": "response", "output": []});
    let upstream = support::upstream([support::json_response(expected.clone())]).await;
    let base = upstream.uri();
    let result = responses(
        &support::resources(),
        &support::http_config(),
        &RecordingSecrets::failing(),
        ResponsesRequest {
            model,
            custom_llm_provider: provider,
            api_base: Some(&base),
            extra_headers: Some(
                json!({"Authorization": "Bearer forwarded-key"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
            ..request
        },
    )
    .await
    .unwrap();
    let ResponsesBody::Response(body) = result.body else {
        panic!("expected JSON")
    };
    assert_eq!(serde_json::to_value(body).unwrap(), expected);
    let sent = support::only_request(&upstream).await;
    assert_eq!(
        sent.json(),
        json!({"model": "test-model", "input": "hello"})
    );
    assert_eq!(
        sent.header_values("authorization"),
        ["Bearer configured-key"]
    );
    assert_eq!(sent.header("content-type"), Some("application/json"));
}

#[rstest]
#[case::preferred_base("OPENAI_BASE_URL")]
#[case::legacy_base("OPENAI_API_BASE")]
#[tokio::test]
async fn resolves_credentials_and_base_from_injected_secrets(
    request: ResponsesRequest<'_>,
    #[case] name: &str,
) {
    let upstream = support::upstream([support::json_response(
        json!({"id": "resp_test", "object": "response", "output": []}),
    )])
    .await;
    let base = format!("{}/v1", upstream.uri());
    let secrets = RecordingSecrets::new([("OPENAI_API_KEY", "secret-key"), (name, base.as_str())]);
    responses(
        &support::resources(),
        &support::http_config(),
        &secrets,
        ResponsesRequest {
            api_key: None,
            api_base: None,
            ..request
        },
    )
    .await
    .unwrap();
    let sent = support::only_request(&upstream).await;
    assert_eq!(sent.target(), "/v1/responses");
    assert_eq!(sent.header("authorization"), Some("Bearer secret-key"));
}

#[rstest]
#[tokio::test]
async fn forwarded_bearer_does_not_require_a_secret_lookup(request: ResponsesRequest<'_>) {
    let upstream = support::upstream([support::json_response(
        json!({"id": "resp_test", "object": "response", "output": []}),
    )])
    .await;
    let base = upstream.uri();
    responses(
        &support::resources(),
        &support::http_config(),
        &RecordingSecrets::failing(),
        ResponsesRequest {
            api_key: None,
            api_base: Some(&base),
            extra_headers: Some(
                json!({"authorization": "Bearer forwarded-key"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
            ..request
        },
    )
    .await
    .unwrap();
    assert_eq!(
        support::only_request(&upstream)
            .await
            .header("authorization"),
        Some("Bearer forwarded-key")
    );
}

#[rstest]
#[tokio::test]
async fn missing_key_fails_before_network(request: ResponsesRequest<'_>) {
    let result = responses(
        &support::resources(),
        &support::http_config(),
        &RecordingSecrets::empty(),
        ResponsesRequest {
            api_key: None,
            ..request
        },
    )
    .await;
    assert!(matches!(
        result,
        Err(RouteError::Auth(litellm_auth::Error::MissingApiKey { .. }))
    ));
}

#[rstest]
#[case::prefixed("anthropic/test-model", None)]
#[case::explicit("test-model", Some("azure"))]
#[tokio::test]
async fn unsupported_provider_never_resolves_credentials(
    request: ResponsesRequest<'_>,
    #[case] model: &str,
    #[case] provider: Option<&str>,
) {
    let result = responses(
        &support::resources(),
        &support::http_config(),
        &RecordingSecrets::failing(),
        ResponsesRequest {
            model,
            custom_llm_provider: provider,
            api_key: None,
            ..request
        },
    )
    .await;
    assert!(matches!(result, Err(RouteError::Unsupported(_))));
}

#[rstest]
#[case::bad_json(false, "text/html", "not json")]
#[case::wrong_stream_content_type(true, "application/json", "{}")]
#[tokio::test]
async fn invalid_upstream_response_is_after_send(
    request: ResponsesRequest<'_>,
    #[case] streaming: bool,
    #[case] content_type: &str,
    #[case] body: &str,
) {
    let upstream =
        support::upstream([ResponseTemplate::new(200).set_body_raw(body, content_type)]).await;
    let base = upstream.uri();
    let result = responses(
        &support::resources(),
        &support::http_config(),
        &RecordingSecrets::empty(),
        ResponsesRequest {
            api_base: Some(&base),
            body: json!({"input": "hi", "stream": streaming})
                .as_object()
                .unwrap()
                .clone(),
            ..request
        },
    )
    .await;
    let error = result.err().expect("upstream response must be rejected");
    assert_eq!(error.phase(), Phase::AfterSend);
    assert!(matches!(error, RouteError::InvalidResponse(_)));
}

#[rstest]
#[tokio::test]
async fn deployment_timeout_is_applied(request: ResponsesRequest<'_>) {
    let upstream =
        support::upstream([ResponseTemplate::new(200).set_delay(Duration::from_secs(1))]).await;
    let base = upstream.uri();
    let result = responses(
        &support::resources(),
        &support::http_config(),
        &RecordingSecrets::empty(),
        ResponsesRequest {
            api_base: Some(&base),
            timeout: Some(Duration::from_millis(20)),
            ..request
        },
    )
    .await;
    assert!(matches!(
        result,
        Err(RouteError::Transport(TransportError::Network(_)))
    ));
}
