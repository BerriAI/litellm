use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use guardrails::{
    GenericApiConfig, Guardrail, GuardrailInput, InputType, RequestContext, UnreachableFallback,
    Verdict,
};
use guardrails::HttpClient;
use guardrails::providers::GenericGuardrailApi;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_input(text: &str) -> GuardrailInput {
    GuardrailInput {
        texts: vec![text.to_owned()],
        ..Default::default()
    }
}

fn make_config(server_uri: &str) -> GenericApiConfig {
    GenericApiConfig {
        api_base: server_uri.to_owned(),
        api_key: None,
        headers: None,
        additional_provider_specific_params: Default::default(),
        unreachable_fallback: None,
    }
}

#[tokio::test]
async fn pass_verdict() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        make_config(&server.uri()),
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(
            &make_input("hello"),
            InputType::Request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

    assert!(matches!(result.verdict, Verdict::Pass));
}

#[tokio::test]
async fn block_verdict() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "BLOCKED",
            "blocked_reason": "harmful content detected"
        })))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        make_config(&server.uri()),
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(
            &make_input("bad stuff"),
            InputType::Request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

    match result.verdict {
        Verdict::Block {
            violation_message, ..
        } => {
            assert_eq!(violation_message, "harmful content detected");
        }
        _ => panic!("expected Block verdict"),
    }
}

#[tokio::test]
async fn mask_verdict() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "GUARDRAIL_INTERVENED",
            "texts": ["my email is [REDACTED]"]
        })))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        make_config(&server.uri()),
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(
            &make_input("my email is foo@bar.com"),
            InputType::Request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

    match result.verdict {
        Verdict::Mask { texts, .. } => {
            assert_eq!(texts, vec!["my email is [REDACTED]"]);
        }
        _ => panic!("expected Mask verdict"),
    }
}

#[tokio::test]
async fn upstream_503_fail_closed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            unreachable_fallback: Some(UnreachableFallback::FailClosed),
            ..make_config(&server.uri())
        },
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(
            &make_input("test"),
            InputType::Request,
            &RequestContext::default(),
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn upstream_503_fail_open() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            unreachable_fallback: Some(UnreachableFallback::FailOpen),
            ..make_config(&server.uri())
        },
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(
            &make_input("test"),
            InputType::Request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

    assert!(matches!(result.verdict, Verdict::Pass));
}

#[tokio::test]
async fn upstream_401_always_fails() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            unreachable_fallback: Some(UnreachableFallback::FailOpen),
            ..make_config(&server.uri())
        },
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(
            &make_input("test"),
            InputType::Request,
            &RequestContext::default(),
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn api_key_sent_as_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .and(wiremock::matchers::header("x-api-key", "secret123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            api_key: Some("secret123".to_owned()),
            ..make_config(&server.uri())
        },
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(
            &make_input("hi"),
            InputType::Request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

    assert!(matches!(result.verdict, Verdict::Pass));
}

#[tokio::test]
async fn request_body_includes_user_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "user_api_key_user_id".to_owned(),
        serde_json::Value::String("user-42".to_owned()),
    );

    let ctx = RequestContext {
        user_api_key_metadata: metadata,
        ..Default::default()
    };

    let provider = GenericGuardrailApi::new(
        make_config(&server.uri()),
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    provider
        .apply(&make_input("hello"), InputType::Request, &ctx)
        .await
        .unwrap();

    let received = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    let rd = body.get("request_data").unwrap().as_object().unwrap();
    assert_eq!(rd.get("user_api_key_user_id").unwrap(), "user-42");
}

#[tokio::test]
async fn request_body_includes_sanitized_headers() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let raw_headers = HashMap::from([
        ("Host".to_owned(), "example.com".to_owned()),
        ("Authorization".to_owned(), "Bearer secret".to_owned()),
    ]);

    let ctx = RequestContext {
        request_headers: Some(raw_headers),
        ..Default::default()
    };

    let provider = GenericGuardrailApi::new(
        make_config(&server.uri()),
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(&make_input("hello"), InputType::Request, &ctx)
        .await
        .unwrap();

    assert!(matches!(result.verdict, Verdict::Pass));

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    let hdrs = body.get("request_headers").unwrap().as_object().unwrap();
    assert_eq!(hdrs.get("Host").unwrap(), "example.com");
    assert_eq!(hdrs.get("Authorization").unwrap(), "[present]");
}

#[tokio::test]
async fn request_body_includes_litellm_version() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let ctx = RequestContext {
        litellm_version: Some("1.55.0".to_owned()),
        ..Default::default()
    };

    let provider = GenericGuardrailApi::new(
        make_config(&server.uri()),
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    provider
        .apply(&make_input("hello"), InputType::Request, &ctx)
        .await
        .unwrap();

    let received = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body.get("litellm_version").unwrap(), "1.55.0");
}

#[tokio::test]
async fn dynamic_params_merged_with_static() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut static_params = serde_json::Map::new();
    static_params.insert(
        "threshold".to_owned(),
        serde_json::Value::Number(serde_json::Number::from(5)),
    );

    let mut dynamic_params = serde_json::Map::new();
    dynamic_params.insert(
        "session_id".to_owned(),
        serde_json::Value::String("sess-1".to_owned()),
    );

    let ctx = RequestContext {
        dynamic_params,
        ..Default::default()
    };

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            additional_provider_specific_params: static_params,
            ..make_config(&server.uri())
        },
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    provider
        .apply(&make_input("hello"), InputType::Request, &ctx)
        .await
        .unwrap();

    let received = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body.get("threshold").unwrap(), 5);
    assert_eq!(body.get("session_id").unwrap(), "sess-1");
}

#[tokio::test]
async fn url_normalization_appends_endpoint_path() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/beta/litellm_basic_guardrail_api"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        make_config(&server.uri()),
        Arc::new(HttpClient::new(Duration::from_secs(5))),
    );

    let result = provider
        .apply(
            &make_input("hello"),
            InputType::Request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

    assert!(matches!(result.verdict, Verdict::Pass));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}
