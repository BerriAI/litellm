use std::sync::Arc;
use std::time::Duration;

use guardrails_core::{
    GenericApiConfig, Guardrail, GuardrailInput, InputType, RequestContext, UnreachableFallback,
    Verdict,
};
use guardrails_http::HttpClient;
use guardrails_providers::GenericGuardrailApi;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_input(text: &str) -> GuardrailInput {
    GuardrailInput {
        texts: vec![text.to_owned()],
        ..Default::default()
    }
}

#[tokio::test]
async fn pass_verdict() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            api_base: server.uri(),
            api_key: None,
            headers: None,
            additional_provider_specific_params: Default::default(),
            unreachable_fallback: None,
        },
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
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "BLOCKED",
            "blocked_reason": "harmful content detected"
        })))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            api_base: server.uri(),
            api_key: None,
            headers: None,
            additional_provider_specific_params: Default::default(),
            unreachable_fallback: None,
        },
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
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "GUARDRAIL_INTERVENED",
            "texts": ["my email is [REDACTED]"]
        })))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            api_base: server.uri(),
            api_key: None,
            headers: None,
            additional_provider_specific_params: Default::default(),
            unreachable_fallback: None,
        },
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
        .and(path("/"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            api_base: server.uri(),
            api_key: None,
            headers: None,
            additional_provider_specific_params: Default::default(),
            unreachable_fallback: Some(UnreachableFallback::FailClosed),
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
        .and(path("/"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            api_base: server.uri(),
            api_key: None,
            headers: None,
            additional_provider_specific_params: Default::default(),
            unreachable_fallback: Some(UnreachableFallback::FailOpen),
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
        .and(path("/"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            api_base: server.uri(),
            api_key: None,
            headers: None,
            additional_provider_specific_params: Default::default(),
            unreachable_fallback: Some(UnreachableFallback::FailOpen),
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
        .and(path("/"))
        .and(wiremock::matchers::header("x-api-key", "secret123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "NONE"
        })))
        .mount(&server)
        .await;

    let provider = GenericGuardrailApi::new(
        GenericApiConfig {
            api_base: server.uri(),
            api_key: Some("secret123".to_owned()),
            headers: None,
            additional_provider_specific_params: Default::default(),
            unreachable_fallback: None,
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
