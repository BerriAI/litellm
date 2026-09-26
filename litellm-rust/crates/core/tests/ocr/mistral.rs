use std::sync::Arc;

use litellm_http::{HttpSettings, Resolution, media::UrlPolicy};
use litellm_llms::{
    base_llm::ocr::{
        settings::OcrSettings,
        transformation::{BaseOcrConfig, OCR_RESPONSE_MAX_BYTES},
    },
    mistral::ocr::transformation::MistralOcrConfig,
};
use rstest::rstest;

use super::*;

#[tokio::test]
async fn direct_mistral_sends_one_request_with_every_option() {
    let upstream = upstream([json_response(json!({
        "pages": [{"index": 0, "markdown": "hello", "custom": "preserved"}],
        "usage_info": {"pages_processed": 1}
    }))])
    .await;

    let result = perform(ocr_request(
        "mistral/model",
        &upstream.uri(),
        json!({"pages": "0,2-4", "extract_header": true, "unknown": "ignored"}),
    ))
    .await
    .unwrap();

    assert_eq!(result.pages[0].markdown, "hello");
    assert_eq!(result.pages[0].extra_fields["custom"], "preserved");
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/v1/ocr");
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
    assert_eq!(
        sent.json(),
        json!({
            "model": "model",
            "document": {"type": "document_url", "document_url": INLINE_PDF},
            "pages": "0,2-4",
            "extract_header": true,
            "unknown": "ignored"
        })
    );
}

#[rstest]
#[case::litellm_format(json!({}), false)]
#[case::native_format(json!({"req_format": "native"}), true)]
#[tokio::test]
async fn the_native_response_is_kept_only_when_requested(
    #[case] options: Value,
    #[case] kept: bool,
) {
    let provider_response = json!({
        "pages": [{"index": 0, "markdown": "hello"}],
        "usage_info": {"pages_processed": 1},
        "provider_only": "preserved"
    });
    let upstream = upstream([json_response(provider_response.clone())]).await;

    let response = perform(ocr_request("mistral/model", &upstream.uri(), options))
        .await
        .unwrap();

    assert_eq!(
        response.provider_native_response.map(Value::Object),
        kept.then_some(provider_response)
    );
}

#[rstest]
#[case::mistral("mistral/model", json!({}))]
#[case::vertex(
    "vertex_ai/mistral-ocr-latest",
    json!({"vertex_project": "test-project", "vertex_location": "us-central1"})
)]
#[tokio::test]
async fn an_upstream_error_keeps_its_status_whole_body_and_headers(
    #[case] model: &str,
    #[case] options: Value,
) {
    let payload = json!({"message": format!("{} END-OF-PROVIDER-BODY", "x".repeat(4096))});
    let expected_body = serde_json::to_string(&payload).unwrap();
    let upstream = upstream([status_response(422, payload)
        .insert_header("Retry-After", "17")
        .insert_header("X-Request-ID", "request-123")
        .insert_header("X-Future-Header", "retained")])
    .await;

    let error = perform(ocr_request(model, &upstream.uri(), options))
        .await
        .unwrap_err();

    assert_eq!(received(&upstream).await.len(), 1);
    let Error::Provider {
        status,
        body,
        headers,
    } = error
    else {
        panic!("expected provider error, got {error:?}");
    };
    assert_eq!(status, 422);
    for (name, value) in [
        ("retry-after", "17"),
        ("x-request-id", "request-123"),
        ("x-future-header", "retained"),
    ] {
        assert!(
            headers
                .iter()
                .any(|(key, actual)| key.eq_ignore_ascii_case(name) && actual == value),
            "{name} missing from {headers:?}"
        );
    }
    assert_eq!(body, expected_body);
}

#[rstest]
#[case::mistral_prefix("mistral/model", None, true)]
#[case::unknown_provider("model", Some("unknown"), false)]
fn decoding_accepts_known_providers_and_rejects_unknown_ones(
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] accepted: bool,
) {
    let request = OcrWireRequest {
        custom_llm_provider: provider.map(Into::into),
        ..wire(
            model,
            "https://example.com",
            json!({"type": "document_url", "document_url": "https://example.com/doc.pdf"}),
            json!({"extract_header": true, "unknown": 42}),
        )
    };

    assert_eq!(decode_request(request).is_ok(), accepted);
}

#[rstest]
#[case::plain_key(&[("MISTRAL_API_KEY", "plain")], "plain")]
#[case::azure_key_wins(&[("MISTRAL_AZURE_API_KEY", "azure"), ("MISTRAL_API_KEY", "plain")], "azure")]
#[case::empty_azure_key_falls_through(&[("MISTRAL_AZURE_API_KEY", ""), ("MISTRAL_API_KEY", "plain")], "plain")]
#[tokio::test]
async fn missing_credentials_come_from_the_injected_secret_source(
    #[case] secrets: &[(&str, &str)],
    #[case] expected_key: &str,
) {
    let upstream = upstream([pages_response()]).await;
    let base = upstream.uri();
    let source = Arc::new(RecordingSecrets::new(
        secrets
            .iter()
            .copied()
            .chain([("MISTRAL_AZURE_API_BASE", base.as_str())]),
    ));
    let client = ocr_client().with_secrets(source.clone());
    let request = decode_request(OcrWireRequest {
        api_key: None,
        api_base: None,
        ..wire(
            "mistral/model",
            &base,
            json!({"type": "document_url", "document_url": INLINE_PDF}),
            json!({}),
        )
    })
    .unwrap();

    litellm_core::ocr::client::perform(&client, request)
        .await
        .unwrap();

    assert_eq!(source.requested(), MistralOcrConfig.secret_names());
    assert_eq!(
        only_request(&upstream).await.header("authorization"),
        Some(format!("Bearer {expected_key}").as_str())
    );
}

#[rstest]
#[tokio::test]
async fn the_client_uses_the_injected_http_pool_configuration() {
    let upstream = upstream([pages_response()]).await;
    let settings = HttpSettings {
        user_agent: Some("host-owned/1".into()),
        ..HttpSettings::default()
    };
    let client = resources()
        .ocr_client(
            &Resolution::from(&settings).config,
            UrlPolicy::default(),
            OcrSettings::default(),
            Arc::new(
                litellm_secrets::source::EnvironmentSecrets::python_compatible(
                    litellm_http::Client::plain_for_test(),
                ),
            ),
        )
        .unwrap();

    litellm_core::ocr::client::perform(
        &client,
        ocr_request("mistral/model", &upstream.uri(), json!({})),
    )
    .await
    .unwrap();

    assert_eq!(
        only_request(&upstream).await.header("user-agent"),
        Some("host-owned/1")
    );
}

#[test]
fn a_valid_response_limit_is_consumed_and_not_forwarded() {
    let request = ocr_request(
        "mistral/model",
        UNREACHABLE_BASE,
        json!({"max_response_bytes": 123}),
    );

    assert_eq!(request.transport.max_response_bytes, 123);
    assert!(!request.optional_params.contains_key("max_response_bytes"));
}

#[rstest]
#[case::zero(json!(0))]
#[case::negative(json!(-1))]
#[case::boolean(json!(true))]
#[case::string(json!("123"))]
#[case::fraction(json!(1.5))]
#[case::above_the_cap(json!(OCR_RESPONSE_MAX_BYTES + 1))]
#[case::null(Value::Null)]
fn an_invalid_response_limit_is_rejected(#[case] limit: Value) {
    let Err(error) = decode_request(wire(
        "mistral/model",
        UNREACHABLE_BASE,
        json!({"type": "document_url", "document_url": INLINE_PDF}),
        json!({"max_response_bytes": limit}),
    )) else {
        panic!("invalid response limit {limit} accepted");
    };

    assert!(error.to_string().contains("max_response_bytes"), "{error}");
}
