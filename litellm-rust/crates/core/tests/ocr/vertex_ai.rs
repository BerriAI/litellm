use litellm_auth::{InputSource, Sourced};
use litellm_core::ocr::arguments::is_supported_request;
use litellm_llms::base_llm::ocr::settings::OcrSettings;
use rstest::rstest;

use super::*;

#[tokio::test]
async fn mistral_is_served_at_the_resolved_project_and_location() {
    let upstream = upstream([json_response(json!({
        "pages": [{"index": 0, "markdown": "hello"}],
        "usage_info": {"pages_processed": 1}
    }))])
    .await;

    let response = perform(ocr_request(
        "vertex_ai/mistral-ocr-maas",
        &upstream.uri(),
        json!({
            "vertex_project": "project-1",
            "vertex_location": "europe-west4",
            "extract_footer": true
        }),
    ))
    .await
    .unwrap();

    assert_eq!(response.pages[0].markdown, "hello");
    let sent = only_request(&upstream).await;
    assert_eq!(
        sent.url.path(),
        "/v1/projects/project-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
    );
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
    assert_eq!(
        sent.json(),
        json!({
            "model": "mistral-ocr-maas",
            "document": {"type": "document_url", "document_url": INLINE_PDF},
            "extract_footer": true
        })
    );
}

#[tokio::test]
async fn configured_project_and_location_apply_when_the_call_sets_neither() {
    let upstream = upstream([pages_response()]).await;
    let client = ocr_client().with_settings(OcrSettings {
        vertex_project: Some("configured-project".into()),
        vertex_location: Some("europe-west4".into()),
        ..OcrSettings::default()
    });

    litellm_core::ocr::client::perform(
        &client,
        ocr_request("vertex_ai/mistral-ocr-maas", &upstream.uri(), json!({})),
    )
    .await
    .unwrap();

    assert_eq!(
        only_request(&upstream).await.url.path(),
        "/v1/projects/configured-project/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
    );
}

#[tokio::test]
async fn a_supplied_authorization_is_forwarded_without_a_static_token() {
    let upstream = upstream([pages_response()]).await;
    let request = with_headers(
        without_api_key(ocr_request(
            "vertex_ai/model",
            &upstream.uri(),
            json!({"vertex_project": "project-1"}),
        )),
        &[("authorization", "Bearer supplied")],
    );

    perform(request).await.unwrap();

    assert_eq!(
        only_request(&upstream).await.header_values("authorization"),
        ["Bearer supplied"]
    );
}

#[tokio::test]
async fn invalid_credentials_fail_before_sending() {
    let error = perform(ocr_request(
        "vertex_ai/model",
        UNREACHABLE_BASE,
        json!({"vertex_credentials": true}),
    ))
    .await
    .unwrap_err();

    assert!(error.to_string().contains("vertex_credentials"), "{error}");
}

#[rstest]
#[tokio::test]
async fn a_request_controlled_api_base_is_rejected_before_vertex_auth(
    #[values("vertex_ai/mistral-ocr-maas", "vertex_ai/deepseek-ocr-maas")] model: &str,
) {
    let mut request = ocr_request(
        model,
        "https://caller.example",
        json!({"vertex_project": "project-1"}),
    );
    request.credentials.api_base = Some(Sourced::new(
        "https://caller.example".into(),
        InputSource::Request,
    ));

    let error = perform(request).await.unwrap_err();

    assert!(
        error
            .to_string()
            .contains("request-controlled Vertex AI endpoint"),
        "{error}"
    );
}

#[tokio::test]
async fn deepseek_is_served_at_the_openai_compatible_endpoint() {
    let upstream = upstream([json_response(json!({
        "choices": [{"message": {"content": "recognized"}}],
        "usage": {"prompt_tokens": 1}
    }))])
    .await;
    let request = with_source(
        ocr_request(
            "vertex_ai/deepseek-ocr-maas",
            &upstream.uri(),
            json!({
                "vertex_project": "project-1",
                "vertex_location": "europe-west4",
                "temperature": 0.1,
                "future_ocr_option": true,
                "extra_body": {"provider_option": "value"}
            }),
        ),
        "gs://bucket/document.pdf",
    );

    let response = perform(request).await.unwrap();

    assert_eq!(response.pages[0].markdown, "recognized");
    assert_eq!(
        response.usage_info.unwrap().extra_fields["prompt_tokens"],
        1
    );
    let sent = only_request(&upstream).await;
    assert_eq!(
        sent.url.path(),
        "/v1/projects/project-1/locations/europe-west4/endpoints/openapi/chat/completions"
    );
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
    let body = sent.json();
    assert_eq!(body["model"], "deepseek-ai/deepseek-ocr-maas");
    assert_eq!(body["temperature"], 0.1);
    assert_eq!(body["future_ocr_option"], true);
    assert_eq!(body["provider_option"], "value");
    assert!(body.get("vertex_project").is_none());
    assert!(body.get("extra_body").is_none());
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type": "image_url", "image_url": "gs://bucket/document.pdf"})
    );
}

#[rstest]
#[case::deepseek("deepseek-ocr-maas", Some("vertex_ai"), true)]
#[case::mistral("mistral-ocr-maas", Some("vertex_ai"), true)]
#[case::prefixed("vertex_ai/mistral-ocr-maas", None, true)]
#[case::unknown_provider("model", Some("unknown"), false)]
fn supported_requests_follow_the_registered_configs(
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] supported: bool,
) {
    assert_eq!(is_supported_request(model, provider), supported);
}
