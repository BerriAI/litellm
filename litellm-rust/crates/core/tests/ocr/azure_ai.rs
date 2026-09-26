use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use litellm_auth::{
    ResolvedCredential, SecretValue, TokenFuture, TokenProvider, TokenProviderHandle,
};
use rstest::rstest;

use super::*;

#[derive(Debug)]
struct CountingToken {
    token: fn(usize) -> String,
    calls: AtomicUsize,
}

impl CountingToken {
    fn new(token: fn(usize) -> String) -> Arc<Self> {
        Arc::new(Self {
            token,
            calls: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl TokenProvider for CountingToken {
    fn acquire(&self) -> TokenFuture<'_> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let token = SecretValue::new((self.token)(call));
        Box::pin(async move {
            Ok(ResolvedCredential::AccessToken {
                token,
                expires_on: None,
            })
        })
    }
}

fn numbered_token(call: usize) -> String {
    format!("callback-{call}")
}

fn azure_request(
    provider: &Arc<CountingToken>,
    api_base: Option<&str>,
    api_key: Option<&str>,
    extra_headers: Value,
    optional_params: Value,
) -> LiteLLMOcrRequest {
    let wire = serde_json::from_value(json!({
        "model": "azure_ai/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": INLINE_PDF},
        "api_key": api_key,
        "api_base": api_base,
        "custom_llm_provider": null,
        "extra_headers": extra_headers,
        "optional_params": optional_params,
        "timeout_seconds": 2.0
    }))
    .unwrap();
    let mut request = decode_request(wire).unwrap();
    request.azure_ad_token_provider = Some(TokenProviderHandle::new(provider.clone()));
    request
}

fn ocr_page() -> ResponseTemplate {
    json_response(json!({"pages": [{"index": 0, "markdown": "hello"}]}))
}

#[tokio::test]
async fn mistral_on_azure_sends_the_prepared_bearer_and_the_mistral_body() {
    let upstream = upstream([json_response(json!({
        "pages": [{"index": 0, "markdown": "hello"}],
        "usage_info": {"pages_processed": 1}
    }))])
    .await;
    let request = with_headers(
        without_api_key(ocr_request(
            "azure_ai/model",
            &upstream.uri(),
            json!({"include_image_base64": true}),
        )),
        &[("Authorization", "Bearer python-prepared-token")],
    );

    let result = perform(request).await.unwrap();

    assert_eq!(result.pages[0].markdown, "hello");
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/providers/mistral/azure/ocr");
    assert_eq!(
        sent.header("authorization"),
        Some("Bearer python-prepared-token")
    );
    assert_eq!(
        sent.json(),
        json!({
            "model": "model",
            "document": {"type": "document_url", "document_url": INLINE_PDF},
            "include_image_base64": true
        })
    );
}

#[tokio::test]
async fn a_static_entra_token_becomes_the_bearer() {
    let upstream = upstream([pages_response()]).await;
    let request = without_api_key(ocr_request(
        "azure_ai/model",
        &upstream.uri(),
        json!({"azure_ad_token": "rust-owned-token"}),
    ));

    perform(request).await.unwrap();

    assert_eq!(
        only_request(&upstream).await.header("authorization"),
        Some("Bearer rust-owned-token")
    );
}

#[tokio::test]
async fn a_guardrail_that_swaps_in_a_remote_document_is_rejected() {
    let host = LocalOcrHost::new(ocr_request("azure_ai/model", UNREACHABLE_BASE, json!({})))
        .with_before_send(|mut wire, _| {
            wire.body["document"] = json!({
                "type": "document_url",
                "document_url": "https://example.com/not-inline.pdf"
            });
            Ok(wire)
        });

    let error = perform_with(host).await.unwrap_err();

    assert!(error.to_string().contains("data URI"), "{error}");
}

#[tokio::test]
async fn the_token_provider_is_the_bearer_and_is_acquired_for_each_request() {
    let provider = CountingToken::new(numbered_token);
    let upstream = upstream([ocr_page(), ocr_page()]).await;
    let base = upstream.uri();

    for _ in 0..2 {
        perform(azure_request(
            &provider,
            Some(&base),
            None,
            Value::Null,
            json!({}),
        ))
        .await
        .unwrap();
    }

    assert_eq!(provider.calls(), 2);
    let authorizations: Vec<String> = received(&upstream)
        .await
        .iter()
        .map(|request| {
            request
                .header("authorization")
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    assert_eq!(authorizations, ["Bearer callback-1", "Bearer callback-2"]);
}

#[rstest]
#[case::api_key_skips_provider(Some("resource-key"), Value::Null, json!({}), "Bearer resource-key", 0)]
#[case::provider_beats_static_token(
    None,
    Value::Null,
    json!({"azure_ad_token": "static-token"}),
    "Bearer callback-1",
    1
)]
#[case::header_wins_on_the_wire_but_provider_still_runs(
    None,
    json!({"Authorization": "Bearer override"}),
    json!({}),
    "Bearer override",
    1
)]
#[tokio::test]
async fn credential_precedence(
    #[case] api_key: Option<&str>,
    #[case] extra_headers: Value,
    #[case] optional_params: Value,
    #[case] expected_authorization: &str,
    #[case] expected_calls: usize,
) {
    let provider = CountingToken::new(numbered_token);
    let upstream = upstream([ocr_page()]).await;

    perform(azure_request(
        &provider,
        Some(&upstream.uri()),
        api_key,
        extra_headers,
        optional_params,
    ))
    .await
    .unwrap();

    assert_eq!(provider.calls(), expected_calls);
    assert_eq!(
        only_request(&upstream).await.header_values("authorization"),
        [expected_authorization]
    );
}

#[rstest]
#[case::missing_api_base(
    false,
    json!({}),
    numbered_token,
    |error: &Error| matches!(error, Error::Auth(litellm_auth::Error::MissingApiBase {
        provider: "Azure AI",
        environment_variable: "AZURE_AI_API_BASE",
    })),
    0
)]
#[case::unsupported_oidc_reference(
    true,
    json!({"azure_ad_token": "oidc/assertion", "client_id": "client", "tenant_id": "tenant"}),
    numbered_token,
    |error: &Error| matches!(error, Error::Auth(litellm_auth::Error::UnsupportedOidcReference)),
    0
)]
#[case::empty_provider_token_ignores_static_token(
    true,
    json!({"azure_ad_token": "static-token"}),
    |_| String::new(),
    |error: &Error| matches!(error, Error::MissingAzureAiCredentials),
    1
)]
#[tokio::test]
async fn credential_failures_send_no_provider_request(
    #[case] with_api_base: bool,
    #[case] optional_params: Value,
    #[case] token: fn(usize) -> String,
    #[case] expected: fn(&Error) -> bool,
    #[case] expected_calls: usize,
) {
    let provider = CountingToken::new(token);
    let upstream = upstream([ocr_page()]).await;
    let base = upstream.uri();

    let error = perform(azure_request(
        &provider,
        with_api_base.then_some(base.as_str()),
        None,
        Value::Null,
        optional_params,
    ))
    .await
    .unwrap_err();

    assert!(expected(&error), "unexpected error: {error:?}");
    assert_eq!(provider.calls(), expected_calls);
    assert!(received(&upstream).await.is_empty());
}
