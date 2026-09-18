use litellm_llms::base_llm::ocr::error::Error;
use serde_json::{Value, json};

use super::test_support::{MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request};
use crate::ocr::route::LocalOcrHost;

#[tokio::test]
async fn facade_executes_azure_mistral_with_prepared_auth() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"hello"}],
        "usage_info":{"pages_processed":1}
    }))])
    .await;
    let mut request = wire_request(
        "azure_ai/model",
        &base,
        json!({"include_image_base64":true}),
    );
    request.credentials.api_key = None;
    request.transport.extra_headers = vec![(
        "Authorization".into(),
        "Bearer python-prepared-token".into(),
    )];

    let result = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(result.pages[0].markdown, "hello");
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("POST /providers/mistral/azure/ocr "));
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer python-prepared-token\r\n")
    );
    let body: Value = serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        json!({
            "model":"model",
            "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
            "include_image_base64":true
        })
    );
}

#[tokio::test]
async fn facade_acquires_supplied_entra_token_for_final_request() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let mut request = wire_request(
        "azure_ai/model",
        &base,
        json!({"azure_ad_token":"rust-owned-token"}),
    );
    request.credentials.api_key = None;

    perform_ocr(request).await.unwrap();
    server.await.unwrap();

    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer rust-owned-token\r\n")
    );
}

#[tokio::test]
async fn rejects_non_inline_body_after_guardrails() {
    let request = wire_request("azure_ai/model", "http://127.0.0.1:1", json!({}));
    let host = LocalOcrHost::new(request).with_before_send(|mut wire, _| {
        wire.body["document"] = json!({
            "type":"document_url",
            "document_url":"https://example.com/not-inline.pdf"
        });
        Ok(wire)
    });
    let error = perform_ocr_with(host).await.unwrap_err();
    assert!(error.to_string().contains("data URI"));
}

mod transformation {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use litellm_auth::{
        ResolvedCredential, SecretValue, TokenFuture, TokenProvider, TokenProviderHandle,
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::ocr::{
        test_support::{MockResponse, header, mock_server, perform_ocr},
        types::LiteLLMOcrRequest,
        wire::decode_request,
    };

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
            "document": {"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
            "api_key": api_key,
            "api_base": api_base,
            "custom_llm_provider": null,
            "extra_headers": extra_headers,
            "optional_params": optional_params,
            "timeout_seconds": 2.0
        }))
        .unwrap();
        LiteLLMOcrRequest {
            azure_ad_token_provider: Some(TokenProviderHandle::new(provider.clone())),
            ..decode_request(wire).unwrap()
        }
    }

    fn ocr_page() -> MockResponse {
        MockResponse::json(json!({"pages":[{"index":0,"markdown":"hello"}]}))
    }

    #[tokio::test]
    async fn token_provider_result_is_the_bearer_and_is_acquired_for_each_request() {
        let provider = CountingToken::new(numbered_token);
        let (base, seen, server) = mock_server(vec![ocr_page(), ocr_page()]).await;

        for _ in 0..2 {
            perform_ocr(azure_request(
                &provider,
                Some(&base),
                None,
                Value::Null,
                json!({}),
            ))
            .await
            .unwrap();
        }
        server.await.unwrap();

        assert_eq!(provider.calls(), 2);
        let requests = seen.lock().unwrap();
        assert_eq!(
            requests
                .iter()
                .map(|request| header(request, "authorization"))
                .collect::<Vec<_>>(),
            [Some("Bearer callback-1"), Some("Bearer callback-2")]
        );
    }

    #[rstest]
    #[case::api_key_skips_provider(Some("resource-key"), Value::Null, json!({}), "Bearer resource-key", 0)]
    #[case::provider_beats_static_token(
        None,
        Value::Null,
        json!({"azure_ad_token":"static-token"}),
        "Bearer callback-1",
        1
    )]
    #[case::header_wins_on_the_wire_but_provider_still_runs(
        None,
        json!({"Authorization":"Bearer override"}),
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
        let (base, seen, server) = mock_server(vec![ocr_page()]).await;

        perform_ocr(azure_request(
            &provider,
            Some(&base),
            api_key,
            extra_headers,
            optional_params,
        ))
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(provider.calls(), expected_calls);
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            header(&requests[0], "authorization"),
            Some(expected_authorization)
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
        json!({"azure_ad_token":"oidc/assertion","client_id":"client","tenant_id":"tenant"}),
        numbered_token,
        |error: &Error| matches!(error, Error::Auth(litellm_auth::Error::UnsupportedOidcReference)),
        0
    )]
    #[case::empty_provider_token_ignores_static_token(
        true,
        json!({"azure_ad_token":"static-token"}),
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
        let (base, seen, server) = mock_server(vec![ocr_page()]).await;

        let error = perform_ocr(azure_request(
            &provider,
            with_api_base.then_some(base.as_str()),
            None,
            Value::Null,
            optional_params,
        ))
        .await
        .unwrap_err();
        server.abort();

        assert!(expected(&error), "unexpected error: {error:?}");
        assert_eq!(provider.calls(), expected_calls);
        assert!(seen.lock().unwrap().is_empty());
    }
}
