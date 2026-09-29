use rstest::rstest;

use super::*;

#[rstest]
#[case::cohere("cohere/parse-v5.0", "/v2/parse")]
#[case::azure_ai("azure_ai/Cohere-parse-v5.0", "/providers/cohere/v2/parse")]
#[tokio::test]
async fn an_image_goes_to_the_parse_endpoint_with_the_bearer_key(
    #[case] model: &str,
    #[case] path: &str,
) {
    let upstream = upstream([pages_response()]).await;
    let request = ocr_request_with_document(
        model,
        &upstream.uri(),
        json!({"type": "image_url", "image_url": "data:image/png;base64,YWJj"}),
        json!({}),
    );

    perform(request).await.unwrap();

    let sent = only_request(&upstream).await;
    assert_eq!(sent.method.as_str(), "POST");
    assert_eq!(sent.url.path(), path);
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
}

#[rstest]
#[tokio::test]
async fn a_non_image_document_is_rejected_before_sending(
    #[values("cohere/parse-v5.0", "azure_ai/Cohere-parse-v5.0")] model: &str,
) {
    let upstream = upstream([pages_response()]).await;

    let error = perform(ocr_request(model, &upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert!(matches!(error, Error::CohereImageOnly), "{error:?}");
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[case::root("", "/v2/parse")]
#[case::version("/v2", "/v2/parse")]
#[case::complete("/v2/parse", "/v2/parse")]
#[case::gateway("/team%2Fname", "/team%2Fname/v2/parse")]
#[tokio::test]
async fn preparation_preserves_legacy_base_and_query(#[case] suffix: &str, #[case] expected: &str) {
    let upstream = upstream([pages_response()]).await;
    let base = format!("{}{suffix}?tenant=a&tenant=b&sig=a%2fb", upstream.uri());
    let request = ocr_request_with_document(
        "cohere/model",
        &base,
        json!({"type":"image_url","image_url":"data:image/png;base64,YWJj"}),
        json!({}),
    );
    perform(request).await.unwrap();
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), expected);
    assert_eq!(sent.url.query(), Some("tenant=a&tenant=b&sig=a%2fb"));
}

#[rstest]
#[tokio::test]
async fn a_hook_destination_is_validated_without_completing_it_again() {
    let upstream = upstream([pages_response()]).await;
    let destination = format!("{}/custom?sig=a%2fb", upstream.uri());
    let request = ocr_request_with_document(
        "cohere/model",
        &upstream.uri(),
        json!({"type":"image_url","image_url":"data:image/png;base64,YWJj"}),
        json!({}),
    );
    perform_with(LocalOcrHost::new(request).with_before_send(move |wire, _| {
        Ok(WireRequest {
            url: destination.clone(),
            ..wire
        })
    }))
    .await
    .unwrap();
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/custom");
    assert_eq!(sent.url.query(), Some("sig=a%2fb"));
}

#[rstest]
#[tokio::test]
async fn an_invalid_hook_destination_never_reaches_transport() {
    let upstream = upstream([pages_response()]).await;
    let request = ocr_request_with_document(
        "cohere/model",
        &upstream.uri(),
        json!({"type":"image_url","image_url":"data:image/png;base64,YWJj"}),
        json!({}),
    );
    let result = perform_with(LocalOcrHost::new(request).with_before_send(|wire, _| {
        Ok(WireRequest {
            url: "ftp://example.test/custom".into(),
            ..wire
        })
    }))
    .await;
    assert!(matches!(result, Err(Error::RequestField { path }) if path == "guardrail.url"));
    assert!(received(&upstream).await.is_empty());
}
