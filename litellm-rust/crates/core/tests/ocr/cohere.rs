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
