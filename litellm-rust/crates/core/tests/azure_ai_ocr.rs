use crate::ocr::backends::OcrIntegration;
use crate::ocr::registry::{AZURE_MISTRAL, MISTRAL};
use crate::ocr::tests::body;
use crate::ocr::types::OcrConnection;
use serde_json::json;

#[tokio::test]
async fn azure_ai_reuses_mistral_body_transform() {
    let document = json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"});
    let params = json!({"include_image_base64":true});
    assert_eq!(
        body(&AZURE_MISTRAL, "model", document.clone(), params.clone())
            .await
            .unwrap(),
        body(&MISTRAL, "model", document, params).await.unwrap()
    );
}
#[tokio::test]
async fn azure_ai_mistral_ocr_uses_generic_api_base() {
    let connection = OcrConnection {
        api_key: Some("key".into()),
        api_base: Some("https://example.com/".into()),
        ..Default::default()
    };
    let prepared = AZURE_MISTRAL
        .prepare(
            &connection,
            &Default::default(),
            "model",
            &Default::default(),
            &|_| None,
        )
        .await
        .unwrap();
    assert_eq!(
        prepared.url,
        "https://example.com/providers/mistral/azure/ocr"
    );
}
