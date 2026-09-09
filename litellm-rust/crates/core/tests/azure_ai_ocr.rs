use super::transformation::AZURE_MISTRAL_OCR_BACKEND;
use crate::ocr::tests::body;
use crate::ocr::transformation::OcrBackend;
use crate::ocr::types::OcrConnection;
use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_BACKEND;
use serde_json::json;

#[tokio::test]
async fn azure_ai_reuses_mistral_body_transform() {
    let document = json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"});
    let params = json!({"include_image_base64":true});
    assert_eq!(
        body(
            &AZURE_MISTRAL_OCR_BACKEND,
            "model",
            document.clone(),
            params.clone()
        )
        .await
        .unwrap(),
        body(&MISTRAL_OCR_BACKEND, "model", document, params)
            .await
            .unwrap()
    );
}
#[test]
fn azure_ai_mistral_ocr_uses_generic_api_base() {
    let connection = OcrConnection {
        api_base: Some("https://example.com/".into()),
        ..Default::default()
    };
    assert_eq!(
        AZURE_MISTRAL_OCR_BACKEND
            .complete_url(&connection, "model", &Default::default())
            .unwrap(),
        "https://example.com/providers/mistral/azure/ocr"
    );
}
