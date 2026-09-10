use crate::ocr::integrations::OcrIntegration;
use crate::ocr::integrations::{MistralDirect as MISTRAL, VertexMistral as VERTEX_MISTRAL};
use crate::ocr::test_support::body;
use crate::ocr::types::OcrConnection;
use crate::providers::vertex_ai::auth::VertexAuthInputs;
use serde_json::json;

#[tokio::test]
async fn vertex_mistral_reuses_mistral_body_transform() {
    let doc = json!({"type":"image_url","image_url":"data:image/png;base64,YWJj"});
    assert_eq!(
        body(
            &VERTEX_MISTRAL,
            "model",
            doc.clone(),
            json!({"extract_footer":true})
        )
        .await
        .unwrap(),
        body(&MISTRAL, "model", doc, json!({"extract_footer":true}))
            .await
            .unwrap()
    );
}
#[tokio::test]
async fn vertex_mistral_url_uses_project_location_and_model() {
    let connection = OcrConnection {
        api_key: Some("token".into()),
        ..Default::default()
    };
    let config = VertexAuthInputs::from_optional_params(
        json!({"vertex_project":"proj-1","vertex_location":"europe-west4"})
            .as_object()
            .unwrap(),
    )
    .unwrap();
    let prepared = OcrIntegration::prepare(
        &VERTEX_MISTRAL,
        &connection,
        &config,
        "mistral-ocr-maas",
        &Default::default(),
        &|_| None,
    )
    .await
    .unwrap();
    assert_eq!(
        prepared.url,
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
    );
}

#[tokio::test]
async fn vertex_mistral_resolves_provider_environment_in_the_backend() {
    let connection = OcrConnection {
        api_key: Some("token".into()),
        ..Default::default()
    };
    let prepared = OcrIntegration::prepare(
        &VERTEX_MISTRAL,
        &connection,
        &Default::default(),
        "model",
        &Default::default(),
        &|name| match name {
            "VERTEXAI_PROJECT" => Some("environment-project".into()),
            "VERTEXAI_LOCATION" => Some("asia-east1".into()),
            _ => None,
        },
    )
    .await
    .unwrap();

    assert_eq!(
        prepared.url,
        "https://asia-east1-aiplatform.googleapis.com/v1/projects/environment-project/locations/asia-east1/publishers/mistralai/models/model:rawPredict"
    );
}
