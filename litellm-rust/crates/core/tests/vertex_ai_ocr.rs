use crate::ocr::registry::{MISTRAL, VERTEX_MISTRAL};
use crate::ocr::tests::body;
use crate::ocr::types::{OcrConnection, VertexOcrSettings};
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
#[test]
fn vertex_mistral_url_uses_project_location_and_model() {
    let connection = OcrConnection {
        vertex: VertexOcrSettings {
            project: Some("proj-1".into()),
            location: Some("europe-west4".into()),
        },
        ..Default::default()
    };
    assert_eq!(
        VERTEX_MISTRAL
            .complete_url(&connection, "mistral-ocr-maas", &Default::default())
            .unwrap(),
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
    );
}
