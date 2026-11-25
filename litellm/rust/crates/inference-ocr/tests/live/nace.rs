use std::sync::Arc;

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_inference_ocr::wire::{OcrWireRequest, decode_request};
use litellm_inference_testing::{RecordingSecrets, http_config, resources};
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};

use crate::support::build_ocr_route;

/// `NACE_API_KEY` as a secret source; tests skip when it is unset.
#[fixture]
fn secrets() -> Option<Arc<RecordingSecrets>> {
    let key = std::env::var("NACE_API_KEY").ok()?;
    Some(Arc::new(RecordingSecrets::new([(
        "NACE_API_KEY",
        key.as_str(),
    )])))
}

fn request(document: Value) -> OcrWireRequest {
    OcrWireRequest {
        model: "nace/parse-low".into(),
        document,
        api_key: None,
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: Map::new(),
        input_sources: Default::default(),
        timeout_seconds: Some(120.0),
    }
}

#[rstest]
#[case::uploaded_text(json!({
    "type": "document_url",
    "document_url": format!(
        "data:text/plain;base64,{}",
        STANDARD.encode("LiteLLM live check. Invoice number SYN-0001.")
    )
}), "SYN-0001", 1)]
#[case::public_link(json!({
    "type": "document_url",
    "document_url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
}), "Dummy PDF file", 1)]
#[tokio::test]
async fn a_document_is_parsed_into_pages_with_billed_units(
    secrets: Option<Arc<RecordingSecrets>>,
    #[case] document: Value,
    #[case] expected_text: &str,
    #[case] units: i64,
) {
    let Some(secrets) = secrets else { return };
    let route = build_ocr_route(
        &resources(),
        &http_config(),
        Default::default(),
        Default::default(),
        secrets,
    );

    let response = route
        .execute(decode_request(request(document)).unwrap(), &(), None)
        .await
        .unwrap();

    assert!(
        response.pages[0].markdown.contains(expected_text),
        "{}",
        response.pages[0].markdown
    );
    assert_eq!(response.usage_info.unwrap().pages_processed, Some(units));
}
