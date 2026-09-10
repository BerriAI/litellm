use rstest::rstest;
use serde_json::{Value, json};

use crate::ocr::adapters::complete_mistral_url;
use crate::ocr::codecs::mistral::{
    MistralOcrParams, MistralOcrResponse, decode_response, encode_request,
};
use crate::ocr::types::OcrDocument;

#[rstest]
#[case(json!({"extract_header":true}))]
#[case(json!({"extract_footer":false}))]
#[case(json!({"pages":[0,2],"include_image_base64":true,"image_limit":2,"image_min_size":100}))]
#[case(json!({"table_format":"html","confidence_scores_granularity":"word","include_blocks":true,"id":"req-123"}))]
#[case(json!({"bbox_annotation_format":{"type":"json_schema","schema":{"type":"object"}},"document_annotation_format":{"type":"json_schema"},"document_annotation_prompt":"extract"}))]
fn request_codec_includes_supported_params(#[case] options: Value) {
    let document: OcrDocument = serde_json::from_value(
        json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
    )
    .unwrap();
    let params: MistralOcrParams = serde_json::from_value(options.clone()).unwrap();
    let result = serde_json::to_value(encode_request("model", document, &params).unwrap()).unwrap();
    assert_eq!(result["model"], "model");
    for (name, value) in options.as_object().unwrap() {
        assert_eq!(&result[name], value);
    }
}

#[test]
fn request_codec_drops_unknown_params() {
    let params: MistralOcrParams =
        serde_json::from_value(json!({"unknown":{},"extract_header":true})).unwrap();
    let document: OcrDocument = serde_json::from_value(
        json!({"type":"image_url","image_url":"data:image/png;base64,YWJj"}),
    )
    .unwrap();
    let result = serde_json::to_value(encode_request("model", document, &params).unwrap()).unwrap();
    assert!(result.get("unknown").is_none());
    assert_eq!(result["extract_header"], true);
}

#[test]
fn response_codec_preserves_provider_fields() {
    let response: MistralOcrResponse = serde_json::from_value(json!({
        "pages":[{"index":0,"markdown":"hello","header":"head","confidence_scores":{"mean":0.99}}],
        "model":"returned-model",
        "usage_info":{"pages_processed":1,"future_counter":5},
        "future_response_field":"kept"
    }))
    .unwrap();
    let result = decode_response("model", response, &MistralOcrParams::default())
        .unwrap()
        .into_json();
    assert_eq!(result["pages"][0]["header"], "head");
    assert_eq!(result["usage_info"]["future_counter"], 5);
    assert_eq!(result["future_response_field"], "kept");
    assert_eq!(result["model"], "returned-model");
}

#[test]
fn response_codec_rejects_null_pages() {
    assert!(serde_json::from_value::<MistralOcrResponse>(json!({"pages":null})).is_err());
}

#[test]
fn complete_url_defaults_and_dedupes_v1() {
    assert_eq!(
        complete_mistral_url(None).unwrap(),
        "https://api.mistral.ai/v1/ocr"
    );
    assert_eq!(
        complete_mistral_url(Some("https://example.com/v1?tenant=a")).unwrap(),
        "https://example.com/v1/ocr?tenant=a"
    );
    assert_eq!(
        complete_mistral_url(Some("https://example.com/v1/ocr?tenant=a")).unwrap(),
        "https://example.com/v1/ocr?tenant=a"
    );
}
