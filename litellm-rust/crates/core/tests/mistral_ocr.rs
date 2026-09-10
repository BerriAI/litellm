use crate::ocr::backends::mistral::complete_url;
use crate::ocr::formats::mistral::MistralOcrFormat;
use crate::ocr::test_support::{format_body, transform_format};
use crate::ocr::types::OcrDocument;
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case(json!({"extract_header":true}))]
#[case(json!({"extract_footer":false}))]
#[case(json!({"pages":[0,2],"include_image_base64":true,"image_limit":2,"image_min_size":100}))]
#[case(json!({"table_format":"html","confidence_scores_granularity":"word","include_blocks":true,"id":"req-123"}))]
#[case(json!({"bbox_annotation_format":{"type":"json_schema","schema":{"type":"object"}},"document_annotation_format":{"type":"json_schema"},"document_annotation_prompt":"extract"}))]
fn transform_ocr_request_includes_each_optional_param(#[case] options: serde_json::Value) {
    let document = json!({"type":"document_url","document_url":"https://example.com/a.pdf"});
    let result = format_body::<MistralOcrFormat>(
        "model",
        serde_json::from_value::<OcrDocument>(document.clone()).unwrap(),
        options.clone(),
    )
    .unwrap();
    assert_eq!(result["model"], "model");
    assert_eq!(result["document"], document);
    for (name, value) in options.as_object().unwrap() {
        assert_eq!(&result[name], value);
    }
}

#[test]
fn map_ocr_params_drops_unknown_params() {
    let result = format_body::<MistralOcrFormat>(
        "model",
        serde_json::from_value::<OcrDocument>(
            json!({"type":"image_url","image_url":"data:image/png;base64,YWJj"}),
        )
        .unwrap(),
        json!({"unknown":{},"extract_header":true}),
    )
    .unwrap();
    assert!(result.get("unknown").is_none());
    assert_eq!(result["extract_header"], true);
}

#[test]
fn transform_ocr_response_preserves_ocr4_page_fields() {
    let page = json!({"index":0,"markdown":"hello","header":"head","footer":"foot","confidence_scores":{"mean":0.99},"blocks":[{"type":"text","content":"hello"}],"images":[{"id":"img-1","image_base64":"YWJj","top_left_x":12}],"dimensions":{"width":100,"height":200,"dpi":96}});
    let response = transform_format::<MistralOcrFormat>("model", json!({"pages":[page.clone()],"model":"returned-model","usage_info":{"pages_processed":1,"future_counter":5}}), json!({}), false).unwrap();
    for (name, value) in page.as_object().unwrap() {
        if name != "images" {
            assert_eq!(&response["pages"][0][name], value);
        }
    }
    assert_eq!(response["pages"][0]["images"][0]["id"], "img-1");
    assert_eq!(response["usage_info"]["future_counter"], 5);
    assert_eq!(response["model"], "returned-model");
}

#[test]
fn transform_ocr_response_normalizes_mistral_json() {
    let result = transform_format::<MistralOcrFormat>(
        "model",
        json!({"pages":[{"index":"0","markdown":"hello"}],"usage_info":{"pages_processed":"1"}}),
        json!({}),
        false,
    )
    .unwrap();
    assert_eq!(result["pages"][0]["index"], 0);
    assert_eq!(result["usage_info"]["pages_processed"], 1);
    assert_eq!(result["pages"][0]["images"], serde_json::Value::Null);
    assert!(result.get("provider_native_response").is_none());
    assert!(
        transform_format::<MistralOcrFormat>("model", json!({"pages":null}), json!({}), false)
            .is_err()
    );
}

#[test]
fn complete_url_defaults_and_dedupes_v1() {
    assert_eq!(complete_url(None).unwrap(), "https://api.mistral.ai/v1/ocr");
    assert_eq!(
        complete_url(Some(" ")).unwrap(),
        "https://api.mistral.ai/v1/ocr"
    );
    assert_eq!(
        complete_url(Some("https://example.com")).unwrap(),
        "https://example.com/v1/ocr"
    );
    assert_eq!(
        complete_url(Some("https://example.com/v1/")).unwrap(),
        "https://example.com/v1/ocr"
    );
    assert_eq!(
        complete_url(Some("https://example.com/v1/ocr?tenant=a")).unwrap(),
        "https://example.com/v1/ocr?tenant=a"
    );
    assert_eq!(
        complete_url(Some("https://example.com/v1?tenant=a")).unwrap(),
        "https://example.com/v1/ocr?tenant=a"
    );
}
