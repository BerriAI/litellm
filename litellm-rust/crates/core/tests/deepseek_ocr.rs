use rstest::rstest;
use serde_json::{Value, json};

use crate::llms::vertex_ai::ocr::deepseek_transformation::{
    DeepSeekOcrParams, DeepSeekOcrResponse, VertexAIDeepSeekOCRConfig, normalize_response,
};
use crate::ocr::types::OcrDocument;

fn document() -> OcrDocument {
    serde_json::from_value(json!({"type":"image_url","image_url":"gs://bucket/a.png"})).unwrap()
}

#[rstest]
#[case("stream", json!(true))]
#[case("temperature", json!(0.1))]
#[case("max_tokens", json!(1024))]
#[case("top_p", json!(0.9))]
#[case("n", json!(2))]
#[case("stop", json!("done"))]
#[case("stop", json!(["done", "stop"]))]
#[case("temperature", json!(null))]
fn request_mapping_matches_python(#[case] name: &str, #[case] value: Value) {
    let params: DeepSeekOcrParams =
        serde_json::from_value(json!({name: value.clone(), "ignored": true})).unwrap();
    let result = serde_json::to_value(
        VertexAIDeepSeekOCRConfig
            .transform_ocr_request("deepseek-ai/deepseek-ocr-maas", document(), &params, &[])
            .unwrap(),
    )
    .unwrap();
    assert_eq!(result["model"], "deepseek-ai/deepseek-ocr-maas");
    assert_eq!(
        result["messages"][0]["content"][0],
        json!({"type":"image_url","image_url":"gs://bucket/a.png"})
    );
    assert_eq!(result[name], value);
    assert!(result.get("ignored").is_none());
}

#[rstest]
#[case(json!({"type":"image_url","image_url":"data:image/png;base64,AA=="}))]
#[case(json!({"type":"document_url","document_url":"data:application/pdf;base64,AA=="}))]
fn request_maps_both_document_types_to_image_content(#[case] document: Value) {
    let source = document
        .get("image_url")
        .or_else(|| document.get("document_url"))
        .unwrap()
        .clone();
    let request = VertexAIDeepSeekOCRConfig
        .transform_ocr_request(
            "deepseek-ai/deepseek-ocr-maas",
            serde_json::from_value(document).unwrap(),
            &DeepSeekOcrParams::default(),
            &[],
        )
        .unwrap();
    let result = serde_json::to_value(request).unwrap();
    assert_eq!(
        result["messages"][0]["content"][0],
        json!({"type":"image_url","image_url":source})
    );
}

#[rstest]
#[case(json!("# hello"), "# hello")]
#[case(json!("{broken"), "{broken")]
#[case(json!(" {\"pages\":[]} "), " {\"pages\":[]} ")]
#[case(json!({"pages":[]}), "")]
#[case(json!("[]"), "[]")]
#[case(json!("{\"pages\":[{\"markdown\":\"json text\"}]}"), "json text")]
#[case(json!({"pages":[{"markdown":"object"}]}), "object")]
fn response_transform_handles_text_json_and_objects(
    #[case] content: Value,
    #[case] expected: &str,
) {
    let has_pages = content
        .as_object()
        .is_some_and(|data| data.contains_key("pages"))
        || content
            .as_str()
            .is_some_and(|text| text.contains("\"pages\""));
    let response: DeepSeekOcrResponse = serde_json::from_value(
        json!({"choices":[{"message":{"content":content}}],"usage":{"prompt_tokens":1}}),
    )
    .unwrap();
    let result = normalize_response("model", response).unwrap().into_json();
    assert_eq!(result["pages"][0]["markdown"], expected);
    assert_eq!(result["pages"][0]["index"], 0);
    if has_pages {
        assert!(result["usage_info"].is_null());
    } else {
        assert_eq!(result["usage_info"]["prompt_tokens"], 1);
    }
}

#[test]
fn structured_result_maps_pages_usage_model_and_annotation() {
    let response: DeepSeekOcrResponse = serde_json::from_value(json!({
        "choices":[{"message":{"content":{
            "pages":[{"index":2,"markdown":"page","images":[{"id":"one"}],"dimensions":{"width":10}}],
            "model":"provider-model",
            "usage_info":{"pages_processed":1},
            "document_annotation":{"language":"en"},
            "future":"kept"
        }}}]
    }))
    .unwrap();
    let result = normalize_response("requested", response)
        .unwrap()
        .into_json();
    assert_eq!(result["pages"][0]["index"], 2);
    assert_eq!(result["pages"][0]["images"][0]["id"], "one");
    assert_eq!(result["model"], "provider-model");
    assert_eq!(result["usage_info"]["pages_processed"], 1);
    assert_eq!(result["document_annotation"]["language"], "en");
    assert!(result.get("future").is_none());
}

#[test]
fn response_transform_rejects_missing_empty_and_malformed_content() {
    for value in [
        json!({"choices":[]}),
        json!({"choices":[{"message":{"content":{}}}]}),
        json!({"choices":[{"message":{"content":""}}]}),
        json!({"choices":[{"message":{"content":"{\"pages\":[{\"markdown\":42}]}"}}]}),
        json!({"choices":[{"message":{"content":{"pages":[{"markdown":42}]}}}]}),
    ] {
        let result = serde_json::from_value::<DeepSeekOcrResponse>(value)
            .map_err(|_| ())
            .and_then(|response| normalize_response("model", response).map_err(|_| ()));
        assert!(result.is_err());
    }
}

#[test]
fn structured_content_preserves_usage_presence_and_shared_page_defaults() {
    for (usage, expected) in [(json!(null), None), (json!({"pages_processed":2}), Some(2))] {
        let response = serde_json::from_value(json!({
            "choices":[{"message":{"content":{
                "pages":[42, {"index":"2", "images":[{"id":"kept"}], "ignored":true}],
                "usage_info":usage
            }}}],
            "usage":{"pages_processed":99}
        }))
        .unwrap();
        let normalized = normalize_response("model", response).unwrap();
        assert_eq!(normalized.pages.len(), 1);
        assert_eq!(normalized.pages[0].index, 2);
        assert_eq!(normalized.pages[0].markdown, "");
        assert!(normalized.pages[0].extra_fields.is_empty());
        assert_eq!(
            normalized
                .usage_info
                .and_then(|usage| usage.pages_processed),
            expected
        );
    }
}
