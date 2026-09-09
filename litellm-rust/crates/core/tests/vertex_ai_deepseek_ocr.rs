use super::transformation::VERTEX_DEEPSEEK_OCR_BACKEND as CONFIG;
use crate::ocr::tests::{body, transform};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case("deepseek-ocr-maas")]
#[case("deepseek-ai/deepseek-ocr-maas")]
#[case("vertex_ai/deepseek-ai/deepseek-ocr-maas")]
#[tokio::test]
async fn vertex_deepseek_request_uses_single_provider_namespace(#[case] model: &str) {
    let result = body(
        &CONFIG,
        model,
        json!({"type":"image_url","image_url":"gs://bucket/a.png"}),
        json!({"temperature":0.1,"max_tokens":1024,"ignored":true}),
    )
    .await
    .unwrap();
    assert_eq!(result["model"], "deepseek-ai/deepseek-ocr-maas");
    assert_eq!(
        result["messages"][0]["content"][0],
        json!({"type":"image_url","image_url":"gs://bucket/a.png"})
    );
    assert_eq!(result["temperature"], 0.1);
    assert!(result.get("ignored").is_none());
}
#[rstest]
#[case(json!("# hello"),"# hello")]
#[case(json!("{broken"),"{broken")]
#[case(json!("{\"pages\":[{\"markdown\":\"json text\"}]}"),"json text")]
#[case(json!({"pages":[{"markdown":"object"}]}),"object")]
fn vertex_deepseek_response_wraps_markdown_content(
    #[case] content: serde_json::Value,
    #[case] expected: &str,
) {
    let result = transform(
        &CONFIG,
        "model",
        json!({"choices":[{"message":{"content":content}}],"usage":{"prompt_tokens":1}}),
        json!({}),
    )
    .unwrap();
    assert_eq!(result["pages"][0]["markdown"], expected);
    assert_eq!(result["pages"][0]["index"], 0);
    assert_eq!(result["usage_info"]["prompt_tokens"], 1);
}
#[test]
fn deepseek_rejects_missing_content_and_invalid_structured_pages() {
    for response in [
        json!({"choices":[]}),
        json!({"choices":[{"message":{"content":""}}]}),
        json!({"choices":[{"message":{"content":{"pages":[{"markdown":42}]}}}]}),
    ] {
        assert!(transform(&CONFIG, "model", response, json!({})).is_err());
    }
}
