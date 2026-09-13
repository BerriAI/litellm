use super::{MistralOcrParams, MistralOcrRequest, MistralOcrResponse};
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{LiteLLMOcrResponse, OcrDocument};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn transform_ocr_request(
    model: &str,
    document: OcrDocument,
    params: &MistralOcrParams,
) -> Result<MistralOcrRequest, OcrRequestError> {
    Ok(MistralOcrRequest {
        model: model.to_string(),
        document,
        params: params.clone(),
    })
}

pub(crate) fn transform_ocr_response(
    model: &str,
    response: MistralOcrResponse,
) -> Result<LiteLLMOcrResponse, OcrResponseError> {
    Ok(LiteLLMOcrResponse {
        pages: response.pages,
        model: response.model.unwrap_or_else(|| model.to_string()),
        document_annotation: response.document_annotation,
        usage_info: response.usage_info,
        object: "ocr".to_string(),
        extra_fields: response.extra_fields,
        provider_native_response: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::{Value, json};

    fn mapped_params(value: Value) -> Value {
        serde_json::to_value(serde_json::from_value::<MistralOcrParams>(value).unwrap()).unwrap()
    }

    fn document() -> OcrDocument {
        serde_json::from_value(
            json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
        )
        .unwrap()
    }

    #[rstest]
    fn extract_header_is_a_supported_ocr_param() {
        assert_eq!(
            mapped_params(json!({"extract_header":true}))["extract_header"],
            true
        );
    }

    #[rstest]
    fn extract_footer_is_a_supported_ocr_param() {
        assert_eq!(
            mapped_params(json!({"extract_footer":false}))["extract_footer"],
            false
        );
    }

    #[rstest]
    fn existing_ocr_params_remain_supported() {
        let mapped = mapped_params(json!({
            "pages":[0,2],
            "include_image_base64":true,
            "image_limit":2,
            "image_min_size":100,
            "bbox_annotation_format":{"type":"json_schema"},
            "document_annotation_format":{"type":"json_schema"}
        }));
        assert_eq!(mapped["pages"], json!([0, 2]));
        assert_eq!(mapped["include_image_base64"], true);
        assert_eq!(mapped["image_limit"], 2);
        assert_eq!(mapped["image_min_size"], 100);
        assert_eq!(mapped["bbox_annotation_format"]["type"], "json_schema");
        assert_eq!(mapped["document_annotation_format"]["type"], "json_schema");
    }

    #[rstest]
    fn map_ocr_params_forwards_extract_header() {
        assert_eq!(
            mapped_params(json!({"extract_header":true}))["extract_header"],
            true
        );
    }

    #[rstest]
    fn map_ocr_params_forwards_extract_footer() {
        assert_eq!(
            mapped_params(json!({"extract_footer":true}))["extract_footer"],
            true
        );
    }

    #[rstest]
    fn map_ocr_params_forwards_extract_header_and_footer() {
        let mapped = mapped_params(json!({"extract_header":true,"extract_footer":false}));
        assert_eq!(mapped["extract_header"], true);
        assert_eq!(mapped["extract_footer"], false);
    }

    #[rstest]
    fn map_ocr_params_drops_unknown_params() {
        let mapped = mapped_params(json!({"extract_header":true,"unsupported_param":"value"}));
        assert_eq!(mapped["extract_header"], true);
        assert!(mapped.get("unsupported_param").is_none());
    }

    #[rstest]
    #[case("table_format", json!("html"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("confidence_scores_granularity", json!("block"))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("include_blocks", json!(true))]
    #[case("id", json!("req-123"))]
    fn new_ocr_params_are_supported(#[case] name: &str, #[case] value: Value) {
        assert_eq!(mapped_params(json!({name:value.clone()}))[name], value);
    }

    #[rstest]
    #[case("table_format", json!("html"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("include_blocks", json!(true))]
    #[case("id", json!("req-123"))]
    fn map_ocr_params_forwards_new_ocr_params(#[case] name: &str, #[case] value: Value) {
        assert_eq!(mapped_params(json!({name:value.clone()}))[name], value);
    }

    #[rstest]
    #[case("pages", json!([0, 2]))]
    #[case("pages", json!("0,2-4"))]
    #[case("include_image_base64", json!(true))]
    #[case("image_limit", json!(2))]
    #[case("image_min_size", json!(100))]
    #[case("bbox_annotation_format", json!({"type":"json_schema"}))]
    #[case("document_annotation_format", json!({"type":"json_schema"}))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("extract_header", json!(true))]
    #[case("extract_footer", json!(false))]
    #[case("table_format", json!("html"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("include_blocks", json!(true))]
    #[case("id", json!("req-123"))]
    fn request_mapping_matches_python(#[case] name: &str, #[case] value: Value) {
        let params: MistralOcrParams =
            serde_json::from_value(json!({name: value.clone()})).unwrap();
        let result =
            serde_json::to_value(transform_ocr_request("model", document(), &params).unwrap())
                .unwrap();
        assert_eq!(result["model"], "model");
        assert_eq!(result[name], value);
    }

    #[rstest]
    #[case("table_format", json!("html"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("id", json!("req-123"))]
    #[case("extract_header", json!(true))]
    #[case("include_blocks", json!(true))]
    #[case("pages", json!([0,1]))]
    fn transform_ocr_request_includes_each_optional_param(
        #[case] name: &str,
        #[case] value: Value,
    ) {
        let params: MistralOcrParams = serde_json::from_value(json!({name:value.clone()})).unwrap();
        let result = serde_json::to_value(
            transform_ocr_request("mistral-ocr-latest", document(), &params).unwrap(),
        )
        .unwrap();
        assert_eq!(result[name], value);
        assert_eq!(result["model"], "mistral-ocr-latest");
    }

    #[rstest]
    fn transform_ocr_request_includes_multiple_new_params() {
        let params: MistralOcrParams = serde_json::from_value(json!({
            "table_format":"html",
            "confidence_scores_granularity":"page",
            "extract_header":true
        }))
        .unwrap();
        let result = serde_json::to_value(
            transform_ocr_request("mistral-ocr-latest", document(), &params).unwrap(),
        )
        .unwrap();
        assert_eq!(result["table_format"], "html");
        assert_eq!(result["confidence_scores_granularity"], "page");
        assert_eq!(result["extract_header"], true);
    }

    #[rstest]
    fn transform_ocr_response_preserves_blocks_and_confidence_scores() {
        let response: MistralOcrResponse = serde_json::from_value(json!({
            "pages":[{
                "index":0,
                "markdown":"hello",
                "images":[{"id":"img-0","image_base64":"data:image/png;base64,AA=="}],
                "dimensions":{"width":612,"height":792,"dpi":72},
                "blocks":[{"type":"title","bbox":{"x":1},"confidence_scores":{"mean":0.98}}],
                "confidence_scores":{"average_page_confidence_score":0.99,"minimum_page_confidence_score":0.97}
            }],
            "model":"returned-model",
            "document_annotation":"{\"language\":\"en\"}",
            "usage_info":{"pages_processed":1}
        }))
        .unwrap();
        let result = transform_ocr_response("model", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0]["blocks"][0]["type"], "title");
        assert_eq!(result["pages"][0]["blocks"][0]["bbox"]["x"], 1);
        assert_eq!(
            result["pages"][0]["blocks"][0]["confidence_scores"]["mean"],
            0.98
        );
        assert_eq!(
            result["pages"][0]["confidence_scores"]["average_page_confidence_score"],
            0.99
        );
        assert_eq!(result["pages"][0]["images"][0]["id"], "img-0");
        assert_eq!(result["pages"][0]["dimensions"]["dpi"], 72);
        assert_eq!(result["model"], "returned-model");
        assert_eq!(result["document_annotation"], "{\"language\":\"en\"}");
        assert_eq!(result["usage_info"]["pages_processed"], 1);
    }

    #[rstest]
    fn transform_ocr_response_preserves_ocr4_page_fields() {
        let page = json!({
            "index":0,
            "markdown":"table page",
            "tables":[{"rows":2,"cols":3}],
            "hyperlinks":["https://example.com"],
            "header":"header",
            "footer":"footer"
        });
        let response: MistralOcrResponse =
            serde_json::from_value(json!({"pages":[page.clone()]})).unwrap();
        let result = transform_ocr_response("model", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0], page);
    }
}
