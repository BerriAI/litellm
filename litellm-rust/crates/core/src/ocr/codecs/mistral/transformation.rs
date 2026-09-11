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

    #[rstest]
    #[case("pages", json!([0, 2]))]
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
        let document: OcrDocument = serde_json::from_value(
            json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
        )
        .unwrap();
        let result =
            serde_json::to_value(transform_ocr_request("model", document, &params).unwrap())
                .unwrap();
        assert_eq!(result["model"], "model");
        assert_eq!(result[name], value);
    }

    #[test]
    fn request_mapping_filters_unknown_fields() {
        let params: MistralOcrParams = serde_json::from_value(json!({"unknown": true})).unwrap();
        let document: OcrDocument = serde_json::from_value(
            json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
        )
        .unwrap();
        let result =
            serde_json::to_value(transform_ocr_request("model", document, &params).unwrap())
                .unwrap();
        assert!(result.get("unknown").is_none());
    }

    #[test]
    fn response_preserves_provider_fields() {
        let response: MistralOcrResponse = serde_json::from_value(json!({
            "pages":[{"index":0,"markdown":"hello","header":"head","confidence_scores":{"mean":0.99}}],
            "model":"returned-model",
            "usage_info":{"pages_processed":1,"future_counter":5},
            "future_response_field":"kept"
        }))
        .unwrap();
        let result = transform_ocr_response("model", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0]["header"], "head");
        assert_eq!(result["usage_info"]["future_counter"], 5);
        assert_eq!(result["future_response_field"], "kept");
        assert_eq!(result["model"], "returned-model");
    }

    #[test]
    fn response_rejects_null_pages() {
        assert!(serde_json::from_value::<MistralOcrResponse>(json!({"pages":null})).is_err());
    }
}
