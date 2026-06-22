use litellm_core::error::{json_type_name, CoreError, CoreResult};
use litellm_core::ocr::transformation::OcrProviderConfig;
use litellm_core::ocr::types::{OcrRequestData, OcrResponseData};
use serde_json::{Map, Value};

const SUPPORTED_OCR_PARAMS: &[&str] = &[
    "pages",
    "include_image_base64",
    "image_limit",
    "image_min_size",
    "bbox_annotation_format",
    "document_annotation_format",
    "document_annotation_prompt",
    "extract_header",
    "extract_footer",
    "table_format",
    "confidence_scores_granularity",
    "id",
];

pub struct MistralOcrConfig;

pub const MISTRAL_OCR_CONFIG: MistralOcrConfig = MistralOcrConfig;

impl OcrProviderConfig for MistralOcrConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        SUPPORTED_OCR_PARAMS
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        if !document.is_object() {
            return Err(CoreError::InvalidType {
                expected: "object",
                actual: json_type_name(&document),
            });
        }

        let mut data = Map::new();
        data.insert("model".to_string(), Value::String(model.to_string()));
        data.insert("document".to_string(), document);
        for (param, value) in optional_params {
            data.insert(param, value);
        }

        Ok(OcrRequestData {
            data: Value::Object(data),
            files: None,
        })
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<OcrResponseData> {
        let response_object = response_json
            .as_object()
            .ok_or_else(|| CoreError::InvalidType {
                expected: "object",
                actual: json_type_name(&response_json),
            })?;

        let pages = response_object
            .get("pages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let model = response_object
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(model)
            .to_string();
        let document_annotation = response_object.get("document_annotation").cloned();
        let usage_info = response_object.get("usage_info").cloned();

        Ok(OcrResponseData {
            pages,
            model,
            document_annotation,
            usage_info,
            object: "ocr".to_string(),
        })
    }
}

pub fn supported_ocr_params() -> &'static [&'static str] {
    MISTRAL_OCR_CONFIG.supported_ocr_params()
}

pub fn map_ocr_params(non_default_params: &Map<String, Value>) -> Map<String, Value> {
    MISTRAL_OCR_CONFIG.map_ocr_params(non_default_params)
}

pub fn transform_ocr_request(
    model: &str,
    document: Value,
    optional_params: Map<String, Value>,
) -> CoreResult<OcrRequestData> {
    MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, optional_params)
}

pub fn transform_ocr_response(model: &str, response_json: Value) -> CoreResult<OcrResponseData> {
    MISTRAL_OCR_CONFIG.transform_ocr_response(model, response_json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn supported_params_match_python_mistral_ocr_config() {
        assert_eq!(
            supported_ocr_params(),
            &[
                "pages",
                "include_image_base64",
                "image_limit",
                "image_min_size",
                "bbox_annotation_format",
                "document_annotation_format",
                "document_annotation_prompt",
                "extract_header",
                "extract_footer",
                "table_format",
                "confidence_scores_granularity",
                "id",
            ]
        );
    }

    #[test]
    fn map_ocr_params_drops_unknown_params() {
        let params = json!({
            "extract_header": true,
            "unsupported_param": "value",
            "pages": [0, 1]
        });
        let mapped = map_ocr_params(params.as_object().unwrap());

        assert_eq!(mapped.get("extract_header"), Some(&json!(true)));
        assert_eq!(mapped.get("pages"), Some(&json!([0, 1])));
        assert!(!mapped.contains_key("unsupported_param"));
    }

    #[test]
    fn transform_ocr_request_builds_mistral_body() {
        let document = json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        });
        let optional_params = json!({
            "include_image_base64": true,
            "table_format": "html"
        })
        .as_object()
        .unwrap()
        .clone();

        let result = transform_ocr_request("mistral-ocr-latest", document.clone(), optional_params)
            .expect("request should transform");

        assert_eq!(
            result.data,
            json!({
                "model": "mistral-ocr-latest",
                "document": document,
                "include_image_base64": true,
                "table_format": "html"
            })
        );
        assert_eq!(result.files, None);
    }

    #[test]
    fn transform_ocr_request_rejects_non_object_document() {
        let err = transform_ocr_request("mistral-ocr-latest", json!("bad"), Map::new())
            .expect_err("string document should be rejected");

        assert_eq!(
            err,
            CoreError::InvalidType {
                expected: "object",
                actual: "string",
            }
        );
    }

    #[test]
    fn transform_ocr_response_normalizes_mistral_json() {
        let response = json!({
            "pages": [{"index": 0, "markdown": "hello"}],
            "model": "mistral-ocr-2505-completion",
            "document_annotation": null,
            "usage_info": {"pages_processed": 1}
        });

        let result = transform_ocr_response("mistral-ocr-latest", response)
            .expect("response should transform");

        assert_eq!(result.pages, vec![json!({"index": 0, "markdown": "hello"})]);
        assert_eq!(result.model, "mistral-ocr-2505-completion");
        assert_eq!(result.document_annotation, Some(Value::Null));
        assert_eq!(result.usage_info, Some(json!({"pages_processed": 1})));
        assert_eq!(result.object, "ocr");
    }
}
