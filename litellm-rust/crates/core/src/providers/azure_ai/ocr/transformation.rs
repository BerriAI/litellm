use crate::error::{CoreError, CoreResult, json_type_name};
use crate::ocr::transformation::OcrProviderTransformation;
use crate::ocr::types::{OcrRequestData, OcrResponseData};
use serde_json::{Map, Value, json};

use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

const AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI: i64 = 96;

const AZURE_DOCUMENT_INTELLIGENCE_SUPPORTED_OCR_PARAMS: &[&str] = &["pages"];

pub struct AzureAiOcrConfig;
pub struct AzureDocumentIntelligenceOcrConfig;

pub const AZURE_AI_OCR_CONFIG: AzureAiOcrConfig = AzureAiOcrConfig;
pub const AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG: AzureDocumentIntelligenceOcrConfig =
    AzureDocumentIntelligenceOcrConfig;

fn document_url_from_mistral_document(document: &Value) -> CoreResult<&str> {
    let object = document.as_object().ok_or_else(|| CoreError::InvalidType {
        expected: "object",
        actual: json_type_name(document),
    })?;
    let doc_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(CoreError::MissingField("document.type"))?;
    let field_name = match doc_type {
        "document_url" => "document_url",
        "image_url" => "image_url",
        other => {
            return Err(CoreError::InvalidRequest(format!(
                "Invalid document type: {other}. Must be 'document_url' or 'image_url'"
            )));
        }
    };
    object
        .get(field_name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(CoreError::MissingField(field_name))
}

fn extract_base64_from_data_uri(data_uri: &str) -> &str {
    data_uri
        .split_once(',')
        .map(|(_, data)| data)
        .unwrap_or(data_uri)
}

fn page_markdown(page: &Map<String, Value>) -> String {
    page.get("lines")
        .and_then(Value::as_array)
        .map(|lines| {
            lines
                .iter()
                .filter_map(|line| line.get("content").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn page_dimensions(page: &Map<String, Value>) -> Value {
    let width = page.get("width").and_then(Value::as_f64).unwrap_or(8.5);
    let height = page.get("height").and_then(Value::as_f64).unwrap_or(11.0);
    let unit = page.get("unit").and_then(Value::as_str).unwrap_or("inch");
    let (width, height) = if unit == "inch" {
        (
            (width * AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI as f64) as i64,
            (height * AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI as f64) as i64,
        )
    } else {
        (width as i64, height as i64)
    };
    json!({
        "width": width,
        "height": height,
        "dpi": AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI,
    })
}

impl OcrProviderTransformation for AzureAiOcrConfig {
    fn get_supported_ocr_params(&self) -> &'static [&'static str] {
        MISTRAL_OCR_CONFIG.get_supported_ocr_params()
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, optional_params)
    }

    fn transform_ocr_response_data(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<OcrResponseData> {
        MISTRAL_OCR_CONFIG.transform_ocr_response_data(model, response_json)
    }
}

impl OcrProviderTransformation for AzureDocumentIntelligenceOcrConfig {
    fn get_supported_ocr_params(&self) -> &'static [&'static str] {
        AZURE_DOCUMENT_INTELLIGENCE_SUPPORTED_OCR_PARAMS
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: Value,
        _optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        let document_url = document_url_from_mistral_document(&document)?;
        let mut data = Map::new();
        if document_url.starts_with("data:") {
            data.insert(
                "base64Source".to_string(),
                Value::String(extract_base64_from_data_uri(document_url).to_string()),
            );
        } else {
            data.insert(
                "urlSource".to_string(),
                Value::String(document_url.to_string()),
            );
        }
        Ok(OcrRequestData {
            data: Value::Object(data),
            files: None,
        })
    }

    fn transform_ocr_response_data(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<OcrResponseData> {
        let response = response_json
            .as_object()
            .ok_or_else(|| CoreError::InvalidType {
                expected: "object",
                actual: json_type_name(&response_json),
            })?;
        let status = response
            .get("status")
            .and_then(Value::as_str)
            .ok_or(CoreError::MissingField("status"))?;
        if status != "succeeded" {
            return Err(CoreError::InvalidResponse(format!(
                "Azure Document Intelligence analysis failed with status: {status}"
            )));
        }

        let azure_pages = response
            .get("analyzeResult")
            .and_then(|result| result.get("pages"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let pages = azure_pages
            .iter()
            .filter_map(Value::as_object)
            .map(|page| {
                let page_number = page.get("pageNumber").and_then(Value::as_i64).unwrap_or(1);
                json!({
                    "index": page_number - 1,
                    "markdown": page_markdown(page),
                    "dimensions": page_dimensions(page),
                })
            })
            .collect::<Vec<_>>();

        Ok(OcrResponseData {
            usage_info: Some(json!({
                "pages_processed": pages.len(),
                "doc_size_bytes": null,
            })),
            pages,
            model: model.to_string(),
            document_annotation: None,
            object: "ocr".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_ai_reuses_mistral_body_transform() {
        let body = AZURE_AI_OCR_CONFIG
            .transform_ocr_request(
                "pixtral-12b-2409",
                json!({"type": "document_url", "document_url": "data:application/pdf;base64,abc"}),
                serde_json::Map::from_iter([("include_image_base64".to_string(), json!(true))]),
            )
            .expect("request transforms")
            .data;

        assert_eq!(body["model"], "pixtral-12b-2409");
        assert_eq!(body["include_image_base64"], true);
        assert_eq!(
            body["document"]["document_url"],
            "data:application/pdf;base64,abc"
        );
    }

    #[test]
    fn document_intelligence_request_uses_base64_source_for_data_uri() {
        let body = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_request(
                "prebuilt-read",
                json!({"type": "document_url", "document_url": "data:application/pdf;base64,abc123"}),
                Map::new(),
            )
            .expect("request transforms")
            .data;

        assert_eq!(body, json!({"base64Source": "abc123"}));
    }

    #[test]
    fn document_intelligence_response_normalizes_pages() {
        let response = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response_data(
                "prebuilt-layout",
                json!({
                    "status": "succeeded",
                    "analyzeResult": {
                        "pages": [{
                            "pageNumber": 2,
                            "width": 8.5,
                            "height": 11,
                            "unit": "inch",
                            "lines": [{"content": "hello"}, {"content": "world"}]
                        }]
                    }
                }),
            )
            .expect("response transforms");

        assert_eq!(response.pages[0]["index"], 1);
        assert_eq!(response.pages[0]["markdown"], "hello\nworld");
        assert_eq!(response.pages[0]["dimensions"]["width"], 816);
        assert_eq!(
            response.usage_info,
            Some(json!({"pages_processed": 1, "doc_size_bytes": null}))
        );
    }
}
