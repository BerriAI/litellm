use crate::error::{CoreError, CoreResult, json_type_name};
use crate::ocr::transformation::OcrProviderTransformation;
use crate::ocr::types::{OcrRequestData, OcrResponseData};
use serde_json::{Map, Value, json};

use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

#[rustfmt::skip]
const DEEPSEEK_SUPPORTED_OCR_PARAMS: &[&str] = &[
    "stream",
    "temperature",
    "max_tokens",
    "top_p",
    "n",
    "stop",
];

pub struct VertexAiOcrConfig;
pub struct VertexAiDeepSeekOcrConfig;

pub const VERTEX_AI_OCR_CONFIG: VertexAiOcrConfig = VertexAiOcrConfig;
pub const VERTEX_AI_DEEPSEEK_OCR_CONFIG: VertexAiDeepSeekOcrConfig = VertexAiDeepSeekOcrConfig;

fn document_content_item(document: &Value) -> CoreResult<Value> {
    let object = document.as_object().ok_or_else(|| CoreError::InvalidType {
        expected: "object",
        actual: json_type_name(document),
    })?;
    let doc_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(CoreError::MissingField("document.type"))?;
    let url_field = match doc_type {
        "image_url" => "image_url",
        "document_url" => "document_url",
        other => {
            return Err(CoreError::InvalidRequest(format!(
                "Unsupported document type: {other}. Expected 'image_url' or 'document_url'"
            )));
        }
    };
    let url = object
        .get(url_field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(CoreError::MissingField(url_field))?;

    Ok(json!({
        "type": "image_url",
        "image_url": url,
    }))
}

fn deepseek_model_name(model: &str) -> String {
    if model.starts_with("deepseek-ai/") {
        model.to_string()
    } else {
        format!("deepseek-ai/{model}")
    }
}

fn first_choice_content(response: &Value) -> CoreResult<Value> {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .cloned()
        .filter(|content| match content {
            Value::String(value) => !value.is_empty(),
            Value::Object(_) => true,
            _ => false,
        })
        .ok_or_else(|| {
            CoreError::InvalidResponse("No content in DeepSeek OCR response".to_string())
        })
}

fn ocr_data_from_content(content: Value, usage: Option<Value>, model: &str) -> Value {
    match content {
        Value::String(content) => {
            if content.trim_start().starts_with('{') {
                serde_json::from_str(&content).unwrap_or_else(|_| {
                    json!({
                        "pages": [{"index": 0, "markdown": content}],
                        "model": model,
                        "usage_info": usage.unwrap_or_else(|| json!({})),
                    })
                })
            } else {
                json!({
                    "pages": [{"index": 0, "markdown": content}],
                    "model": model,
                    "usage_info": usage.unwrap_or_else(|| json!({})),
                })
            }
        }
        Value::Object(_) => content,
        other => json!({
            "pages": [{"index": 0, "markdown": other.to_string()}],
            "model": model,
            "usage_info": usage.unwrap_or_else(|| json!({})),
        }),
    }
}

impl OcrProviderTransformation for VertexAiOcrConfig {
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

impl OcrProviderTransformation for VertexAiDeepSeekOcrConfig {
    fn get_supported_ocr_params(&self) -> &'static [&'static str] {
        DEEPSEEK_SUPPORTED_OCR_PARAMS
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        let mut data = Map::new();
        data.insert(
            "model".to_string(),
            Value::String(deepseek_model_name(model)),
        );
        data.insert(
            "messages".to_string(),
            json!([{"role": "user", "content": [document_content_item(&document)?]}]),
        );
        for (key, value) in optional_params {
            if DEEPSEEK_SUPPORTED_OCR_PARAMS.contains(&key.as_str()) {
                data.insert(key, value);
            }
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
        let usage = response.get("usage").cloned();
        let content = first_choice_content(&response_json)?;
        let mut ocr_data = ocr_data_from_content(content.clone(), usage.clone(), model);

        if !ocr_data.get("pages").is_some_and(Value::is_array) {
            ocr_data = json!({
                "pages": [{
                    "index": 0,
                    "markdown": match content {
                        Value::String(value) => value,
                        other => other.to_string(),
                    }
                }],
                "model": ocr_data.get("model").and_then(Value::as_str).unwrap_or(model),
                "usage_info": ocr_data.get("usage_info").cloned().or(usage).unwrap_or_else(|| json!({})),
            });
        }

        let object = ocr_data.as_object().ok_or_else(|| CoreError::InvalidType {
            expected: "object",
            actual: json_type_name(&ocr_data),
        })?;
        let pages = object
            .get("pages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let usage_info = object
            .get("usage_info")
            .cloned()
            .or_else(|| response.get("usage").cloned());
        Ok(OcrResponseData {
            pages,
            model: object
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(model)
                .to_string(),
            document_annotation: object.get("document_annotation").cloned(),
            usage_info,
            object: "ocr".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_mistral_reuses_mistral_body_transform() {
        let body = VERTEX_AI_OCR_CONFIG
            .transform_ocr_request(
                "mistral-ocr-maas",
                json!({"type": "image_url", "image_url": "data:image/png;base64,abc"}),
                Map::new(),
            )
            .expect("request transforms")
            .data;

        assert_eq!(body["model"], "mistral-ocr-maas");
        assert_eq!(body["document"]["image_url"], "data:image/png;base64,abc");
    }

    #[test]
    fn vertex_deepseek_request_uses_ocr_endpoint_shape() {
        let body = VERTEX_AI_DEEPSEEK_OCR_CONFIG
            .transform_ocr_request(
                "deepseek-ocr-maas",
                json!({"type": "document_url", "document_url": "gs://bucket/doc.pdf"}),
                Map::from_iter([("temperature".to_string(), json!(0.1))]),
            )
            .expect("request transforms")
            .data;

        assert_eq!(body["model"], "deepseek-ai/deepseek-ocr-maas");
        assert_eq!(body["temperature"], 0.1);
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({"type": "image_url", "image_url": "gs://bucket/doc.pdf"})
        );
    }

    #[test]
    fn vertex_deepseek_response_wraps_markdown_content() {
        let response = VERTEX_AI_DEEPSEEK_OCR_CONFIG
            .transform_ocr_response_data(
                "deepseek-ocr-maas",
                json!({
                    "choices": [{"message": {"content": "# OCR text"}}],
                    "usage": {"prompt_tokens": 1}
                }),
            )
            .expect("response transforms");

        assert_eq!(
            response.pages,
            vec![json!({"index": 0, "markdown": "# OCR text"})]
        );
        assert_eq!(response.model, "deepseek-ocr-maas");
        assert_eq!(response.usage_info, Some(json!({"prompt_tokens": 1})));
    }
}
