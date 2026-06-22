use litellm_core::error::{json_type_name, CoreError};
use litellm_core::ocr::transformation::OcrProviderConfig;
use litellm_core::CoreResult;
use serde_json::{Map, Value};

use crate::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

pub fn transform(payload: Value) -> CoreResult<Value> {
    let payload = payload_object(&payload)?;
    let provider = required_string(payload, "provider")?;
    let operation = required_string(payload, "operation")?;

    match provider {
        "mistral" => transform_with_provider(&MISTRAL_OCR_CONFIG, operation, payload),
        _ => Err(CoreError::InvalidResponse(format!(
            "unsupported OCR provider: {provider}"
        ))),
    }
}

fn transform_with_provider(
    provider_config: &impl OcrProviderConfig,
    operation: &str,
    payload: &Map<String, Value>,
) -> CoreResult<Value> {
    match operation {
        "map_params" => {
            let params = required_object(payload, "non_default_params")?;
            Ok(Value::Object(provider_config.map_ocr_params(&params)))
        }
        "transform_request" => {
            let model = required_string(payload, "model")?;
            let document = required_value(payload, "document")?;
            let optional_params = required_object(payload, "optional_params")?;
            let transformed =
                provider_config.transform_ocr_request(model, document, optional_params)?;
            Ok(serde_json::json!({
                "data": transformed.data,
                "files": transformed.files,
            }))
        }
        "transform_response" => {
            let model = required_string(payload, "model")?;
            let response_json = required_value(payload, "response_json")?;
            let transformed = provider_config.transform_ocr_response(model, response_json)?;
            Ok(transformed.into_json())
        }
        _ => Err(CoreError::InvalidResponse(format!(
            "unsupported OCR operation: {operation}"
        ))),
    }
}

fn payload_object(payload: &Value) -> CoreResult<&Map<String, Value>> {
    payload.as_object().ok_or_else(|| CoreError::InvalidType {
        expected: "object",
        actual: json_type_name(payload),
    })
}

fn required_string<'a>(payload: &'a Map<String, Value>, key: &'static str) -> CoreResult<&'a str> {
    let value = payload.get(key).ok_or(CoreError::MissingField(key))?;
    value.as_str().ok_or_else(|| CoreError::InvalidType {
        expected: "string",
        actual: json_type_name(value),
    })
}

fn required_object(
    payload: &Map<String, Value>,
    key: &'static str,
) -> CoreResult<Map<String, Value>> {
    let value = payload.get(key).ok_or(CoreError::MissingField(key))?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::InvalidType {
            expected: "object",
            actual: json_type_name(value),
        })
}

fn required_value(payload: &Map<String, Value>, key: &'static str) -> CoreResult<Value> {
    payload
        .get(key)
        .cloned()
        .ok_or(CoreError::MissingField(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn transform_dispatches_mistral_map_params() {
        let result = transform(json!({
            "provider": "mistral",
            "operation": "map_params",
            "non_default_params": {
                "extract_header": true,
                "unsupported_param": "value"
            }
        }))
        .expect("payload should transform");

        assert_eq!(result, json!({"extract_header": true}));
    }

    #[test]
    fn transform_dispatches_mistral_request() {
        let document = json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        });

        let result = transform(json!({
            "provider": "mistral",
            "operation": "transform_request",
            "model": "mistral-ocr-latest",
            "document": document,
            "optional_params": {"include_image_base64": true}
        }))
        .expect("payload should transform");

        assert_eq!(
            result,
            json!({
                "data": {
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "document_url",
                        "document_url": "https://example.com/doc.pdf"
                    },
                    "include_image_base64": true
                },
                "files": null
            })
        );
    }

    #[test]
    fn transform_rejects_unknown_provider() {
        let err = transform(json!({
            "provider": "azure_ai",
            "operation": "map_params",
            "non_default_params": {}
        }))
        .expect_err("unsupported provider should fail");

        assert_eq!(
            err,
            CoreError::InvalidResponse("unsupported OCR provider: azure_ai".to_string())
        );
    }
}
