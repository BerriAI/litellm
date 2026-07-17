use std::collections::BTreeMap;

use crate::error::{json_type_name, CoreError, CoreResult};
use crate::ocr::transformation::{OcrDocumentPreparation, OcrProviderConfig};
use crate::ocr::types::{OcrRequestData, OcrResponseData};
use serde_json::{json, Map, Value};

const REDUCTO_API_BASE: &str = "https://platform.reducto.ai";
const REDUCTO_API_KEY_ENV: &str = "REDUCTO_API_KEY";
const REDUCTO_PARSE_V3_PARAMS: &[&str] = &["formatting", "retrieval", "settings"];
const REDUCTO_PARSE_LEGACY_PARAMS: &[&str] = &["enhance"];

pub struct ReductoParseV3Config;
pub struct ReductoParseLegacyConfig;

pub const REDUCTO_PARSE_V3_CONFIG: ReductoParseV3Config = ReductoParseV3Config;
pub const REDUCTO_PARSE_LEGACY_CONFIG: ReductoParseLegacyConfig = ReductoParseLegacyConfig;

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(REDUCTO_API_KEY_ENV).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| {
            CoreError::Auth(
                "Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()"
                    .to_string(),
            )
        })
}

pub fn complete_url(api_base: Option<&str>) -> String {
    let base = non_empty(api_base).unwrap_or(REDUCTO_API_BASE);
    format!("{}/parse", base.trim_end_matches('/'))
}

fn source_url<'a>(document: &'a Value, model: &str) -> CoreResult<&'a str> {
    let object = document.as_object().ok_or_else(|| CoreError::InvalidType {
        expected: "object",
        actual: json_type_name(document),
    })?;
    object
        .get("document_url")
        .or_else(|| object.get("image_url"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            CoreError::InvalidRequest(format!(
                "Reducto expected OCR preprocessing to produce document_url or image_url for model={model}"
            ))
        })
}

fn page_no(block: &Value) -> Option<i64> {
    block
        .get("bbox")
        .and_then(|bbox| bbox.get("page"))
        .and_then(|page| match page {
            Value::Number(number) => number.as_i64(),
            Value::String(value) => value.parse::<i64>().ok(),
            _ => None,
        })
}

fn block_content(block: &Value) -> Option<&str> {
    block.get("content").and_then(Value::as_str)
}

fn build_pages_from_reducto(result: &Value) -> Vec<Value> {
    let chunks = result
        .get("chunks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut blocks_by_page: BTreeMap<i64, Vec<Value>> = BTreeMap::new();

    for chunk in &chunks {
        let Some(blocks) = chunk.get("blocks").and_then(Value::as_array) else {
            continue;
        };
        for block in blocks {
            if let Some(page) = page_no(block) {
                blocks_by_page.entry(page).or_default().push(block.clone());
            }
        }
    }

    if blocks_by_page.is_empty() {
        let markdown = chunks
            .iter()
            .filter_map(|chunk| chunk.get("content").and_then(Value::as_str))
            .filter(|content| !content.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if markdown.is_empty() {
            return Vec::new();
        }
        return vec![json!({"index": 0, "markdown": markdown})];
    }

    blocks_by_page
        .into_iter()
        .map(|(page, blocks)| {
            let markdown = blocks
                .iter()
                .filter_map(block_content)
                .filter(|content| !content.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            json!({
                "index": (page - 1).max(0),
                "markdown": markdown,
                "blocks": blocks,
            })
        })
        .collect()
}

impl OcrProviderConfig for ReductoParseV3Config {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        REDUCTO_PARSE_V3_PARAMS
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        let mut data = Map::new();
        data.insert(
            "input".to_string(),
            Value::String(source_url(&document, model)?.to_string()),
        );
        for (key, value) in optional_params {
            data.insert(key, value);
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
        transform_reducto_response(model, response_json)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(complete_url(api_base))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_api_key(api_key, env_lookup)
    }

    fn document_preparation(&self) -> OcrDocumentPreparation {
        OcrDocumentPreparation::ReductoUpload
    }
}

impl OcrProviderConfig for ReductoParseLegacyConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        REDUCTO_PARSE_LEGACY_PARAMS
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        let mut data = Map::new();
        data.insert(
            "document_url".to_string(),
            Value::String(source_url(&document, model)?.to_string()),
        );
        if let Some(enhance) = optional_params.get("enhance") {
            data.insert("options".to_string(), json!({"enhance": enhance}));
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
        transform_reducto_response(model, response_json)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(complete_url(api_base))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_api_key(api_key, env_lookup)
    }

    fn document_preparation(&self) -> OcrDocumentPreparation {
        OcrDocumentPreparation::ReductoUpload
    }
}

fn transform_reducto_response(model: &str, response_json: Value) -> CoreResult<OcrResponseData> {
    let response = response_json
        .as_object()
        .ok_or_else(|| CoreError::unexpected_response_type(&response_json))?;
    let result = response.get("result").unwrap_or(&response_json);
    let usage = response.get("usage").cloned().unwrap_or_else(|| json!({}));
    Ok(OcrResponseData {
        pages: build_pages_from_reducto(result),
        model: model.to_string(),
        document_annotation: None,
        usage_info: Some(json!({
            "pages_processed": usage.get("num_pages").cloned().unwrap_or(Value::Null),
            "credits": usage.get("credits").cloned().unwrap_or(Value::Null),
        })),
        object: "ocr".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_v3_request_uses_uploaded_file_id() {
        let body = REDUCTO_PARSE_V3_CONFIG
            .transform_ocr_request(
                "parse-v3",
                json!({"type": "document_url", "document_url": "reducto://file-1"}),
                Map::from_iter([("settings".to_string(), json!({"ocr": true}))]),
            )
            .expect("request transforms")
            .data;

        assert_eq!(
            body,
            json!({"input": "reducto://file-1", "settings": {"ocr": true}})
        );
    }

    #[test]
    fn parse_legacy_request_maps_enhance_into_options() {
        let body = REDUCTO_PARSE_LEGACY_CONFIG
            .transform_ocr_request(
                "parse-legacy",
                json!({"type": "document_url", "document_url": "reducto://file-1"}),
                Map::from_iter([("enhance".to_string(), json!(true))]),
            )
            .expect("request transforms")
            .data;

        assert_eq!(
            body,
            json!({"document_url": "reducto://file-1", "options": {"enhance": true}})
        );
    }

    #[test]
    fn reducto_response_groups_blocks_by_page() {
        let response = REDUCTO_PARSE_V3_CONFIG
            .transform_ocr_response(
                "parse-v3",
                json!({
                    "result": {
                        "chunks": [{
                            "blocks": [
                                {"content": "p1", "bbox": {"page": 1}},
                                {"content": "p2", "bbox": {"page": 2}}
                            ]
                        }]
                    },
                    "usage": {"num_pages": 2, "credits": 1}
                }),
            )
            .expect("response transforms");

        assert_eq!(response.pages[0]["index"], 0);
        assert_eq!(response.pages[0]["markdown"], "p1");
        assert_eq!(response.pages[1]["index"], 1);
        assert_eq!(
            response.usage_info,
            Some(json!({"pages_processed": 2, "credits": 1}))
        );
    }
}
