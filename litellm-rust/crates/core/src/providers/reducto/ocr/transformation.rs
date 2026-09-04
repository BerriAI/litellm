use std::collections::BTreeMap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde_json::{Map, Value, json};

use crate::error::{Error, json_type_name};
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::{OcrRequestData, OcrResponseData};

pub const REDUCTO_API_BASE: &str = "https://platform.reducto.ai";
pub const REDUCTO_API_KEY_ENV: &str = "REDUCTO_API_KEY";
pub const REDUCTO_ID_PREFIX: &str = "reducto://";

const PARSE_V3_SUPPORTED_OCR_PARAMS: &[&str] = &["formatting", "retrieval", "settings"];
const PARSE_LEGACY_SUPPORTED_OCR_PARAMS: &[&str] = &["enhance"];
const MISSING_KEY_MESSAGE: &str = "Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()";
const DATA_URI_UPLOAD_REQUIRED: &str =
    "Reducto data URI upload must complete before OCR request transformation";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReductoDocumentSource {
    FileId(String),
    Upload { bytes: Vec<u8>, mime_type: String },
}

#[derive(Clone, PartialEq, Eq)]
pub struct ReductoUploadRequest {
    pub url: String,
    pub authorization: String,
    pub file_name: &'static str,
    pub bytes: Vec<u8>,
    pub mime_type: String,
}

pub struct ReductoParseV3Config;
pub struct ReductoParseLegacyConfig;

pub const REDUCTO_PARSE_V3_CONFIG: ReductoParseV3Config = ReductoParseV3Config;
pub const REDUCTO_PARSE_LEGACY_CONFIG: ReductoParseLegacyConfig = ReductoParseLegacyConfig;

pub fn config_for_model(model: &str) -> Option<&'static dyn OcrProviderConfig> {
    match model {
        "parse-v3" => Some(&REDUCTO_PARSE_V3_CONFIG),
        "parse-legacy" => Some(&REDUCTO_PARSE_LEGACY_CONFIG),
        _ => None,
    }
}

pub fn normalize_api_base(api_base: Option<&str>) -> String {
    api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(REDUCTO_API_BASE)
        .trim_end_matches('/')
        .to_string()
}

pub fn parse_url(api_base: Option<&str>) -> String {
    format!("{}/parse", normalize_api_base(api_base))
}

pub fn upload_url(api_base: Option<&str>) -> String {
    format!("{}/upload", normalize_api_base(api_base))
}

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| {
            env_lookup(REDUCTO_API_KEY_ENV)
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty())
        })
        .ok_or_else(|| Error::Auth(MISSING_KEY_MESSAGE.to_string()))
}

pub fn extract_document_source(document: &Value) -> Result<ReductoDocumentSource, Error> {
    let document = document.as_object().ok_or_else(|| Error::InvalidType {
        expected: "object",
        actual: json_type_name(document),
    })?;
    let source = document
        .get("document_url")
        .and_then(Value::as_str)
        .filter(|source| !source.is_empty())
        .or_else(|| document.get("image_url").and_then(Value::as_str))
        .ok_or_else(|| {
            Error::InvalidRequest(
                "Reducto expected OCR preprocessing to produce document_url or image_url"
                    .to_string(),
            )
        })?;
    classify_document_source(source)
}

pub fn classify_document_source(source: &str) -> Result<ReductoDocumentSource, Error> {
    if source.starts_with(REDUCTO_ID_PREFIX) {
        return Ok(ReductoDocumentSource::FileId(source.to_string()));
    }
    if source.starts_with("http://") || source.starts_with("https://") {
        return Err(Error::InvalidRequest(
            "Reducto requires type='file' (auto-uploaded) or a reducto:// id. Plain http(s) URLs are not supported; upload the file first."
                .to_string(),
        ));
    }
    if !source.starts_with("data:") {
        return Err(Error::InvalidRequest(
            "Reducto requires a reducto:// id or a base64 data URI after OCR preprocessing."
                .to_string(),
        ));
    }

    let (header, encoded) = source
        .split_once(',')
        .ok_or_else(|| Error::InvalidRequest("Invalid Reducto data URI provided.".to_string()))?;
    if !header.split(';').any(|part| part == "base64") {
        return Err(Error::InvalidRequest(
            "Reducto only supports base64-encoded data URIs.".to_string(),
        ));
    }

    let mime_type = header
        .strip_prefix("data:")
        .and_then(|header| header.split(';').next())
        .filter(|mime| !mime.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    let bytes = BASE64_STANDARD.decode(encoded).map_err(|_| {
        Error::InvalidRequest("Invalid Reducto base64 payload provided.".to_string())
    })?;

    Ok(ReductoDocumentSource::Upload { bytes, mime_type })
}

pub fn build_upload_request(
    source: ReductoDocumentSource,
    authorization: &str,
    api_base: Option<&str>,
) -> Option<ReductoUploadRequest> {
    let ReductoDocumentSource::Upload { bytes, mime_type } = source else {
        return None;
    };

    Some(ReductoUploadRequest {
        url: upload_url(api_base),
        authorization: authorization.to_string(),
        file_name: "document",
        bytes,
        mime_type,
    })
}

pub fn extract_upload_file_id(response_json: &Value) -> Result<&str, Error> {
    response_json
        .as_object()
        .and_then(|response| response.get("file_id"))
        .and_then(Value::as_str)
        .filter(|file_id| !file_id.is_empty())
        .ok_or_else(|| {
            Error::InvalidResponse(format!(
                "Reducto /upload returned 200 without a file_id; got payload={response_json}"
            ))
        })
}

pub fn build_parse_v3_request(
    file_id: &str,
    optional_params: Map<String, Value>,
) -> OcrRequestData {
    let data = std::iter::once(("input".to_string(), Value::String(file_id.to_string())))
        .chain(optional_params)
        .collect();
    OcrRequestData {
        data: Value::Object(data),
        files: None,
    }
}

pub fn build_parse_legacy_request(
    file_id: &str,
    optional_params: &Map<String, Value>,
) -> OcrRequestData {
    let options = optional_params
        .get("enhance")
        .filter(|enhance| !enhance.is_null())
        .map(|enhance| json!({"options": {"enhance": enhance}}));
    let data = match options {
        Some(Value::Object(options)) => std::iter::once((
            "document_url".to_string(),
            Value::String(file_id.to_string()),
        ))
        .chain(options)
        .collect(),
        _ => Map::from_iter([(
            "document_url".to_string(),
            Value::String(file_id.to_string()),
        )]),
    };
    OcrRequestData {
        data: Value::Object(data),
        files: None,
    }
}

fn source_file_id(document: &Value) -> Result<String, Error> {
    match extract_document_source(document)? {
        ReductoDocumentSource::FileId(file_id) => Ok(file_id),
        ReductoDocumentSource::Upload { .. } => Err(Error::Unsupported(DATA_URI_UPLOAD_REQUIRED)),
    }
}

fn page_number(block: &Map<String, Value>) -> Option<i64> {
    let page = block.get("bbox")?.as_object()?.get("page")?;
    page.as_i64()
        .or_else(|| page.as_u64().and_then(|page| i64::try_from(page).ok()))
        .or_else(|| page.as_str().and_then(|page| page.parse().ok()))
}

fn chunks(result: &Map<String, Value>) -> &[Value] {
    result
        .get("chunks")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

fn build_pages(result: &Map<String, Value>) -> Vec<Value> {
    let blocks_by_page = chunks(result)
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|chunk| chunk.get("blocks").and_then(Value::as_array))
        .flatten()
        .filter_map(|block| block.as_object().map(|object| (block, object)))
        .filter_map(|(block, object)| page_number(object).map(|page| (page, block.clone())))
        .fold(
            BTreeMap::<i64, Vec<Value>>::new(),
            |mut pages, (page, block)| {
                pages.entry(page).or_default().push(block);
                pages
            },
        );

    if blocks_by_page.is_empty() {
        let markdown = chunks(result)
            .iter()
            .filter_map(Value::as_object)
            .filter_map(|chunk| chunk.get("content").and_then(Value::as_str))
            .filter(|content| !content.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        return if markdown.is_empty() {
            Vec::new()
        } else {
            vec![json!({"index": 0, "markdown": markdown})]
        };
    }

    blocks_by_page
        .into_iter()
        .map(|(page, blocks)| {
            let markdown = blocks
                .iter()
                .filter_map(Value::as_object)
                .filter_map(|block| block.get("content").and_then(Value::as_str))
                .filter(|content| !content.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            json!({
                "index": page.saturating_sub(1).max(0),
                "markdown": markdown,
                "blocks": blocks,
            })
        })
        .collect()
}

pub fn transform_reducto_response(
    model: &str,
    response_json: Value,
) -> Result<OcrResponseData, Error> {
    let response = response_json
        .as_object()
        .ok_or_else(|| Error::InvalidType {
            expected: "object",
            actual: json_type_name(&response_json),
        })?;
    let empty_result = Map::new();
    let result = match response.get("result") {
        Some(Value::Object(result)) => result,
        Some(Value::Null) => &empty_result,
        Some(_) => {
            return Err(Error::InvalidResponse(
                "Reducto result must be an object".to_string(),
            ));
        }
        None => response,
    };
    let usage = response
        .get("usage")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let usage_info = Some(json!({
        "pages_processed": usage.get("num_pages").cloned().unwrap_or(Value::Null),
        "credits": usage.get("credits").cloned().unwrap_or(Value::Null),
    }));

    Ok(OcrResponseData {
        pages: build_pages(result),
        model: model.to_string(),
        document_annotation: None,
        usage_info,
        object: "ocr".to_string(),
        extra_fields: Map::new(),
        provider_native_response: Some(response_json),
    })
}

impl OcrProviderConfig for ReductoParseV3Config {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        PARSE_V3_SUPPORTED_OCR_PARAMS
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> Result<OcrRequestData, Error> {
        let file_id = source_file_id(&document)?;
        Ok(build_parse_v3_request(&file_id, optional_params))
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<OcrResponseData, Error> {
        transform_reducto_response(model, response_json)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(parse_url(api_base))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        resolve_api_key(api_key, env_lookup)
    }
}

impl OcrProviderConfig for ReductoParseLegacyConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        PARSE_LEGACY_SUPPORTED_OCR_PARAMS
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> Result<OcrRequestData, Error> {
        let file_id = source_file_id(&document)?;
        Ok(build_parse_legacy_request(&file_id, &optional_params))
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<OcrResponseData, Error> {
        transform_reducto_response(model, response_json)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(parse_url(api_base))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        resolve_api_key(api_key, env_lookup)
    }
}
