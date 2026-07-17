use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use litellm_core::error::CoreError;
use litellm_core::CoreResult;
use reqwest::Url;
use serde_json::{Map, Value};

use super::common_utils::{document_url_field, read_error_body};
use super::http_client;

fn decode_reducto_data_uri(source_url: &str) -> CoreResult<(Vec<u8>, String)> {
    let (header, encoded) = source_url.split_once(',').ok_or_else(|| {
        CoreError::InvalidRequest("Invalid Reducto data URI provided.".to_string())
    })?;
    if !header.contains(";base64") {
        return Err(CoreError::InvalidRequest(
            "Reducto only supports base64-encoded data URIs.".to_string(),
        ));
    }
    let mime = header
        .strip_prefix("data:")
        .and_then(|value| value.split(';').next())
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    let bytes = BASE64_STANDARD.decode(encoded).map_err(|_| {
        CoreError::InvalidRequest("Invalid Reducto base64 payload provided.".to_string())
    })?;
    Ok((bytes, mime))
}

fn reducto_upload_url(parse_url: &str) -> CoreResult<Url> {
    let mut url = Url::parse(parse_url)
        .map_err(|err| CoreError::InvalidRequest(format!("invalid Reducto parse URL: {err}")))?;
    let path = url.path().trim_end_matches('/');
    let base_path = path
        .strip_suffix("/parse")
        .unwrap_or(path)
        .trim_end_matches('/');
    url.set_path(&format!("{base_path}/upload"));
    url.set_query(None);
    Ok(url)
}

fn reducto_auth_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    headers
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case("authorization"))
        .cloned()
        .collect()
}

async fn upload_reducto_bytes(
    bytes: Vec<u8>,
    mime: String,
    parse_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<String> {
    let upload_url = reducto_upload_url(parse_url)?;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("document")
        .mime_str(&mime)
        .map_err(|err| {
            CoreError::InvalidRequest(format!("invalid Reducto upload MIME type: {err}"))
        })?;
    let form = reqwest::multipart::Form::new().part("file", part);
    let mut request_builder = http_client().post(upload_url).multipart(form);
    for (key, value) in reducto_auth_headers(headers) {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: read_error_body(response).await,
        });
    }
    let text = response
        .text()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let response_json: Value = serde_json::from_str(&text).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid Reducto upload response JSON: {err}"))
    })?;
    response_json
        .get("file_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            CoreError::InvalidResponse("Reducto /upload returned 200 without a file_id".to_string())
        })
}

pub(super) async fn upload_reducto_document(
    document: Value,
    parse_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<Value> {
    let Some((field, source_url)) = document_url_field(&document)? else {
        return Err(CoreError::InvalidRequest(
            "Reducto expected OCR preprocessing to produce document_url or image_url".to_string(),
        ));
    };
    if source_url.starts_with("reducto://") {
        return Ok(document);
    }
    if source_url.starts_with("http://") || source_url.starts_with("https://") {
        return Err(CoreError::InvalidRequest(
            "Reducto requires type='file' (auto-uploaded) or a reducto:// id. Plain http(s) URLs are not supported; upload the file first."
                .to_string(),
        ));
    }
    if !source_url.starts_with("data:") {
        return Err(CoreError::InvalidRequest(
            "Reducto requires a reducto:// id or a base64 data URI after OCR preprocessing."
                .to_string(),
        ));
    }

    let (bytes, mime) = decode_reducto_data_uri(source_url)?;
    let file_id = upload_reducto_bytes(bytes, mime, parse_url, headers, timeout).await?;
    let object = document
        .as_object()
        .ok_or_else(|| CoreError::InvalidRequest("OCR document must be an object".to_string()))?;
    let transformed: Map<String, Value> = object
        .iter()
        .map(|(key, value)| {
            if key == field {
                (key.clone(), Value::String(file_id.clone()))
            } else {
                (key.clone(), value.clone())
            }
        })
        .collect();
    Ok(Value::Object(transformed))
}
