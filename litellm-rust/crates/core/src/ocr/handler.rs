use std::time::Duration;

use serde_json::{Map, Value, json};

use super::client::http_client;
use super::common_utils::{
    convert_document_url_to_data_uri, poll_document_intelligence, string_headers,
};
use super::types::PreparedOcrRequest;
use crate::error::Error;
use crate::http_utils::{http_request, truncate_error_body};
use crate::ocr::transformation::{OcrProviderConfig, OcrResponseHandling};
use crate::providers::reducto::ocr::transformation::{
    build_upload_request, extract_document_source, extract_upload_file_id,
};

fn public_response(
    config: &'static dyn OcrProviderConfig,
    model: &str,
    optional_params: &Map<String, Value>,
    response_json: Value,
) -> Result<Value, Error> {
    let mut response =
        config.transform_ocr_response_with_params(model, response_json, optional_params)?;
    response.provider_native_response = None;
    Ok(response.into_json())
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) async fn execute_ocr_provider_call(request: PreparedOcrRequest) -> Result<Value, Error> {
    let PreparedOcrRequest {
        model,
        config,
        document,
        api_key,
        api_base,
        extra_headers,
        url_params,
        optional_params,
        requires_reducto_upload,
        timeout,
    } = request;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let upstream_headers = config.validate_environment(
        string_headers(extra_headers)?,
        api_key.as_deref(),
        &env_lookup,
    )?;
    let url = config.complete_url(api_base.as_deref(), &model, &url_params, &env_lookup)?;
    let document = if config.requires_data_uri_document() {
        convert_document_url_to_data_uri(document).await?
    } else if requires_reducto_upload {
        upload_reducto_document(document, api_base.as_deref(), timeout, &upstream_headers).await?
    } else {
        document
    };
    let body = config
        .transform_ocr_request(&model, document, optional_params.clone())?
        .data;
    let mut request_builder = http_client().post(&url).json(&body);
    for (key, value) in upstream_headers.iter().filter(|(key, _)| {
        !key.eq_ignore_ascii_case("content-type") && !key.eq_ignore_ascii_case("content-length")
    }) {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = http_request(request_builder)
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    let status = response.status();
    if config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll
        && status.as_u16() == 202
    {
        let operation_url = response
            .headers()
            .get("operation-location")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| {
                Error::InvalidResponse(
                    "Azure Document Intelligence returned 202 but no Operation-Location header found"
                        .to_string(),
                )
            })?;
        let response_json =
            poll_document_intelligence(&operation_url, &url, &upstream_headers, timeout).await?;
        return public_response(config, &model, &optional_params, response_json);
    }

    let text = response
        .text()
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| Error::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    public_response(config, &model, &optional_params, response_json)
}

async fn upload_reducto_document(
    document: Value,
    api_base: Option<&str>,
    timeout: Option<Duration>,
    upstream_headers: &[(String, String)],
) -> Result<Value, Error> {
    let source = extract_document_source(&document)?;
    let authorization = upstream_headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .map(|(_, value)| value.as_str())
        .ok_or_else(|| {
            Error::Auth("Reducto upload requires an Authorization header".to_string())
        })?;
    let Some(upload) = build_upload_request(source, authorization, api_base) else {
        return Ok(document);
    };
    let part = reqwest::multipart::Part::bytes(upload.bytes)
        .file_name(upload.file_name)
        .mime_str(&upload.mime_type)
        .map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let form = reqwest::multipart::Form::new().part("file", part);
    let request_builder = upstream_headers
        .iter()
        .filter(|(name, _)| {
            !name.eq_ignore_ascii_case("content-type")
                && !name.eq_ignore_ascii_case("content-length")
        })
        .fold(
            http_client().post(upload.url).multipart(form),
            |builder, (name, value)| builder.header(name, value),
        );
    let request_builder = match timeout {
        Some(duration) => request_builder.timeout(duration),
        None => request_builder,
    };
    let response = http_request(request_builder)
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&body),
        });
    }
    let response_json: Value = serde_json::from_str(&body).map_err(|error| {
        Error::InvalidResponse(format!("invalid Reducto upload response JSON: {error}"))
    })?;
    let file_id = extract_upload_file_id(&response_json)?;
    Ok(json!({"type": "document_url", "document_url": file_id}))
}
