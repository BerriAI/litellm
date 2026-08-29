use crate::CoreResult;
use crate::error::CoreError;
use crate::http_utils::{map_send_error, upstream_http};
use crate::ocr::transformation::OcrResponseHandling;
use crate::providers::azure_ai::ocr::poll_document_intelligence;
use serde_json::Value;

use super::client::http_client;
use super::types::ProviderOcrRequest;

pub(crate) async fn execute_ocr_provider_call(request: ProviderOcrRequest) -> CoreResult<Value> {
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder.send().await.map_err(map_send_error)?;

    let status = response.status();
    if request.config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll
        && status.as_u16() == 202
    {
        let operation_url = response
            .headers()
            .get("operation-location")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| {
                CoreError::invalid_response(
                    "Azure Document Intelligence returned 202 but no Operation-Location header found"
                        .to_string(),
                )
        })?;
        let response_json = poll_document_intelligence(
            http_client(),
            &operation_url,
            &request.url,
            &request.upstream_headers,
            request.timeout,
        )
        .await?;
        return Ok(request
            .config
            .transform_ocr_response(&request.model, response_json)?
            .into_json());
    }

    let text = response
        .text()
        .await
        .map_err(|err| CoreError::network(err.to_string()))?;

    if !status.is_success() {
        return Err(upstream_http(status, &text));
    }

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| CoreError::invalid_response(format!("invalid OCR response JSON: {err}")))?;

    Ok(request
        .config
        .transform_ocr_response(&request.model, response_json)?
        .into_json())
}
