use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrResponseHandling;
use litellm_core::CoreResult;
use serde_json::Value;

use super::client::http_client;
use super::common_utils::{
    classify_reqwest_error, poll_document_intelligence, truncate_error_body,
};
use super::types::ProviderOcrRequest;

pub(crate) async fn execute_ocr_provider_call(request: ProviderOcrRequest) -> CoreResult<Value> {
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(classify_reqwest_error)?;

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
                CoreError::InvalidResponse(
                    "Azure Document Intelligence returned 202 but no Operation-Location header found"
                        .to_string(),
                )
            })?;
        let response_json = poll_document_intelligence(
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

    let text = response.text().await.map_err(classify_reqwest_error)?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    if text.trim().is_empty() {
        return Err(CoreError::InvalidResponse(
            "OCR provider returned an empty success response".to_string(),
        ));
    }

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(request
        .config
        .transform_ocr_response(&request.model, response_json)?
        .into_json())
}
