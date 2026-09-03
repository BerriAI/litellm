use litellm_core::error::Error;
use litellm_core::ocr::handler::send_ocr_request;
use litellm_core::ocr::observers::{OcrObserver, OcrPreCall};
use litellm_core::ocr::transformation::OcrResponseHandling;
use litellm_core::ocr::types::OcrRequestData;
use serde_json::Value;

use super::common_utils::poll_document_intelligence;
use super::hooks::OcrLifecycleHooks;
use super::types::PreparedOcrRequest;
use crate::client::http_client;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) async fn execute_ocr_provider_call(
    request: PreparedOcrRequest,
    hooks: &OcrLifecycleHooks,
    observer: &mut impl OcrObserver,
) -> Result<Value, Error> {
    let request = hooks.prepare_provider_request(request).await?;
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let event = OcrPreCall {
        model: request.model.clone(),
        request: OcrRequestData {
            data: request.body,
            files: None,
        },
        api_base: request.url.clone(),
        headers: request.upstream_headers.iter().cloned().collect(),
    };
    let response = send_ocr_request(request_builder, &event, observer).await?;

    let status = response.status;
    if request.config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll
        && status.as_u16() == 202
    {
        let operation_url = response
            .headers
            .get("operation-location")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| {
                Error::InvalidResponse(
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

    let response_json: Value = serde_json::from_str(&response.body)
        .map_err(|err| Error::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(request
        .config
        .transform_ocr_response(&request.model, response_json)?
        .into_json())
}
