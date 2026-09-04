use litellm_core::Error;
use litellm_core::http_utils::http_request;
use litellm_core::ocr::transformation::OcrResponseHandling;
use serde_json::Value;

use super::common_utils::poll_document_intelligence;
use super::hooks::OcrLifecycleHooks;
use super::types::PreparedOcrRequest;
use crate::client::http_client;
use crate::error::invalid_json_response;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) async fn execute_ocr_provider_call(
    request: PreparedOcrRequest,
    hooks: &OcrLifecycleHooks,
) -> Result<Value, Error> {
    let request = hooks.prepare_provider_request(request).await?;
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        if !key.eq_ignore_ascii_case("content-type") {
            request_builder = request_builder.header(key, value);
        }
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = http_request(request_builder)
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

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
            .transform_ocr_response_with_params(
                &request.model,
                response_json,
                &request.optional_params,
            )?
            .into_json());
    }

    let text = response
        .text()
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: text,
        });
    }

    let response_json: Value =
        serde_json::from_str(&text).map_err(|error| invalid_json_response("OCR", error))?;

    Ok(request
        .config
        .transform_ocr_response_with_params(
            &request.model,
            response_json,
            &request.optional_params,
        )?
        .into_json())
}
