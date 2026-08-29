use crate::CoreResult;
use crate::error::{CoreError, json_type_name};
use crate::http_utils::{http_client, json_response, map_send_error};
use crate::ocr::transformation::OcrResponseHandling;
use crate::providers::azure_ai::ocr::poll_document_intelligence;
use serde_json::Value;

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
        let Value::Object(response_json) = response_json else {
            return Err(CoreError::invalid_response(format!(
                "invalid OCR response type: expected object, got {}",
                json_type_name(&response_json)
            )));
        };
        return Ok(request
            .config
            .transform_ocr_response(&request.model, response_json)?
            .into_json());
    }

    let response_json = json_response(response, "invalid OCR response JSON").await?;
    let Value::Object(response_json) = response_json else {
        return Err(CoreError::invalid_response(format!(
            "invalid OCR response type: expected object, got {}",
            json_type_name(&response_json)
        )));
    };

    Ok(request
        .config
        .transform_ocr_response(&request.model, response_json)?
        .into_json())
}
