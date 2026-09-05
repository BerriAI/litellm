use litellm_core::call_lifecycle::CallLifecycleContext;
use litellm_core::error::Error;
use litellm_core::ocr::transformation::OcrResponseHandling;
use litellm_core::provider_callbacks::ProviderAttemptObserver;
use litellm_core::provider_callbacks::handler::{
    ProviderAttemptContext, ProviderRequest, send_provider_request,
};
use serde_json::Value;

use super::common_utils::poll_document_intelligence;
use super::hooks::OcrLifecycleHooks;
use super::types::PreparedOcrRequest;
use crate::client::http_client;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) async fn execute_ocr_provider_call<Observer>(
    request: PreparedOcrRequest,
    context: &CallLifecycleContext,
    hooks: &OcrLifecycleHooks,
    observer: &mut Observer,
) -> Result<Value, Error>
where
    Observer: ProviderAttemptObserver,
    Observer::Error: std::fmt::Display,
{
    let request = hooks.prepare_provider_request(request).await?;
    let provider_request = ProviderRequest {
        api_key: None,
        provider: request.custom_llm_provider.clone(),
        model: request.model.clone(),
        body: serde_json::from_value(request.body).map_err(|error| {
            Error::InvalidRequest(format!("OCR provider request must be an object: {error}"))
        })?,
        api_base: request.url.clone(),
        headers: request.upstream_headers.iter().cloned().collect(),
    };
    let mut request_builder = http_client().post(&request.url);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = send_provider_request(
        request_builder,
        provider_request,
        ProviderAttemptContext {
            call_id: context.litellm_call_id.clone(),
            trace_id: None,
            attempt: 1,
        },
        observer,
    )
    .await?;

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
            .transform_ocr_response_with_params(
                &request.model,
                response_json,
                &request.optional_params,
            )?
            .into_json());
    }

    let response_json: Value = serde_json::from_str(&response.body)
        .map_err(|err| Error::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(request
        .config
        .transform_ocr_response_with_params(
            &request.model,
            response_json,
            &request.optional_params,
        )?
        .into_json())
}
