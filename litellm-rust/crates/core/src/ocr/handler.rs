use reqwest::{RequestBuilder, StatusCode, header::HeaderMap};
use serde_json::Value;

use super::client::http_client;
use super::common_utils::poll_document_intelligence;
use super::observers::{OcrObserver, OcrPostCall, OcrPreCall};
use super::transformation::OcrResponseHandling;
use super::types::{OcrRequestData, ProviderOcrRequest};
use crate::Error;
use crate::http_utils::{http_request, truncate_error_body};

pub struct OcrHttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn send_ocr_request(
    request: RequestBuilder,
    event: &OcrPreCall,
    observer: &mut impl OcrObserver,
) -> Result<OcrHttpResponse, Error> {
    if observer.pre_call(event).await.is_err() {
        tracing::warn!("OCR pre-call observer failed");
    }
    let response = http_request(request).await.map_err(transport_error)?;
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.text().await.map_err(transport_error)?;
    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&body),
        });
    }
    let event = OcrPostCall {
        original_response: body,
    };
    if observer.post_call(&event).await.is_err() {
        tracing::warn!("OCR post-call observer failed");
    }
    Ok(OcrHttpResponse {
        status,
        headers,
        body: event.original_response,
    })
}

fn transport_error(error: reqwest::Error) -> Error {
    Error::Network(if error.is_timeout() {
        "Request timed out".into()
    } else {
        error.to_string()
    })
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn execute_ocr_provider_call(
    request: ProviderOcrRequest,
    observer: &mut impl OcrObserver,
) -> Result<Value, Error> {
    let mut request_builder = http_client().post(request.url()).json(request.body());
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let event = OcrPreCall {
        model: request.model().to_string(),
        request: OcrRequestData {
            data: request.body().clone(),
            files: None,
        },
        api_base: request.url().to_string(),
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
            request.url(),
            &request.upstream_headers,
            request.timeout,
        )
        .await?;
        return Ok(request
            .config
            .transform_ocr_response(request.model(), response_json)?
            .into_json());
    }

    let response_json: Value = serde_json::from_str(&response.body)
        .map_err(|err| Error::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(request
        .config
        .transform_ocr_response(request.model(), response_json)?
        .into_json())
}
