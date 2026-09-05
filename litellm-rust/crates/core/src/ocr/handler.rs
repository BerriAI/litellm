use std::time::Duration;

use reqwest::{
    Method, StatusCode, Url,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::Value;

use super::client::http_client;
use super::common_utils::poll_document_intelligence;
use super::observers::{OcrObserver, OcrPostCall, OcrPreCall};
use super::transformation::OcrResponseHandling;
use super::types::{OcrRequestData, ProviderOcrRequest};
use crate::Error;
use crate::auth::{
    BodyDecision, NoAuthorization, OperationControl, OutboundBody, OutboundOperation,
    OutboundOperationKind, send_once,
};
use crate::constants::OCR_TIMEOUT_SECS;
use crate::http_utils::truncate_error_body;

pub struct OcrHttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn send_ocr_request(
    origin: &Url,
    request: OutboundOperation,
    event: &OcrPreCall,
    observer: &mut impl OcrObserver,
    control: &OperationControl,
) -> Result<OcrHttpResponse, Error> {
    let response = send_once(
        http_client(),
        origin,
        request,
        |_| async {
            if observer.pre_call(event).await.is_err() {
                tracing::warn!("OCR pre-call observer failed");
            }
            Ok(BodyDecision::Unchanged)
        },
        &NoAuthorization,
        control,
    )
    .await?;
    let status = response.status;
    let headers = response.headers;
    let body = String::from_utf8_lossy(&response.body).into_owned();
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

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn execute_ocr_provider_call(
    request: ProviderOcrRequest,
    observer: &mut impl OcrObserver,
) -> Result<Value, Error> {
    let url = Url::parse(request.url())
        .map_err(|error| Error::InvalidRequest(format!("invalid OCR provider URL: {error}")))?;
    let mut headers = HeaderMap::new();
    for (key, value) in &request.upstream_headers {
        let name = key.parse::<HeaderName>().map_err(|error| {
            Error::InvalidRequest(format!("invalid OCR provider header name: {error}"))
        })?;
        let value = value.parse::<HeaderValue>().map_err(|error| {
            Error::InvalidRequest(format!("invalid OCR provider header value: {error}"))
        })?;
        headers.append(name, value);
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
    let body =
        request.body().as_object().cloned().ok_or_else(|| {
            Error::InvalidRequest("OCR provider body must be a JSON object".into())
        })?;
    let timeout = request
        .timeout
        .unwrap_or(Duration::from_secs(OCR_TIMEOUT_SECS));
    let operation = OutboundOperation {
        method: Method::POST,
        url: url.clone(),
        headers,
        body: OutboundBody::JsonObject(body),
        operation: OutboundOperationKind::Submission,
    };
    let response = send_ocr_request(
        &url,
        operation,
        &event,
        observer,
        &OperationControl::with_timeout(timeout),
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
