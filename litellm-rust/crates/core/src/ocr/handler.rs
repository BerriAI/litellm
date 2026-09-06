use std::time::Duration;

use reqwest::{
    Method, StatusCode, Url,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::Value;

use super::client::http_client;
use super::observers::{OcrObserver, OcrPostCall, OcrPreCall};
use super::transformation::OcrResponseHandling;
use super::types::{OcrRequestData, PendingOcrUpload, ProviderOcrRequest};
use crate::Error;
use crate::auth::{
    BodyDecision, OperationControl, OutboundBody, OutboundOperation, OutboundOperationKind,
    send_once,
};
use crate::constants::OCR_TIMEOUT_SECS;
use crate::http_utils::truncate_error_body;
use crate::provider_callbacks::CallbackDecision;

pub struct OcrHttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn send_ocr_request<Observer>(
    origin: &Url,
    request: OutboundOperation,
    model: &str,
    request_context: &crate::request_context::LiteLlmRequestContext,
    observer: &mut Observer,
    authorization: &dyn crate::auth::AuthorizationProvider,
    control: &OperationControl,
) -> Result<OcrHttpResponse, Error>
where
    Observer: OcrObserver,
    Observer::Error: std::fmt::Display,
{
    let response = send_once(
        http_client(),
        origin,
        request,
        |view| {
            let observer = &mut *observer;
            async move {
                let data = match view.body {
                    OutboundBody::JsonObject(body) => Value::Object(body),
                    OutboundBody::Bodyless => Value::Null,
                    OutboundBody::Encoded { .. } => Value::Null,
                };
                let event = OcrPreCall {
                    call_id: request_context.litellm_call_id.clone(),
                    trace_id: request_context.trace_id.clone(),
                    requested_model: request_context.request_model.clone(),
                    attribution: request_context.attribution.clone(),
                    capabilities: request_context.capabilities.clone(),
                    model: model.to_string(),
                    request: OcrRequestData { data, files: None },
                    api_base: view.url.to_string(),
                    headers: view
                        .headers
                        .iter()
                        .filter_map(|(name, value)| {
                            value
                                .to_str()
                                .ok()
                                .map(|value| (name.to_string(), value.to_string()))
                        })
                        .collect(),
                };
                match observer.pre_call(&event).await.map_err(callback_error)? {
                    CallbackDecision::Unchanged => Ok(BodyDecision::Unchanged),
                    CallbackDecision::Replace { payload } => Ok(BodyDecision::Replace(payload)),
                    CallbackDecision::Reject { message, .. } => Ok(BodyDecision::Reject(message)),
                }
            }
        },
        authorization,
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
        call_id: request_context.litellm_call_id.clone(),
        trace_id: request_context.trace_id.clone(),
        requested_model: request_context.request_model.clone(),
        attribution: request_context.attribution.clone(),
        capabilities: request_context.capabilities.clone(),
        original_response: body.clone(),
    };
    let body = match observer.post_call(&event).await.map_err(callback_error)? {
        CallbackDecision::Unchanged => body,
        CallbackDecision::Replace {
            payload: Value::String(replacement),
        } => replacement,
        CallbackDecision::Replace { payload } => {
            serde_json::to_string(&payload).map_err(|error| {
                Error::InvalidResponse(format!("OCR callback response is invalid: {error}"))
            })?
        }
        CallbackDecision::Reject { message, .. } => return Err(Error::InvalidResponse(message)),
    };
    Ok(OcrHttpResponse {
        status,
        headers,
        body,
    })
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn execute_ocr_provider_call<Observer>(
    mut request: ProviderOcrRequest,
    observer: &mut Observer,
) -> Result<Value, Error>
where
    Observer: OcrObserver,
    Observer::Error: std::fmt::Display,
{
    if let Some(upload) = request.pending_upload.take() {
        let file_id = execute_ocr_upload(&request, upload, observer).await?;
        let document = serde_json::json!({
            "type": "document_url",
            "document_url": file_id,
        });
        request.body = Some(
            request
                .config
                .transform_ocr_request(request.model(), document, request.optional_params.clone())?
                .data,
        );
    }
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
    let control = OperationControl::with_timeout(timeout);
    let response = send_ocr_request(
        &url,
        operation,
        request.model(),
        &request.context,
        observer,
        request.authorization.as_ref(),
        &control,
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
        let response_json =
            poll_document_intelligence_with_auth(operation_url, url, &request, observer, &control)
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

async fn execute_ocr_upload<Observer>(
    request: &ProviderOcrRequest,
    upload: PendingOcrUpload,
    observer: &mut Observer,
) -> Result<String, Error>
where
    Observer: OcrObserver,
    Observer::Error: std::fmt::Display,
{
    let url = Url::parse(&upload.url)
        .map_err(|error| Error::InvalidRequest(format!("invalid OCR upload URL: {error}")))?;
    let boundary = "litellm-rust-ocr-boundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"document\"\r\nContent-Type: {}\r\n\r\n",
        upload.mime_type
    )
    .into_bytes();
    body.extend_from_slice(&upload.bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let content_type = format!("multipart/form-data; boundary={boundary}")
        .parse()
        .expect("static multipart content type is valid");
    let control = OperationControl::with_timeout(
        request
            .timeout
            .unwrap_or(Duration::from_secs(OCR_TIMEOUT_SECS)),
    );
    let response = send_ocr_request(
        &url,
        OutboundOperation {
            method: Method::POST,
            url: url.clone(),
            headers: HeaderMap::new(),
            body: OutboundBody::Encoded {
                bytes: body,
                content_type,
            },
            operation: OutboundOperationKind::Upload,
        },
        request.model(),
        &request.context,
        observer,
        request.authorization.as_ref(),
        &control,
    )
    .await?;
    let payload: Value = serde_json::from_str(&response.body)
        .map_err(|error| Error::InvalidResponse(format!("invalid OCR upload JSON: {error}")))?;
    Ok(
        crate::providers::reducto::ocr::transformation::extract_upload_file_id(&payload)?
            .to_string(),
    )
}

async fn poll_document_intelligence_with_auth<Observer>(
    operation_url: String,
    origin: Url,
    request: &ProviderOcrRequest,
    observer: &mut Observer,
    control: &OperationControl,
) -> Result<Value, Error>
where
    Observer: OcrObserver,
    Observer::Error: std::fmt::Display,
{
    let url = Url::parse(&operation_url)
        .map_err(|_| Error::InvalidResponse("invalid OCR operation URL".into()))?;
    loop {
        let mut request_headers = HeaderMap::new();
        for (name, value) in &request.upstream_headers {
            request_headers.append(
                name.parse::<HeaderName>().map_err(|_| {
                    Error::InvalidRequest("invalid OCR provider header name".into())
                })?,
                value.parse::<HeaderValue>().map_err(|_| {
                    Error::InvalidRequest("invalid OCR provider header value".into())
                })?,
            );
        }
        let response = send_ocr_request(
            &origin,
            OutboundOperation {
                method: Method::GET,
                url: url.clone(),
                headers: request_headers,
                body: OutboundBody::Bodyless,
                operation: OutboundOperationKind::Poll,
            },
            request.model(),
            &request.context,
            observer,
            request.authorization.as_ref(),
            control,
        )
        .await?;
        let data: Value = serde_json::from_str(&response.body)
            .map_err(|_| Error::InvalidResponse("invalid OCR polling response JSON".into()))?;
        match data
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("succeeded") => return Ok(data),
            Some("failed" | "canceled" | "cancelled") => {
                return Err(Error::InvalidResponse(
                    "Azure Document Intelligence operation failed".into(),
                ));
            }
            Some("notstarted" | "running") => {}
            _ => {
                return Err(Error::InvalidResponse(
                    "Azure Document Intelligence returned an invalid operation status".into(),
                ));
            }
        }
        let delay = response
            .headers
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .and_then(|value| Duration::try_from_secs_f64(value).ok())
            .unwrap_or(Duration::from_secs(1));
        tokio::select! {
            _ = control.cancellation.cancelled() => {
                return Err(Error::Network("Request cancelled".into()));
            }
            result = tokio::time::timeout_at(control.deadline.into(), tokio::time::sleep(delay)) => {
                result.map_err(|_| Error::Network("Request timed out".into()))?;
            }
        }
    }
}

fn callback_error(error: impl std::fmt::Display) -> Error {
    Error::InvalidResponse(format!("OCR callback failed: {error}"))
}
