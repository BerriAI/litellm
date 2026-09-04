use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::{RequestBuilder, StatusCode, header::HeaderMap};
use serde_json::Value;

use crate::Error;
use crate::http_utils::{http_request, truncate_error_body};
use crate::provider_callbacks::{
    CallbackDecision, ProviderAttemptObserver, ProviderError, ProviderPostCall, ProviderPreCall,
};

pub struct OcrHttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn send_ocr_request<Observer>(
    request: RequestBuilder,
    event: &ProviderPreCall,
    observer: &mut Observer,
) -> Result<OcrHttpResponse, Error>
where
    Observer: ProviderAttemptObserver,
    Observer::Error: std::fmt::Display,
{
    let response = match http_request(request).await {
        Ok(response) => response,
        Err(error) => {
            let mapped = transport_error(error);
            notify_error(observer, event, &mapped, "provider_request", true).await?;
            return Err(mapped);
        }
    };
    let status = response.status();
    let headers = response.headers().clone();
    let body = match response.text().await {
        Ok(body) => body,
        Err(error) => {
            let mapped = transport_error(error);
            notify_error(observer, event, &mapped, "response_body", true).await?;
            return Err(mapped);
        }
    };
    if !status.is_success() {
        let error = Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&body),
        };
        notify_error(observer, event, &error, "provider_response", true).await?;
        return Err(error);
    }
    let post_call = ProviderPostCall {
        provider: event.provider.clone(),
        model: event.model.clone(),
        call_id: event.call_id.clone(),
        trace_id: event.trace_id.clone(),
        attempt: event.attempt,
        started_at: event.started_at,
        response: Value::String(body.clone()),
        status_code: status.as_u16(),
        headers: header_values(&headers),
        ended_at: epoch_seconds(),
    };
    let body = match observer
        .post_call(&post_call)
        .await
        .map_err(callback_error)?
    {
        CallbackDecision::Unchanged => body,
        CallbackDecision::Replace {
            payload: Value::String(replacement),
        } => replacement,
        CallbackDecision::Replace { payload } => {
            serde_json::to_string(&payload).map_err(|error| {
                Error::InvalidResponse(format!("callback response is invalid: {error}"))
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

async fn notify_error<Observer>(
    observer: &mut Observer,
    context: &ProviderPreCall,
    error: &Error,
    stage: &'static str,
    committed: bool,
) -> Result<(), Error>
where
    Observer: ProviderAttemptObserver,
    Observer::Error: std::fmt::Display,
{
    let event = ProviderError {
        provider: context.provider.clone(),
        model: context.model.clone(),
        call_id: context.call_id.clone(),
        trace_id: context.trace_id.clone(),
        attempt: context.attempt,
        started_at: context.started_at,
        message: error.to_string(),
        stage,
        committed,
        status_code: match error {
            Error::Http { status, .. } => Some(*status),
            _ => None,
        },
        will_retry: false,
        ended_at: epoch_seconds(),
    };
    observer.error(&event).await.map_err(callback_error)
}

fn header_values(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect()
}

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn callback_error(error: impl std::fmt::Display) -> Error {
    Error::InvalidResponse(format!("OCR callback failed: {error}"))
}

fn transport_error(error: reqwest::Error) -> Error {
    Error::Network(if error.is_timeout() {
        "Request timed out".into()
    } else {
        error.to_string()
    })
}
