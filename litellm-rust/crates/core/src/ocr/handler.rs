use reqwest::{RequestBuilder, StatusCode, header::HeaderMap};

use super::observers::{OcrObserver, OcrPostCall, OcrPreCall};
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
