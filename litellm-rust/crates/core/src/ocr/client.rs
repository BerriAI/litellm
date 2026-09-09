use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use std::net::IpAddr;
use std::sync::OnceLock;
use std::time::Duration;

use super::types::{OcrConnection, OcrDocument};
use crate::constants::{
    OCR_CONNECT_TIMEOUT_SECS, OCR_ERROR_BODY_MAX_CHARS, OCR_HTTP_TIMEOUT_SECS,
    OCR_MAX_FETCH_REDIRECTS,
};
use crate::error::TransportError;
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::Url;

pub(crate) fn http_client() -> Result<&'static reqwest::Client, TransportError> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(OCR_CONNECT_TIMEOUT_SECS))
                .timeout(Duration::from_secs(OCR_HTTP_TIMEOUT_SECS))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| TransportError::Network(error.clone()))
}

pub(crate) fn network_error(error: reqwest::Error) -> TransportError {
    TransportError::from(error)
}

pub(crate) fn truncate_error_body(body: &str) -> String {
    let truncated: String = body.chars().take(OCR_ERROR_BODY_MAX_CHARS).collect();
    if truncated.len() == body.len() {
        truncated
    } else {
        format!("{truncated}... (truncated)")
    }
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_multicast()
                || ip.is_unspecified()
        }
        IpAddr::V6(ip) => {
            let first_segment = ip.segments()[0];
            let is_unique_local = (first_segment & 0xfe00) == 0xfc00;
            let is_link_local = (first_segment & 0xffc0) == 0xfe80;
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || is_unique_local
                || is_link_local
                || ip
                    .to_ipv4_mapped()
                    .or_else(|| ip.to_ipv4())
                    .map(|v4| is_blocked_ip(IpAddr::V4(v4)))
                    .unwrap_or(false)
        }
    }
}

fn blocked_url_error() -> OcrError {
    OcrRequestError::BlockedDocumentUrl.into()
}
async fn validate_safe_fetch_url(url: &Url) -> Result<(), OcrError> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(blocked_url_error());
    }
    let host = url.host_str().ok_or_else(blocked_url_error)?;
    if let Ok(ip) = host.parse::<IpAddr>() {
        return if is_blocked_ip(ip) {
            Err(blocked_url_error())
        } else {
            Ok(())
        };
    }
    let port = url.port_or_known_default().ok_or_else(blocked_url_error)?;
    let addresses = tokio::net::lookup_host((host, port))
        .await
        .map_err(|error| TransportError::Network(error.to_string()))?
        .collect::<Vec<_>>();
    if addresses.is_empty() || addresses.iter().any(|address| is_blocked_ip(address.ip())) {
        return Err(blocked_url_error());
    }
    Ok(())
}

async fn safe_get_document_url(
    source: &str,
    timeout: Duration,
) -> Result<reqwest::Response, OcrError> {
    let mut url = Url::parse(source).map_err(|_| OcrRequestError::RequestField {
        path: "document URL".into(),
    })?;
    for _ in 0..OCR_MAX_FETCH_REDIRECTS {
        validate_safe_fetch_url(&url).await?;
        let response = http_client()?
            .get(url.clone())
            .timeout(timeout)
            .send()
            .await
            .map_err(network_error)?;
        if !response.status().is_redirection() {
            return Ok(response);
        }
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| OcrResponseError::MissingRedirectLocation)?;
        url = url
            .join(location)
            .map_err(|_| OcrResponseError::InvalidRedirect)?;
    }
    Err(OcrRequestError::TooManyRedirects.into())
}

fn enforce_download_size(length: u64, max_bytes: u64) -> Result<(), OcrError> {
    if max_bytes == 0 {
        return Err(OcrRequestError::DownloadDisabled.into());
    }
    if length > max_bytes {
        return Err(OcrRequestError::DownloadTooLarge.into());
    }
    Ok(())
}

async fn download_data_uri(source: &str, connection: &OcrConnection) -> Result<String, OcrError> {
    enforce_download_size(0, connection.max_download_bytes)?;
    let mut response = safe_get_document_url(source, connection.timeout).await?;
    if !response.status().is_success() {
        let status = response.status().as_u16();
        return Err(TransportError::Http {
            status,
            body: "OCR document download failed".into(),
        }
        .into());
    }
    enforce_download_size(
        response.content_length().unwrap_or(0),
        connection.max_download_bytes,
    )?;
    let mime = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(network_error)? {
        enforce_download_size(
            bytes.len() as u64 + chunk.len() as u64,
            connection.max_download_bytes,
        )?;
        bytes.extend_from_slice(&chunk);
    }
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

pub(crate) async fn convert_document_url_to_data_uri(
    document: OcrDocument,
    connection: &OcrConnection,
) -> Result<OcrDocument, OcrError> {
    let source = document.source();
    if !source.starts_with("http://") && !source.starts_with("https://") {
        return Ok(document);
    }
    let data_uri = tokio::time::timeout(connection.timeout, download_data_uri(source, connection))
        .await
        .map_err(|_| TransportError::Network("OCR document download timed out".into()))??;
    Ok(document.with_source(data_uri))
}
