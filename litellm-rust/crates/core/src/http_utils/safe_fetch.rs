//! Bounded public-Internet fetches for payloads that must be embedded upstream.

use std::io;
use std::net::IpAddr;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use reqwest::Url;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::CoreResult;
use crate::constants::{
    DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB, HTTP_CLIENT_CONNECT_TIMEOUT_SECS,
    HTTP_CLIENT_TIMEOUT_SECS, MAX_SAFE_FETCH_REDIRECTS,
};
use crate::error::CoreError;
use crate::http_utils::{map_send_error, upstream_http};

#[derive(Debug)]
struct PublicIpResolver;

impl Resolve for PublicIpResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(boxed_io_error)?
                .collect::<Vec<_>>();
            if addresses.is_empty() {
                return Err(boxed_io_error(io::Error::new(
                    io::ErrorKind::NotFound,
                    "host resolved to no addresses",
                )));
            }
            if addresses.iter().any(|address| is_blocked_ip(address.ip())) {
                return Err(boxed_io_error(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "host resolved to a non-public address",
                )));
            }
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

fn boxed_io_error(error: io::Error) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(error)
}

fn safe_fetch_client() -> CoreResult<&'static reqwest::Client> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .dns_resolver(Arc::new(PublicIpResolver))
                .no_proxy()
                .timeout(Duration::from_secs(HTTP_CLIENT_TIMEOUT_SECS))
                .connect_timeout(Duration::from_secs(HTTP_CLIENT_CONNECT_TIMEOUT_SECS))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| CoreError::connect(error.clone()))
}

fn max_document_download_bytes() -> u64 {
    let max_size_mb = std::env::var("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB);
    (max_size_mb.max(0.0) * 1024.0 * 1024.0) as u64
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
                    .is_some_and(|v4| is_blocked_ip(IpAddr::V4(v4)))
        }
    }
}

fn blocked_url_error(url: &Url) -> CoreError {
    CoreError::invalid_request(format!(
        "OCR document URL rejected by SSRF protection: {url}"
    ))
}

fn validate_safe_fetch_url(url: &Url) -> CoreResult<()> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(blocked_url_error(url));
    }
    let host = url.host_str().ok_or_else(|| blocked_url_error(url))?;
    let ip_literal = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    if ip_literal.parse::<IpAddr>().is_ok_and(is_blocked_ip) {
        return Err(blocked_url_error(url));
    }
    Ok(())
}

fn redirect_location(response: &reqwest::Response, url: &Url) -> CoreResult<Url> {
    let location = response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            CoreError::invalid_response("OCR document redirect missing Location header".to_string())
        })?;
    url.join(location).map_err(|error| {
        CoreError::invalid_response(format!("invalid OCR document redirect: {error}"))
    })
}

async fn safe_get(url: &str) -> CoreResult<(Url, reqwest::Response)> {
    let client = safe_fetch_client()?;
    let mut current_url = Url::parse(url).map_err(|error| {
        CoreError::invalid_request(format!("invalid OCR document URL: {error}"))
    })?;

    for _ in 0..MAX_SAFE_FETCH_REDIRECTS {
        validate_safe_fetch_url(&current_url)?;
        let response = client
            .get(current_url.clone())
            .send()
            .await
            .map_err(map_send_error)?;
        if !response.status().is_redirection() {
            return Ok((current_url, response));
        }
        current_url = redirect_location(&response, &current_url)?;
    }

    Err(CoreError::invalid_request(
        "Too many redirects while fetching OCR document URL".to_string(),
    ))
}

fn enforce_download_size(content_length: u64, max_bytes: u64, url: &Url) -> CoreResult<()> {
    if max_bytes == 0 {
        return Err(CoreError::invalid_request(format!(
            "OCR document URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0). url={url}"
        )));
    }
    if content_length > max_bytes {
        let size_mb = content_length as f64 / (1024.0 * 1024.0);
        let max_size_mb = max_bytes as f64 / (1024.0 * 1024.0);
        return Err(CoreError::invalid_request(format!(
            "OCR document size ({size_mb:.2}MB) exceeds maximum allowed size ({max_size_mb:.2}MB). url={url}"
        )));
    }
    Ok(())
}

async fn read_response_with_limit(
    mut response: reqwest::Response,
    url: &Url,
) -> CoreResult<Vec<u8>> {
    let max_bytes = max_document_download_bytes();
    let content_length = response.content_length();
    enforce_download_size(content_length.unwrap_or_default(), max_bytes, url)?;

    let capacity = content_length
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or_default();
    let mut bytes = Vec::with_capacity(capacity);
    let mut bytes_downloaded = 0_u64;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| CoreError::network(error.to_string()))?
    {
        bytes_downloaded += chunk.len() as u64;
        enforce_download_size(bytes_downloaded, max_bytes, url)?;
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

fn data_uri(content_type: &str, bytes: &[u8]) -> String {
    let prefix = format!("data:{content_type};base64,");
    let encoded_length = base64::encoded_len(bytes.len(), true).unwrap_or_default();
    let mut encoded = String::with_capacity(prefix.len().saturating_add(encoded_length));
    encoded.push_str(&prefix);
    BASE64_STANDARD.encode_string(bytes, &mut encoded);
    encoded
}

pub async fn fetch_url_as_data_uri(url: &str) -> CoreResult<String> {
    let (final_url, response) = safe_get(url).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(upstream_http(status, &body));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    let bytes = read_response_with_limit(response, &final_url).await?;
    Ok(data_uri(&content_type, &bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequestError;

    #[test]
    fn blocks_private_and_metadata_ips() {
        assert!(is_blocked_ip("127.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("10.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("::1".parse().unwrap()));
        assert!(is_blocked_ip("fd00::1".parse().unwrap()));
        assert!(is_blocked_ip("fe80::1".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:10.0.0.1".parse().unwrap()));
        assert!(!is_blocked_ip("8.8.8.8".parse().unwrap()));
        assert!(!is_blocked_ip("::ffff:8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn encodes_into_one_preallocated_data_uri() {
        assert_eq!(
            data_uri("image/png", b"hello"),
            "data:image/png;base64,aGVsbG8="
        );
    }

    #[test]
    fn rejects_direct_private_address_before_connecting() {
        let url = Url::parse("http://127.0.0.1/image.png").unwrap();
        let error = validate_safe_fetch_url(&url).unwrap_err();
        assert!(matches!(
            error,
            CoreError::Request(RequestError::InvalidRequest(message))
                if message.contains("SSRF protection")
        ));
    }

    #[test]
    fn rejects_direct_ipv6_loopback_before_connecting() {
        let url = Url::parse("http://[::1]/image.png").unwrap();
        assert!(validate_safe_fetch_url(&url).is_err());
    }

    #[tokio::test]
    async fn resolver_rejects_non_public_results() {
        let name = "localhost".parse::<Name>().unwrap();
        let Err(error) = PublicIpResolver.resolve(name).await else {
            panic!("localhost must not resolve through the public-IP resolver");
        };
        assert!(error.to_string().contains("non-public address"));
    }
}
