use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrProviderConfig;
use litellm_core::CoreResult;
use reqwest::Url;
use serde_json::{Map, Value};

use litellm_core::providers::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use litellm_core::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use litellm_core::providers::vertex_ai::ocr::transformation as vertex_ai;
use litellm_core::providers::vertex_ai::ocr::transformation::{
    VERTEX_AI_DEEPSEEK_OCR_CONFIG, VERTEX_AI_OCR_CONFIG,
};

use super::http_client;

const ERROR_BODY_MAX_CHARS: usize = 256;
const AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB: f64 = 50.0;
const MAX_SAFE_FETCH_REDIRECTS: usize = 10;
const AZURE_WIRE_SERVER_IP: Ipv4Addr = Ipv4Addr::new(168, 63, 129, 16);

struct ValidatedFetchDestination {
    url: Url,
    host: String,
    addresses: Vec<SocketAddr>,
}

pub(super) fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

pub(super) fn ocr_provider_config(
    provider: &str,
    model: &str,
) -> Option<&'static dyn OcrProviderConfig> {
    match provider {
        "mistral" => Some(&MISTRAL_OCR_CONFIG),
        "azure_ai" if is_azure_document_intelligence_model(model) => {
            Some(&AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG)
        }
        "azure_ai/doc-intelligence" => Some(&AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG),
        "azure_ai" => Some(&AZURE_AI_OCR_CONFIG),
        "vertex_ai" if vertex_ai::is_deepseek_model(model) => Some(&VERTEX_AI_DEEPSEEK_OCR_CONFIG),
        "vertex_ai" => Some(&VERTEX_AI_OCR_CONFIG),
        _ => None,
    }
}

fn is_azure_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<Vec<(String, String)>> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidRequest(format!(
                        "OCR extra_headers.{key} must be a string, got {}",
                        litellm_core::error::json_type_name(&value)
                    ))
                })
        })
        .collect()
}

pub(super) fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

fn document_url_field(document: &Value) -> CoreResult<Option<(&str, &str)>> {
    let Some(object) = document.as_object() else {
        return Ok(None);
    };
    let Some(doc_type) = object.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let field = match doc_type {
        "document_url" => "document_url",
        "image_url" => "image_url",
        _ => return Ok(None),
    };
    let Some(url) = object.get(field).and_then(Value::as_str) else {
        return Ok(None);
    };
    Ok(Some((field, url)))
}

fn is_url_requiring_fetch(url: &str) -> bool {
    !url.starts_with("data:") && (url.starts_with("http://") || url.starts_with("https://"))
}

fn max_document_download_bytes() -> u64 {
    let max_size_mb = std::env::var("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB);
    (max_size_mb.max(0.0) * 1024.0 * 1024.0) as u64
}

fn is_private_ipv4(ip: Ipv4Addr, prefix: [u8; 4], prefix_len: u32) -> bool {
    let ip = u32::from(ip);
    let prefix = u32::from(Ipv4Addr::from(prefix));
    let mask = u32::MAX.checked_shl(32 - prefix_len).unwrap_or(0);
    (ip & mask) == (prefix & mask)
}

fn is_global_ipv4(ip: Ipv4Addr) -> bool {
    !(is_private_ipv4(ip, [0, 0, 0, 0], 8)
        || is_private_ipv4(ip, [10, 0, 0, 0], 8)
        || is_private_ipv4(ip, [100, 64, 0, 0], 10)
        || is_private_ipv4(ip, [127, 0, 0, 0], 8)
        || is_private_ipv4(ip, [169, 254, 0, 0], 16)
        || is_private_ipv4(ip, [172, 16, 0, 0], 12)
        || is_private_ipv4(ip, [192, 0, 0, 0], 24)
        || is_private_ipv4(ip, [192, 0, 2, 0], 24)
        || is_private_ipv4(ip, [192, 88, 99, 0], 24)
        || is_private_ipv4(ip, [192, 168, 0, 0], 16)
        || is_private_ipv4(ip, [198, 18, 0, 0], 15)
        || is_private_ipv4(ip, [198, 51, 100, 0], 24)
        || is_private_ipv4(ip, [203, 0, 113, 0], 24)
        || is_private_ipv4(ip, [224, 0, 0, 0], 4)
        || is_private_ipv4(ip, [240, 0, 0, 0], 4))
}

fn is_private_ipv6(ip: Ipv6Addr, prefix: u128, prefix_len: u32) -> bool {
    let ip = u128::from(ip);
    let mask = u128::MAX.checked_shl(128 - prefix_len).unwrap_or(0);
    (ip & mask) == (prefix & mask)
}

fn is_global_ipv6(ip: Ipv6Addr) -> bool {
    if ip
        .to_ipv4_mapped()
        .or_else(|| ip.to_ipv4())
        .map(is_global_ipv4)
        .is_some_and(|is_global| !is_global)
    {
        return false;
    }

    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || is_private_ipv6(ip, 0x0064_ff9b_0001_0000_0000_0000_0000_0000, 48)
        || is_private_ipv6(ip, 0x0100_0000_0000_0000_0000_0000_0000_0000, 64)
        || is_private_ipv6(ip, 0x2001_0000_0000_0000_0000_0000_0000_0000, 23)
        || is_private_ipv6(ip, 0x2001_0db8_0000_0000_0000_0000_0000_0000, 32)
        || is_private_ipv6(ip, 0x2002_0000_0000_0000_0000_0000_0000_0000, 16)
        || is_private_ipv6(ip, 0xfc00_0000_0000_0000_0000_0000_0000_0000, 7)
        || is_private_ipv6(ip, 0xfe80_0000_0000_0000_0000_0000_0000_0000, 10))
}

fn is_cloud_metadata_exception(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip == AZURE_WIRE_SERVER_IP,
        IpAddr::V6(ip) => ip
            .to_ipv4_mapped()
            .or_else(|| ip.to_ipv4())
            .is_some_and(|ip| ip == AZURE_WIRE_SERVER_IP),
    }
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    if is_cloud_metadata_exception(ip) {
        return true;
    }
    match ip {
        IpAddr::V4(ip) => !is_global_ipv4(ip),
        IpAddr::V6(ip) => !is_global_ipv6(ip),
    }
}

fn blocked_url_error(url: &Url) -> CoreError {
    CoreError::InvalidRequest(format!(
        "OCR document URL rejected by SSRF protection: {url}"
    ))
}

async fn validate_safe_fetch_url(url: &Url) -> CoreResult<ValidatedFetchDestination> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(blocked_url_error(url));
    }

    let host = url.host_str().ok_or_else(|| blocked_url_error(url))?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| blocked_url_error(url))?;
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_blocked_ip(ip) {
            return Err(blocked_url_error(url));
        }
        return Ok(ValidatedFetchDestination {
            url: url.clone(),
            host: host.to_string(),
            addresses: vec![SocketAddr::new(ip, port)],
        });
    }

    let addresses = tokio::net::lookup_host((host, port))
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let mut safe_addresses = Vec::new();
    for address in addresses {
        if is_blocked_ip(address.ip()) {
            return Err(blocked_url_error(url));
        }
        safe_addresses.push(address);
    }
    if safe_addresses.is_empty() {
        return Err(blocked_url_error(url));
    }
    Ok(ValidatedFetchDestination {
        url: url.clone(),
        host: host.to_string(),
        addresses: safe_addresses,
    })
}

fn redirect_location(response: &reqwest::Response, url: &Url) -> CoreResult<Url> {
    let location = response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            CoreError::InvalidResponse("OCR document redirect missing Location header".to_string())
        })?;
    url.join(location)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid OCR document redirect: {err}")))
}

async fn safe_get_document_url(url: &str) -> CoreResult<(Url, reqwest::Response)> {
    let mut current_url = Url::parse(url)
        .map_err(|err| CoreError::InvalidRequest(format!("invalid OCR document URL: {err}")))?;

    for _ in 0..MAX_SAFE_FETCH_REDIRECTS {
        let destination = validate_safe_fetch_url(&current_url).await?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .resolve_to_addrs(&destination.host, &destination.addresses)
            .build()
            .map_err(|err| CoreError::Network(err.to_string()))?;
        let response = client
            .get(destination.url.clone())
            .send()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        if !response.status().is_redirection() {
            return Ok((destination.url, response));
        }
        current_url = redirect_location(&response, &current_url)?;
    }

    Err(CoreError::InvalidRequest(
        "Too many redirects while fetching OCR document URL".to_string(),
    ))
}

fn enforce_download_size(content_length: u64, max_bytes: u64, url: &Url) -> CoreResult<()> {
    if max_bytes == 0 {
        return Err(CoreError::InvalidRequest(format!(
            "OCR document URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0). url={url}"
        )));
    }
    if content_length > max_bytes {
        let size_mb = content_length as f64 / (1024.0 * 1024.0);
        let max_size_mb = max_bytes as f64 / (1024.0 * 1024.0);
        return Err(CoreError::InvalidRequest(format!(
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
    if let Some(content_length) = response.content_length() {
        enforce_download_size(content_length, max_bytes, url)?;
    } else {
        enforce_download_size(0, max_bytes, url)?;
    }

    let mut bytes = Vec::new();
    let mut bytes_downloaded: u64 = 0;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?
    {
        bytes_downloaded += chunk.len() as u64;
        enforce_download_size(bytes_downloaded, max_bytes, url)?;
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

pub(super) async fn convert_document_url_to_data_uri(document: Value) -> CoreResult<Value> {
    let Some((field, url)) = document_url_field(&document)? else {
        return Ok(document);
    };
    if !is_url_requiring_fetch(url) {
        return Ok(document);
    }

    let (final_url, response) = safe_get_document_url(url).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&body),
        });
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
    let data_uri = format!(
        "data:{content_type};base64,{}",
        BASE64_STANDARD.encode(bytes)
    );

    let mut transformed = document
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::InvalidRequest("OCR document must be an object".to_string()))?;
    transformed.insert(field.to_string(), Value::String(data_uri));
    Ok(Value::Object(transformed))
}

fn same_origin(left: &str, right: &str) -> bool {
    let Ok(left) = reqwest::Url::parse(left) else {
        return false;
    };
    let Ok(right) = reqwest::Url::parse(right) else {
        return false;
    };
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn retry_after_secs(response: &reqwest::Response) -> u64 {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2)
}

fn operation_status(response_json: &Value) -> CoreResult<&str> {
    let status = response_json
        .get("status")
        .and_then(Value::as_str)
        .ok_or(CoreError::MissingField("status"))?;
    match status {
        "succeeded" => Ok("succeeded"),
        "running" | "notStarted" => Ok("running"),
        "failed" => {
            let message = response_json
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown error");
            Err(CoreError::InvalidResponse(format!(
                "Azure Document Intelligence analysis failed: {message}"
            )))
        }
        other => Err(CoreError::InvalidResponse(format!(
            "Unknown operation status: {other}"
        ))),
    }
}

pub(super) async fn poll_document_intelligence(
    operation_url: &str,
    original_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<Value> {
    if !same_origin(operation_url, original_url) {
        return Err(CoreError::InvalidResponse(
            "Azure Document Intelligence: rejected cross-origin polling URL".to_string(),
        ));
    }

    let start = Instant::now();
    let timeout = timeout.unwrap_or(Duration::from_secs(
        AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS,
    ));
    loop {
        if start.elapsed() > timeout {
            return Err(CoreError::Network(format!(
                "Azure Document Intelligence operation polling timed out after {} seconds",
                timeout.as_secs()
            )));
        }

        let mut request_builder = http_client().get(operation_url);
        for (key, value) in headers {
            if key.eq_ignore_ascii_case("ocp-apim-subscription-key") {
                request_builder = request_builder.header(key, value);
            }
        }
        let response = request_builder
            .send()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        let retry_after = retry_after_secs(&response);
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        if !status.is_success() {
            return Err(CoreError::Http {
                status: status.as_u16(),
                body: truncate_error_body(&text),
            });
        }
        let response_json: Value = serde_json::from_str(&text).map_err(|err| {
            CoreError::InvalidResponse(format!("invalid Azure DI poll response JSON: {err}"))
        })?;
        if operation_status(&response_json)? == "succeeded" {
            return Ok(response_json);
        }
        tokio::time::sleep(Duration::from_secs(retry_after)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    async fn read_http_headers(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let n = socket.read(&mut buffer).await.expect("reads request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        String::from_utf8(request).expect("request is utf8")
    }

    #[test]
    fn blocks_private_and_metadata_ips() {
        assert!(is_blocked_ip("127.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("10.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("100.64.0.1".parse().unwrap()));
        assert!(is_blocked_ip("192.0.2.1".parse().unwrap()));
        assert!(is_blocked_ip("169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("168.63.129.16".parse().unwrap()));
        assert!(is_blocked_ip("::1".parse().unwrap()));
        assert!(is_blocked_ip("fd00::1".parse().unwrap()));
        assert!(is_blocked_ip("fe80::1".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:168.63.129.16".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:10.0.0.1".parse().unwrap()));
        assert!(!is_blocked_ip("8.8.8.8".parse().unwrap()));
        assert!(!is_blocked_ip("::ffff:8.8.8.8".parse().unwrap()));
    }

    #[tokio::test]
    async fn convert_document_url_rejects_loopback_fetch() {
        let error = convert_document_url_to_data_uri(json!({
            "type": "image_url",
            "image_url": "http://127.0.0.1/image.png"
        }))
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            CoreError::InvalidRequest(message)
                if message.contains("SSRF protection")
        ));
    }

    #[tokio::test]
    async fn convert_document_url_leaves_data_uri_untouched() {
        let document = json!({
            "type": "image_url",
            "image_url": "data:image/png;base64,abcd"
        });

        let transformed = convert_document_url_to_data_uri(document.clone())
            .await
            .unwrap();

        assert_eq!(transformed, document);
    }

    #[tokio::test]
    async fn document_intelligence_poll_does_not_follow_redirects_with_credentials() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");
        let operation_url = format!("http://{addr}/operations/1");
        let redirect_url = format!("http://{addr}/leak");

        let server = tokio::spawn(async move {
            let (mut poll_socket, _) = listener.accept().await.expect("accepts poll request");
            let poll_request = read_http_headers(&mut poll_socket).await;
            let poll_response = format!(
                "HTTP/1.1 302 Found\r\nlocation: {redirect_url}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
            poll_socket
                .write_all(poll_response.as_bytes())
                .await
                .expect("writes redirect response");

            let second_connection =
                tokio::time::timeout(Duration::from_millis(200), listener.accept()).await;
            (poll_request, second_connection.is_ok())
        });

        let error = poll_document_intelligence(
            &operation_url,
            &format!("http://{addr}/documentintelligence/documentModels/prebuilt-read:analyze"),
            &[(
                "Ocp-Apim-Subscription-Key".to_string(),
                "di-key".to_string(),
            )],
            Some(Duration::from_secs(5)),
        )
        .await
        .expect_err("poll redirect is returned to caller");

        assert_eq!(
            error,
            CoreError::Http {
                status: 302,
                body: "".to_string()
            }
        );

        let (poll_request, followed_redirect) = server.await.expect("server task completes");
        assert!(
            poll_request
                .to_ascii_lowercase()
                .contains("ocp-apim-subscription-key: di-key"),
            "{poll_request}"
        );
        assert!(
            !followed_redirect,
            "credentialed DI poll followed redirect: {poll_request}"
        );
    }
}
