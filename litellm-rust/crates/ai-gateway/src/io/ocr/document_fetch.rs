use std::net::SocketAddr;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use litellm_core::error::CoreError;
use litellm_core::CoreResult;
use reqwest::Url;
use serde_json::{Map, Value};

use crate::constants::{
    DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB, MAX_SAFE_FETCH_REDIRECTS,
    OCR_DOCUMENT_FETCH_NETWORK_ERROR,
};

use super::common_utils::{document_url_field, read_error_body};
use super::ssrf::{parse_fetchable_url, pinned_client, resolve_validated};

fn document_fetch_network_error() -> CoreError {
    CoreError::Network(OCR_DOCUMENT_FETCH_NETWORK_ERROR.to_string())
}

fn max_document_download_bytes() -> u64 {
    let max_size_mb = std::env::var("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB);
    (max_size_mb.max(0.0) * 1024.0 * 1024.0) as u64
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
        .map_err(|_| CoreError::InvalidResponse("invalid OCR document redirect".to_string()))
}

async fn safe_get_document_url(url: &str) -> CoreResult<(Url, reqwest::Response)> {
    fetch_with_redirects(url, |candidate| async move {
        resolve_validated(&candidate).await
    })
    .await
}

async fn fetch_with_redirects<P, Fut>(url: &str, resolve: P) -> CoreResult<(Url, reqwest::Response)>
where
    P: Fn(Url) -> Fut,
    Fut: std::future::Future<Output = CoreResult<Vec<SocketAddr>>>,
{
    let mut current_url = Url::parse(url).map_err(|_| {
        CoreError::InvalidRequest("OCR document URL rejected by SSRF protection".to_string())
    })?;

    for _ in 0..MAX_SAFE_FETCH_REDIRECTS {
        let addresses = resolve(current_url.clone()).await?;
        let client = pinned_client(&current_url, &addresses)?;
        let response = client
            .get(current_url.clone())
            .send()
            .await
            .map_err(|_| document_fetch_network_error())?;
        if !response.status().is_redirection() {
            return Ok((current_url, response));
        }
        current_url = redirect_location(&response, &current_url)?;
    }

    Err(CoreError::InvalidRequest(
        "Too many redirects while fetching OCR document URL".to_string(),
    ))
}

fn enforce_download_size(content_length: u64, max_bytes: u64) -> CoreResult<()> {
    if max_bytes == 0 {
        return Err(CoreError::InvalidRequest(
            "OCR document URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0)".to_string(),
        ));
    }
    if content_length > max_bytes {
        let size_mb = content_length as f64 / (1024.0 * 1024.0);
        let max_size_mb = max_bytes as f64 / (1024.0 * 1024.0);
        return Err(CoreError::InvalidRequest(format!(
            "OCR document size ({size_mb:.2}MB) exceeds maximum allowed size ({max_size_mb:.2}MB)"
        )));
    }
    Ok(())
}

async fn read_response_with_limit(
    mut response: reqwest::Response,
    max_bytes: u64,
) -> CoreResult<Vec<u8>> {
    enforce_download_size(response.content_length().unwrap_or(0), max_bytes)?;

    let mut bytes = Vec::new();
    let mut bytes_downloaded: u64 = 0;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| document_fetch_network_error())?
    {
        bytes_downloaded += chunk.len() as u64;
        enforce_download_size(bytes_downloaded, max_bytes)?;
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

pub(super) async fn convert_document_url_to_data_uri(document: Value) -> CoreResult<Value> {
    let Some((field, url)) = document_url_field(&document)? else {
        return Ok(document);
    };
    if url.starts_with("data:") {
        return Ok(document);
    }
    parse_fetchable_url(url)?;

    let (_final_url, response) = safe_get_document_url(url).await?;
    let status = response.status();
    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: read_error_body(response).await,
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
    let bytes = read_response_with_limit(response, max_document_download_bytes()).await?;
    let data_uri = format!(
        "data:{content_type};base64,{}",
        BASE64_STANDARD.encode(bytes)
    );

    let object = document
        .as_object()
        .ok_or_else(|| CoreError::InvalidRequest("OCR document must be an object".to_string()))?;
    let transformed: Map<String, Value> = object
        .iter()
        .map(|(key, value)| {
            if key == field {
                (key.clone(), Value::String(data_uri.clone()))
            } else {
                (key.clone(), value.clone())
            }
        })
        .collect();
    Ok(Value::Object(transformed))
}

#[cfg(test)]
mod tests {
    use super::super::ssrf::{blocked_url_error, resolve_validated};
    use super::super::test_support::{http_response, spawn_counting_server};
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn loopback_fetch_rejected_without_connecting() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n",
            b"hi",
        ))
        .await;
        let url = format!("http://127.0.0.1:{}/doc.png", server.addr.port());
        let error = convert_document_url_to_data_uri(json!({
            "type": "image_url",
            "image_url": url,
        }))
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "got {error:?}"
        );
        assert_eq!(server.connection_count(), 0);
    }

    #[tokio::test]
    async fn mapped_ipv6_loopback_fetch_rejected_without_connecting() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n",
            b"hi",
        ))
        .await;
        let url = format!("http://[::ffff:127.0.0.1]:{}/doc.png", server.addr.port());
        let error = convert_document_url_to_data_uri(json!({
            "type": "image_url",
            "image_url": url,
        }))
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "got {error:?}"
        );
        assert_eq!(server.connection_count(), 0);
    }

    #[tokio::test]
    async fn unsupported_scheme_rejected_without_fetch() {
        for raw in ["file:///etc/passwd", "gopher://8.8.8.8/x"] {
            let error = convert_document_url_to_data_uri(json!({
                "type": "image_url",
                "image_url": raw,
            }))
            .await
            .unwrap_err();
            assert!(
                matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
                "{raw} got {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn redirect_to_loopback_rejected_without_connecting_to_target() {
        let target = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n",
            b"secret",
        ))
        .await;
        let location = format!("http://127.0.0.1:{}/internal", target.addr.port());
        let redirector = spawn_counting_server(
            format!("HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\n\r\n")
                .into_bytes(),
        )
        .await;
        let redirector_addr = redirector.addr;
        let redirector_port = redirector_addr.port();
        let start_url = format!("http://127.0.0.1:{redirector_port}/doc.png");

        let pin = move |candidate: Url| async move {
            if candidate.port() == Some(redirector_port) {
                return Ok(vec![redirector_addr]);
            }
            resolve_validated(&candidate).await
        };
        let error = fetch_with_redirects(&start_url, pin).await.unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "got {error:?}"
        );
        assert!(redirector.connection_count() >= 1);
        assert_eq!(target.connection_count(), 0);
    }

    #[tokio::test]
    async fn download_exceeding_cap_without_content_length_is_rejected() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            &vec![b'a'; 4096],
        ))
        .await;
        let url_str = format!("http://127.0.0.1:{}/big", server.addr.port());
        let response = reqwest::Client::new().get(&url_str).send().await.unwrap();
        assert!(response.content_length().is_none());

        let error = read_response_with_limit(response, 1024).await.unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("exceeds maximum")),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn download_within_cap_without_content_length_succeeds() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            &vec![b'a'; 512],
        ))
        .await;
        let url_str = format!("http://127.0.0.1:{}/small", server.addr.port());
        let response = reqwest::Client::new().get(&url_str).send().await.unwrap();
        assert!(response.content_length().is_none());

        let bytes = read_response_with_limit(response, 1024).await.unwrap();

        assert_eq!(bytes.len(), 512);
    }

    #[tokio::test]
    async fn domain_resolving_to_blocked_address_is_rejected_without_connecting() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n",
            b"secret",
        ))
        .await;
        let start_url = format!("http://localhost:{}/doc", server.addr.port());

        let error = fetch_with_redirects(&start_url, |candidate| async move {
            resolve_validated(&candidate).await
        })
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "a domain whose DNS answer is a blocked address must be rejected, got {error:?}"
        );
        assert_eq!(server.connection_count(), 0);
    }

    #[tokio::test]
    async fn request_connects_only_to_pinned_address_without_a_second_dns_lookup() {
        let server = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n",
            b"ok",
        ))
        .await;
        let pinned_addr = server.addr;
        let url = format!("http://pinned.invalid:{}/doc", pinned_addr.port());

        let (_final_url, response) =
            fetch_with_redirects(&url, move |_candidate| async move { Ok(vec![pinned_addr]) })
                .await
                .expect("request must connect via the pinned address");

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(server.connection_count(), 1);
    }

    #[tokio::test]
    async fn redirect_across_ports_repins_without_reusing_stale_client() {
        let second = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n",
            b"second",
        ))
        .await;
        let second_addr = second.addr;
        let first = spawn_counting_server(http_response(
            &format!(
                "HTTP/1.1 302 Found\r\nLocation: http://pinned.invalid:{}/next\r\nContent-Length: 0\r\n\r\n",
                second_addr.port()
            ),
            b"",
        ))
        .await;
        let first_addr = first.addr;
        let start_url = format!("http://pinned.invalid:{}/start", first_addr.port());

        let (final_url, response) = fetch_with_redirects(&start_url, move |candidate| async move {
            match candidate.port() {
                Some(port) if port == first_addr.port() => Ok(vec![first_addr]),
                Some(port) if port == second_addr.port() => Ok(vec![second_addr]),
                _ => Err(blocked_url_error()),
            }
        })
        .await
        .expect("redirect across ports must repin to the second server");

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(final_url.port(), Some(second_addr.port()));
        assert_eq!(first.connection_count(), 1);
        assert_eq!(second.connection_count(), 1);
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
}
