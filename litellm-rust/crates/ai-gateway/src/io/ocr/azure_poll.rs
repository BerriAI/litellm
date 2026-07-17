use std::net::SocketAddr;
use std::time::{Duration, Instant};

use litellm_core::error::CoreError;
use litellm_core::CoreResult;
use reqwest::Url;
use serde_json::Value;

use crate::constants::{
    AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS, AZURE_POLL_DEFAULT_RETRY_AFTER_SECS,
    AZURE_POLL_NETWORK_ERROR,
};

use super::common_utils::read_error_body;
use super::ssrf::{blocked_url_error, pinned_client, resolve_validated};

fn same_origin(left: &str, right: &str) -> bool {
    let Ok(left) = Url::parse(left) else {
        return false;
    };
    let Ok(right) = Url::parse(right) else {
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
        .unwrap_or(AZURE_POLL_DEFAULT_RETRY_AFTER_SECS)
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

fn poll_timeout_error(overall: Duration) -> CoreError {
    CoreError::Network(format!(
        "Azure Document Intelligence operation polling timed out after {} seconds",
        overall.as_secs()
    ))
}

pub(super) async fn poll_document_intelligence(
    operation_url: &str,
    original_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<Value> {
    poll_document_intelligence_with(
        operation_url,
        original_url,
        headers,
        timeout,
        |url| async move { resolve_validated(&url).await },
    )
    .await
}

pub(super) async fn poll_document_intelligence_with<P, Fut>(
    operation_url: &str,
    original_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
    resolve: P,
) -> CoreResult<Value>
where
    P: Fn(Url) -> Fut,
    Fut: std::future::Future<Output = CoreResult<Vec<SocketAddr>>>,
{
    if !same_origin(operation_url, original_url) {
        return Err(CoreError::InvalidRequest(
            "Azure Document Intelligence: rejected unsafe polling target".to_string(),
        ));
    }
    let parsed = Url::parse(operation_url).map_err(|_| blocked_url_error())?;

    let start = Instant::now();
    let overall = timeout.unwrap_or(Duration::from_secs(
        AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS,
    ));
    loop {
        let Some(remaining) = overall.checked_sub(start.elapsed()) else {
            return Err(poll_timeout_error(overall));
        };
        if remaining.is_zero() {
            return Err(poll_timeout_error(overall));
        }

        let addresses = resolve(parsed.clone()).await?;
        let client = pinned_client(&parsed, &addresses)?;
        let mut request_builder = client.get(parsed.clone()).timeout(remaining);
        for (key, value) in headers {
            if key.eq_ignore_ascii_case("ocp-apim-subscription-key") {
                request_builder = request_builder.header(key, value);
            }
        }
        let response = request_builder
            .send()
            .await
            .map_err(|_| CoreError::Network(AZURE_POLL_NETWORK_ERROR.to_string()))?;
        let retry_after = retry_after_secs(&response);
        let status = response.status();
        if !status.is_success() {
            return Err(CoreError::Http {
                status: status.as_u16(),
                body: read_error_body(response).await,
            });
        }
        let text = response
            .text()
            .await
            .map_err(|_| CoreError::Network(AZURE_POLL_NETWORK_ERROR.to_string()))?;
        let response_json: Value = serde_json::from_str(&text).map_err(|err| {
            CoreError::InvalidResponse(format!("invalid Azure DI poll response JSON: {err}"))
        })?;
        if operation_status(&response_json)? == "succeeded" {
            return Ok(response_json);
        }

        let remaining_after = overall
            .checked_sub(start.elapsed())
            .unwrap_or(Duration::ZERO);
        if remaining_after.is_zero() {
            return Err(poll_timeout_error(overall));
        }
        let sleep = Duration::from_secs(retry_after).min(remaining_after);
        tokio::time::sleep(sleep).await;
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{http_response, spawn_counting_server};
    use super::*;

    #[tokio::test]
    async fn foreign_operation_location_rejected_without_connecting() {
        let foreign = spawn_counting_server(http_response(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n",
            b"{}",
        ))
        .await;
        let operation_url = format!("http://127.0.0.1:{}/op", foreign.addr.port());
        let original_url = format!("http://127.0.0.1:{}/analyze", foreign.addr.port() ^ 1);

        let error = poll_document_intelligence(&operation_url, &original_url, &[], None)
            .await
            .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("polling target")),
            "got {error:?}"
        );
        assert_eq!(foreign.connection_count(), 0);
    }

    #[tokio::test]
    async fn same_origin_operation_location_polls_pinned_target() {
        let body = br#"{"status":"succeeded","analyzeResult":{}}"#;
        let server = spawn_counting_server(http_response(
            &format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()),
            body,
        ))
        .await;
        let server_addr = server.addr;
        let operation_url = format!("http://127.0.0.1:{}/op", server_addr.port());

        let result = poll_document_intelligence_with(
            &operation_url,
            &operation_url,
            &[],
            None,
            move |_url| async move { Ok(vec![server_addr]) },
        )
        .await;

        assert!(result.is_ok(), "got {result:?}");
        assert!(server.connection_count() >= 1);
    }

    #[tokio::test]
    async fn poll_forwards_only_subscription_key_header() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let capture = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 4096];
            let n = socket.read(&mut buffer).await.unwrap();
            let request = String::from_utf8_lossy(&buffer[..n]).to_string();
            let body = br#"{"status":"succeeded","analyzeResult":{}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.write_all(body).await.unwrap();
            socket.flush().await.unwrap();
            request
        });

        let operation_url = format!("http://127.0.0.1:{}/op", addr.port());
        let headers = vec![
            (
                "ocp-apim-subscription-key".to_string(),
                "di-secret".to_string(),
            ),
            ("authorization".to_string(), "Bearer leak".to_string()),
        ];

        let result = poll_document_intelligence_with(
            &operation_url,
            &operation_url,
            &headers,
            None,
            move |_url| async move { Ok(vec![addr]) },
        )
        .await
        .expect("poll succeeds");
        assert_eq!(result["status"], "succeeded");

        let request = capture.await.unwrap().to_ascii_lowercase();
        assert!(
            request.contains("ocp-apim-subscription-key: di-secret"),
            "{request}"
        );
        assert!(!request.contains("authorization:"), "{request}");
    }

    #[tokio::test]
    async fn poll_rejects_operation_location_resolving_to_blocked_address() {
        let operation_url = "http://127.0.0.1:9/op";

        let error = poll_document_intelligence(operation_url, operation_url, &[], None)
            .await
            .unwrap_err();

        assert!(
            matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
            "loopback poll target must be rejected, got {error:?}"
        );
    }

    #[tokio::test]
    async fn poll_timeout_override_is_enforced() {
        let body = br#"{"status":"running"}"#;
        let server = spawn_counting_server(http_response(
            &format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\nRetry-After: 0\r\n\r\n",
                body.len()
            ),
            body,
        ))
        .await;
        let server_addr = server.addr;
        let operation_url = format!("http://127.0.0.1:{}/op", server_addr.port());

        let error = poll_document_intelligence_with(
            &operation_url,
            &operation_url,
            &[],
            Some(Duration::from_millis(150)),
            move |_url| async move { Ok(vec![server_addr]) },
        )
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::Network(message) if message.contains("timed out")),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn poll_bounds_retry_after_by_remaining_timeout() {
        let body = br#"{"status":"running"}"#;
        let server = spawn_counting_server(http_response(
            &format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\nRetry-After: 3600\r\n\r\n",
                body.len()
            ),
            body,
        ))
        .await;
        let server_addr = server.addr;
        let operation_url = format!("http://127.0.0.1:{}/op", server_addr.port());

        let started = Instant::now();
        let error = poll_document_intelligence_with(
            &operation_url,
            &operation_url,
            &[],
            Some(Duration::from_millis(200)),
            move |_url| async move { Ok(vec![server_addr]) },
        )
        .await
        .unwrap_err();

        assert!(
            matches!(&error, CoreError::Network(message) if message.contains("timed out")),
            "got {error:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a huge Retry-After must be bounded by the remaining timeout"
        );
    }
}
