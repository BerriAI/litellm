//! Shared mapper from `reqwest` failures to typed [`CoreError`] contracts,
//! used by every endpoint in this crate that performs HTTP I/O.

use litellm_core::error::CoreError;

pub(crate) fn map_reqwest_error(err: reqwest::Error) -> CoreError {
    if err.is_timeout() {
        return CoreError::Timeout;
    }
    if let Some(status) = err.status() {
        return CoreError::Http {
            status: status.as_u16(),
            body: String::new(),
        };
    }
    if err.is_decode() {
        return CoreError::InvalidResponse(err.without_url().to_string());
    }
    CoreError::Network(err.without_url().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const URL_CANARY: &str = "canary-host.invalid";
    const QUERY_CANARY: &str = "token=SECRET_CANARY_123";

    async fn local_server(response: &'static str, delay: Option<Duration>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let mut buffer = [0_u8; 1024];
            let _ = socket.read(&mut buffer).await;
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            let _ = socket.write_all(response.as_bytes()).await;
        });
        format!("http://{addr}")
    }

    fn assert_no_canaries(text: &str) {
        assert!(!text.contains(URL_CANARY), "leaked host: {text}");
        assert!(!text.contains("SECRET_CANARY_123"), "leaked query: {text}");
    }

    #[tokio::test]
    async fn maps_request_timeout_to_timeout() {
        let base = local_server(
            "HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n",
            Some(Duration::from_secs(3)),
        )
        .await;
        let err = reqwest::Client::new()
            .get(format!("{base}/?{QUERY_CANARY}"))
            .timeout(Duration::from_millis(50))
            .send()
            .await
            .expect_err("timeout surfaces");
        let mapped = map_reqwest_error(err);
        assert_eq!(mapped, CoreError::Timeout);
        assert_eq!(mapped.public_status_code(), Some(408));
        assert_no_canaries(&mapped.to_string());
        assert_no_canaries(&mapped.public_message());
    }

    #[tokio::test]
    async fn maps_connect_failure_to_network() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");
        drop(listener);
        let err = reqwest::Client::new()
            .get(format!("http://{addr}/?{QUERY_CANARY}"))
            .send()
            .await
            .expect_err("connect failure surfaces");
        assert!(err.is_connect());
        let mapped = map_reqwest_error(err);
        assert!(matches!(mapped, CoreError::Network(_)));
        assert_eq!(mapped.public_status_code(), None);
        assert_no_canaries(&mapped.to_string());
        assert_no_canaries(&mapped.public_message());
    }

    #[tokio::test]
    async fn maps_status_error_to_typed_http() {
        let base = local_server(
            "HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
            None,
        )
        .await;
        let err = reqwest::Client::new()
            .get(format!("{base}/?{QUERY_CANARY}"))
            .send()
            .await
            .expect("response arrives")
            .error_for_status()
            .expect_err("status error surfaces");
        let mapped = map_reqwest_error(err);
        assert_eq!(
            mapped,
            CoreError::Http {
                status: 503,
                body: String::new()
            }
        );
        assert_eq!(mapped.public_status_code(), Some(503));
        assert_no_canaries(&mapped.to_string());
        assert_no_canaries(&mapped.public_message());
    }

    #[tokio::test]
    async fn maps_body_decode_failure_to_invalid_response() {
        let base = local_server(
            "HTTP/1.1 200 OK\r\ncontent-length: 100\r\nconnection: close\r\n\r\nshort",
            None,
        )
        .await;
        let response = reqwest::Client::new()
            .get(format!("{base}/?{QUERY_CANARY}"))
            .send()
            .await
            .expect("response arrives");
        let err = response.text().await.expect_err("body read fails");
        assert!(err.is_decode());
        let mapped = map_reqwest_error(err);
        assert!(matches!(mapped, CoreError::InvalidResponse(_)));
        assert_eq!(mapped.public_status_code(), Some(500));
        assert_no_canaries(&mapped.to_string());
        assert_no_canaries(&mapped.public_message());
    }

    #[tokio::test]
    async fn maps_unresolvable_host_to_network_without_url() {
        let err = reqwest::Client::new()
            .get(format!("http://{URL_CANARY}/?{QUERY_CANARY}"))
            .send()
            .await
            .expect_err("dns failure surfaces");
        let mapped = map_reqwest_error(err);
        assert!(matches!(mapped, CoreError::Network(_)));
        assert_no_canaries(&mapped.to_string());
        assert_no_canaries(&mapped.public_message());
    }
}
