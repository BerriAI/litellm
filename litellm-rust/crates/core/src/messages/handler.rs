use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
use crate::error::{CoreError, CoreResult, as_response_error};
use crate::http_utils::classify_send_error;

use super::client::http_client;
use super::common_utils::truncate_error_body;
use super::types::{AnthropicMessagesResponse, ProviderMessagesRequest};

pub(super) async fn execute_messages_provider_call(
    request: ProviderMessagesRequest,
) -> CoreResult<AnthropicMessagesResponse> {
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder.send().await.map_err(classify_send_error)?;

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

    let response = serde_json::from_str(&text).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid messages response JSON: {err}"))
    })?;
    request
        .config
        .transform_response(&request.model, response)
        .map_err(as_response_error)
}

pub(super) async fn execute_messages_provider_stream(
    request: ProviderMessagesRequest,
) -> CoreResult<reqwest::Response> {
    if request.provider != ANTHROPIC_MESSAGES_PROVIDER {
        return Err(CoreError::InvalidRequest(
            "streaming messages is not supported for this provider".to_string(),
        ));
    }

    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        let text = response
            .text()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use super::*;
    use crate::messages::transformation::AnthropicMessagesProviderConfig;

    struct RejectingResponseConfig;

    impl AnthropicMessagesProviderConfig for RejectingResponseConfig {
        fn complete_url(
            &self,
            _api_base: Option<&str>,
            _model: &str,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> CoreResult<String> {
            unreachable!()
        }

        fn resolve_api_key(
            &self,
            _api_key: Option<&str>,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> CoreResult<String> {
            unreachable!()
        }

        fn transform_response(
            &self,
            _model: &str,
            _response: AnthropicMessagesResponse,
        ) -> CoreResult<AnthropicMessagesResponse> {
            Err(CoreError::MissingField("normalized_content"))
        }
    }

    static REJECTING_RESPONSE_CONFIG: RejectingResponseConfig = RejectingResponseConfig;

    fn request(url: String, timeout: Duration) -> ProviderMessagesRequest {
        ProviderMessagesRequest {
            provider: "anthropic".to_string(),
            model: "claude-test".to_string(),
            config: &REJECTING_RESPONSE_CONFIG,
            url,
            body: json!({}),
            upstream_headers: Vec::new(),
            timeout: Some(timeout),
        }
    }

    async fn read_http_request(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let read = socket.read(&mut buffer).await.expect("reads request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        String::from_utf8(request).expect("request is utf8")
    }

    #[tokio::test]
    async fn post_response_transform_errors_are_non_retryable() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let _ = read_http_request(&mut socket).await;
            let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-test"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
        });

        let error = execute_messages_provider_call(request(
            format!("http://{addr}/v1/messages"),
            Duration::from_secs(5),
        ))
        .await
        .expect_err("response transform should fail");

        server.await.expect("server task completes");
        assert!(
            matches!(error, CoreError::InvalidResponse(message) if message.contains("normalized_content"))
        );
    }

    #[tokio::test]
    async fn refused_connections_are_safe_to_fallback() {
        let port = {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
            listener.local_addr().expect("has an address").port()
        };
        let error = execute_messages_provider_call(request(
            format!("http://127.0.0.1:{port}"),
            Duration::from_secs(1),
        ))
        .await
        .expect_err("nothing is listening");

        assert!(matches!(error, CoreError::Connect(_)));
    }

    #[tokio::test]
    async fn established_request_timeouts_are_network_errors() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let addr = listener.local_addr().expect("has an address");
        let (request_received_tx, request_received_rx) = tokio::sync::oneshot::channel();
        let (release_server_tx, release_server_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let received = read_http_request(&mut socket).await;
            request_received_tx.send(received).expect("reports request");
            release_server_rx.await.expect("server is released");
        });

        let error = tokio::time::timeout(
            Duration::from_secs(2),
            execute_messages_provider_call(request(
                format!("http://{addr}"),
                Duration::from_millis(100),
            )),
        )
        .await
        .expect("client call completes")
        .expect_err("established request times out");

        let received = tokio::time::timeout(Duration::from_secs(2), request_received_rx)
            .await
            .expect("server observes request")
            .expect("server reports request");
        assert!(received.starts_with("POST / "), "{received}");
        release_server_tx.send(()).expect("releases server");
        server.await.expect("server task completes");
        assert!(matches!(error, CoreError::Network(_)));
    }
}
