use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::logging::{LogEvent, LogSink};
use futures_util::StreamExt;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::error::CoreError;
use crate::utils::truncate_error_body;

use super::common_utils::{has_bearer_auth, has_header, messages_provider_config, string_headers};
use super::types::MessagesRequest;
use super::{messages, messages_stream};

async fn read_http_request(socket: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    let header_end = loop {
        let n = socket.read(&mut buffer).await.expect("reads request");
        if n == 0 {
            break request.len();
        }
        request.extend_from_slice(&buffer[..n]);
        if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    while request.len().saturating_sub(header_end) < content_length {
        let n = socket.read(&mut buffer).await.expect("reads body");
        if n == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..n]);
    }
    String::from_utf8(request).expect("request is utf8")
}

fn write_response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

#[test]
fn provider_config_resolves_anthropic_and_azure_ai() {
    assert!(messages_provider_config("anthropic").is_some());
    assert!(messages_provider_config("azure_ai").is_some());
    assert!(messages_provider_config("openai").is_none());
}

#[test]
fn truncate_error_body_caps_long_payloads() {
    let body = "x".repeat(400);
    let truncated = truncate_error_body(&body);
    assert!(truncated.ends_with("... (truncated)"));
    let prefix_chars = truncated
        .strip_suffix("... (truncated)")
        .expect("truncated marker present")
        .chars()
        .count();
    assert_eq!(prefix_chars, 256);
}

#[test]
fn string_headers_rejects_non_string_values() {
    let headers = json!({"x-count": 3}).as_object().unwrap().clone();
    let err = string_headers(Some(headers)).expect_err("non-string header rejected");
    assert!(matches!(err, CoreError::InvalidRequest(_)));
}

#[test]
fn has_header_is_case_insensitive() {
    let headers = vec![("X-Api-Key".to_string(), "secret".to_string())];
    assert!(has_header(&headers, "x-api-key"));
    assert!(!has_header(&headers, "authorization"));
}

#[test]
fn has_bearer_auth_requires_a_nonempty_bearer_token() {
    assert!(has_bearer_auth(&[(
        "Authorization".to_string(),
        "Bearer tok".to_string()
    )]));
    assert!(has_bearer_auth(&[(
        "authorization".to_string(),
        "bearer tok".to_string()
    )]));
    assert!(!has_bearer_auth(&[(
        "authorization".to_string(),
        "Bearer ".to_string()
    )]));
    assert!(!has_bearer_auth(&[(
        "authorization".to_string(),
        String::new()
    )]));
    assert!(!has_bearer_auth(&[(
        "authorization".to_string(),
        "Basic abc".to_string()
    )]));
    assert!(!has_bearer_auth(&[(
        "x-api-key".to_string(),
        "sk".to_string()
    )]));
}

#[tokio::test]
async fn messages_round_trip_builds_azure_request_and_passes_response_through() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        let response_body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"model":"claude-sonnet-4-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}"#;
        socket
            .write_all(write_response(response_body).as_bytes())
            .await
            .expect("writes response");
        request
    });

    let response = messages(MessagesRequest {
        model: "claude-sonnet-4-5",
        body: json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral", "scope": "global"}
                }]
            }]
        }),
        api_key: Some("sk-azure"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("azure_ai"),
        extra_headers: None,
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: None,
        logging_sink: None,
    })
    .await
    .expect("messages request succeeds");

    assert_eq!(response.content[0]["text"], "hi");
    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));

    let request = server.await.expect("server task completes");
    let (head, body) = request.split_once("\r\n\r\n").expect("has body");
    assert!(head.starts_with("POST /anthropic/v1/messages "), "{head}");
    let head_lower = head.to_ascii_lowercase();
    assert!(head_lower.contains("x-api-key: sk-azure"), "{head}");
    assert!(
        head_lower.contains("anthropic-version: 2023-06-01"),
        "{head}"
    );
    assert!(
        head_lower.contains("content-type: application/json"),
        "{head}"
    );

    let sent_body: Value = serde_json::from_str(body).expect("body is json");
    assert_eq!(
        sent_body["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

#[tokio::test]
async fn messages_round_trip_builds_native_anthropic_request() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        let response_body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"model":"claude-sonnet-4-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}"#;
        socket
            .write_all(write_response(response_body).as_bytes())
            .await
            .expect("writes response");
        request
    });

    let response = messages(MessagesRequest {
        model: "claude-sonnet-4-5",
        body: json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}]
        }),
        api_key: Some("sk-ant"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: None,
        logging_sink: None,
    })
    .await
    .expect("messages request succeeds");

    assert_eq!(response.content[0]["text"], "hi");
    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));

    let request = server.await.expect("server task completes");
    let (head, _) = request.split_once("\r\n\r\n").expect("has body");
    assert!(head.starts_with("POST /v1/messages "), "{head}");
    let head_lower = head.to_ascii_lowercase();
    assert!(head_lower.contains("x-api-key: sk-ant"), "{head}");
    assert!(
        head_lower.contains("anthropic-version: 2023-06-01"),
        "{head}"
    );
}

#[tokio::test]
async fn messages_does_not_duplicate_auth_when_x_api_key_supplied() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        let response_body =
            r#"{"id":"msg_2","type":"message","role":"assistant","content":[],"model":"m"}"#;
        socket
            .write_all(write_response(response_body).as_bytes())
            .await
            .expect("writes response");
        request
    });

    let mut headers = Map::new();
    headers.insert(
        "x-api-key".to_string(),
        Value::String("from-python".to_string()),
    );
    headers.insert(
        "anthropic-beta".to_string(),
        Value::String("token-efficient-tools-2025-02-19".to_string()),
    );

    messages(MessagesRequest {
        model: "claude-sonnet-4-5",
        body: json!({"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []}),
        api_key: Some("rust-fallback-key"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("azure_ai"),
        extra_headers: Some(headers),
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: None,
        logging_sink: None,
    })
    .await
    .expect("messages request succeeds");

    let request = server.await.expect("server task completes");
    let head = request
        .split_once("\r\n\r\n")
        .expect("has body")
        .0
        .to_ascii_lowercase();
    let api_key_count = head
        .lines()
        .filter(|line| line.starts_with("x-api-key:"))
        .count();
    assert_eq!(api_key_count, 1, "{head}");
    assert!(head.contains("x-api-key: from-python"), "{head}");
    assert!(
        head.contains("anthropic-beta: token-efficient-tools-2025-02-19"),
        "{head}"
    );
    assert!(!head.contains("rust-fallback-key"), "{head}");
}

#[tokio::test]
async fn messages_forwards_entra_id_bearer_without_requiring_api_key() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        let response_body =
            r#"{"id":"msg_3","type":"message","role":"assistant","content":[],"model":"m"}"#;
        socket
            .write_all(write_response(response_body).as_bytes())
            .await
            .expect("writes response");
        request
    });

    let mut headers = Map::new();
    headers.insert(
        "Authorization".to_string(),
        Value::String("Bearer entra-token".to_string()),
    );

    messages(MessagesRequest {
        model: "claude-sonnet-4-5",
        body: json!({"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []}),
        api_key: None,
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("azure_ai"),
        extra_headers: Some(headers),
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: None,
        logging_sink: None,
    })
    .await
    .expect("entra id request succeeds without api key");

    let request = server.await.expect("server task completes");
    let head = request
        .split_once("\r\n\r\n")
        .expect("has body")
        .0
        .to_ascii_lowercase();
    assert!(head.contains("authorization: bearer entra-token"), "{head}");
    assert!(!head.contains("x-api-key"), "{head}");
}

#[tokio::test]
async fn messages_requires_auth_when_no_key_and_no_header() {
    let err = messages(MessagesRequest {
        model: "claude-sonnet-4-5",
        body: json!({"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []}),
        api_key: None,
        api_base: Some("http://127.0.0.1:1"),
        custom_llm_provider: Some("azure_ai"),
        extra_headers: None,
        timeout: Some(Duration::from_millis(50)),
        litellm_call_id: None,
        logging_sink: None,
    })
    .await
    .expect_err("missing auth errors");

    assert!(matches!(err, CoreError::Auth(_)));
}

#[tokio::test]
async fn messages_ignores_malformed_authorization_and_uses_api_key() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        let response_body =
            r#"{"id":"msg_4","type":"message","role":"assistant","content":[],"model":"m"}"#;
        socket
            .write_all(write_response(response_body).as_bytes())
            .await
            .expect("writes response");
        request
    });

    let mut headers = Map::new();
    headers.insert(
        "Authorization".to_string(),
        Value::String("Bearer ".to_string()),
    );

    messages(MessagesRequest {
        model: "claude-sonnet-4-5",
        body: json!({"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []}),
        api_key: Some("sk-azure"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("azure_ai"),
        extra_headers: Some(headers),
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: None,
        logging_sink: None,
    })
    .await
    .expect("falls back to api key");

    let request = server.await.expect("server task completes");
    let head = request
        .split_once("\r\n\r\n")
        .expect("has body")
        .0
        .to_ascii_lowercase();
    assert!(head.contains("x-api-key: sk-azure"), "{head}");
}

#[tokio::test]
async fn messages_maps_provider_error_status_to_http_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let _ = read_http_request(&mut socket).await;
        let body = "unauthorized";
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes response");
    });

    let err = messages(MessagesRequest {
        model: "claude-sonnet-4-5",
        body: json!({"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []}),
        api_key: Some("sk-azure"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("azure_ai"),
        extra_headers: None,
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: None,
        logging_sink: None,
    })
    .await
    .expect_err("provider error propagates");

    assert!(matches!(err, CoreError::Http { status: 401, .. }));
}

#[tokio::test]
async fn messages_rejects_unsupported_provider() {
    let err = messages(MessagesRequest {
        model: "claude-3-5-sonnet",
        body: json!({"model": "claude-3-5-sonnet", "max_tokens": 8, "messages": []}),
        api_key: Some("sk"),
        api_base: Some("http://127.0.0.1:1"),
        custom_llm_provider: Some("openai"),
        extra_headers: None,
        timeout: Some(Duration::from_millis(50)),
        litellm_call_id: None,
        logging_sink: None,
    })
    .await
    .expect_err("unsupported provider errors");

    assert!(matches!(err, CoreError::InvalidProvider(provider) if provider == "openai"));
}

#[derive(Clone, Default)]
struct RecordingSink(Arc<Mutex<Vec<LogEvent>>>);

impl LogSink for RecordingSink {
    fn emit(&self, event: &LogEvent) {
        self.0.lock().expect("recording lock").push(event.clone());
    }
}

#[tokio::test]
async fn messages_debug_logs_transformed_request_and_redacted_response() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        let response = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"visible"}],"model":"claude","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"token":"response-secret"}"#;
        socket
            .write_all(write_response(response).as_bytes())
            .await
            .expect("writes response");
        request
    });
    let sink = RecordingSink::default();
    let response = messages(MessagesRequest {
        model: "claude",
        body: json!({
            "model": "claude",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "prompt visible", "token": "request-secret"}]
        }),
        api_key: Some("request-secret"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("anthropic"),
        extra_headers: Some(
            json!({
                "Authorization": "Bearer bearer-secret",
                "X-Amz-Security-Token": "session-secret"
            })
            .as_object()
            .expect("headers")
            .clone(),
        ),
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: Some("debug-success"),
        logging_sink: Some(Arc::new(sink.clone())),
    })
    .await
    .expect("messages succeeds");
    assert_eq!(response.content[0]["text"], "visible");
    let request = server.await.expect("server task");
    let events = sink.0.lock().expect("recording lock");
    let serialized = serde_json::to_string(&*events).expect("events serialize");
    assert!(serialized.contains("prompt visible"));
    assert!(!serialized.contains("request-secret"));
    assert!(!serialized.contains("bearer-secret"));
    assert!(!serialized.contains("session-secret"));
    assert!(!serialized.contains("response-secret"));
    let LogEvent::Request(request_event) = &events[0] else {
        panic!("request event first");
    };
    assert!(request_event.url.ends_with("/v1/messages"));
    assert_eq!(request_event.headers["content-type"], "application/json");
    let sent_body = request.split_once("\r\n\r\n").expect("request body").1;
    assert!(sent_body.contains("prompt visible"));
    let response_event = events
        .iter()
        .find_map(|event| match event {
            LogEvent::Response(value) => Some(value),
            _ => None,
        })
        .expect("response event");
    assert_eq!(response_event.status, 200);
    assert!(response_event.duration_ms < 10_000);
}

#[tokio::test]
async fn messages_debug_wrong_shape_emits_response_then_failure() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let _ = read_http_request(&mut socket).await;
        socket
            .write_all(write_response(r#"{"id":"wrong-shape"}"#).as_bytes())
            .await
            .expect("writes response");
    });
    let sink = RecordingSink::default();
    let err = messages(MessagesRequest {
        model: "claude",
        body: json!({"model": "claude", "max_tokens": 8, "messages": []}),
        api_key: Some("request-secret"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: Some("wrong-shape"),
        logging_sink: Some(Arc::new(sink.clone())),
    })
    .await
    .expect_err("wrong response shape errors");
    assert!(matches!(err, CoreError::InvalidResponse(_)));

    server.await.expect("server task");
    let events = sink.0.lock().expect("recording lock");
    assert!(matches!(events[0], LogEvent::Request(_)));
    assert!(matches!(events[1], LogEvent::Response(_)));
    assert!(matches!(events[2], LogEvent::Error(_)));
    assert_eq!(events.len(), 3);
}

#[tokio::test]
async fn messages_debug_http_failure_emits_one_error_and_none_emits_nothing() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let _ = read_http_request(&mut socket).await;
        let body = r#"{"token":"provider-secret","error":"no"}"#;
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes response");
    });
    let sink = RecordingSink::default();
    let err = messages(MessagesRequest {
        model: "claude",
        body: json!({"model": "claude", "max_tokens": 8, "messages": []}),
        api_key: Some("secret"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: Some("debug-error"),
        logging_sink: Some(Arc::new(sink.clone())),
    })
    .await
    .expect_err("request fails");
    assert!(matches!(err, CoreError::Http { status: 401, .. }));
    server.await.expect("server task");
    {
        let events = sink.0.lock().expect("recording lock");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, LogEvent::Error(_)))
                .count(),
            1
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, LogEvent::Response(_)))
        );
    }

    let none_sink = messages(MessagesRequest {
        model: "claude",
        body: json!({"model": "claude", "max_tokens": 8, "messages": []}),
        api_key: Some("secret"),
        api_base: Some("http://127.0.0.1:1"),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        timeout: Some(Duration::from_millis(20)),
        litellm_call_id: Some("debug-disabled"),
        logging_sink: None,
    })
    .await
    .expect_err("disabled test endpoint fails");
    assert!(matches!(none_sink, CoreError::Network(_)));
}

#[tokio::test]
async fn messages_debug_stream_counts_clean_and_interrupted_streams() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let _ = read_http_request(&mut socket).await;
        let body = b"data: one\n\ndata: two\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("headers");
        socket.write_all(body).await.expect("body");
    });
    let sink = RecordingSink::default();
    let upstream = messages_stream(MessagesRequest {
        model: "claude",
        body: json!({"model": "claude", "max_tokens": 8, "stream": true, "messages": []}),
        api_key: Some("secret"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: Some("debug-stream"),
        logging_sink: Some(Arc::new(sink.clone())),
    })
    .await
    .expect("stream starts");
    let chunks = crate::logging::stream::count_forwarded_stream(
        upstream.response.bytes_stream(),
        upstream.logger,
    );
    let forwarded = futures_util::StreamExt::collect::<Vec<_>>(chunks).await;
    assert!(!forwarded.is_empty());
    assert!(forwarded.iter().all(Result::is_ok));
    server.await.expect("server task");
    let events = sink.0.lock().expect("recording lock");
    let completed = events
        .iter()
        .find_map(|event| match event {
            LogEvent::StreamCompleted(value) => Some(value),
            _ => None,
        })
        .expect("completion");
    assert!(completed.bytes_received > 0);
    assert!(completed.frames_received > 0);
    assert!(completed.events_decoded > 0);
}

#[tokio::test]
async fn messages_debug_interrupted_stream_emits_error_without_completion() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let _ = read_http_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: 100\r\nconnection: close\r\n\r\nshort",
            )
            .await
            .expect("writes partial response");
    });
    let sink = RecordingSink::default();
    let upstream = messages_stream(MessagesRequest {
        model: "claude",
        body: json!({"model": "claude", "max_tokens": 8, "stream": true, "messages": []}),
        api_key: Some("secret"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: Some("debug-interrupted"),
        logging_sink: Some(Arc::new(sink.clone())),
    })
    .await
    .expect("stream starts");
    let results = crate::logging::stream::count_forwarded_stream(
        upstream.response.bytes_stream(),
        upstream.logger,
    )
    .collect::<Vec<_>>()
    .await;
    assert!(results.iter().any(Result::is_err));
    server.await.expect("server task");
    let events = sink.0.lock().expect("recording lock");
    assert!(events.iter().any(|event| matches!(
        event,
        LogEvent::Error(value) if value.kind == "stream_error"
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, LogEvent::StreamCompleted(_)))
    );
}
