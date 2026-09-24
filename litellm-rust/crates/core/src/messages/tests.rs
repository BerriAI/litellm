use std::{sync::Arc, time::Duration};

use futures_util::future::BoxFuture;
use litellm_http::request::{has_bearer_auth, has_header};
use litellm_secrets::{SecretValue, source::SecretSource};
use serde_json::{Map, Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

use super::{
    Error,
    common_utils::{messages_provider_config, string_headers, truncate_error_body},
    messages,
    route::{LocalMessagesHost, MessagesCall, MessagesOutput, messages_machine},
};
use crate::messages::types::{MessagesRequest, MessagesShaping};

struct RecordingSecrets {
    values: Vec<(&'static str, String)>,
    fails: bool,
    requested: std::sync::Mutex<Vec<String>>,
}

impl RecordingSecrets {
    fn new(values: Vec<(&'static str, String)>, fails: bool) -> Self {
        Self {
            values,
            fails,
            requested: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl SecretSource for RecordingSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
        Box::pin(async move {
            self.requested.lock().unwrap().push(name.to_string());
            if self.fails {
                return Err(litellm_secrets::Error::ManagedSecretMissing);
            }
            Ok(self
                .values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| SecretValue::new(value.clone())))
        })
    }
}

fn secrets_call() -> MessagesCall {
    let Value::Object(body) = json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    }) else {
        unreachable!("literal object")
    };
    MessagesCall {
        model: "claude-sonnet-4-5".into(),
        body,
        api_key: None,
        api_base: None,
        custom_llm_provider: Some("anthropic".into()),
        extra_headers: None,
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
    }
}

#[tokio::test]
async fn route_reads_the_provider_credential_and_base_from_the_secret_source() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        let response_body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
        socket
            .write_all(write_response(response_body).as_bytes())
            .await
            .expect("writes response");
        request
    });
    let secrets = Arc::new(RecordingSecrets::new(
        vec![
            ("ANTHROPIC_API_KEY", "sk-from-manager".to_string()),
            ("ANTHROPIC_BASE_URL", format!("http://{addr}")),
        ],
        false,
    ));

    let output = litellm_host::run::run(
        messages_machine(secrets.clone()),
        &LocalMessagesHost::new(secrets_call()),
    )
    .await
    .expect("messages request succeeds");

    assert!(matches!(output, MessagesOutput::Message(_)));
    let request = server.await.expect("server task completes");
    assert!(
        request
            .to_ascii_lowercase()
            .contains("x-api-key: sk-from-manager"),
        "{request}"
    );
    assert_eq!(
        *secrets.requested.lock().unwrap(),
        [
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_BASE",
            "ANTHROPIC_BASE_URL"
        ]
    );
}

#[tokio::test]
async fn route_reads_no_secret_when_the_caller_supplies_the_key_and_base() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        let response_body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
        socket
            .write_all(write_response(response_body).as_bytes())
            .await
            .expect("writes response");
        request
    });
    let failing = Arc::new(RecordingSecrets::new(Vec::new(), true));

    let output = litellm_host::run::run(
        messages_machine(failing.clone()),
        &LocalMessagesHost::new(MessagesCall {
            api_key: Some("sk-caller".into()),
            api_base: Some(format!("http://{addr}")),
            ..secrets_call()
        }),
    )
    .await
    .expect("messages request succeeds without touching the secret manager");

    assert!(matches!(output, MessagesOutput::Message(_)));
    assert!(
        server
            .await
            .expect("server task completes")
            .to_ascii_lowercase()
            .contains("x-api-key: sk-caller")
    );
    assert!(failing.requested.lock().unwrap().is_empty());
}

#[tokio::test]
async fn route_surfaces_a_secret_manager_failure_before_the_call() {
    let Err(error) = litellm_host::run::run(
        messages_machine(Arc::new(RecordingSecrets::new(Vec::new(), true))),
        &LocalMessagesHost::new(secrets_call()),
    )
    .await
    else {
        panic!("a secret manager failure fails the call");
    };
    assert!(
        matches!(&error, Error::Secret(source) if matches!(source.source_error(), litellm_secrets::Error::ManagedSecretMissing)),
        "{error:?}"
    );
}

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
    assert_eq!(
        err,
        Error::Headers(litellm_http::request::HeaderError {
            context: "messages",
            name: "x-count".to_string(),
            actual: "number",
        })
    );
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
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
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
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
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
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
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
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
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
        provider_specific_header: None,
        timeout: Some(Duration::from_millis(50)),
        shaping: MessagesShaping::default(),
    })
    .await
    .expect_err("missing auth errors");

    assert!(matches!(err, Error::Auth(_)));
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
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
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
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
    })
    .await
    .expect_err("provider error propagates");

    assert!(matches!(
        err,
        Error::Transport(litellm_http::transport::Error::Http { status: 401, .. })
    ));
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
        provider_specific_header: None,
        timeout: Some(Duration::from_millis(50)),
        shaping: MessagesShaping::default(),
    })
    .await
    .expect_err("unsupported provider errors");

    assert!(matches!(err, Error::InvalidProvider(provider) if provider == "openai"));
}
