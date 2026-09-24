use litellm_http::request::string_headers as shared_string_headers;
pub(super) use litellm_http::request::truncate_error_body;
use litellm_llms::{
    anthropic::experimental_pass_through::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    azure_ai::anthropic::messages_transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG,
    base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig,
};
use serde_json::{Map, Value};

use super::Error;

const HEADER_CONTEXT: &str = "messages";

pub(super) fn messages_provider_config(
    provider: &str,
) -> Option<&'static dyn BaseAnthropicMessagesConfig> {
    match provider {
        "anthropic" => Some(&ANTHROPIC_MESSAGES_CONFIG),
        "azure_ai" => Some(&AZURE_ANTHROPIC_MESSAGES_CONFIG),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use futures_util::future::BoxFuture;
    use litellm_secrets::{SecretValue, source::SecretSource};
    use serde_json::{Value, json};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
    };

    use super::{messages_provider_config, string_headers, truncate_error_body};
    use crate::messages::{
        Error,
        route::{LocalMessagesHost, MessagesCall, MessagesOutput, messages_machine},
        types::MessagesShaping,
    };

    struct RecordingSecrets {
        values: Vec<(&'static str, String)>,
        requested: std::sync::Mutex<Vec<String>>,
    }

    impl SecretSource for RecordingSecrets {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
            Box::pin(async move {
                self.requested.lock().unwrap().push(name.to_string());
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

    #[tokio::test]
    async fn route_reads_the_provider_credential_and_base_from_the_secret_source() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let request = read_http_request(&mut socket).await;
            let response_body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            request
        });
        let secrets = Arc::new(RecordingSecrets {
            values: vec![
                ("ANTHROPIC_API_KEY", "sk-from-manager".to_string()),
                ("ANTHROPIC_BASE_URL", format!("http://{addr}")),
            ],
            requested: std::sync::Mutex::new(Vec::new()),
        });

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
        let requested = secrets.requested.lock().unwrap().clone();
        assert_eq!(
            requested,
            messages_provider_config("anthropic")
                .unwrap()
                .secret_names()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        );
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
}
