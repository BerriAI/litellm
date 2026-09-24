use std::{sync::OnceLock, time::Duration};

use litellm_http::{request::truncate_error_body, transport::Error as TransportError};
use litellm_llms::{
    anthropic::batches::transformation::{
        ANTHROPIC_BATCHES_TRANSFORMATION, AnthropicBatchesConfig, AnthropicMessageBatch,
        LiteLlmMessageBatch,
    },
    base_llm::{anthropic_messages::transformation::Headers, chat::transformation::Error as LlmError},
};
use reqwest::Method;
use time::OffsetDateTime;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Request(#[from] LlmError),
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error("invalid Anthropic batch response: {0}")]
    InvalidResponse(String),
}

pub struct Connection<'a> {
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub extra_headers: Headers,
    pub timeout: Option<Duration>,
}

pub struct RetrieveBatchRequest<'a> {
    pub batch_id: &'a str,
    pub connection: Connection<'a>,
}

pub struct CreateBatchRequest<'a> {
    pub model: Option<&'a str>,
    pub input_jsonl: &'a str,
    pub connection: Connection<'a>,
}

pub fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

pub async fn retrieve_batch(
    client: &reqwest::Client,
    request: RetrieveBatchRequest<'_>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<LiteLlmMessageBatch, Error> {
    let config = &ANTHROPIC_BATCHES_TRANSFORMATION;
    let connection = request.connection;
    let url = config.retrieve_batch_url(connection.api_base, request.batch_id, env_lookup)?;
    let headers =
        config.validate_environment(connection.extra_headers, connection.api_key, env_lookup)?;
    let batch = send(client, Method::GET, url, headers, None, connection.timeout).await?;
    Ok(config.transform_retrieve_batch_response(batch, now()))
}

pub async fn create_batch(
    client: &reqwest::Client,
    request: CreateBatchRequest<'_>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<LiteLlmMessageBatch, Error> {
    let config = &ANTHROPIC_BATCHES_TRANSFORMATION;
    let connection = request.connection;
    let body = config.transform_create_batch_request(request.model, request.input_jsonl)?;
    let url = config.create_batch_url(connection.api_base, env_lookup)?;
    let headers =
        config.validate_environment(connection.extra_headers, connection.api_key, env_lookup)?;
    let body = serde_json::to_vec(&body)
        .map_err(|error| LlmError::InvalidRequest(format!("unserializable batch: {error}")))?;
    let batch = send(client, Method::POST, url, headers, Some(body), connection.timeout).await?;
    Ok(config.transform_create_batch_response(batch, now()))
}

fn now() -> i64 {
    OffsetDateTime::now_utc().unix_timestamp()
}

async fn send(
    client: &reqwest::Client,
    method: Method,
    url: String,
    headers: Headers,
    body: Option<Vec<u8>>,
    timeout: Option<Duration>,
) -> Result<AnthropicMessageBatch, Error> {
    let request = headers
        .iter()
        .fold(client.request(method, url), |builder, (name, value)| {
            builder.header(name, value)
        });
    let request = body.into_iter().fold(request, reqwest::RequestBuilder::body);
    let request = timeout.into_iter().fold(request, reqwest::RequestBuilder::timeout);
    let response = request
        .send()
        .await
        .map_err(TransportError::from_reqwest_before_dispatch)?;
    let status = response.status();
    let text = response.text().await.map_err(TransportError::from)?;
    if !status.is_success() {
        return Err(TransportError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        }
        .into());
    }
    serde_json::from_str(&text).map_err(|error| Error::InvalidResponse(error.to_string()))
}

#[cfg(test)]
mod tests {
    use litellm_auth::Error as AuthError;
    use litellm_llms::anthropic::batches::transformation::{BatchRequestCounts, BatchStatus};
    use rstest::rstest;
    use serde_json::{Value, json};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        task::JoinHandle,
    };

    use super::*;

    const CREATED: i64 = 1_727_172_000;
    const BATCH: &str = r#"{"id":"msgbatch_1","type":"message_batch","processing_status":"in_progress","created_at":"2024-09-24T10:00:00Z","request_counts":{"processing":2,"succeeded":1}}"#;

    #[derive(Debug, PartialEq)]
    struct Received {
        request_line: String,
        headers: Vec<(String, String)>,
        body: String,
    }

    async fn stub(status: &'static str, body: &'static str) -> (String, JoinHandle<Received>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut raw = Vec::new();
            let mut buffer = [0_u8; 4096];
            let received = loop {
                let n = socket.read(&mut buffer).await.unwrap();
                raw.extend_from_slice(&buffer[..n]);
                let text = String::from_utf8_lossy(&raw).to_string();
                let Some((head, body)) = text.split_once("\r\n\r\n") else {
                    continue;
                };
                let mut lines = head.lines();
                let request_line = lines.next().unwrap().to_string();
                let mut headers: Vec<(String, String)> = lines
                    .filter_map(|line| line.split_once(": "))
                    .map(|(name, value)| (name.to_lowercase(), value.to_string()))
                    .filter(|(name, _)| !matches!(name.as_str(), "host" | "content-length"))
                    .collect();
                headers.sort();
                let length = head
                    .lines()
                    .find_map(|line| line.to_lowercase().strip_prefix("content-length: ").map(str::to_string))
                    .map_or(0, |value| value.parse::<usize>().unwrap());
                if body.len() >= length || n == 0 {
                    break Received {
                        request_line,
                        headers,
                        body: body.to_string(),
                    };
                }
            };
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            received
        });
        (base, handle)
    }

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + Sync + use<> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        }
    }

    fn connection<'a>(api_key: Option<&'a str>, api_base: Option<&'a str>) -> Connection<'a> {
        Connection {
            api_key,
            api_base,
            extra_headers: vec![],
            timeout: Some(Duration::from_secs(5)),
        }
    }

    fn headers(auth: (&str, &str)) -> Vec<(String, String)> {
        let mut headers: Vec<(String, String)> = [
            ("accept", "application/json"),
            ("anthropic-beta", "message-batches-2024-09-24"),
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
            auth,
        ]
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .into();
        headers.sort();
        headers
    }

    fn in_progress_batch() -> LiteLlmMessageBatch {
        LiteLlmMessageBatch {
            id: "msgbatch_1".into(),
            object: "batch".into(),
            endpoint: "/v1/messages".into(),
            input_file_id: "None".into(),
            completion_window: "24h".into(),
            status: BatchStatus::InProgress,
            output_file_id: "msgbatch_1".into(),
            created_at: CREATED,
            in_progress_at: Some(CREATED),
            expires_at: None,
            completed_at: None,
            expired_at: None,
            cancelling_at: None,
            cancelled_at: None,
            request_counts: BatchRequestCounts {
                total: 3,
                completed: 1,
                failed: 0,
            },
        }
    }

    #[rstest]
    #[case::explicit_key(Some("sk-ant-param"), &[], ("x-api-key", "sk-ant-param"))]
    #[case::key_from_environment(None, &[("ANTHROPIC_API_KEY", "sk-ant-env")], ("x-api-key", "sk-ant-env"))]
    #[tokio::test]
    async fn retrieve_gets_the_batch_and_maps_it(
        #[case] api_key: Option<&'static str>,
        #[case] environment: &'static [(&'static str, &'static str)],
        #[case] auth: (&'static str, &'static str),
    ) {
        let (base, server) = stub("200 OK", BATCH).await;
        let batch = retrieve_batch(
            http_client(),
            RetrieveBatchRequest {
                batch_id: "msgbatch_1",
                connection: connection(api_key, Some(&base)),
            },
            &env(environment),
        )
        .await
        .unwrap();

        assert_eq!(batch, in_progress_batch());
        assert_eq!(
            server.await.unwrap(),
            Received {
                request_line: "GET /v1/messages/batches/msgbatch_1 HTTP/1.1".into(),
                headers: headers(auth),
                body: String::new(),
            }
        );
    }

    #[tokio::test]
    async fn retrieve_falls_back_to_the_base_from_the_environment() {
        let (base, server) = stub("200 OK", BATCH).await;
        retrieve_batch(
            http_client(),
            RetrieveBatchRequest {
                batch_id: "msgbatch_1",
                connection: connection(Some("sk"), None),
            },
            &env(&[("ANTHROPIC_API_BASE", &base)]),
        )
        .await
        .unwrap();

        assert_eq!(
            server.await.unwrap().request_line,
            "GET /v1/messages/batches/msgbatch_1 HTTP/1.1"
        );
    }

    #[rstest]
    #[case::missing_key(
        "msgbatch_1",
        None,
        Error::Request(LlmError::Auth(AuthError::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        })),
    )]
    #[case::dot_segment(
        "..",
        Some("sk"),
        Error::Request(LlmError::InvalidRequest("batch_id cannot be a dot path segment".into())),
    )]
    #[tokio::test]
    async fn retrieve_rejects_requests_before_calling_out(
        #[case] batch_id: &str,
        #[case] api_key: Option<&str>,
        #[case] expected: Error,
    ) {
        let error = retrieve_batch(
            http_client(),
            RetrieveBatchRequest {
                batch_id,
                connection: connection(api_key, Some("http://127.0.0.1:9")),
            },
            &env(&[]),
        )
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), expected.to_string());
        assert!(matches!(error, Error::Request(_)));
    }

    #[rstest]
    #[case::not_found(
        "404 Not Found",
        r#"{"type":"error","error":{"type":"not_found_error"}}"#,
        "upstream request failed with status 404: {\"type\":\"error\",\"error\":{\"type\":\"not_found_error\"}}",
    )]
    #[case::server_error("500 Internal Server Error", "boom", "upstream request failed with status 500: boom")]
    #[tokio::test]
    async fn retrieve_surfaces_upstream_failures(
        #[case] status: &'static str,
        #[case] body: &'static str,
        #[case] message: &str,
    ) {
        let (base, _server) = stub(status, body).await;
        let error = retrieve_batch(
            http_client(),
            RetrieveBatchRequest {
                batch_id: "msgbatch_1",
                connection: connection(Some("sk"), Some(&base)),
            },
            &env(&[]),
        )
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), message);
        assert!(!matches!(error, Error::Request(_)));
    }

    #[tokio::test]
    async fn retrieve_rejects_a_success_body_that_is_not_a_batch() {
        let (base, _server) = stub("200 OK", "[]").await;
        let error = retrieve_batch(
            http_client(),
            RetrieveBatchRequest {
                batch_id: "msgbatch_1",
                connection: connection(Some("sk"), Some(&base)),
            },
            &env(&[]),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, Error::InvalidResponse(_)), "{error:?}");
    }

    #[tokio::test]
    async fn create_posts_the_translated_requests_and_maps_the_batch() {
        let (base, server) = stub("200 OK", BATCH).await;
        let input = json!({
            "custom_id": "r1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "alias", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        })
        .to_string();
        let batch = create_batch(
            http_client(),
            CreateBatchRequest {
                model: Some("claude-deployed"),
                input_jsonl: &input,
                connection: Connection {
                    extra_headers: vec![("x-trace".into(), "t1".into())],
                    ..connection(Some("sk-ant"), Some(&base))
                },
            },
            &env(&[]),
        )
        .await
        .unwrap();

        assert_eq!(batch, in_progress_batch());
        let received = server.await.unwrap();
        let mut expected_headers = headers(("x-api-key", "sk-ant"));
        expected_headers.push(("x-trace".into(), "t1".into()));
        expected_headers.sort();
        assert_eq!(
            (
                received.request_line,
                received.headers,
                serde_json::from_str::<Value>(&received.body).unwrap()
            ),
            (
                "POST /v1/messages/batches HTTP/1.1".to_string(),
                expected_headers,
                json!({"requests": [{"custom_id": "r1", "params": {
                    "model": "claude-deployed",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                    "max_tokens": 4,
                }}]}),
            )
        );
    }

    #[tokio::test]
    async fn create_rejects_an_untranslatable_input_before_calling_out() {
        let error = create_batch(
            http_client(),
            CreateBatchRequest {
                model: None,
                input_jsonl: "",
                connection: connection(Some("sk"), Some("http://127.0.0.1:9")),
            },
            &env(&[]),
        )
        .await
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid request: batch input file has no requests"
        );
        assert!(matches!(error, Error::Request(_)));
    }
}
