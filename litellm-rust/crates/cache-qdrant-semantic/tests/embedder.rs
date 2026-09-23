use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{Error, semantic::Embedder};
use litellm_cache_qdrant_semantic::{OpenAiEmbedder, OpenAiEmbedderConfig};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

struct TestHttpServer {
    address: std::net::SocketAddr,
    request: Arc<Mutex<Option<Vec<u8>>>>,
    task: tokio::task::JoinHandle<()>,
}

impl TestHttpServer {
    async fn response(status: &str, body: &str) -> Self {
        Self::response_after(status, body, Duration::ZERO).await
    }

    async fn response_after(status: &str, body: &str, delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let request = Arc::new(Mutex::new(None));
        let captured = request.clone();
        let status = status.to_owned();
        let body = body.to_owned();
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request_bytes = read_request(&mut stream).await;
            *captured.lock().unwrap() = Some(request_bytes);
            tokio::time::sleep(delay).await;
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        Self {
            address,
            request,
            task,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.address)
    }
}

impl Drop for TestHttpServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn read_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 1024];
        let count = stream.read(&mut chunk).await.unwrap();
        assert_ne!(count, 0);
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.split_once(':')
                .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| value.trim())
        })
        .unwrap()
        .parse::<usize>()
        .unwrap();
    while bytes.len() < header_end + content_length {
        let mut chunk = [0_u8; 1024];
        let count = stream.read(&mut chunk).await.unwrap();
        assert_ne!(count, 0);
        bytes.extend_from_slice(&chunk[..count]);
    }
    bytes
}

fn config(base: String, timeout: Option<Duration>) -> OpenAiEmbedderConfig {
    OpenAiEmbedderConfig {
        api_base: base,
        api_key: "test-key".to_owned(),
        model: "test-model".to_owned(),
        timeout,
    }
}

#[rstest]
#[tokio::test]
async fn posts_embeddings_request_and_parses_vector() {
    let server = TestHttpServer::response("200 OK", r#"{"data":[{"embedding":[0.1,0.2]}]}"#).await;
    let embedder = OpenAiEmbedder::new(
        reqwest::Client::new(),
        config(
            format!("{}/", server.base_url()),
            Some(Duration::from_secs(1)),
        ),
    );
    assert_eq!(embedder.model(), "test-model");
    assert_eq!(
        embedder
            .async_embed("hello", Some(&json!({"ignored": true})))
            .await
            .unwrap(),
        vec![0.1, 0.2]
    );
    let request = server.request.lock().unwrap().clone().unwrap();
    let request_text = String::from_utf8(request).unwrap();
    assert!(request_text.starts_with("POST /embeddings HTTP/1.1\r\n"));
    assert!(request_text.contains("\r\nauthorization: Bearer test-key\r\n"));
    let body = request_text.split("\r\n\r\n").nth(1).unwrap();
    let body: Value = serde_json::from_str(body).unwrap();
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["input"], "hello");
    assert_eq!(body["encoding_format"], "float");
}

#[rstest]
#[case::error_status("500 Internal Server Error", "{}", 0, None, Err(Error::Unavailable))]
#[case::timed_out(
    "200 OK",
    r#"{"data":[{"embedding":[0.1,0.2]}]}"#,
    500,
    Some(Duration::from_millis(200)),
    Err(Error::Unavailable)
)]
#[case::within_timeout(
    "200 OK",
    r#"{"data":[{"embedding":[0.1,0.2]}]}"#,
    100,
    Some(Duration::from_secs(1)),
    Ok(vec![0.1, 0.2])
)]
#[case::missing_embedding("200 OK", r#"{"data":[]}"#, 0, None, Err(Error::Unavailable))]
#[tokio::test]
async fn status_timeout_and_body_errors_are_unavailable(
    #[case] status: &str,
    #[case] body: &str,
    #[case] delay_ms: u64,
    #[case] timeout: Option<Duration>,
    #[case] expected: Result<Vec<f32>, Error>,
) {
    let server =
        TestHttpServer::response_after(status, body, Duration::from_millis(delay_ms)).await;
    let embedder = OpenAiEmbedder::new(reqwest::Client::new(), config(server.base_url(), timeout));
    assert_eq!(embedder.async_embed("hello", None).await, expected);
}

#[rstest]
fn sync_embedding_is_unsupported() {
    let embedder = OpenAiEmbedder::new(
        reqwest::Client::new(),
        config("http://127.0.0.1:9".to_owned(), None),
    );
    assert_eq!(
        embedder.embed("hello", None),
        Err(Error::UnsupportedOperation)
    );
}

#[rstest]
#[tokio::test]
async fn uses_the_injected_client() {
    let server = TestHttpServer::response("200 OK", r#"{"data":[{"embedding":[0.1,0.2]}]}"#).await;
    let client = reqwest::Client::builder()
        .user_agent("litellm-embedder-test")
        .build()
        .unwrap();
    let embedder = OpenAiEmbedder::new(client, config(server.base_url(), None));
    assert_eq!(
        embedder.async_embed("hello", None).await.unwrap(),
        vec![0.1, 0.2]
    );
    let request = server.request.lock().unwrap().clone().unwrap();
    let request_text = String::from_utf8(request).unwrap();
    assert!(request_text.contains("\r\nuser-agent: litellm-embedder-test\r\n"));
}
