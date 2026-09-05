use super::*;
use crate::auth::AuthPreflight;
use serde_json::json;

#[test]
fn document_contract_declines_python_owned_objects() {
    for value in [
        json!({"type":"file", "file":"/tmp/file"}),
        json!({"type":"image_url", "image_url":{"url":"x"}}),
    ] {
        assert!(matches!(decode_document(value), AuthPreflight::Declined(_)));
    }
    let value = json!({"type":"document_url", "document_url":"https://example.com/test.pdf"});
    let AuthPreflight::Ready(document) = decode_document(value.clone()) else {
        panic!("URL document should be admitted");
    };
    assert_eq!(serde_json::to_value(document).unwrap(), value);
}

use crate::auth::{
    AuthBinding, AuthFuture, AuthHttpClient, AuthRuntime, CredentialResolver, CredentialSpec,
    StaticHeaderAuthorizer, SystemClock,
};
use crate::provider_callbacks::NoopProviderAttemptObserver;
use crate::request_context::LiteLlmRequestContext;
use crate::request_options::RequestOptions;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

struct Credentials(Arc<AtomicUsize>);
impl CredentialResolver for Credentials {
    fn resolve(
        &self,
        credential: CredentialSpec,
        _: Arc<AuthRuntime>,
    ) -> AuthFuture<'_, AuthBinding> {
        Box::pin(async move {
            self.0.fetch_add(1, Ordering::SeqCst);
            let CredentialSpec::Header { name, value } = credential;
            Ok(AuthBinding::new(Arc::new(StaticHeaderAuthorizer::new(
                name,
                value,
                Vec::new(),
            ))))
        })
    }
}

fn runtime(calls: Arc<AtomicUsize>) -> Arc<AuthRuntime> {
    Arc::new(AuthRuntime {
        http: AuthHttpClient::new(
            reqwest::Client::builder().no_proxy(),
            Duration::from_secs(1),
            Duration::from_secs(2),
        )
        .unwrap(),
        clock: Arc::new(SystemClock),
        credentials: Arc::new(Credentials(calls)),
    })
}

fn plan(
    model: &str,
    base: &str,
    connection: serde_json::Map<String, serde_json::Value>,
) -> AuthPreflight<OcrPlan> {
    preflight(
        OcrRequest {
            model,
            document: OcrDocument::DocumentUrl {
                document_url: "https://example.com/test.pdf".into(),
            },
            optional_params: serde_json::Map::new(),
            options: RequestOptions {
                api_base: Some(base.into()),
                provider_connection: connection,
                ..Default::default()
            },
        },
        &|name| name.ends_with("API_KEY").then(|| "env-test-key".into()),
    )
    .unwrap()
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut wire = Vec::new();
    let mut buffer = [0; 4096];
    loop {
        let count = socket.read(&mut buffer).await.unwrap();
        assert_ne!(count, 0);
        wire.extend_from_slice(&buffer[..count]);
        let text = String::from_utf8_lossy(&wire);
        if let Some((headers, body)) = text.split_once("\r\n\r\n") {
            let length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(|value| value.parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            if body.len() >= length {
                return String::from_utf8(wire).unwrap();
            }
        }
    }
}

async fn reply(socket: &mut tokio::net::TcpStream, status: u16, headers: &str, body: &str) {
    socket.write_all(format!("HTTP/1.1 {status} Response\r\nconnection: close\r\ncontent-length: {}\r\n{headers}\r\n{body}", body.len()).as_bytes()).await.unwrap();
}

#[tokio::test]
async fn mistral_url_input_resolves_auth_in_rust_and_normalizes_response() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let wire = read_request(&mut socket).await;
        reply(
            &mut socket,
            200,
            "",
            r#"{"pages":[{"index":0,"markdown":"hello"}],"usage_info":{"pages_processed":1}}"#,
        )
        .await;
        wire
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let AuthPreflight::Ready(plan) = plan("mistral/mistral-ocr-latest", &base, Default::default())
    else {
        panic!("should admit Mistral");
    };
    let response = ocr(
        plan,
        &LiteLlmRequestContext::default(),
        runtime(calls.clone()),
        &mut NoopProviderAttemptObserver,
    )
    .await
    .unwrap();
    assert_eq!(response.model, "mistral-ocr-latest");
    assert_eq!(response.pages[0]["markdown"], "hello");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let wire = server.await.unwrap();
    assert!(wire.starts_with("POST /v1/ocr HTTP/1.1"));
    assert!(wire.contains("authorization: Bearer env-test-key"));
    let body: serde_json::Value =
        serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        json!({"model":"mistral-ocr-latest","document":{"type":"document_url","document_url":"https://example.com/test.pdf"}})
    );
}

#[tokio::test]
async fn document_intelligence_reuses_one_session_for_bodyless_same_origin_polls() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let location = format!("operation-location: {base}/operation/1\r\n");
    let server = tokio::spawn(async move {
        let mut requests = Vec::new();
        for attempt in 0..2 {
            let (mut socket, _) = listener.accept().await.unwrap();
            requests.push(read_request(&mut socket).await);
            if attempt == 0 {
                reply(&mut socket, 202, &location, "").await;
            } else {
                reply(&mut socket, 200, "", r#"{"status":"succeeded","analyzeResult":{"pages":[{"pageNumber":1,"lines":[{"content":"hello"}]}]}}"#).await;
            }
        }
        requests
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let AuthPreflight::Ready(plan) = plan(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        Default::default(),
    ) else {
        panic!("should admit Azure");
    };
    let response = ocr(
        plan,
        &LiteLlmRequestContext::default(),
        runtime(calls.clone()),
        &mut NoopProviderAttemptObserver,
    )
    .await
    .unwrap();
    assert_eq!(response.pages[0]["markdown"], "hello");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let wire = server.await.unwrap();
    assert!(wire[0].starts_with("POST /documentintelligence/"));
    assert!(wire[1].starts_with("GET /operation/1 HTTP/1.1"));
    assert!(wire[1].ends_with("\r\n\r\n"));
    assert!(
        wire.iter()
            .all(|request| request.contains("ocp-apim-subscription-key: env-test-key"))
    );
}

#[tokio::test]
async fn untrusted_poll_destination_is_terminal_before_second_request() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let foreign = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let location = format!(
        "operation-location: http://{}/operation\r\n",
        foreign.local_addr().unwrap()
    );
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        reply(&mut socket, 202, &location, "").await;
    });
    let AuthPreflight::Ready(plan) = plan(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        Default::default(),
    ) else {
        panic!("should admit Azure");
    };
    let error = ocr(
        plan,
        &LiteLlmRequestContext::default(),
        runtime(Arc::new(AtomicUsize::new(0))),
        &mut NoopProviderAttemptObserver,
    )
    .await
    .unwrap_err();
    assert!(matches!(error, crate::Error::InvalidResponse(_)));
    server.await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(30), foreign.accept())
            .await
            .is_err()
    );
}

#[test]
fn unsupported_auth_is_declined_during_preflight() {
    for connection in [
        json!({"azure_ad_token":"token"}),
        json!({"azure_client_secret":"secret"}),
        json!({"vertex_credentials":"credentials"}),
    ] {
        assert!(matches!(
            plan(
                "azure_ai/pixtral",
                "https://example.com",
                connection.as_object().unwrap().clone()
            ),
            AuthPreflight::Declined(_)
        ));
    }
    assert!(matches!(
        plan(
            "vertex_ai/mistral",
            "https://example.com",
            Default::default()
        ),
        AuthPreflight::Declined(_)
    ));
}

#[tokio::test]
async fn oversized_retry_after_is_bounded_by_the_call_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let location = format!("operation-location: {base}/operation/1\r\n");
    let server = tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request(&mut socket).await;
            if attempt == 0 {
                reply(&mut socket, 202, &location, "").await;
            } else {
                reply(
                    &mut socket,
                    200,
                    "retry-after: 1e18\r\n",
                    r#"{"status":"running"}"#,
                )
                .await;
            }
        }
    });
    let AuthPreflight::Ready(mut plan) = plan(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        Default::default(),
    ) else {
        panic!("should admit Azure");
    };
    plan.timeout = Some(Duration::from_millis(100));
    let result = ocr(
        plan,
        &LiteLlmRequestContext::default(),
        runtime(Arc::new(AtomicUsize::new(0))),
        &mut NoopProviderAttemptObserver,
    )
    .await;
    assert!(matches!(result, Err(crate::Error::Network(_))));
    server.await.unwrap();
}
