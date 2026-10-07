use litellm_lens::{
    State, Storage,
    auth::{Credential, Snapshot, unix_seconds},
    config::http_client,
    router,
};
use litellm_traces::Tenant;
use litellm_traces_clickhouse::Config;
use rstest::rstest;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const KEY: &str = "lens-trace-test-credential";
const SERVICE_TOKEN: &str = "test-only-service-credential-32-characters";

struct Server {
    url: String,
    state: Arc<State>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(clickhouse: &str, ready: bool) -> Server {
    let storage = Storage::new(
        Config::new("litellm".into(), clickhouse, 14, 65_536).unwrap(),
        http_client().unwrap(),
        SERVICE_TOKEN.into(),
    );
    let state = Arc::new(State::new(storage, SERVICE_TOKEN.into()));
    state.schema_ready.store(ready, Ordering::Release);
    state
        .credentials
        .replace(Snapshot {
            issued_at: unix_seconds(),
            keys: vec![Credential {
                token_hash: format!("{:x}", Sha256::digest(KEY)),
                tenant: Tenant {
                    team_id: "authenticated-team".into(),
                    user_id: "authenticated-user".into(),
                    api_key_hash: "authenticated-key".into(),
                    ..Tenant::default()
                },
                expires_at: None,
            }],
        })
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = router(state.clone());
    let task = tokio::spawn(async {
        axum::serve(listener, app).await.unwrap();
    });
    Server { url, state, task }
}

fn export() -> serde_json::Value {
    json!({"resourceSpans": [{"resource": {"attributes": [
        {"key": "service.name", "value": {"stringValue": "lens-receiver-test"}},
        {"key": "litellm.team_id", "value": {"stringValue": "spoofed-team"}}
    ]}, "scopeSpans": [{"spans": [{
        "traceId": "1234567890abcdef1234567890abcdef", "spanId": "1234567890abcdef",
        "name": "receiver boundary", "startTimeUnixNano": "1791388800000000000",
        "endTimeUnixNano": "1791388801000000000", "status": {"code": 1}
    }]}]}]})
}

#[rstest]
#[tokio::test]
async fn ingestion_confirms_storage_and_overwrites_exporter_tenant() {
    let store = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(100)))
        .expect(1)
        .mount(&store)
        .await;
    let server = serve(&store.uri(), true).await;
    let before = std::time::Instant::now();
    let response = http_client()
        .unwrap()
        .post(format!("{}/v1/traces", server.url))
        .bearer_auth(KEY)
        .json(&export())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(before.elapsed() >= Duration::from_millis(100));
    let requests = store.received_requests().await.unwrap();
    let mut decoded = String::new();
    std::io::Read::read_to_string(
        &mut flate2::read::GzDecoder::new(requests[0].body.as_slice()),
        &mut decoded,
    )
    .unwrap();
    let row: serde_json::Value = serde_json::from_str(decoded.trim()).unwrap();
    assert_eq!(row["TeamId"], "authenticated-team");
    assert_eq!(row["UserId"], "authenticated-user");
    assert_eq!(row["ApiKeyHash"], "authenticated-key");
}

#[rstest]
#[case::refused(503)]
#[case::disk_full(507)]
#[tokio::test]
async fn storage_failure_returns_retryable_otlp_error(#[case] status: u16) {
    let store = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status))
        .mount(&store)
        .await;
    let server = serve(&store.uri(), true).await;
    let response = http_client()
        .unwrap()
        .post(format!("{}/v1/traces", server.url))
        .bearer_auth(KEY)
        .json(&export())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    assert_eq!(response.headers()["retry-after"], "5");
    assert!(response.json::<serde_json::Value>().await.unwrap()["message"].is_string());
}

#[rstest]
#[tokio::test]
async fn no_storage_or_credentials_does_not_prevent_service_liveness() {
    let server = serve("http://127.0.0.1:1", false).await;
    server.state.credentials.clear();
    let client = http_client().unwrap();
    assert_eq!(
        client
            .get(format!("{}/health/live", server.url))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        client
            .get(format!("{}/health/ready", server.url))
            .send()
            .await
            .unwrap()
            .status(),
        503
    );
    assert_eq!(
        client
            .post(format!("{}/v1/traces", server.url))
            .bearer_auth(KEY)
            .json(&export())
            .send()
            .await
            .unwrap()
            .status(),
        503
    );
}

#[rstest]
#[tokio::test]
async fn ingestion_key_cannot_read_or_export_gateway_records() {
    let store = MockServer::start().await;
    let server = serve(&store.uri(), true).await;
    let client = http_client().unwrap();
    for path in ["/internal/read", "/internal/spend"] {
        let response = client
            .post(format!("{}{path}", server.url))
            .bearer_auth(KEY)
            .json(&json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401);
    }
    assert!(store.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn malformed_and_oversized_uploads_never_reach_storage() {
    let store = MockServer::start().await;
    let server = serve(&store.uri(), true).await;
    let client = http_client().unwrap();
    let malformed = client
        .post(format!("{}/v1/traces", server.url))
        .bearer_auth(KEY)
        .header("content-type", "application/json")
        .body("{")
        .send()
        .await
        .unwrap();
    assert_eq!(malformed.status(), 400);
    let oversized = client
        .post(format!("{}/v1/traces", server.url))
        .bearer_auth(KEY)
        .body(vec![b' '; 16 * 1024 * 1024 + 1])
        .send()
        .await
        .unwrap();
    assert_eq!(oversized.status(), 413);
    assert!(store.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn replacing_credentials_revokes_previous_keys() {
    let server = serve("http://127.0.0.1:1", true).await;
    server
        .state
        .credentials
        .replace(Snapshot {
            issued_at: unix_seconds(),
            keys: vec![],
        })
        .unwrap();
    let response = http_client()
        .unwrap()
        .post(format!("{}/v1/traces", server.url))
        .bearer_auth(KEY)
        .json(&export())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
}
