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
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_string_contains, method, query_param},
};

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
async fn agent_picker_query_preserves_scope_through_the_internal_read_route() {
    let store = MockServer::start().await;
    let result = json!({"data": [{
        "agent_name": "research-agent", "runs": "3", "failed_runs": "1",
        "last_seen_ms": "1791405060000", "frameworks": ["openai-agents"]
    }]});
    Mock::given(method("POST"))
        .and(body_string_contains("FROM agent_traces_by_key"))
        .and(body_string_contains("o.AgentName"))
        .and(query_param("param_all_teams", "0"))
        .and(query_param("param_user_id", "agent-owner"))
        .and(query_param("param_team_ids", "['managed-team']"))
        .and(query_param("param_start_ms", "123"))
        .and(query_param("param_end_ms", "456"))
        .and(query_param("param_limit", "100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&result))
        .expect(1)
        .mount(&store)
        .await;
    let server = serve(&store.uri(), true).await;
    let response = http_client()
        .unwrap()
        .post(format!("{}/internal/read", server.url))
        .bearer_auth(SERVICE_TOKEN)
        .json(&json!({
            "operation": "query", "name": "trace_agents", "parameters": {
                "all_teams": 0, "user_id": "agent-owner", "team_ids": ["managed-team"],
                "start_ms": 123, "end_ms": 456, "limit": 100
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.json::<serde_json::Value>().await.unwrap(), result);
}

#[rstest]
#[tokio::test]
async fn feedback_summary_query_preserves_scope_through_the_internal_read_route() {
    let store = MockServer::start().await;
    let result = json!({"data": [{
        "trace_id": "1234567890abcdef1234567890abcdef", "trace_ref": "REF",
        "count": "2", "average": 5.5, "lowest": "2"
    }]});
    Mock::given(method("POST"))
        .and(body_string_contains("FROM lens_feedback FINAL"))
        .and(query_param("param_all_teams", "0"))
        .and(query_param("param_team", "feedback-team"))
        .and(query_param(
            "param_trace_ids",
            "['1234567890abcdef1234567890abcdef']",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(&result))
        .expect(1)
        .mount(&store)
        .await;
    let server = serve(&store.uri(), true).await;
    let response = http_client()
        .unwrap()
        .post(format!("{}/internal/read", server.url))
        .bearer_auth(SERVICE_TOKEN)
        .json(&json!({
            "operation": "query", "name": "feedback_summary", "parameters": {
                "all_teams": 0, "team": "feedback-team", "key_hash": "",
                "trace_ids": ["1234567890abcdef1234567890abcdef"]
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.json::<serde_json::Value>().await.unwrap(), result);
}

#[rstest]
#[tokio::test]
async fn feedback_rows_are_written_to_the_feedback_table() {
    let store = MockServer::start().await;
    Mock::given(method("POST"))
        .and(query_param(
            "query",
            "INSERT INTO `litellm`.lens_feedback FORMAT JSONEachRow",
        ))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&store)
        .await;
    let server = serve(&store.uri(), true).await;
    let response = http_client()
        .unwrap()
        .post(format!("{}/internal/feedback", server.url))
        .bearer_auth(SERVICE_TOKEN)
        .json(&json!([{
            "TeamId": "team", "ApiKeyHash": "", "TraceId": "1234567890abcdef1234567890abcdef",
            "Author": "customer-1042", "Score": 2, "Comment": "wrong command",
            "CreatedAt": "2026-10-07T21:57:01.414Z", "UpdatedAt": "2026-10-07T21:57:01.414Z",
            "IsDeleted": 0
        }]))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 204);
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
#[tokio::test]
async fn shared_ingress_prefix_exposes_uploads_without_internal_control_routes() {
    let store = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&store)
        .await;
    let server = serve(&store.uri(), true).await;
    let client = http_client().unwrap();
    let upload = client
        .post(format!("{}/lens-ingest/v1/traces", server.url))
        .bearer_auth(KEY)
        .json(&export())
        .send()
        .await
        .unwrap();
    assert_eq!(upload.status(), 200);
    let internal = client
        .get(format!("{}/lens-ingest/internal/status", server.url))
        .bearer_auth(SERVICE_TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(internal.status(), 404);
    let preflight = client
        .request(
            http::Method::OPTIONS,
            format!("{}/lens-ingest/v1/traces", server.url),
        )
        .header("origin", "https://dashboard.example")
        .header("access-control-request-method", "POST")
        .header(
            "access-control-request-headers",
            "authorization,content-type",
        )
        .send()
        .await
        .unwrap();
    assert_eq!(preflight.headers()["access-control-allow-origin"], "*");
    assert!(
        !preflight
            .headers()
            .contains_key("access-control-allow-credentials")
    );
}

#[rstest]
#[tokio::test]
async fn only_the_service_secret_can_replace_ingestion_credentials() {
    let server = serve("http://127.0.0.1:1", true).await;
    let client = http_client().unwrap();
    let snapshot = json!({"issued_at": unix_seconds(), "keys": []});
    let denied = client
        .post(format!("{}/internal/credentials", server.url))
        .bearer_auth(KEY)
        .json(&snapshot)
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 401);
    let accepted = client
        .post(format!("{}/internal/credentials", server.url))
        .bearer_auth(SERVICE_TOKEN)
        .json(&snapshot)
        .send()
        .await
        .unwrap();
    assert_eq!(accepted.status(), 204);
    let revoked = client
        .post(format!("{}/v1/traces", server.url))
        .bearer_auth(KEY)
        .json(&export())
        .send()
        .await
        .unwrap();
    assert_eq!(revoked.status(), 401);
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
    for path in ["/internal/read", "/internal/spend", "/internal/feedback"] {
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

#[rstest]
#[tokio::test]
async fn newly_created_key_is_retryable_until_this_replica_has_refreshed() {
    let server = serve("http://127.0.0.1:1", true).await;
    let now = unix_seconds();
    let token = format!("lens-trace-{now}-new-key");
    let client = http_client().unwrap();
    let pending = client
        .post(format!("{}/v1/traces", server.url))
        .bearer_auth(&token)
        .json(&export())
        .send()
        .await
        .unwrap();
    assert_eq!(pending.status(), 429);
    assert_eq!(pending.headers()["retry-after"], "5");
    let older = format!("lens-trace-{}-invalid-key", now - 100);
    let denied = client
        .post(format!("{}/v1/traces", server.url))
        .bearer_auth(&older)
        .json(&export())
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 401);
    assert!(
        server
            .state
            .credentials
            .replace(Snapshot {
                issued_at: now - 1,
                keys: vec![],
            })
            .is_err()
    );
    let headers = http::HeaderMap::from_iter([(
        http::header::AUTHORIZATION,
        http::HeaderValue::from_str(&format!("Bearer {KEY}")).unwrap(),
    )]);
    assert!(server.state.credentials.tenant(&headers).is_ok());
}
