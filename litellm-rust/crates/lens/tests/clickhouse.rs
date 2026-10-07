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
    collections::BTreeMap,
    sync::{Arc, atomic::Ordering},
};

#[rstest]
#[case::own_trace("isolated-ingestion-key", vec![], true)]
#[case::own_span("isolated-ingestion-key", vec!["aabbccdd00112233"], true)]
#[case::missing_span("isolated-ingestion-key", vec!["ffffffffffffffff"], false)]
#[case::other_key("other-ingestion-key", vec![], false)]
#[tokio::test]
#[ignore = "requires an isolated ClickHouse instance in LENS_TEST_CLICKHOUSE_URL"]
async fn traces_round_trip_through_real_clickhouse_with_scoped_reads(
    #[case] key: &str,
    #[case] spans: Vec<&str>,
    #[case] expected: bool,
) {
    let url = std::env::var("LENS_TEST_CLICKHOUSE_URL").expect("set LENS_TEST_CLICKHOUSE_URL");
    let client = http_client().unwrap();
    let database = format!("lens_test_{}", uuid::Uuid::new_v4().simple());
    let config = Config::new(database.clone(), &url, 14, 65_536).unwrap();
    let storage = Storage::new(
        config.clone(),
        client.clone(),
        "isolated-test-internal-secret-32-bytes".into(),
    );
    storage.ensure_schema().await.unwrap();
    let state = Arc::new(State::new(
        storage,
        "isolated-test-internal-secret-32-bytes".into(),
    ));
    state.schema_ready.store(true, Ordering::Release);
    state
        .credentials
        .replace(Snapshot {
            issued_at: unix_seconds(),
            keys: ["isolated-ingestion-key", "other-ingestion-key"]
                .into_iter()
                .map(|key| Credential {
                    token_hash: format!("{:x}", Sha256::digest(key)),
                    tenant: Tenant {
                        team_id: "team-a".into(),
                        user_id: "user-a".into(),
                        api_key_hash: format!("{:x}", Sha256::digest(key)),
                        ..Tenant::default()
                    },
                    expires_at: None,
                })
                .collect(),
        })
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let service = tokio::spawn(async move {
        axum::serve(listener, router(state)).await.unwrap();
    });
    let now = unix_seconds() * 1_000_000_000;
    let trace_id = "aabbccdd00112233aabbccdd00112233";
    let payload = json!({"resourceSpans": [{"resource": {"attributes": [{"key":"service.name","value":{"stringValue":"isolated-agent"}}]},"scopeSpans":[{"spans":[{
        "traceId":trace_id,"spanId":"aabbccdd00112233","name":"Real storage validation",
        "startTimeUnixNano":now.to_string(),"endTimeUnixNano":(now+1_000_000).to_string(),
        "attributes":[{"key":"gen_ai.input.messages","value":{"stringValue":"[{\"role\":\"user\",\"content\":\"Count three apples\"}]"}}],
        "status":{"code":1}
    }]}]}]});
    let written = client
        .post(format!("{endpoint}/v1/traces"))
        .bearer_auth("isolated-ingestion-key")
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(written.status(), 200, "{}", written.text().await.unwrap());
    let receipt = client
        .post(format!("{endpoint}/v1/traces/receipt"))
        .bearer_auth(key)
        .json(&json!({"trace_id": trace_id, "span_ids": spans}))
        .send()
        .await
        .unwrap();
    assert_eq!(receipt.status(), 200);
    assert_eq!(
        receipt.json::<serde_json::Value>().await.unwrap(),
        json!({"received": expected})
    );
    let read = json!({"operation":"list","scope":{"all_teams":0,"user_id":"user-a","team_ids":[]},"start_ms":now/1_000_000-1000,"end_ms":now/1_000_000+1000,"cursor":null,"limit":50});
    let found = client
        .post(format!("{endpoint}/internal/read"))
        .bearer_auth("isolated-test-internal-secret-32-bytes")
        .json(&read)
        .send()
        .await
        .unwrap();
    assert_eq!(found.status(), 200, "{}", found.text().await.unwrap());
    let visible: serde_json::Value = found.json().await.unwrap();
    assert!(visible.to_string().contains(trace_id), "{visible}");
    let mut other = read.clone();
    other["scope"] = json!({"all_teams":0,"user_id":"different-user","team_ids":[]});
    let hidden: serde_json::Value = client
        .post(format!("{endpoint}/internal/read"))
        .bearer_auth("isolated-test-internal-secret-32-bytes")
        .json(&other)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!hidden.to_string().contains(trace_id), "{hidden}");
    let count = litellm_storage_clickhouse::execute_read(
        &client,
        config.storage().reader(),
        "SELECT count() AS count FROM otel_traces",
        &BTreeMap::new(),
    )
    .await
    .unwrap();
    assert!(count.contains('1'), "{count}");
    service.abort();
    litellm_storage_clickhouse::execute_statement(
        &client,
        config.storage().writer(),
        &format!("DROP DATABASE {database}"),
        std::time::Duration::from_secs(10),
    )
    .await
    .unwrap();
}
