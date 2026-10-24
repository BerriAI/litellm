use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    Router, body::Body, extract::State, http::StatusCode, response::Response, routing::get,
};
use litellm_config::Config;
use litellm_gateway_model_catalog::{Source, bundled};
use litellm_http::{Client, ClientVariant};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::{net::TcpListener, sync::Notify, task::JoinHandle, time::timeout};
use tokio_util::sync::CancellationToken;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[derive(Clone)]
struct MapGate {
    started: Arc<Notify>,
    release: Arc<Notify>,
    body: bytes::Bytes,
}

async fn hold_map(State(gate): State<MapGate>) -> Response<Body> {
    gate.started.notify_one();
    gate.release.notified().await;
    Response::builder()
        .status(StatusCode::OK)
        .header("etag", "\"remote-map\"")
        .body(Body::from(gate.body))
        .unwrap()
}

async fn serve(router: Router) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address: SocketAddr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{address}"), task)
}

async fn source(client: &Client, origin: &str) -> Value {
    let response = client
        .get(format!("{origin}/model/cost_map/source"))
        .header("authorization", "Bearer gateway-key")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_slice(&response.bytes().await.unwrap()).unwrap()
}

#[rstest]
#[tokio::test]
async fn held_catalog_fetch_does_not_block_provider_or_source_routes() {
    let provider = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&provider)
        .await;

    let mut map: Value = serde_json::from_slice(litellm_model_catalog::BUNDLED_REGISTRY).unwrap();
    map.as_object_mut().unwrap().insert(
        "__invalid_cost_map_row".to_owned(),
        json!({"litellm_provider": "openai", "input_cost_per_token": -1}),
    );
    let gate = MapGate {
        started: Arc::new(Notify::new()),
        release: Arc::new(Notify::new()),
        body: bytes::Bytes::from(serde_json::to_vec(&map).unwrap()),
    };
    let (map_origin, map_server) = serve(
        Router::new()
            .route("/", get(hold_map))
            .with_state(gate.clone()),
    )
    .await;
    let map_url = format!("{map_origin}/");
    let config = Config::from_yaml(&format!(
        r#"
model_list:
  - model_name: test-model
    litellm_params:
      model: openai_like/gpt-4o-mini
      api_base: {provider_url}/v1
      api_key: test-key
general_settings:
  master_key: gateway-key
environment_variables:
  LITELLM_MODEL_COST_MAP_URL: {map_url}
  LITELLM_LOCAL_MODEL_COST_MAP: 'false'
  LITELLM_OFFLINE_MODE: 'false'
litellm_settings:
  model_cost_map_url: https://config.example.test/prices.json
"#,
        provider_url = provider.uri(),
        map_url = map_url,
    ))
    .unwrap();
    let settings = litellm_gateway::cost_map::settings(&config).unwrap();
    assert_eq!(settings.remote_url(), Some(map_url.as_str()));
    let (writer, mut reader) = bundled(&settings, time::OffsetDateTime::now_utc());
    let inference = litellm_gateway::build_inference(&config, reader.clone()).unwrap();
    let client = inference
        .resources
        .pool
        .client(&inference.http, ClientVariant::Provider)
        .unwrap();
    let shutdown = CancellationToken::new();
    let revalidation =
        litellm_gateway::cost_map::spawn(&settings, writer, &inference, shutdown.clone())
            .unwrap()
            .unwrap();
    timeout(Duration::from_secs(5), gate.started.notified())
        .await
        .unwrap();

    let (origin, gateway_server) = serve(litellm_gateway::router(
        inference.clone(),
        &config,
        None,
        None,
    ))
    .await;
    let source_before = source(&client, &origin).await;
    assert_eq!(source_before["source"], "local");
    assert_eq!(
        source_before["fallback_reason"],
        "remote revalidation pending"
    );

    let completion = client
        .post(format!("{origin}/v1/chat/completions"))
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}]
            })
            .to_string(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(completion.status(), StatusCode::OK);
    let completion_body = completion.bytes().await.unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&completion_body).unwrap()["choices"][0]["message"]["content"],
        "ok"
    );

    gate.release.notify_one();
    timeout(Duration::from_secs(10), reader.changed())
        .await
        .unwrap()
        .unwrap();
    revalidation.await.unwrap();
    let served = reader.current();
    assert_eq!(served.provenance.source, Source::Remote);
    assert_eq!(served.provenance.fallback_reason, None);
    assert_eq!(served.installed.etag.as_deref(), Some("\"remote-map\""));
    let source_after = source(&client, &origin).await;
    assert_eq!(source_after["source"], "remote");
    assert_eq!(source_after["fallback_reason"], Value::Null);
    assert_eq!(source_after["etag"], "\"remote-map\"");
    assert_eq!(
        source_after["model_count"].as_u64().unwrap() as usize,
        served.installed.catalog.served().count()
    );
    assert_eq!(
        source_after["source_revision"],
        litellm_model_catalog::git_blob_id(&gate.body)
    );
    assert!(
        !served
            .installed
            .catalog
            .model_names()
            .any(|name| name == "__invalid_cost_map_row")
    );
    assert_eq!(
        served.installed.catalog.served().count(),
        source_before["model_count"].as_u64().unwrap() as usize
    );

    shutdown.cancel();
    map_server.abort();
    gateway_server.abort();
}

#[rstest]
#[case::local_only("true", "false")]
#[case::offline("false", "true")]
#[tokio::test]
async fn disabled_modes_skip_remote_fetch(#[case] local_only: &str, #[case] offline: &str) {
    let map = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/registry"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&map)
        .await;
    let config = Config::from_yaml(&format!(
        "general_settings: {{master_key: gateway-key}}\nenvironment_variables: {{LITELLM_MODEL_COST_MAP_URL: {}/registry, LITELLM_LOCAL_MODEL_COST_MAP: '{}', LITELLM_OFFLINE_MODE: '{}'}}",
        map.uri(),
        local_only,
        offline,
    ))
    .unwrap();
    let settings = litellm_gateway::cost_map::settings(&config).unwrap();
    let (writer, reader) = bundled(&settings, time::OffsetDateTime::now_utc());
    let inference = litellm_gateway::build_inference(&config, reader).unwrap();
    assert!(
        litellm_gateway::cost_map::spawn(&settings, writer, &inference, CancellationToken::new())
            .unwrap()
            .is_none()
    );
    let client = inference
        .resources
        .pool
        .client(&inference.http, ClientVariant::Provider)
        .unwrap();
    let (origin, server) = serve(litellm_gateway::router(inference, &config, None, None)).await;
    let source = source(&client, &origin).await;
    assert_eq!(source["source"], "local");
    assert_eq!(source["is_env_forced"], true);
    assert_eq!(source["fallback_reason"], Value::Null);
    map.verify().await;
    server.abort();
}
