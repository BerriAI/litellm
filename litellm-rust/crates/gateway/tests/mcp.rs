use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use litellm_config::Config;
use litellm_gateway_mcp::{
    HttpConfig,
    rmcp::{
        ErrorData, RoleServer, ServerHandler,
        model::*,
        service::RequestContext,
        transport::streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
        },
    },
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

#[derive(Clone)]
struct Upstream;

impl ServerHandler for Upstream {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(
            ["echo", "hidden"]
                .into_iter()
                .map(|name| {
                    Tool::new(
                        name,
                        name,
                        Arc::new(serde_json::from_value(json!({"type":"object"})).unwrap()),
                    )
                })
                .collect(),
        ))
    }

    async fn call_tool(
        &self,
        params: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let headers = &context
            .extensions
            .get::<axum::http::request::Parts>()
            .unwrap()
            .headers;
        let result = json!({"arguments": params.arguments, "authorization": headers.get("authorization").unwrap().to_str().unwrap(), "static": headers.get("x-static").unwrap().to_str().unwrap()});
        Ok(CallToolResult::success(vec![ContentBlock::text(result.to_string())]).into())
    }
}

struct Fixture {
    url: String,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

#[fixture]
async fn upstream() -> Fixture {
    let shutdown = CancellationToken::new();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let transport = StreamableHttpService::new(
        || Ok(Upstream),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default()
            .with_allowed_hosts(["127.0.0.1", "upstream.invalid"])
            .with_cancellation_token(shutdown.clone()),
    );
    let app = Router::new().route_service("/mcp", transport);
    let stop = shutdown.clone();
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(stop.cancelled_owned())
            .await
            .unwrap();
    });
    Fixture {
        url,
        shutdown,
        task,
    }
}

struct Served {
    app: Router,
    mcp: litellm_gateway_mcp::ConfiguredGateway,
    shutdown: CancellationToken,
    upstream: Fixture,
}

impl Served {
    async fn close(self) {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.mcp.close())
            .await
            .unwrap();
        self.upstream.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.upstream.task)
            .await
            .unwrap()
            .unwrap();
    }
}

#[fixture]
async fn served(#[future] upstream: Fixture) -> Served {
    let upstream = upstream.await;
    let config = Config::from_yaml(&format!(
        r#"
environment_variables:
  GATEWAY_KEY: admission-secret
  UPSTREAM_KEY: upstream-secret
  STATIC_VALUE: static-secret
general_settings:
  master_key: os.environ/GATEWAY_KEY
mcp_servers:
  docs:
    server_id: configured-id
    alias: knowledge
    transport: http
    url: {}
    auth_type: bearer_token
    authentication_token: os.environ/UPSTREAM_KEY
    static_headers:
      x-static: os.environ/STATIC_VALUE
    allowed_tools: [echo]
    timeout: 2
    max_concurrent_requests: 1
"#,
        upstream.url
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let shutdown = CancellationToken::new();
    let mcp = litellm_gateway::build_mcp(
        &config,
        inference.secrets.clone(),
        shutdown.clone(),
        &inference.resources.pool,
        &inference.http,
    )
    .await
    .unwrap()
    .unwrap();
    let mcp_router = litellm_gateway_mcp::router(
        mcp.operations.clone(),
        HttpConfig {
            cancellation_token: shutdown.clone(),
            ..Default::default()
        },
    );
    let app = litellm_gateway::router(inference, &config, None, Some(mcp_router));
    Served {
        app,
        mcp,
        shutdown,
        upstream,
    }
}

#[rstest]
#[case::streamable_post("/mcp", "POST")]
#[case::streamable_get("/mcp", "GET")]
#[case::streamable_delete("/mcp", "DELETE")]
#[case::legacy_sse("/mcp/sse", "GET")]
#[case::legacy_messages("/mcp/sse/messages", "POST")]
#[case::rest_list("/mcp-rest/tools/list", "GET")]
#[case::enabled("/mcp/enabled", "GET")]
#[tokio::test]
async fn every_mcp_route_requires_admission_credentials(
    #[future] served: Served,
    #[case] path: &str,
    #[case] method: &str,
) {
    let served = served.await;
    let response = served
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("host", "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    served.close().await;
}

#[rstest]
#[tokio::test]
async fn serves_configured_upstream_with_separate_admission_and_outbound_credentials(
    #[future] served: Served,
) {
    let served = served.await;
    let app = served.app.clone();
    let list = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/mcp-rest/tools/list?server_id=configured-id")
                .header("authorization", "Bearer admission-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list.status(), 200);
    let body: Value =
        serde_json::from_slice(&to_bytes(list.into_body(), 65536).await.unwrap()).unwrap();
    assert_eq!(body["tools"].as_array().unwrap().len(), 1);
    assert_eq!(body["tools"][0]["name"], "echo");
    assert_eq!(body["tools"][0]["mcp_info"]["alias"], "knowledge");

    let response = app.clone().oneshot(Request::post("/mcp-rest/tools/call").header("authorization", "Bearer admission-secret").header("content-type", "application/json").body(Body::from(json!({"server_id":"configured-id", "name":"echo", "arguments":{"value":"hello"}}).to_string())).unwrap()).await.unwrap();
    assert_eq!(response.status(), 200);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
    let result: Value = serde_json::from_str(body["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(
        result,
        json!({"arguments":{"value":"hello"}, "authorization":"Bearer upstream-secret", "static":"static-secret"})
    );
    let blocked = app
        .oneshot(
            Request::post("/mcp-rest/tools/call")
                .header("authorization", "Bearer admission-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"server_id":"configured-id", "name":"hidden"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(blocked.status(), 403);

    served.close().await;
}

#[rstest]
#[tokio::test]
async fn mcp_tools_are_discovered_through_the_configured_proxy(#[future] upstream: Fixture) {
    let upstream = upstream.await;
    let config = Config::from_yaml(
        "general_settings: {master_key: gateway-key}\nmcp_servers: {docs: {url: 'http://upstream.invalid/mcp', timeout: 2}}",
    )
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let proxy = upstream.url.trim_end_matches("/mcp");
    let http = litellm_http::HttpClientConfig {
        proxies: litellm_http::EnvironmentProxies::from_environment(&|name: &str| {
            (name == "HTTP_PROXY").then(|| proxy.to_owned())
        }),
        ..inference.http.clone()
    };
    let shutdown = CancellationToken::new();
    let mcp = litellm_gateway::build_mcp(
        &config,
        inference.secrets.clone(),
        shutdown.clone(),
        &inference.resources.pool,
        &http,
    )
    .await
    .unwrap()
    .unwrap();
    let app = litellm_gateway_mcp::router(mcp.operations.clone(), HttpConfig::default());
    let response = app
        .oneshot(
            Request::get("/mcp-rest/tools/list?server_id=docs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
    assert_eq!(body["tools"][0]["name"], "echo");
    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(5), mcp.close())
        .await
        .unwrap();
    upstream.shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(5), upstream.task)
        .await
        .unwrap()
        .unwrap();
}

#[rstest]
#[rstest]
#[tokio::test]
async fn mcp_uses_the_configured_ca_bundle() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing-ca.pem");
    let config = Config::from_yaml(
        "general_settings: {master_key: gateway-key}\nmcp_servers: {docs: {url: 'http://127.0.0.1:1/mcp'}}",
    )
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let http = litellm_http::HttpClientConfig {
        verify: litellm_http::Verify::CaBundle(missing.clone()),
        ..inference.http.clone()
    };
    let error = litellm_gateway::build_mcp(
        &config,
        inference.secrets.clone(),
        CancellationToken::new(),
        &inference.resources.pool,
        &http,
    )
    .await
    .err()
    .unwrap();
    let litellm_gateway::Error::Http(error) = error else {
        panic!("expected an HTTP configuration error");
    };
    assert_eq!(error.tls_source(), Some(litellm_http::TlsSource::CaBundle));
    assert_eq!(
        error.to_string(),
        format!("could not read {}", missing.display())
    );
    assert_eq!(
        litellm_error::find::<std::io::Error>(&error)
            .unwrap()
            .kind(),
        std::io::ErrorKind::NotFound,
    );
}

#[rstest]
#[case::unsupported_auth("auth_type: oauth2", "auth mode")]
#[case::unsupported_policy("allowed_params: {echo: [safe]}", "allowed_params")]
#[case::unsupported_transport("transport: sse", "legacy SSE")]
#[case::zero_timeout("timeout: 0", "timeout")]
#[case::negative_timeout("timeout: -1", "timeout")]
#[case::zero_concurrency("max_concurrent_requests: 0", "max_concurrent_requests")]
#[case::protocol_override("mcp_info: {protocol_version: auto}", "protocol and cost")]
#[case::costs("mcp_info: {mcp_server_cost_info: {default: 1}}", "protocol and cost")]
#[case::missing_token("auth_type: api_key", "authentication_token")]
#[case::missing_env(
    "auth_type: api_key\n    authentication_token: os.environ/LITELLM_TEST_NONEXISTENT_MCP_SECRET",
    "referenced secret"
)]
#[case::reserved_header("static_headers: {host: secret-value}", "header")]
#[tokio::test]
async fn rejects_unsupported_or_invalid_settings_before_serving(
    #[case] settings: &str,
    #[case] expected: &str,
) {
    let config = Config::from_yaml(&format!("general_settings: {{master_key: gateway-key}}\nmcp_servers:\n  docs:\n    url: http://127.0.0.1:1/mcp\n    {settings}\n")).unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let result = litellm_gateway::build_mcp(
        &config,
        Arc::new(NoSecrets),
        CancellationToken::new(),
        &inference.resources.pool,
        &inference.http,
    )
    .await;
    let error = result.err().expect("startup must fail");
    let report = litellm_error::Report::new(&error).to_string();
    assert!(report.contains(expected), "{report}");
    assert!(!report.contains("secret-value"));
}

#[rstest]
#[tokio::test]
async fn mcp_is_optional_and_requires_admission_credentials_when_enabled() {
    let empty = Config::from_yaml("{}").unwrap();
    let inference = litellm_gateway::build_inference(&empty).unwrap();
    assert!(
        litellm_gateway::build_mcp(
            &empty,
            inference.secrets.clone(),
            CancellationToken::new(),
            &inference.resources.pool,
            &inference.http
        )
        .await
        .unwrap()
        .is_none()
    );
    let enabled =
        Config::from_yaml("mcp_servers: {docs: {url: 'http://127.0.0.1:1/mcp'}}").unwrap();
    let error = litellm_gateway::build_mcp(
        &enabled,
        inference.secrets.clone(),
        CancellationToken::new(),
        &inference.resources.pool,
        &inference.http,
    )
    .await
    .err()
    .unwrap();
    assert!(matches!(error, litellm_gateway::Error::Auth(_)));
}

struct NoSecrets;
impl litellm_secrets::source::SecretSource for NoSecrets {
    fn get_secret_str<'a>(
        &'a self,
        _: &'a str,
    ) -> futures_util::future::BoxFuture<
        'a,
        Result<Option<litellm_secret_value::SecretValue>, litellm_secrets::Error>,
    > {
        Box::pin(async { Ok(None) })
    }
}

#[rstest]
fn applies_configured_http_environment_before_building_clients() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing-ca.pem");
    let config = Config::from_yaml(&format!(
        "environment_variables: {{SSL_VERIFY: '{}'}}\nlitellm_settings: {{ssl_verify: false}}",
        missing.display(),
    ))
    .unwrap();
    let error = litellm_gateway::build_inference(&config).err().unwrap();
    assert!(matches!(
        &error,
        litellm_gateway::Error::Http(http)
            if http.tls_source() == Some(litellm_http::TlsSource::CaBundle)
                && http.to_string().contains(&missing.display().to_string())
    ));
}

#[rstest]
#[case::client_policy("general_settings: {master_key: key, mcp_allowed_clients: [approved]}")]
#[case::guardrail("guardrails: [{guardrail_name: policy}]")]
#[case::local_tools("mcp_tools: [{name: custom}]")]
#[tokio::test]
async fn refuses_unimplemented_global_mcp_policy(#[case] settings: &str) {
    let config = Config::from_yaml(&format!(
        "{settings}\nmcp_servers: {{docs: {{url: 'http://127.0.0.1:1/mcp'}}}}\n"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let error = litellm_gateway::build_mcp(
        &config,
        Arc::new(NoSecrets),
        CancellationToken::new(),
        &inference.resources.pool,
        &inference.http,
    )
    .await
    .err()
    .unwrap();
    assert!(matches!(
        error,
        litellm_gateway::Error::McpSetting(_) | litellm_gateway::Error::McpGuardrails
    ));
}
