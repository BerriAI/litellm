use super::support;

use super::support::{Harness, harness};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use litellm_gateway_mcp::{
    Context, Error, GatewayFuture, HttpConfig, NativeGateway, Registry, Server, ServerResolver,
    router,
};
use rstest::rstest;
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

#[rstest]
#[case::aggregate("/mcp", 4)]
#[case::trailing_slash("/mcp/", 4)]
#[case::scope_suffix("/alpha/mcp", 2)]
#[case::scope_prefix("/mcp/alpha", 2)]
#[case::server_id("/id-alpha/mcp", 2)]
#[tokio::test]
async fn streamable_http_lists_scoped_tools(
    #[future] harness: Harness,
    #[case] path: &str,
    #[case] count: usize,
) {
    let harness = harness.await;
    let response = harness
        .app
        .oneshot(
            Request::post(path)
                .header("host", "localhost")
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .header("mcp-protocol-version", "2025-11-25")
                .body(Body::from(
                    json!({"jsonrpc":"2.0","id":42,"method":"tools/list"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 65536).await.unwrap();
    let wire = String::from_utf8(body.to_vec()).unwrap();
    let data = wire
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    let result: Value = serde_json::from_str(data).unwrap();
    assert_eq!(result["id"], 42);
    assert_eq!(result["result"]["tools"].as_array().unwrap().len(), count);
}

#[rstest]
#[tokio::test]
async fn rest_lists_bare_names_and_calls_selected_server(#[future] harness: Harness) {
    let harness = harness.await;
    let listing = harness
        .app
        .clone()
        .oneshot(
            Request::get("/mcp-rest/tools/list?server_id=id-alpha")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(listing.status(), StatusCode::OK);
    let listing: Value =
        serde_json::from_slice(&to_bytes(listing.into_body(), 65536).await.unwrap()).unwrap();
    assert_eq!(listing["tools"][0]["name"], "echo-value");
    assert_eq!(listing["tools"][0]["mcp_info"]["server_id"], "id-alpha");
    assert_eq!(listing["tools"].as_array().unwrap().len(), 2);
    let response = harness
        .app
        .oneshot(
            Request::post("/mcp-rest/tools/call")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"server_id":"id-alpha", "name":"echo-value", "arguments":{"foo":"bar"}})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
    let forwarded: Value =
        serde_json::from_str(response["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(forwarded["arguments"], json!({"foo":"bar"}));
}

#[rstest]
#[case::foreign_origin("localhost", Some("https://attacker.example"))]
#[case::dns_rebinding("attacker.example", None)]
#[tokio::test]
async fn rejects_untrusted_browser_origins_and_hosts(
    #[future] harness: Harness,
    #[case] host: &str,
    #[case] origin: Option<&str>,
) {
    let harness = harness.await;
    let request = Request::post("/mcp")
        .header("host", host)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream");
    let request = match origin {
        Some(origin) => request.header("origin", origin),
        None => request,
    };
    let response = harness
        .app
        .oneshot(
            request
                .body(Body::from(
                    json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

struct PerRequestResolver(Arc<[Server]>);

impl ServerResolver for PerRequestResolver {
    fn resolve<'a>(&'a self, context: &'a Context) -> GatewayFuture<'a, Arc<[Server]>> {
        Box::pin(async move {
            match context
                .parts
                .headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
            {
                Some("Bearer allowed") => Ok(self.0.clone()),
                _ => Err(Error::Forbidden),
            }
        })
    }
}

#[rstest]
#[case::mcp("/mcp", "POST")]
#[case::rest("/mcp-rest/tools/list", "GET")]
#[tokio::test]
async fn resolver_receives_each_requests_identity(
    #[future] harness: Harness,
    #[case] path: &str,
    #[case] method: &str,
) {
    let harness = harness.await;
    let server = Server {
        info: litellm_gateway_mcp::ServerInfo {
            server_id: "id-alpha".into(),
            server_name: "alpha".into(),
            alias: None,
        },
        peer: harness.connections[0].peer().clone(),
        allowed_tools: None,
    };
    let registry = Registry::new(vec![server]).unwrap();
    let servers = registry.resolve(&support::context(None)).await.unwrap();
    let operations = Arc::new(NativeGateway::new(
        Arc::new(PerRequestResolver(servers)),
        Duration::from_secs(2),
    ));
    let app = router(operations, HttpConfig::default());
    for token in ["allowed", "denied", "allowed"] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("host", "localhost")
                    .header("authorization", format!("Bearer {token}"))
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .header("mcp-protocol-version", "2025-11-25")
                    .body(Body::from(
                        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = String::from_utf8(
            to_bytes(response.into_body(), 65536)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        if token == "denied" {
            if method == "GET" {
                assert_eq!(status, StatusCode::FORBIDDEN);
            }
            assert!(body.contains("not allowed"), "{body}");
        } else {
            assert_eq!(status, StatusCode::OK);
            assert!(body.contains("echo-value"), "{body}");
        }
    }
}

fn rpc_request(path: &str, token: &str, session: Option<&str>, body: Value) -> Request<Body> {
    let request = Request::post(path)
        .header("host", "localhost")
        .header("authorization", format!("Bearer {token}"))
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .header("mcp-protocol-version", "2025-11-25");
    let request = match session {
        Some(session) => request.header("mcp-session-id", session),
        None => request,
    };
    request.body(Body::from(body.to_string())).unwrap()
}

#[rstest]
#[case::different_owner("other", "/alpha/mcp", false)]
#[case::forged_alternate_key("other", "/alpha/mcp", true)]
#[case::different_scope("owner", "/alpha-beta/mcp", false)]
#[tokio::test]
async fn binds_sessions_to_owner_and_scope(
    #[future] harness: Harness,
    #[case] token: &str,
    #[case] path: &str,
    #[case] forged: bool,
) {
    let harness = harness.await;
    let response = harness.app.clone().oneshot(rpc_request("/alpha/mcp", "owner", None, json!({
        "jsonrpc":"2.0", "id":1, "method":"initialize", "params":{
            "protocolVersion":"2025-11-25", "capabilities":{}, "clientInfo":{"name":"fixture","version":"1"}
        }
    }))).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let session = response.headers()["mcp-session-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let initialized = harness
        .app
        .clone()
        .oneshot(rpc_request(
            "/alpha/mcp",
            "owner",
            Some(&session),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        ))
        .await
        .unwrap();
    assert_eq!(initialized.status(), StatusCode::ACCEPTED);
    let mut denied_request = rpc_request(
        path,
        token,
        Some(&session),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
    );
    if forged {
        denied_request
            .headers_mut()
            .insert("x-litellm-api-key", "Bearer owner".parse().unwrap());
    }
    let denied = harness.app.clone().oneshot(denied_request).await.unwrap();
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    let allowed = harness
        .app
        .clone()
        .oneshot(rpc_request(
            "/alpha/mcp",
            "owner",
            Some(&session),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/list"}),
        ))
        .await
        .unwrap();
    assert_eq!(allowed.status(), StatusCode::OK);
    let body =
        String::from_utf8(to_bytes(allowed.into_body(), 65536).await.unwrap().to_vec()).unwrap();
    assert!(body.contains("alpha-echo-value"), "{body}");
    let delete = Request::delete("/alpha/mcp")
        .header("host", "localhost")
        .header("authorization", "Bearer owner")
        .header("mcp-session-id", &session)
        .body(Body::empty())
        .unwrap();
    assert!(
        harness
            .app
            .clone()
            .oneshot(delete)
            .await
            .unwrap()
            .status()
            .is_success()
    );
    let expired = harness
        .app
        .oneshot(rpc_request(
            "/alpha/mcp",
            "owner",
            Some(&session),
            json!({"jsonrpc":"2.0","id":4,"method":"tools/list"}),
        ))
        .await
        .unwrap();
    assert_eq!(expired.status(), StatusCode::NOT_FOUND);
}

#[rstest]
#[case::narrowed("/mcp", "alpha", false)]
#[case::multiple("/mcp", "alpha,alpha-beta", false)]
#[case::path_broadening("/alpha/mcp", "alpha-beta", true)]
#[case::unknown("/mcp", "unknown", true)]
#[tokio::test]
async fn header_scope_cannot_broaden_path_scope(
    #[future] harness: Harness,
    #[case] path: &str,
    #[case] scope: &str,
    #[case] denied: bool,
) {
    let harness = harness.await;
    let mut request = rpc_request(
        path,
        "owner",
        None,
        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
    );
    request
        .headers_mut()
        .insert("x-mcp-servers", scope.parse().unwrap());
    let response = harness.app.oneshot(request).await.unwrap();
    let body = String::from_utf8(
        to_bytes(response.into_body(), 65536)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert_eq!(body.contains("not allowed"), denied, "{body}");
    if !denied {
        assert!(body.contains("alpha-echo-value"));
    }
}

#[rstest]
#[case::missing_server(json!({"name":"echo-value"}))]
#[case::missing_name(json!({"server_id":"alpha"}))]
#[tokio::test]
async fn rest_requires_server_and_tool(#[future] harness: Harness, #[case] body: Value) {
    let harness = harness.await;
    let response = harness
        .app
        .oneshot(
            Request::post("/mcp-rest/tools/call")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[rstest]
#[tokio::test]
async fn legacy_sse_initializes_calls_tools_and_rejects_other_owners(#[future] harness: Harness) {
    use futures_util::StreamExt;
    let harness = harness.await;
    let response = harness
        .app
        .clone()
        .oneshot(
            Request::get("/mcp/sse")
                .header("host", "localhost")
                .header("authorization", "Bearer owner")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut events = sse_stream::SseStream::new(response.into_body());
    let endpoint = events.next().await.unwrap().unwrap();
    assert_eq!(endpoint.event.as_deref(), Some("endpoint"));
    let path = endpoint.data.unwrap();
    let response = harness
        .app
        .clone()
        .oneshot(rpc_request(
            &path,
            "other",
            None,
            json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let response = harness.app.clone().oneshot(rpc_request(&path,"owner",None,json!({
        "jsonrpc":"2.0", "id":1, "method":"initialize", "params":{
            "protocolVersion":"2025-11-25", "capabilities":{}, "clientInfo":{"name":"fixture","version":"1"}
        }
    }))).await.unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let initialized = events.next().await.unwrap().unwrap();
    let initialized: Value = serde_json::from_str(&initialized.data.unwrap()).unwrap();
    assert_eq!(
        initialized["result"]["serverInfo"]["name"],
        "litellm-mcp-server"
    );
    let response = harness
        .app
        .clone()
        .oneshot(rpc_request(
            &path,
            "owner",
            None,
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let response = harness.app.clone().oneshot(rpc_request(&path,"owner",None,json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"alpha-echo-value","arguments":{"source":"sse"}}}))).await.unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let called = tokio::time::timeout(Duration::from_secs(2), events.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let called: Value = serde_json::from_str(&called.data.unwrap()).unwrap();
    let forwarded: Value =
        serde_json::from_str(called["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(forwarded["arguments"], json!({"source":"sse"}));
    drop(events);
}
