use std::net::SocketAddr;

use axum::Router;
use axum::http::StatusCode;
use axum::routing::get;
use litellm_gateway_auth::{MasterKeyState, RequireMasterKey, hash_token};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

#[derive(Clone)]
struct State(Option<&'static str>);

impl MasterKeyState for State {
    fn master_key(&self) -> Option<&str> {
        self.0
    }
}

async fn protected(_: RequireMasterKey) -> StatusCode {
    StatusCode::OK
}

async fn spawn_server(master_key: Option<&'static str>) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let app = Router::new()
        .route("/protected", get(protected))
        .with_state(State(master_key));
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("server");
    });
    (address, server)
}

#[test]
fn hash_token_matches_python_sha256_hexdigest() {
    assert_eq!(
        hash_token("sk-1234"),
        "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
    );
    let hash = hash_token("sk-secret");
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|character| character.is_ascii_hexdigit()));
    assert_ne!(hash, "sk-secret");
}

#[tokio::test]
async fn extractor_enforces_auth_over_http() {
    let (configured_address, configured_server) = spawn_server(Some("master-key")).await;
    let configured_url = format!("http://{configured_address}/protected");
    let client = reqwest::Client::new();

    let accepted = client
        .get(&configured_url)
        .bearer_auth("master-key")
        .send()
        .await
        .expect("accepted response");
    let invalid = client
        .get(&configured_url)
        .bearer_auth("wrong-key")
        .send()
        .await
        .expect("invalid response");
    let missing = client
        .get(&configured_url)
        .send()
        .await
        .expect("missing response");

    assert_eq!(accepted.status(), StatusCode::OK);
    assert_eq!(invalid.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
    configured_server.abort();

    let (unconfigured_address, unconfigured_server) = spawn_server(None).await;
    let unconfigured = client
        .get(format!("http://{unconfigured_address}/protected"))
        .bearer_auth("master-key")
        .send()
        .await
        .expect("unconfigured response");

    assert_eq!(unconfigured.status(), StatusCode::INTERNAL_SERVER_ERROR);
    unconfigured_server.abort();
}
