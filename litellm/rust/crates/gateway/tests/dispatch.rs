use std::{sync::Arc, time::Duration};

use litellm_config::Config;
use litellm_gateway_forward::{Settings, Upstream};
use litellm_http::ClientVariant;
use rstest::rstest;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

const PYTHON: &str = "answered by python";
const MASTER_KEY: &str = "Bearer gateway-key";
const JWT: &str = "Bearer header.claims.signature";

async fn python() -> Upstream {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, axum::Router::new().fallback(|| async { PYTHON }))
            .await
            .unwrap();
    });
    format!("http://{address}").parse().unwrap()
}

async fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[rstest]
#[case::master_key(&[("authorization", MASTER_KEY)], "", "", false)]
#[case::wrong_key(&[("authorization", "Bearer wrong")], "", "", true)]
#[case::no_key(&[], "", "", true)]
#[case::body(&[("authorization", MASTER_KEY)], "{}", "", true)]
#[case::key_python_reads_first(&[("authorization", MASTER_KEY), ("x-litellm-api-key", "sk-other")], "", "", true)]
#[case::jwt(&[("authorization", JWT)], "", "enable_jwt_auth: true", true)]
#[case::oauth2_proxy(&[("authorization", MASTER_KEY)], "", "enable_oauth2_proxy_auth: true", true)]
#[case::custom_auth(&[("authorization", MASTER_KEY)], "", "custom_auth: my_module.auth", true)]
#[tokio::test]
async fn serves_only_callers_it_authenticates_and_forwards_the_rest(
    #[case] headers: &[(&'static str, &'static str)],
    #[case] body: &'static str,
    #[case] setting: &str,
    #[case] forwarded: bool,
) {
    let config = Config::from_yaml(&format!(
        "model_list: []\ngeneral_settings:\n  master_key: gateway-key\n  {setting}\n"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let auth = litellm_gateway::auth(&config, inference.secrets.as_ref(), &|_| None)
        .await
        .unwrap();
    let port = free_port().await;
    let settings = Settings {
        upstream: python().await,
        host: "127.0.0.1".into(),
        port,
        upstream_keepalive: Duration::from_secs(5),
        shutdown_drain: Duration::ZERO,
        config: None,
    };
    let shutdown = CancellationToken::new();
    let gateway = Arc::clone(&inference);
    tokio::spawn(litellm_gateway_forward::run_app(
        settings,
        move |forward| {
            litellm_gateway::dispatching_router(gateway, &config, &|_| None, auth, forward)
        },
        shutdown.clone().cancelled_owned(),
    ));
    let client = inference
        .resources
        .pool
        .client(&inference.http, ClientVariant::Provider)
        .unwrap();
    let send = || {
        headers
            .iter()
            .fold(
                client.post(format!("http://127.0.0.1:{port}/v1/messages")),
                |request, (name, value)| request.header(*name, *value),
            )
            .header("content-type", "application/json")
            .body(body)
            .send()
    };
    let response = loop {
        match send().await {
            Ok(response) => break response,
            Err(_) => tokio::task::yield_now().await,
        }
    };

    assert_eq!(response.text().await.unwrap() == PYTHON, forwarded);
    shutdown.cancel();
}
