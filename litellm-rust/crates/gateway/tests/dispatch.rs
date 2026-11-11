use std::{sync::Arc, time::Duration};

use litellm_config::Config;
use litellm_gateway_forward::{Settings, Upstream};
use litellm_http::ClientVariant;
use rstest::rstest;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

const PYTHON: &str = "answered by python";

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
#[case::master_key("Bearer gateway-key", "", false)]
#[case::rejected_key("Bearer wrong", "", true)]
#[case::body("Bearer gateway-key", "{}", true)]
#[tokio::test]
async fn the_gate_serves_only_callers_it_authenticates_and_forwards_the_rest(
    #[case] authorization: &str,
    #[case] body: &'static str,
    #[case] forwarded: bool,
) {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
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
    };
    let shutdown = CancellationToken::new();
    let gateway = Arc::clone(&inference);
    tokio::spawn(litellm_gateway_forward::run_app(
        settings,
        move |forward| {
            litellm_gateway::dispatching_router(gateway, None, &config, &|_| None, auth, forward)
        },
        shutdown.clone().cancelled_owned(),
    ));
    let client = inference
        .resources
        .pool
        .client(&inference.http, ClientVariant::Provider)
        .unwrap();
    let send = || {
        client
            .post(format!("http://127.0.0.1:{port}/v1/messages"))
            .header("authorization", authorization)
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
