use std::{sync::Arc, time::Duration};

use litellm_config::Config;
use litellm_gateway_forward::{Forward, Settings, Upstream, VERDICT_HEADER};
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
#[case::master_key(&[("authorization", MASTER_KEY)], "", "", "native")]
#[case::wrong_key(&[("authorization", "Bearer wrong")], "", "", "python; reason=unauthenticated")]
#[case::no_key(&[], "", "", "python; reason=no_key")]
#[case::body(&[("authorization", MASTER_KEY)], "{}", "", "python; reason=body")]
#[case::litellm_header(&[("x-litellm-api-key", "gateway-key")], "", "", "native")]
#[case::provider_header(&[("x-api-key", "gateway-key")], "", "", "native")]
#[case::key_python_reads_first(&[("authorization", MASTER_KEY), ("x-litellm-api-key", "sk-other")], "", "", "python; reason=unauthenticated")]
#[case::jwt(&[("authorization", JWT)], "", "enable_jwt_auth: true", "python; reason=license")]
#[case::master_key_with_jwt_auth(&[("authorization", MASTER_KEY)], "", "enable_jwt_auth: true", "native")]
#[case::oauth2_introspection(&[("authorization", MASTER_KEY)], "", "enable_oauth2_auth: true", "python; reason=license")]
#[case::oauth2_proxy(&[("authorization", MASTER_KEY)], "", "enable_oauth2_proxy_auth: true", "python; reason=plan")]
#[case::unsupported_setting(&[("authorization", MASTER_KEY)], "", "custom_auth: my_module.auth", "python; reason=configuration")]
#[tokio::test]
async fn serves_only_requests_it_admits_and_names_who_answered_each(
    #[case] headers: &[(&'static str, &'static str)],
    #[case] body: &'static str,
    #[case] setting: &str,
    #[case] verdict: &str,
) {
    let config = Config::from_yaml(&format!(
        "model_list: []\ngeneral_settings:\n  master_key: gateway-key\n  {setting}\n"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).await.unwrap();
    let auth = litellm_gateway_auth::Auth::from_config(&config, inference.secrets.clone());
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
    let supported = litellm_gateway::unsupported(&config, &|_| None).is_empty();
    let admission = litellm_gateway::Admission::new(&config, auth);
    let native = litellm_gateway::admitted_router(Arc::clone(&inference));
    tokio::spawn(litellm_gateway_forward::run_app(
        settings,
        move |python: Forward| match supported {
            true => python.behind(native, move |request| admission.clone().admit(request)),
            false => python.without_native(),
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

    assert_eq!(response.headers()[VERDICT_HEADER], verdict);
    assert_eq!(
        response.text().await.unwrap() == PYTHON,
        verdict.starts_with("python")
    );
    shutdown.cancel();
}
