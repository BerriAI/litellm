use std::{sync::Arc, time::Duration};

use litellm_config::Config;
use litellm_gateway_inference::{Error, Gateway};
use litellm_http::ClientVariant;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};

#[fixture]
fn inference() -> Arc<Gateway> {
    litellm_gateway::build_inference(&Config::from_yaml("model_list: []").unwrap()).unwrap()
}

#[rstest]
#[case::authorized("/v1/messages", Some("Bearer gateway-key"), Some("gateway-key"), 400)]
#[case::missing_token("/v1/messages", None, Some("gateway-key"), 401)]
#[case::wrong_token("/v1/messages", Some("Bearer wrong"), Some("gateway-key"), 401)]
#[case::ocr("/ocr", None, Some("gateway-key"), 401)]
#[case::chat("/v1/chat/completions", None, Some("gateway-key"), 401)]
#[case::deployment(
    "/openai/deployments/model/chat/completions",
    None,
    Some("gateway-key"),
    401
)]
#[case::transcription("/audio/transcriptions", None, Some("gateway-key"), 401)]
#[case::unsupported_route("/responses", None, Some("gateway-key"), 401)]
#[case::unknown_path("/unknown", None, Some("gateway-key"), 404)]
#[case::unknown_path_unconfigured("/unknown", None, None, 404)]
#[case::unconfigured("/v1/messages", Some("Bearer gateway-key"), None, 500)]
#[tokio::test]
async fn authenticates_before_serving_mounted_inference_routes(
    inference: Arc<Gateway>,
    #[case] path: &str,
    #[case] authorization: Option<&str>,
    #[case] master_key: Option<&str>,
    #[case] status: u16,
) {
    let config = Config::from_yaml(&format!(
        "model_list: []\ngeneral_settings:\n  master_key: {}\n",
        master_key.unwrap_or("null")
    ))
    .unwrap();
    let client = inference
        .resources
        .pool
        .client(&inference.http, ClientVariant::Provider)
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown, stopped) = oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, litellm_gateway::router(inference, &config))
            .with_graceful_shutdown(async move {
                let _ = stopped.await;
            })
            .await
    });

    let request = client
        .post(format!("http://{address}{path}"))
        .timeout(Duration::from_secs(5))
        .header("x-request-id", "gateway-test")
        .json(&json!({"model": "unconfigured-model"}));
    let request = match authorization {
        Some(value) => request.header("authorization", value),
        None => request,
    };
    let response = request.send().await.unwrap();
    assert_eq!(response.status().as_u16(), status);
    if status == 400 {
        let expected = Error::UnknownModel("unconfigured-model".into());
        assert_eq!(
            response.json::<Value>().await.unwrap(),
            expected.body(Some("gateway-test"))
        );
    } else {
        let text = response.text().await.unwrap();
        assert!(!text.contains("gateway-key"));
        assert!(!text.contains("unconfigured-model"));
    }

    shutdown.send(()).unwrap();
    timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
