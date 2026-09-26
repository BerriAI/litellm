use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use axum::{body::Body, http::Request};
use litellm_config::Config;
use litellm_gateway_inference::{Error, Gateway};
use litellm_http::ClientVariant;
use litellm_tracing::{Logger, Metadata, Record, Sink};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};
use tower::ServiceExt;

struct LogSink(mpsc::Sender<Value>);

impl Sink for LogSink {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn emit(&self, record: &Record) {
        self.0
            .send(json!({"message": record.message, "fields": record.fields}))
            .unwrap();
    }
}

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

#[rstest]
#[tokio::test]
async fn logs_request_outcome_without_credentials_or_query(inference: Arc<Gateway>) {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
            .unwrap();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages?token=query-secret")
        .header("authorization", "Bearer header-secret")
        .body(Body::empty())
        .unwrap();
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(LogSink(sender));

    let response = logger
        .instrument(litellm_gateway::router(inference, &config).oneshot(request))
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 401);
    let record = receiver.try_recv().unwrap();
    assert_eq!(record["message"], "response headers");
    assert_eq!(record["fields"]["method"], "POST");
    assert_eq!(record["fields"]["path"], "/v1/messages");
    assert_eq!(record["fields"]["status"], 401);
    assert!(record["fields"]["time_to_headers_ms"].as_f64().unwrap() >= 0.0);
    assert!(record["fields"]["request_id"].as_str().is_some());
    assert!(receiver.try_recv().is_err());
    assert!(!record.to_string().contains("header-secret"));
    assert!(!record.to_string().contains("query-secret"));
}
