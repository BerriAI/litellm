use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use axum::{body::Body, http::Request};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
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
        axum::serve(listener, litellm_gateway::router(inference, &config, None))
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
        .instrument(litellm_gateway::router(inference, &config, None).oneshot(request))
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

#[rstest]
#[tokio::test]
async fn mounts_ui_without_exposing_credentials_or_authorizing_inference(inference: Arc<Gateway>) {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: inference-secret\n")
            .unwrap();
    let assets = tempfile::TempDir::new().unwrap();
    std::fs::write(assets.path().join("index.html"), "dashboard").unwrap();
    let backend = litellm_gateway_auth::UiBackend::new(
        "admin".into(),
        litellm_auth_types::SecretValue::new("ui-password"),
    )
    .unwrap();
    let ui = litellm_gateway_ui::router(
        backend,
        tower_sessions_moka_store::MokaStore::new(Some(10_000)),
        false,
    )
    .merge(litellm_gateway_ui::dashboard_assets(assets.path()));
    let app = litellm_gateway::router(inference, &config, Some(ui));
    let page = app
        .clone()
        .oneshot(Request::get("/ui/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(page.status().as_u16(), 200);
    assert_eq!(
        axum::body::to_bytes(page.into_body(), 65536).await.unwrap(),
        "dashboard"
    );
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(LogSink(sender));
    let response = logger
        .instrument(
            app.clone().oneshot(
                Request::post("/v2/login")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"username": "admin", "password": "ui-password"}).to_string(),
                    ))
                    .unwrap(),
            ),
        )
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert!(
        response
            .headers()
            .get_all("set-cookie")
            .iter()
            .all(|cookie| !cookie.to_str().unwrap().contains("Secure"))
    );
    let cookie = response
        .headers()
        .get_all("set-cookie")
        .iter()
        .filter_map(|value| value.to_str().unwrap().split(';').next())
        .collect::<Vec<_>>()
        .join("; ");
    let login: Value = serde_json::from_slice(
        &logger
            .instrument(axum::body::to_bytes(response.into_body(), 65536))
            .await
            .unwrap(),
    )
    .unwrap();
    let jwt = login["token"].as_str().unwrap();
    let token: Value = serde_json::from_slice(
        &URL_SAFE_NO_PAD
            .decode(jwt.split('.').nth(1).unwrap())
            .unwrap(),
    )
    .unwrap();
    let csrf = token["key"].as_str().unwrap();
    let logs = receiver.try_iter().collect::<Vec<_>>();
    let logged = serde_json::to_string(&logs).unwrap();
    assert!(!logged.contains("ui-password"));
    assert!(!logged.contains(jwt));
    let ui_response = app
        .clone()
        .oneshot(
            Request::get("/session/info")
                .header("cookie", &cookie)
                .header("authorization", format!("Bearer {csrf}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ui_response.status().as_u16(), 200);
    let inference_response = app
        .clone()
        .oneshot(
            Request::post("/v1/messages")
                .header("cookie", cookie)
                .header("authorization", format!("Bearer {csrf}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(inference_response.status().as_u16(), 401);
    let master_response = app
        .oneshot(
            Request::get("/session/info")
                .header("authorization", "Bearer inference-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(master_response.status().as_u16(), 401);
}

#[rstest]
#[case::assets("/ui/", "GET")]
#[case::login("/v2/login", "POST")]
#[case::session("/session/info", "GET")]
#[case::discovery("/.well-known/litellm-ui-config", "GET")]
#[tokio::test]
async fn ui_routes_are_absent_when_not_mounted(
    inference: Arc<Gateway>,
    #[case] path: &str,
    #[case] method: &str,
) {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
            .unwrap();
    let response = litellm_gateway::router(inference, &config, None)
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 404);
}
