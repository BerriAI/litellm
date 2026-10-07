use std::sync::mpsc;

use litellm_tracing::{Diagnostics, DiagnosticsConfig, Level, Metadata, Record, Sink};
use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use prost::Message;
use rstest::rstest;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

struct Compatibility(mpsc::Sender<String>);

impl Sink for Compatibility {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= &Level::WARN
    }
    fn emit(&self, record: &Record) {
        self.0.send(record.message.clone()).unwrap();
    }
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_runtime_preserves_output_filters_and_stops_export_without_replacing_the_logger() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/logs"))
        .and(wiremock::matchers::header(
            "Authorization",
            "os.environ/LITERAL_TOKEN",
        ))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let config: DiagnosticsConfig = serde_json::from_value(json!({
        "enabled": true,
        "policy": {"minimum_level": "DEBUG", "sample_rate": 0},
        "destinations": [{"transport": "otlp", "name": "logs", "endpoint": format!("{}/v1/logs", server.uri()), "headers": {"Authorization": "os.environ/TOKEN"}}]
    })).unwrap();
    let config = config
        .resolve(&|name| (name == "TOKEN").then(|| "os.environ/LITERAL_TOKEN".to_owned()))
        .unwrap();
    let diagnostics = Diagnostics::default();
    let runtime = diagnostics.clone();
    tokio::task::spawn_blocking(move || runtime.configure(config))
        .await
        .unwrap()
        .unwrap();
    let (sender, output) = mpsc::channel();
    let logger = diagnostics.logger(Compatibility(sender));
    logger.emit(
        Level::DEBUG,
        "export policy must not widen console admission",
        Default::default(),
    );
    logger.emit(Level::WARN, "console only", Default::default());
    let invalid =
        serde_json::from_value(json!({"enabled": true, "policy": {"sample_rate": 2}})).unwrap();
    assert!(diagnostics.configure(invalid).is_err());
    assert!(diagnostics.active());
    logger.emit(Level::ERROR, "both", Default::default());
    let runtime = diagnostics.clone();
    tokio::task::spawn_blocking(move || {
        runtime.force_flush().unwrap();
        runtime.configure(DiagnosticsConfig::default()).unwrap();
        runtime.shutdown().unwrap();
    })
    .await
    .unwrap();
    assert!(!diagnostics.active());
    logger.emit(Level::ERROR, "after disable", Default::default());
    assert_eq!(
        output.try_iter().collect::<Vec<_>>(),
        ["console only", "both", "after disable"]
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let payload = ExportLogsServiceRequest::decode(requests[0].body.as_slice()).unwrap();
    let records = &payload.resource_logs[0].scope_logs[0].log_records;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].severity_number, 17);
}

#[rstest]
#[case::disabled(false, 1.0, 0)]
#[case::enabled(true, 1.0, 1)]
#[case::sampled_out(true, 0.0, 0)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shape_capture_requires_opt_in_and_uses_export_sampling_only(
    #[case] enabled: bool,
    #[case] sample_rate: f64,
    #[case] expected: usize,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/logs"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let config = serde_json::from_value(json!({"enabled":true, "payload_shapes": enabled,
        "policy":{"sample_rate":sample_rate},
        "destinations":[{"transport":"otlp", "name":"shapes", "endpoint":format!("{}/v1/logs", server.uri())}]})).unwrap();
    let runtime = Diagnostics::default();
    let configure = runtime.clone();
    tokio::task::spawn_blocking(move || configure.configure(config))
        .await
        .unwrap()
        .unwrap();
    let (sender, compatibility) = mpsc::channel();
    let logger = runtime.logger(Compatibility(sender));
    logger
        .instrument(litellm_tracing::payload::capture(async {
            litellm_tracing::payload::record(
                litellm_tracing::payload::PayloadStage::RequestSent,
                &json!({"messages":[{"content":"private"}]}),
            );
        }))
        .await;
    tokio::task::spawn_blocking(move || {
        runtime.force_flush().unwrap();
        runtime.shutdown().unwrap();
    })
    .await
    .unwrap();
    assert!(compatibility.try_recv().is_err());
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), expected);
    if expected > 0 {
        let payload = ExportLogsServiceRequest::decode(requests[0].body.as_slice()).unwrap();
        let records = &payload.resource_logs[0].scope_logs[0].log_records;
        assert_eq!(records.len(), 1);
        assert!(
            records[0]
                .attributes
                .iter()
                .any(|attribute| attribute.key == "payload.field_paths")
        );
        assert!(!format!("{payload:?}").contains("private"));
    }
}
