use std::{collections::HashMap, sync::Arc};

use litellm_tracing::{Error, ExportPolicy, ExportSink, Level, Logger, OtlpSink};
use opentelemetry_proto::tonic::{
    collector::logs::v1::ExportLogsServiceRequest, common::v1::any_value,
};
use prost::Message;
use rstest::rstest;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

#[rstest]
#[case::negative(-0.1)]
#[case::above_one(1.1)]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
fn invalid_sampling_configuration_is_rejected(#[case] rate: f64) {
    assert!(matches!(
        ExportPolicy::new(Level::INFO, vec![], rate),
        Err(Error::InvalidSampleRate)
    ));
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn otlp_exports_owned_redacted_records_after_destination_filtering_and_sampling() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/logs"))
        .and(header("authorization", "Bearer test-token"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let endpoint = format!("{}/v1/logs", server.uri());
    let sink = tokio::task::spawn_blocking(move || {
        Arc::new(
            OtlpSink::new(
                endpoint,
                HashMap::from([("authorization".to_owned(), "Bearer test-token".to_owned())]),
                "diagnostic-test".to_owned(),
                ExportPolicy::new(Level::WARN, vec!["LiteLLM".into()], 0.0).unwrap(),
            )
            .unwrap(),
        )
    })
    .await
    .unwrap();
    let logger = Logger::new(sink.clone());
    let fields = json!({"source.target": "LiteLLM", "source.timestamp": 1_700_000_000.5, "extra": {"api_key": "secret123", "nested": [true, 3]}, "stack": null, "exception.stacktrace": "Authorization: Bearer abcdefghijklmnop"}).as_object().unwrap().clone();
    logger.emit(Level::INFO, "level filtered", fields.clone());
    logger.emit(Level::WARN, "sampled out", fields.clone());
    logger.emit(
        Level::ERROR,
        "Authorization: Bearer abcdefghijklmnop",
        fields,
    );
    logger.emit(
        Level::ERROR,
        "target filtered",
        json!({"source.target": "Other"})
            .as_object()
            .unwrap()
            .clone(),
    );
    tokio::task::spawn_blocking(move || {
        sink.force_flush().unwrap();
        sink.shutdown().unwrap();
    })
    .await
    .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let exported = ExportLogsServiceRequest::decode(requests[0].body.as_slice()).unwrap();
    let logs = &exported.resource_logs[0].scope_logs[0].log_records;
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0].time_unix_nano, 1_700_000_000_500_000_000);
    assert_eq!(logs[0].severity_number, 17);
    assert!(
        !logs[0]
            .attributes
            .iter()
            .any(|attribute| attribute.key == "stack")
    );
    let Some(any_value::Value::StringValue(message)) = &logs[0].body.as_ref().unwrap().value else {
        panic!("string body")
    };
    assert_eq!(message, "Authorization: REDACTED");
    let exception = logs[0]
        .attributes
        .iter()
        .find(|attribute| attribute.key == "exception.stacktrace")
        .unwrap();
    assert_eq!(
        exception.value.as_ref().unwrap().value,
        Some(any_value::Value::StringValue(
            "Authorization: REDACTED".into()
        ))
    );
    let extra = logs[0]
        .attributes
        .iter()
        .find(|attribute| attribute.key == "extra")
        .unwrap();
    let Some(any_value::Value::KvlistValue(extra)) = &extra.value.as_ref().unwrap().value else {
        panic!("nested extra")
    };
    assert_eq!(
        extra
            .values
            .iter()
            .find(|attribute| attribute.key == "api_key")
            .unwrap()
            .value
            .as_ref()
            .unwrap()
            .value,
        Some(any_value::Value::StringValue("REDACTED".into()))
    );
    let Some(any_value::Value::ArrayValue(nested)) = &extra
        .values
        .iter()
        .find(|attribute| attribute.key == "nested")
        .unwrap()
        .value
        .as_ref()
        .unwrap()
        .value
    else {
        panic!("typed list")
    };
    assert_eq!(
        nested.values[0].value,
        Some(any_value::Value::BoolValue(true))
    );
    assert_eq!(nested.values[1].value, Some(any_value::Value::IntValue(3)));
}

#[cfg(feature = "posthog")]
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn posthog_uses_the_official_sdk_to_export_personless_redacted_diagnostics() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/batch/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": 1})))
        .mount(&server)
        .await;
    let host = server.uri();
    let sink = tokio::task::spawn_blocking(move || {
        Arc::new(
            litellm_tracing::PostHogSink::new(
                "phc_test".to_owned(),
                host,
                "diagnostic-test".to_owned(),
                ExportPolicy::new(Level::WARN, vec!["LiteLLM".into()], 0.0).unwrap(),
            )
            .unwrap(),
        )
    })
    .await
    .unwrap();
    let logger = Logger::new(sink.clone());
    logger.emit(
        Level::WARN,
        "sampled out",
        json!({"source.target": "LiteLLM"})
            .as_object()
            .unwrap()
            .clone(),
    );
    logger.emit(
        Level::ERROR,
        "Authorization: Bearer abcdefghijklmnop",
        json!({"source.target": "LiteLLM", "source.timestamp": 1_700_000_000.5, "extra": {"api_key": "secret123", "retry": true}})
            .as_object()
            .unwrap()
            .clone(),
    );
    logger.emit(
        Level::ERROR,
        "target filtered",
        json!({"source.target": "Other"})
            .as_object()
            .unwrap()
            .clone(),
    );
    tokio::task::spawn_blocking(move || {
        sink.force_flush().unwrap();
        sink.shutdown().unwrap();
    })
    .await
    .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["api_key"], "phc_test");
    let events = body["batch"].as_array().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["event"], "litellm diagnostic");
    assert_eq!(events[0]["timestamp"], "2023-11-14T22:13:20.500");
    assert_eq!(events[0]["distinct_id"], "diagnostic-test");
    assert_eq!(events[0]["properties"]["$process_person_profile"], false);
    assert_eq!(
        events[0]["properties"]["message"],
        "Authorization: REDACTED"
    );
    assert_eq!(
        events[0]["properties"]["fields"]["extra"],
        json!({"api_key": "REDACTED", "retry": true})
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn logs_with_the_same_trace_have_stable_sampling_decisions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/logs"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let endpoint = format!("{}/v1/logs", server.uri());
    let sink = tokio::task::spawn_blocking(move || {
        Arc::new(
            OtlpSink::new(
                endpoint,
                HashMap::new(),
                "sampling-test".to_owned(),
                ExportPolicy::new(Level::INFO, vec![], 0.5).unwrap(),
            )
            .unwrap(),
        )
    })
    .await
    .unwrap();
    let logger = Logger::new(sink.clone());
    for trace in ["trace-0", "trace-1"] {
        let scoped = logger.with_fields(json!({"trace_id": trace}).as_object().unwrap().clone());
        scoped.emit(Level::INFO, "one", serde_json::Map::new());
        scoped.emit(Level::INFO, "two", serde_json::Map::new());
    }
    tokio::task::spawn_blocking(move || {
        sink.force_flush().unwrap();
        sink.shutdown().unwrap();
    })
    .await
    .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let exported = ExportLogsServiceRequest::decode(requests[0].body.as_slice()).unwrap();
    let logs = &exported.resource_logs[0].scope_logs[0].log_records;
    assert_eq!(logs.len(), 2);
    for log in logs {
        let trace = log
            .attributes
            .iter()
            .find(|attribute| attribute.key == "trace_id")
            .unwrap();
        assert_eq!(
            trace.value.as_ref().unwrap().value,
            Some(any_value::Value::StringValue("trace-1".into()))
        );
    }
}
