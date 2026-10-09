use std::sync::{Arc, mpsc};

use litellm_tracing::{
    Diagnostics, DiagnosticsConfig, Error, ExportSink, Logger, Metadata, Record, Sink,
    analytics::{self, Analytics, AnalyticsInputs, AnalyticsSurface},
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::oss(None, None, false, true, false)]
#[case::enterprise(None, None, true, false, false)]
#[case::enterprise_opt_in(None, Some("true"), true, true, false)]
#[case::oss_opt_out(None, Some("false"), false, false, false)]
#[case::dnt_wins(Some("1"), Some("true"), false, false, false)]
#[case::dnt_enterprise(Some("true"), Some("true"), true, false, false)]
#[case::invalid_explicit(None, Some("garbage"), false, false, true)]
#[case::invalid_enterprise(None, Some("garbage"), true, false, true)]
#[case::invalid_dnt(Some("garbage"), Some("true"), false, false, true)]
#[case::whitespace(Some(" TRUE "), Some("true"), false, false, false)]
#[case::dnt_false(Some("0"), None, false, true, false)]
#[case::empty_explicit(None, Some(""), false, false, true)]
#[case::empty_dnt(Some(""), Some("true"), false, false, true)]
#[case::short_dnt(Some("y"), Some("true"), false, false, false)]
#[case::short_explicit_true(None, Some("t"), true, true, false)]
#[case::short_explicit_false(None, Some("f"), false, false, false)]
#[case::short_dnt_false(Some("n"), None, false, true, false)]
#[case::python_whitespace_dnt(Some("\u{1c}TRUE\u{1f}"), Some("true"), false, false, false)]
#[case::unicode_whitespace_opt_out(None, Some("\u{a0}off\u{2003}"), false, false, false)]
fn controls_fail_closed_without_license_verification(
    #[case] dnt: Option<&str>,
    #[case] explicit: Option<&str>,
    #[case] license: bool,
    #[case] enabled: bool,
    #[case] invalid: bool,
) {
    let decision = AnalyticsInputs {
        do_not_track: dnt.map(Into::into),
        explicit: explicit.map(Into::into),
        license_configured: license,
    }
    .decision();
    assert_eq!(decision.enabled, enabled);
    assert_eq!(decision.invalid_setting, invalid);
}

#[rstest]
#[case::empty("")]
#[case::invalid("not-a-license")]
#[case::unresolved("os.environ/MISSING_LICENSE")]
fn license_presence_is_sufficient_even_when_unusable(#[case] value: &str) {
    let inputs = AnalyticsInputs::from_sources(
        |name| (name == "LITELLM_LICENSE").then(|| value.into()),
        false,
    );
    assert!(!inputs.decision().enabled);
}

struct Capture(mpsc::Sender<Value>);
impl Sink for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn emit(&self, record: &Record) {
        self.0.send(Value::Object(record.fields.clone())).unwrap();
    }
}
impl ExportSink for Capture {
    fn force_flush(&self) -> Result<(), Error> {
        Ok(())
    }
    fn shutdown(&self) -> Result<(), Error> {
        Ok(())
    }
}

#[rstest]
fn disabled_decision_does_not_construct_a_client_and_is_frozen() {
    let runtime = Analytics::default();
    assert!(
        !runtime
            .initialize_with(false, AnalyticsSurface::PythonSdk, "1.0".into(), || panic!(
                "must not construct a client"
            ))
            .unwrap()
    );
    assert!(
        !runtime
            .initialize_with(true, AnalyticsSurface::PythonSdk, "1.0".into(), || panic!(
                "must not reevaluate a frozen decision"
            ))
            .unwrap()
    );
    runtime.shutdown().unwrap();
}

#[rstest]
fn enabled_without_build_configuration_is_a_frozen_noop() {
    let runtime = Analytics::default();
    assert!(
        !runtime
            .initialize(
                AnalyticsInputs::default().decision(),
                AnalyticsSurface::RustSdk,
                "1.0".into(),
                None
            )
            .unwrap()
    );
    assert!(runtime.initialized());
    runtime.force_flush().unwrap();
    runtime.shutdown().unwrap();
}

#[rstest]
#[case::python(AnalyticsSurface::PythonSdk, false, "python_sdk")]
#[case::rust(AnalyticsSurface::RustSdk, true, "rust_sdk")]
fn projection_drops_all_logging_context_and_operator_shutdown_is_independent(
    #[case] surface: AnalyticsSurface,
    #[case] rust: bool,
    #[case] wire_surface: &str,
) {
    let (sender, received) = mpsc::channel();
    let (compatibility, console) = mpsc::channel();
    let runtime = Diagnostics::default();
    runtime
        .analytics()
        .initialize_with(true, surface, "1.0".into(), || {
            Ok(Some(Arc::new(Capture(sender))))
        })
        .unwrap();
    let logger = runtime.logger(Capture(compatibility)).with_fields(
        json!({"api_key":"private", "trace_id":"private", "messages":["private"]})
            .as_object()
            .unwrap()
            .clone(),
    );
    runtime.configure(DiagnosticsConfig::default()).unwrap();
    runtime.shutdown().unwrap();
    logger.scope(|| {
        analytics::runtime_started(rust);
        analytics::api_used("acompletion");
        tracing::info!("private diagnostic");
    });
    let records = received.try_iter().collect::<Vec<_>>();
    assert_eq!(
        records[0],
        json!({"schema_version":1,"event":"runtime_started","surface":wire_surface,"version":"1.0"})
    );
    assert_eq!(records.len(), if rust { 1 } else { 2 });
    if !rust {
        assert_eq!(
            records[1],
            json!({"schema_version":1,"event":"api_used","surface":wire_surface,"version":"1.0","route":"chat_completions"})
        );
    }
    assert_eq!(console.try_iter().count(), 1);
    runtime.analytics().shutdown().unwrap();
    logger.scope(|| analytics::runtime_started(rust));
    assert_eq!(received.try_iter().count(), 0);
}

#[rstest]
#[case::rust(AnalyticsSurface::RustGateway, 1)]
#[case::python(AnalyticsSurface::PythonGateway, 0)]
fn native_route_summaries_are_counted_only_by_rust_hosts(
    #[case] surface: AnalyticsSurface,
    #[case] count: usize,
) {
    let (sender, received) = mpsc::channel();
    let analytics = Analytics::default();
    analytics
        .initialize_with(true, surface, "1.0".into(), || {
            Ok(Some(Arc::new(Capture(sender))))
        })
        .unwrap();
    Logger::new(analytics).scope(|| {
        let span = tracing::info_span!(target: "litellm_inference_chat", "litellm.route", route="chat_completions", model="private", provider="private");
        let _guard = span.enter();
    });
    let records = received.try_iter().collect::<Vec<_>>();
    assert_eq!(records.len(), count);
    if count != 0 {
        assert_eq!(
            records[0],
            json!({"schema_version":1,"event":"api_used","surface":"rust_gateway","version":"1.0","route":"chat_completions"})
        );
    }
}

#[cfg(feature = "posthog")]
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn official_transport_sends_only_projected_product_properties() {
    use litellm_tracing::analytics::AnalyticsProject;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/batch/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status":1})))
        .mount(&server)
        .await;
    let endpoint = server.uri();
    let analytics = Analytics::default();
    let start = analytics.clone();
    tokio::task::spawn_blocking(move || {
        start.initialize(
            AnalyticsInputs::default().decision(),
            AnalyticsSurface::PythonSdk,
            "1.0".into(),
            Some(AnalyticsProject {
                endpoint,
                token: "phc_test".into(),
            }),
        )
    })
    .await
    .unwrap()
    .unwrap();
    let logger = Logger::new(analytics.clone()).with_fields(
        json!({"api_key":"private", "request.field_paths":["private"]})
            .as_object()
            .unwrap()
            .clone(),
    );
    logger.scope(|| {
        analytics::runtime_started(false);
        analytics::api_used("private/customer/model");
    });
    tokio::task::spawn_blocking(move || analytics.shutdown())
        .await
        .unwrap()
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let events = requests
        .iter()
        .flat_map(|request| {
            serde_json::from_slice::<Value>(&request.body).unwrap()["batch"]
                .as_array()
                .unwrap()
                .clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["event"], "litellm.runtime.started");
    assert_eq!(events[0]["distinct_id"], events[1]["distinct_id"]);
    assert_eq!(
        events[1]["properties"],
        json!({"schema_version":1,"event":"api_used","surface":"python_sdk","version":"1.0","route":"other","$process_person_profile":false,"$geoip_disable":true,"$is_server":true})
    );
    assert!(!serde_json::to_string(&events).unwrap().contains("private"));
}
