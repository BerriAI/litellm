use litellm_lens::{
    config::http_client,
    control::{Control, JobClient},
    model, wire,
    worker::Worker,
};
use rstest::rstest;
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{method, path, query_param},
};

const QUOTE: &str = "refund_status=failed; agent_reply=Your refund is complete";

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/claim.json")).unwrap()
}

fn quote() -> Value {
    json!({"execution_id":"run-test","span_id":"span-test","quote":QUOTE,"role":"support"})
}

fn finding() -> Value {
    json!({"title":"Refund success was falsely reported", "description":"The agent said the refund completed even though its tool returned a failure", "check_id":"refund", "kind":"issue", "evidence":[quote()], "brief":{"problem":"A failed refund was reported as successful", "user_goal":"Receive a refund", "what_happened":"The refund tool failed but the assistant reported success", "test_cases":[{"input":"A refund request whose payment tool returns failed", "expected":"The agent must explain the failure without claiming a completed refund"}]}})
}

fn client(server: &MockServer) -> JobClient {
    JobClient::new(
        Control::new(
            http_client().unwrap(),
            server.uri().parse().unwrap(),
            "test-worker-key".into(),
        ),
        "lens-test",
        "job-test",
        2,
    )
    .unwrap()
}

#[rstest]
#[tokio::test]
async fn worker_reviews_original_unicode_content_repairs_citations_and_submits_verified_finding() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/lens/worker/claim"))
        .and(query_param(
            "protocol_version",
            wire::PROTOCOL_VERSION.to_string(),
        ))
        .and(query_param("worker_release", "test-release"))
        .respond_with(ResponseTemplate::new(200).set_body_json(fixture()))
        .expect(1)
        .mount(&server)
        .await;
    let sample: Value = serde_json::from_str(include_str!("fixtures/sample.json")).unwrap();
    Mock::given(method("GET"))
        .and(path("/lens/worker/lens-test/job-test/sample"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&sample))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/lens/worker/lens-test/job-test/reviews"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
        .mount(&server)
        .await;
    let text = format!("{}{}{}", "é".repeat(7990), QUOTE, "終".repeat(8000));
    Mock::given(method("GET")).and(path("/lens/worker/lens-test/job-test/content")).respond_with(move |request: &Request| {
        let offset: usize = request.url.query_pairs().find(|(k, _)| k == "offset").unwrap().1.parse().unwrap();
        assert!(offset > 0, "full evidence uses the API's one-based content offset");
        let start = offset - 1;
        let content: String = text.chars().skip(start).take(8000).collect();
        ResponseTemplate::new(200).set_body_json(json!({"execution":sample["executions"][0],"parts":[{"execution_id":"run-test","span_id":"span-test","name":"refund","kind":"tool","content":content,"truncated":start+8000<text.chars().count()}]}))
    }).mount(&server).await;
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/progress"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let extract_calls = calls.clone();
    Mock::given(method("POST")).and(path("/lens/worker/lens-test/job-test/model")).respond_with(move |request: &Request| {
        let model: wire::ModelRequest = request.body_json().unwrap();
        let content = match model.purpose {
            wire::ModelRequestPurpose::Extract => match extract_calls.fetch_add(1, Ordering::SeqCst) {
                0 => json!({"tools":[{"action":"read","execution_id":"run-test"}]}),
                1 => json!({"result":{"observations":[{"check_id":"refund","summary":"False refund claim","evidence":[{"execution_id":"run-test","span_id":"span-test","quote":"fabricated quotation"}]}]}}),
                _ => json!({"result":{"reasoning":"The original tool failure contradicts the agent response", "observations":[{"check_id":"refund","summary":"False refund claim","evidence":[quote()]}]}}),
            },
            wire::ModelRequestPurpose::Cluster => json!({"candidates":[{"check_id":"refund","title":"False refund claim","hypothesis":"The agent ignored a tool failure","execution_ids":["p0"]}]}),
            wire::ModelRequestPurpose::Investigate => json!({"result":{"findings":[finding()]}}),
        };
        ResponseTemplate::new(200).set_body_json(json!({"content":content.to_string(),"cost":0}))
    }).mount(&server).await;
    let saved = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured = saved.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/result"))
        .respond_with(move |request: &Request| {
            captured.lock().unwrap().push(request.body_json().unwrap());
            ResponseTemplate::new(200).set_body_json(json!({}))
        })
        .expect(1)
        .mount(&server)
        .await;
    let worker = Worker::new(
        Control::new(
            http_client().unwrap(),
            server.uri().parse().unwrap(),
            "test-worker-key".into(),
        ),
        "test-release".into(),
    );
    assert!(worker.run_once().await.unwrap());
    let saved = saved.lock().unwrap();
    let result: wire::Result = serde_json::from_value(saved[0].clone()).unwrap();
    assert_eq!(result.error, "");
    assert_eq!(result.findings.len(), 1);
    assert_eq!(&*result.findings[0].evidence[0].quote, QUOTE);
    assert_eq!(result.coverage.screened, 1);
    assert_eq!(result.coverage.investigated, 1);
    assert_eq!(result.review_versions.len(), 1);
    assert_eq!(result.assessments[0].issue_checks, vec!["refund"]);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[rstest]
#[case::wrong_title(json!({"title": []}))]
#[case::empty_evidence(json!({"evidence": []}))]
#[case::empty_test_cases(json!({"brief": {"problem":"Refund success was falsely reported", "user_goal":"Receive refund", "what_happened":"Failure hidden", "test_cases":[]}}))]
#[tokio::test]
async fn model_contract_rejects_malformed_findings_and_repairs(#[case] change: Value) {
    let server = MockServer::start().await;
    let mut invalid = finding();
    for (key, value) in change.as_object().unwrap() {
        invalid[key] = value.clone();
    }
    let count = Arc::new(AtomicUsize::new(0));
    let calls = count.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/model"))
        .respond_with(move |_request: &Request| {
            let value = if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                invalid.clone()
            } else {
                finding()
            };
            ResponseTemplate::new(200)
                .set_body_json(json!({"content":json!({"findings":[value]}).to_string(), "cost":0}))
        })
        .expect(2)
        .mount(&server)
        .await;
    let request = model::request(
        wire::ModelRequestPurpose::Investigate,
        json!({"task":"Inspect evidence"}),
    )
    .unwrap();
    let (result, _) =
        model::structured::<wire::Findings>(&client(&server), request, "Findings", |_| None)
            .await
            .unwrap();
    assert_eq!(result.findings.len(), 1);
    assert!(!result.findings[0].evidence.is_empty());
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn incompatible_claim_is_failed_without_calling_models() {
    let server = MockServer::start().await;
    let mut claim = fixture();
    claim["unknown_protocol_field"] = true.into();
    Mock::given(method("POST"))
        .and(path("/lens/worker/claim"))
        .respond_with(ResponseTemplate::new(200).set_body_json(claim))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/result"))
        .respond_with(|request: &Request| {
            let result: wire::Result = request.body_json().unwrap();
            assert!(result.error.contains("Update the worker"));
            ResponseTemplate::new(200).set_body_json(json!({}))
        })
        .expect(1)
        .mount(&server)
        .await;
    let worker = Worker::new(
        Control::new(
            http_client().unwrap(),
            server.uri().parse().unwrap(),
            "test-worker-key".into(),
        ),
        "test-release".into(),
    );
    assert!(worker.run_once().await.unwrap());
    assert!(
        !server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .any(|r| r.url.path().ends_with("/model"))
    );
}

#[rstest]
#[tokio::test]
async fn proxy_prefix_is_preserved_for_every_control_request() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/gateway/prefix/lens/status"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok":true})))
        .expect(1)
        .mount(&server)
        .await;
    let control = Control::new(
        http_client().unwrap(),
        format!("{}/gateway/prefix", server.uri()).parse().unwrap(),
        "test-key".into(),
    );
    let result: Value = control.get("/lens/status").await.unwrap();
    assert_eq!(result["ok"], true);
}

#[rstest]
#[case::sanitized(json!({"detail":{"lens_error":"Configure pricing before investigation"},"secret":"must-not-appear"}), true)]
#[case::raw_provider_error(json!({"detail":"must-not-appear"}), false)]
#[case::oversized(json!({"detail":{"lens_error":"must-not-appear".repeat(4096)}}), false)]
#[tokio::test]
async fn model_failures_expose_only_bounded_sanitized_gateway_diagnostics(
    #[case] body: Value,
    #[case] expected_diagnostic: bool,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/model"))
        .respond_with(ResponseTemplate::new(400).set_body_json(body))
        .mount(&server)
        .await;
    let request =
        model::request(wire::ModelRequestPurpose::Extract, json!({"task":"Review"})).unwrap();
    let error = client(&server).model(&request).await.unwrap_err();
    assert_eq!(
        error
            .to_string()
            .contains("Configure pricing before investigation"),
        expected_diagnostic
    );
    assert!(!error.to_string().contains("must-not-appear"));
    assert!(matches!(
        error,
        litellm_lens::Error::Control { status: 400, .. }
    ));
}
