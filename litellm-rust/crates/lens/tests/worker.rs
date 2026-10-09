use litellm_lens::{
    config::http_client,
    control::{Control, JobClient},
    model, pipeline, wire,
    worker::Worker,
};
use rstest::rstest;
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
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
#[case::healthy_reads(false, false)]
#[case::review_read_fails(true, false)]
#[case::candidate_read_fails(false, true)]
#[tokio::test]
async fn failed_reads_remain_retryable_after_storage_recovers(
    #[case] fail_review: bool,
    #[case] fail_candidate: bool,
) {
    let server = MockServer::start().await;
    let mut claim: wire::Claim = serde_json::from_value(fixture()).unwrap();
    let sample: wire::Sample = serde_json::from_str(include_str!("fixtures/sample.json")).unwrap();
    let execution = sample.executions[0].clone();
    let unavailable = Arc::new(AtomicBool::new(false));
    let storage_unavailable = unavailable.clone();
    Mock::given(method("GET"))
        .and(path("/lens/worker/lens-test/job-test/content"))
        .respond_with(move |_: &Request| {
            if storage_unavailable.load(Ordering::SeqCst) {
                return ResponseTemplate::new(503);
            }
            ResponseTemplate::new(200).set_body_json(json!({
                "execution": execution,
                "parts": [{"execution_id": "run-test", "span_id": "span-test", "name": "refund",
                    "kind": "tool", "content": QUOTE, "truncated": false}],
            }))
        })
        .mount(&server)
        .await;
    let reviews = Arc::new(Mutex::new(Vec::<wire::Review>::new()));
    let recorded_reviews = reviews.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/progress"))
        .respond_with(move |request: &Request| {
            let progress: wire::Progress = request.body_json().unwrap();
            if let Some(review) = progress.review {
                recorded_reviews.lock().unwrap().push(review);
            }
            ResponseTemplate::new(200).set_body_json(json!({}))
        })
        .mount(&server)
        .await;
    let outage_enabled = Arc::new(AtomicBool::new(true));
    let inject_outage = outage_enabled.clone();
    let fail_content = unavailable.clone();
    let extraction_calls = AtomicUsize::new(0);
    let investigation_calls = AtomicUsize::new(0);
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/model"))
        .respond_with(move |request: &Request| {
            let model: wire::ModelRequest = request.body_json().unwrap();
            let content = match model.purpose {
                wire::ModelRequestPurpose::Extract if extraction_calls.fetch_add(1, Ordering::SeqCst).is_multiple_of(2) => {
                    fail_content.store(fail_review && inject_outage.load(Ordering::SeqCst), Ordering::SeqCst);
                    json!({"tools": [{"action": "read", "execution_id": "run-test"}]})
                }
                wire::ModelRequestPurpose::Extract if fail_review && inject_outage.load(Ordering::SeqCst) => {
                    json!({"result": {"observations": []}})
                }
                wire::ModelRequestPurpose::Extract => json!({"result": {"observations": [
                    {"check_id": "refund", "summary": "False refund claim", "evidence": [quote()]},
                ]}}),
                wire::ModelRequestPurpose::Cluster => json!({"candidates": [
                    {"check_id": "refund", "title": "False refund claim", "hypothesis": "Failure hidden", "execution_ids": ["p0"]},
                ]}),
                wire::ModelRequestPurpose::Investigate if investigation_calls.fetch_add(1, Ordering::SeqCst).is_multiple_of(2) => {
                    fail_content.store(fail_candidate && inject_outage.load(Ordering::SeqCst), Ordering::SeqCst);
                    json!({"tools": [{"action": "read", "execution_id": "run-test"}]})
                }
                wire::ModelRequestPurpose::Investigate => json!({"result": {"findings": []}}),
            };
            ResponseTemplate::new(200).set_body_json(json!({"content": content.to_string(), "cost": 0}))
        })
        .mount(&server)
        .await;
    let result = pipeline::analyze(&claim, sample.clone(), client(&server))
        .await
        .unwrap();
    assert!(result.findings.is_empty());
    if fail_review || fail_candidate {
        assert!(result.error.contains("run-test"));
        assert!(result.review_versions.is_empty());
    } else {
        assert!(result.error.is_empty());
        assert_eq!(result.review_versions.len(), 1);
    }
    let mut saved = reviews.lock().unwrap()[0].clone();
    saved.consolidated = result
        .review_versions
        .iter()
        .any(|r| r.execution_id == saved.execution_id);
    if fail_review {
        assert!(saved.extraction.is_none());
        assert!(saved.cannot_assess);
    }
    claim.reviews = Some(vec![saved]);
    unavailable.store(false, Ordering::SeqCst);
    outage_enabled.store(false, Ordering::SeqCst);
    let recovered = pipeline::analyze(&claim, sample, client(&server))
        .await
        .unwrap();
    assert!(recovered.error.is_empty());
    assert_eq!(recovered.review_versions.len(), 1);
    assert_eq!(
        recovered.coverage.investigated,
        i64::from(fail_review || fail_candidate)
    );
}

#[rstest]
#[case::budget_exhausted(402, 1)]
#[case::model_access_denied(403, 1)]
#[case::model_retries_exhausted(503, 5)]
#[tokio::test]
async fn candidate_control_failure_stops_the_run_without_publishing_partial_findings(
    #[case] status: u16,
    #[case] failed_requests: usize,
) {
    let server = MockServer::start().await;
    let mut claim = fixture();
    claim["job"]["settings"]["concurrency"] = 1.into();
    Mock::given(method("POST"))
        .and(path("/lens/worker/claim"))
        .respond_with(ResponseTemplate::new(200).set_body_json(claim))
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
    Mock::given(method("GET"))
        .and(path("/lens/worker/lens-test/job-test/content"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "execution": sample["executions"][0],
            "parts": [{"execution_id": "run-test", "span_id": "span-test", "name": "refund",
                "kind": "tool", "content": QUOTE, "truncated": false}],
        })))
        .mount(&server)
        .await;
    let progress = Arc::new(Mutex::new(Vec::<wire::Progress>::new()));
    let received_progress = progress.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/progress"))
        .respond_with(move |request: &Request| {
            received_progress
                .lock()
                .unwrap()
                .push(request.body_json().unwrap());
            ResponseTemplate::new(200).set_body_json(json!({}))
        })
        .mount(&server)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let model_calls = calls.clone();
    let extraction_calls = AtomicUsize::new(0);
    let cluster_calls = Arc::new(AtomicUsize::new(0));
    let clustering = cluster_calls.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/model"))
        .respond_with(move |request: &Request| {
            let model: wire::ModelRequest = request.body_json().unwrap();
            let content = match model.purpose {
                wire::ModelRequestPurpose::Extract if extraction_calls.fetch_add(1, Ordering::SeqCst) == 0 => {
                    json!({"tools": [{"action": "read", "execution_id": "run-test"}]})
                }
                wire::ModelRequestPurpose::Extract => json!({"result": {"observations": [
                    {"check_id": "refund", "summary": "False refund claim", "evidence": [quote()]},
                    {"check_id": "refund", "summary": "Missing failure recovery", "evidence": [quote()]},
                    {"check_id": "refund", "summary": "Unverified payment", "evidence": [quote()]},
                ]}}),
                wire::ModelRequestPurpose::Cluster => {
                    clustering.fetch_add(1, Ordering::SeqCst);
                    json!({"candidates": [
                        {"check_id": "refund", "title": "False refund claim", "hypothesis": "Failure hidden", "execution_ids": ["p0"]},
                        {"check_id": "refund", "title": "Missing failure recovery", "hypothesis": "No recovery", "execution_ids": ["p1"]},
                        {"check_id": "refund", "title": "Unverified payment", "hypothesis": "Not checked", "execution_ids": ["p2"]},
                    ]})
                }
                wire::ModelRequestPurpose::Investigate => {
                    if model_calls.fetch_add(1, Ordering::SeqCst) != 0 {
                        return ResponseTemplate::new(status).set_body_json(json!({
                            "detail": {"lens_error": "Test model access failure"},
                        }));
                    }
                    json!({"result": {"findings": [finding()]}})
                }
            };
            ResponseTemplate::new(200).set_body_json(json!({"content": content.to_string(), "cost": 0}))
        })
        .mount(&server)
        .await;
    let results = Arc::new(Mutex::new(Vec::<wire::Result>::new()));
    let received_results = results.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/result"))
        .respond_with(move |request: &Request| {
            received_results
                .lock()
                .unwrap()
                .push(request.body_json().unwrap());
            ResponseTemplate::new(200).set_body_json(json!({}))
        })
        .expect(1)
        .mount(&server)
        .await;
    let worker = Worker::new(
        Control::new(
            http_client().unwrap(),
            server.uri().parse().unwrap(),
            "worker-test".into(),
        ),
        "test-release".into(),
    );
    assert!(worker.run_once().await.unwrap());
    let results = results.lock().unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].error.contains(&format!("HTTP {status}")));
    assert!(results[0].findings.is_empty());
    assert!(results[0].review_versions.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 1 + failed_requests);
    assert_eq!(cluster_calls.load(Ordering::SeqCst), 1);
    assert!(!progress.lock().unwrap().iter().any(|progress| {
        progress.stage.as_deref() == Some("Consolidating findings across runs")
    }));
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
        .expect(2)
        .mount(&server)
        .await;
    let sample: Value = serde_json::from_str(include_str!("fixtures/sample.json")).unwrap();
    Mock::given(method("GET"))
        .and(path("/lens/worker/lens-test/job-test/sample"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&sample))
        .mount(&server)
        .await;
    let reviews = Arc::new(Mutex::new(Vec::<wire::Review>::new()));
    let previous = reviews.clone();
    Mock::given(method("GET"))
        .and(path("/lens/worker/lens-test/job-test/reviews"))
        .respond_with(move |_: &Request| {
            ResponseTemplate::new(200).set_body_json(previous.lock().unwrap().clone())
        })
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
    let recorded = Arc::new(Mutex::new(Vec::<wire::Review>::new()));
    let progress_reviews = recorded.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/progress"))
        .respond_with(move |request: &Request| {
            let progress: wire::Progress = request.body_json().unwrap();
            if let Some(review) = progress.review {
                progress_reviews.lock().unwrap().push(review);
            }
            ResponseTemplate::new(200).set_body_json(json!({}))
        })
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
        .expect(2)
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
    let result: wire::Result = serde_json::from_value(saved.lock().unwrap()[0].clone()).unwrap();
    assert_eq!(result.error, "");
    assert_eq!(result.findings.len(), 1);
    assert_eq!(&*result.findings[0].evidence[0].quote, QUOTE);
    assert_eq!(result.coverage.screened, 1);
    assert_eq!(result.coverage.investigated, 1);
    assert_eq!(result.review_versions.len(), 1);
    assert_eq!(result.assessments[0].issue_checks, vec!["refund"]);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let mut prior = recorded.lock().unwrap()[0].clone();
    assert!(!prior.spans.is_empty());
    prior.consolidated = true;
    reviews.lock().unwrap().push(prior.clone());
    recorded.lock().unwrap().clear();
    assert!(worker.run_once().await.unwrap());
    let reused = recorded.lock().unwrap()[0].clone();
    assert!(reused.reused);
    assert_eq!(
        serde_json::to_value(&reused.spans).unwrap(),
        serde_json::to_value(&prior.spans).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&reused.extraction).unwrap(),
        serde_json::to_value(&prior.extraction).unwrap()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let result: wire::Result = serde_json::from_value(saved.lock().unwrap()[1].clone()).unwrap();
    assert_eq!(result.error, "");
    assert_eq!(result.coverage.reused, 1);
    assert!(result.findings.is_empty());
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

#[rstest]
#[tokio::test]
async fn configured_private_dns_names_are_reachable_without_following_redirects() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/private-service"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/redirect"))
        .respond_with(ResponseTemplate::new(302).insert_header("location", "/private-service"))
        .mount(&server)
        .await;
    let base = server.uri().replace("127.0.0.1", "localhost");
    let client = http_client().unwrap();
    let response = client
        .get(format!("{base}/private-service"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let redirected = client.get(format!("{base}/redirect")).send().await.unwrap();
    assert_eq!(redirected.status(), 302);
}

#[rstest]
#[tokio::test]
async fn checkpoint_history_preserves_only_the_supplied_finding_summary() {
    use litellm_lens::{activity::Tracker, agent, evidence::Workspace};

    let server = MockServer::start().await;
    let mut saved = finding();
    saved["id"] = json!("saved-finding");
    saved["first_seen"] = json!("2026-01-01T00:00:00Z");
    saved["last_seen"] = json!("2026-01-01T00:00:00Z");
    saved["revision"] = json!(1);
    let mut input = fixture();
    input["findings"] = json!([saved]);
    let claim: wire::Claim = serde_json::from_value(input).unwrap();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/progress"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/model"))
        .respond_with(move |request: &Request| {
            let model: wire::ModelRequest = request.body_json().unwrap();
            let message: Value =
                serde_json::from_str(&model.messages.last().unwrap().content).unwrap();
            let turn = match observed.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    assert_eq!(message["existing_findings"][0]["id"], "saved-finding");
                    assert!(message["existing_findings"][0].get("evidence").is_none());
                    json!({"checkpoint": "Recover the saved finding summary"})
                }
                1 => json!({"tools": [{"action": "history", "include_initial": true,
                    "turn_start": 0, "turn_end": 0}]}),
                2 => {
                    let history: Value =
                        serde_json::from_str(message["tool_results"][0].as_str().unwrap()).unwrap();
                    let recovered = &history["initial_context"]["existing_findings"][0];
                    assert_eq!(recovered["id"], "saved-finding");
                    assert_eq!(recovered["title"], "Refund success was falsely reported");
                    for field in ["evidence", "occurrences", "investigation_runs"] {
                        assert!(
                            recovered.get(field).is_none(),
                            "{field} escaped into history"
                        );
                    }
                    assert_eq!(
                        history["initial_context"]["supplied"]["task_id"],
                        "summary-test"
                    );
                    json!({"result": {"observations": []}})
                }
                _ => panic!("Unexpected retry while recovering a finding summary"),
            };
            ResponseTemplate::new(200)
                .set_body_json(json!({"content": turn.to_string(), "cost": 0}))
        })
        .expect(3)
        .mount(&server)
        .await;
    let client = client(&server);
    let workspace = Workspace::new(vec![], client.clone());
    let tracker = Tracker::start(
        &client,
        "summary-test".into(),
        wire::ActivityPhase::Review,
        "Recover summary".into(),
        vec![],
    )
    .await
    .unwrap();
    let output: wire::Extraction = agent::run(
        &claim,
        &workspace,
        agent::Assignment {
            stage: "test",
            task: "Recover only supplied finding details".into(),
            purpose: wire::ModelRequestPurpose::Extract,
            supplied: json!({"task_id": "summary-test"}),
        },
        &tracker,
    )
    .await
    .unwrap();
    assert!(output.observations.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[rstest]
#[tokio::test]
async fn oversized_combined_tool_replies_remain_readable_after_a_checkpoint() {
    use litellm_lens::{
        activity::Tracker,
        agent,
        evidence::{MAX_TOOL_BYTES, Workspace},
    };
    let server = MockServer::start().await;
    let claim: wire::Claim = serde_json::from_value(fixture()).unwrap();
    let sample: wire::Sample = serde_json::from_str(include_str!("fixtures/sample.json")).unwrap();
    let filler_size = MAX_TOOL_BYTES * 3 / 5;
    let page_calls = Arc::new(AtomicUsize::new(0));
    let page_count = page_calls.clone();
    let execution = sample.executions[0].clone();
    Mock::given(method("GET"))
        .and(path("/lens/worker/lens-test/job-test/content"))
        .respond_with(move |_: &Request| {
            let marker = if page_count.fetch_add(1, Ordering::SeqCst) == 0 {
                "FIRST_REPLY"
            } else {
                "ARCHIVED_SECOND_REPLY"
            };
            ResponseTemplate::new(200).set_body_json(json!({"execution":execution,"parts":[{
                "execution_id":"run-test","span_id":"span-test","name":format!("{marker}{}", "x".repeat(filler_size)),"kind":"tool","content":"evidence","truncated":false
            }]}))
        }).expect(2).mount(&server).await;
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/progress"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;
    let model_calls = Arc::new(AtomicUsize::new(0));
    let model_count = model_calls.clone();
    Mock::given(method("POST"))
        .and(path("/lens/worker/lens-test/job-test/model"))
        .respond_with(move |request: &Request| {
            let model: wire::ModelRequest = request.body_json().unwrap();
            let turn = match model_count.fetch_add(1, Ordering::SeqCst) {
                0 => json!({"tools":[{"action":"catalog","execution_id":"run-test"},{"action":"catalog","execution_id":"run-test"}],"checkpoint":"Inspect the archived second reply"}),
                1 => {
                    let reply: Value = serde_json::from_str(&model.messages.last().unwrap().content).unwrap();
                    assert!(reply["tool_results"][1].as_str().unwrap().contains("Combined tool output exceeds"), "{}", reply["tool_results"][1].as_str().unwrap().chars().take(600).collect::<String>());
                    json!({"tools":[{"action":"history","turn_start":0,"turn_end":1,"char_start":filler_size,"char_end":filler_size+6000}]})
                },
                2 => {
                    let reply: Value = serde_json::from_str(&model.messages.last().unwrap().content).unwrap();
                    let history: Value = serde_json::from_str(reply["tool_results"][0].as_str().unwrap()).unwrap();
                    assert!(history["excerpt"].as_str().unwrap().contains("ARCHIVED_SECOND_REPLY"));
                    json!({"result":{"observations":[]}})
                },
                _ => panic!("Unexpected model retry"),
            };
            ResponseTemplate::new(200).set_body_json(json!({"content":turn.to_string(),"cost":0}))
        }).expect(3).mount(&server).await;
    let client = client(&server);
    let workspace = Workspace::new(sample.executions, client.clone());
    let tracker = Tracker::start(
        &client,
        "test".into(),
        wire::ActivityPhase::Review,
        "Archive".into(),
        vec![],
    )
    .await
    .unwrap();
    let output: wire::Extraction = agent::run(
        &claim,
        &workspace,
        agent::Assignment {
            stage: "test",
            task: "Read two tools and recover the second from history".into(),
            purpose: wire::ModelRequestPurpose::Extract,
            supplied: json!({}),
        },
        &tracker,
    )
    .await
    .unwrap();
    assert!(output.observations.is_empty());
    assert_eq!(page_calls.load(Ordering::SeqCst), 2);
    assert_eq!(model_calls.load(Ordering::SeqCst), 3);
}
