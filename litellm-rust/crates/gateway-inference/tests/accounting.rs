mod accounting_support;
mod support;

use accounting_support::{Gate, Ledger, Policy, cost, failure, ledger, runtime};
use axum::body::to_bytes;
use futures_util::StreamExt;
use litellm_accounting::{
    ApplyResult, Cost, Effect, EffectState, Outcome, ProviderWork, ReportedUsage, SettlementStatus,
};
use rstest::rstest;
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn accounted_app(
    model: &str,
    base: &str,
    accounting: Option<litellm_gateway_inference::accounting::ResponsesAccounting>,
) -> axum::Router {
    support::app_with_accounting(
        model,
        base,
        accounting,
        litellm_gateway_auth::Permissions::All,
    )
}

const EFFECTS: [Effect; 3] = [
    Effect::ReconcileBudget,
    Effect::RecordSpend,
    Effect::ReleaseBudgetReservation,
];

async fn upstream(stream: bool) -> (MockServer, Value, String) {
    let server = MockServer::start().await;
    let completed = json!({"id": "response", "model": "test-model", "output": [], "usage": {"input_tokens": 10, "output_tokens": 2}, "custom": true});
    let events = format!(
        "event: response.completed\ndata: {}\n\n",
        json!({"type": "response.completed", "response": completed})
    );
    let template = if stream {
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(&events)
    } else {
        ResponseTemplate::new(200).set_body_json(&completed)
    };
    Mock::given(method("POST"))
        .respond_with(template)
        .mount(&server)
        .await;
    (server, completed, events)
}

fn request(stream: bool) -> Value {
    json!({"model": "public/model", "input": "hello", "stream": stream})
}

#[rstest]
#[case::ordinary(false)]
#[case::streaming(true)]
#[tokio::test]
async fn accounting_without_callbacks_preserves_payload_and_records_once(
    ledger: Arc<Mutex<Ledger>>,
    #[case] stream: bool,
    #[values("/responses", "/v1/responses")] route: &str,
) {
    let (server, completed, events) = upstream(stream).await;
    let accounting = runtime(&ledger, Policy::default(), None);
    let app = accounted_app("openai/test-model", &server.uri(), Some(accounting.clone()));
    let response = support::post(app, route, request(stream)).await;
    assert_eq!(response.status(), 200);
    if stream {
        assert_eq!(
            to_bytes(response.into_body(), 4096).await.unwrap().as_ref(),
            events.as_bytes()
        );
    } else {
        assert_eq!(support::json(response).await, completed);
    }
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.admitted.len(), 1);
    assert_eq!(state.admitted[0].0, "public/model");
    assert_eq!(state.admitted[0].1, "openai/test-model");
    assert!(!state.admitted[0].2.is_empty());
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.spend.len(), 1);
    assert_eq!(
        state.spend[0].usage(),
        &ReportedUsage::Known(completed["usage"].clone())
    );
    assert_eq!(state.spend[0].charges().total(), Ok(cost(4)));
    assert_eq!(state.prepared_credentials, [false]);
    assert!(state.reports.is_empty());
}

#[rstest]
#[case::healthy(false)]
#[case::callback_failure(true)]
#[tokio::test]
async fn terminal_callback_receives_accounting_after_reservation_release(
    ledger: Arc<Mutex<Ledger>>,
    #[case] fail: bool,
) {
    let (server, _, _) = upstream(false).await;
    let accounting = runtime(&ledger, Policy::default(), Some(fail));
    let app = accounted_app("openai/test-model", &server.uri(), Some(accounting.clone()));
    let response = support::post(app, "/responses", request(false)).await;
    assert_eq!(response.status(), 200);
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.spend.len(), 1);
    assert_eq!(state.reports.len(), 1);
    assert_eq!(state.reports[0].status, SettlementStatus::Committed);
    assert_eq!(state.callback_release_counts, [1]);
    assert!(EFFECTS.iter().all(|effect| matches!(
        state.reports[0].progress.effect(*effect),
        EffectState::Committed
    )));
}

#[rstest]
#[case::budget(Effect::ReconcileBudget)]
#[case::spend(Effect::RecordSpend)]
#[case::release(Effect::ReleaseBudgetReservation)]
#[tokio::test]
async fn financial_failure_is_visible_and_all_effects_are_attempted(
    ledger: Arc<Mutex<Ledger>>,
    #[case] failed: Effect,
    #[values(false, true)] uncertain: bool,
) {
    let (server, _, _) = upstream(false).await;
    let result = if uncertain {
        ApplyResult::Indeterminate(failure())
    } else {
        ApplyResult::NotApplied(failure())
    };
    let accounting = runtime(
        &ledger,
        Policy {
            result: Some((failed, result)),
            ..Default::default()
        },
        Some(false),
    );
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 500);
    assert_eq!(support::json(response).await["error"]["code"], 500);
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.reports[0].status, SettlementStatus::NeedsAttention);
    assert_eq!(
        state.released.len(),
        usize::from(failed != Effect::ReleaseBudgetReservation)
    );
}

#[rstest]
#[case::fact_failure(Policy { bad_fact: true, ..Default::default() })]
#[case::price_failure(Policy { bad_price: true, ..Default::default() })]
#[case::unknown_price(Policy { unknown_price: true, ..Default::default() })]
#[tokio::test]
async fn unknown_accounting_is_not_zero_or_success_and_still_releases(
    ledger: Arc<Mutex<Ledger>>,
    #[case] policy: Policy,
) {
    let (server, _, _) = upstream(false).await;
    let accounting = runtime(&ledger, policy, Some(false));
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 500);
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.reports[0].status, SettlementStatus::Unpriced);
    assert_eq!(state.spend[0].charges().total(), Ok(Cost::Unknown));
    assert_eq!(state.spend[0].work(), ProviderWork::Unknown);
}

#[rstest]
#[tokio::test]
async fn avoided_provider_work_preserves_usage_and_independent_service_cost(
    ledger: Arc<Mutex<Ledger>>,
) {
    let (server, completed, _) = upstream(false).await;
    let accounting = runtime(
        &ledger,
        Policy {
            avoided: true,
            ..Default::default()
        },
        None,
    );
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 200);
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(
        state.spend[0].usage(),
        &ReportedUsage::Known(completed["usage"].clone())
    );
    assert_eq!(state.spend[0].charges().provider(), cost(0));
    assert_eq!(state.spend[0].charges().services(), cost(1));
    assert_eq!(state.spend[0].charges().total(), Ok(cost(1)));
}

#[rstest]
#[tokio::test]
async fn admission_rejection_never_reaches_provider(ledger: Arc<Mutex<Ledger>>) {
    let (server, _, _) = upstream(false).await;
    let accounting = runtime(
        &ledger,
        Policy {
            reject: true,
            ..Default::default()
        },
        None,
    );
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 500);
    accounting.shutdown().await;
    assert!(server.received_requests().await.unwrap().is_empty());
    let state = ledger.lock().unwrap();
    assert!(state.admitted.is_empty());
    assert!(state.effects.is_empty());
}

#[rstest]
#[tokio::test]
async fn provider_failure_preserves_http_error_and_accounting_cleanup(ledger: Arc<Mutex<Ledger>>) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .mount(&server)
        .await;
    let accounting = runtime(
        &ledger,
        Policy {
            result: Some((Effect::RecordSpend, ApplyResult::NotApplied(failure()))),
            ..Default::default()
        },
        Some(false),
    );
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 429);
    assert!(
        support::json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("slow down")
    );
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.reports[0].terminal.outcome(), Outcome::Failed);
    assert_eq!(state.reports[0].status, SettlementStatus::NeedsAttention);
}

#[rstest]
#[case::before_chunks(false)]
#[case::after_usage(true)]
#[tokio::test]
async fn dropping_stream_retains_delivered_usage_and_releases(
    ledger: Arc<Mutex<Ledger>>,
    #[case] consume: bool,
) {
    let (server, completed, events) = upstream(true).await;
    let accounting = runtime(&ledger, Policy::default(), Some(false));
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(true),
    )
    .await;
    let mut body = response.into_body().into_data_stream();
    if consume {
        assert_eq!(
            body.next().await.unwrap().unwrap().as_ref(),
            events.as_bytes()
        );
    }
    drop(body);
    tokio::time::timeout(Duration::from_secs(5), accounting.shutdown())
        .await
        .unwrap();
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.spend[0].outcome(), Outcome::Cancelled);
    assert_eq!(state.spend[0].charges().total(), Ok(cost(4)));
    let usage = if consume {
        ReportedUsage::Known(completed["usage"].clone())
    } else {
        ReportedUsage::Unknown
    };
    assert_eq!(state.spend[0].usage(), &usage);
    assert_eq!(state.reports.len(), 1);
}

#[rstest]
#[case::admission(false)]
#[case::settlement(true)]
#[tokio::test]
async fn cancelling_a_wait_does_not_cancel_accounting_or_repeat_writes(
    ledger: Arc<Mutex<Ledger>>,
    #[case] settlement: bool,
) {
    let (server, _, _) = upstream(false).await;
    let gate = Arc::new(Gate::new());
    let policy = if settlement {
        Policy {
            spend_gate: Some(gate.clone()),
            ..Default::default()
        }
    } else {
        Policy {
            admission_gate: Some(gate.clone()),
            ..Default::default()
        }
    };
    let accounting = runtime(&ledger, policy, Some(false));
    let app = accounted_app("openai/test-model", &server.uri(), Some(accounting.clone()));
    let call = tokio::spawn(support::post(app, "/responses", request(false)));
    tokio::time::timeout(Duration::from_secs(5), gate.entered.notified())
        .await
        .unwrap();
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    gate.resume.add_permits(1);
    tokio::time::timeout(Duration::from_secs(5), accounting.shutdown())
        .await
        .unwrap();
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.spend.len(), 1);
    assert_eq!(state.reports.len(), 1);
    assert_eq!(
        state.spend[0].outcome(),
        if settlement {
            Outcome::Succeeded
        } else {
            Outcome::Cancelled
        }
    );
}

#[rstest]
#[tokio::test]
async fn shutdown_stops_new_admissions(ledger: Arc<Mutex<Ledger>>) {
    let (server, _, _) = upstream(false).await;
    let accounting = runtime(&ledger, Policy::default(), None);
    accounting.shutdown().await;
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting)),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 500);
    assert!(ledger.lock().unwrap().admitted.is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn authorization_precedes_accounting_admission(ledger: Arc<Mutex<Ledger>>) {
    let (server, _, _) = upstream(false).await;
    let accounting = runtime(&ledger, Policy::default(), None);
    let app = support::app_with_accounting(
        "openai/test-model",
        &server.uri(),
        Some(accounting.clone()),
        litellm_gateway_auth::Permissions::None,
    );
    let response = support::post(app, "/responses", request(false)).await;
    assert_eq!(response.status(), 403);
    accounting.shutdown().await;
    assert!(ledger.lock().unwrap().admitted.is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn missing_accounting_service_preserves_existing_response() {
    let (server, completed, _) = upstream(false).await;
    let response = support::post(
        support::app("openai/test-model", &server.uri()),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 200);
    assert_eq!(support::json(response).await, completed);
}

#[rstest]
#[case::queue_accepted(ApplyResult::Accepted, SettlementStatus::Accepted)]
#[case::committed(ApplyResult::Committed, SettlementStatus::Committed)]
#[tokio::test]
async fn terminal_callback_distinguishes_queue_acceptance_from_commit(
    ledger: Arc<Mutex<Ledger>>,
    #[case] result: ApplyResult<litellm_gateway_inference::PluginError>,
    #[case] status: SettlementStatus,
) {
    let (server, _, _) = upstream(false).await;
    let accounting = runtime(
        &ledger,
        Policy {
            result: Some((Effect::RecordSpend, result)),
            ..Default::default()
        },
        Some(false),
    );
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 200);
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.reports[0].status, status);
}

#[rstest]
#[tokio::test]
async fn stream_settlement_failure_is_one_sse_error_after_original_chunks(
    ledger: Arc<Mutex<Ledger>>,
) {
    let (server, _, events) = upstream(true).await;
    let accounting = runtime(
        &ledger,
        Policy {
            result: Some((Effect::RecordSpend, ApplyResult::NotApplied(failure()))),
            ..Default::default()
        },
        Some(false),
    );
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(true),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = to_bytes(response.into_body(), 8192).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.starts_with(&events));
    assert_eq!(text.matches("event: error").count(), 1);
    assert!(text.contains("NeedsAttention"));
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.reports[0].terminal.outcome(), Outcome::Succeeded);
}

#[rstest]
#[tokio::test]
async fn partial_stream_transport_failure_preserves_usage_and_charge_policy(
    ledger: Arc<Mutex<Ledger>>,
) {
    let gate = Arc::new(Gate::new());
    let partial = json!({"type": "response.in_progress", "response": {"usage": {"input_tokens": 10, "output_tokens": 1}}});
    let event = format!("event: response.in_progress\ndata: {partial}\n\n");
    let frame = event.clone();
    let upstream_gate = gate.clone();
    let app = axum::Router::new().route(
        "/responses",
        axum::routing::post(move || {
            let frame = frame.clone();
            let gate = upstream_gate.clone();
            async move {
                let chunks = futures_util::stream::once(async move {
                    Ok::<_, std::io::Error>(bytes::Bytes::from(frame))
                })
                .chain(futures_util::stream::once(async move {
                    gate.resume.acquire().await.unwrap().forget();
                    Err(std::io::Error::other("upstream disconnected"))
                }));
                (
                    [("content-type", "text/event-stream")],
                    axum::body::Body::from_stream(chunks),
                )
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let accounting = runtime(&ledger, Policy::default(), Some(false));
    let response = support::post(
        accounted_app("openai/test-model", &url, Some(accounting.clone())),
        "/responses",
        request(true),
    )
    .await;
    assert_eq!(response.status(), 200);
    let mut body = response.into_body().into_data_stream();
    assert_eq!(body.next().await.unwrap().unwrap(), event);
    gate.resume.add_permits(1);
    let error = tokio::time::timeout(Duration::from_secs(5), body.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        std::str::from_utf8(&error)
            .unwrap()
            .starts_with("event: error\n")
    );
    assert!(body.next().await.is_none());
    drop(body);
    accounting.shutdown().await;
    server.abort();
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.spend.len(), 1);
    assert_eq!(state.spend[0].outcome(), Outcome::Failed);
    assert_eq!(
        state.spend[0].usage(),
        &ReportedUsage::Known(partial["response"]["usage"].clone())
    );
    assert_eq!(state.spend[0].charges().total(), Ok(cost(4)));
}

#[rstest]
#[tokio::test]
async fn pricing_failure_preserves_reported_usage(ledger: Arc<Mutex<Ledger>>) {
    let (server, completed, _) = upstream(false).await;
    let accounting = runtime(
        &ledger,
        Policy {
            bad_price: true,
            ..Default::default()
        },
        Some(false),
    );
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 500);
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(
        state.spend[0].usage(),
        &ReportedUsage::Known(completed["usage"].clone())
    );
    assert_eq!(state.reports[0].terminal.usage(), state.spend[0].usage());
    assert!(state.reports[0].assessment_error.is_some());
    assert_eq!(state.spend[0].charges().total(), Ok(Cost::Unknown));
    assert_eq!(state.released, ["reservation"]);
}

#[rstest]
#[tokio::test]
async fn separate_calls_have_separate_sessions_and_callback_identity(ledger: Arc<Mutex<Ledger>>) {
    let (server, _, _) = upstream(false).await;
    let accounting = runtime(&ledger, Policy::default(), Some(false));
    let app = accounted_app("openai/test-model", &server.uri(), Some(accounting.clone()));
    let (first, second) = tokio::join!(
        support::post(app.clone(), "/responses", request(false)),
        support::post(app, "/responses", request(false))
    );
    assert_eq!(first.status(), 200);
    assert_eq!(second.status(), 200);
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.admitted.len(), 2);
    assert_eq!(state.spend.len(), 2);
    assert_eq!(state.released.len(), 2);
    assert_eq!(state.reports.len(), 2);
    assert_ne!(state.reports[0].call_id, state.reports[1].call_id);
    for effect in EFFECTS {
        assert_eq!(
            state
                .effects
                .iter()
                .filter(|actual| **actual == effect)
                .count(),
            2
        );
    }
}

#[rstest]
#[tokio::test]
async fn unreserved_native_calls_still_record_spend(ledger: Arc<Mutex<Ledger>>) {
    let (server, completed, _) = upstream(false).await;
    let accounting = runtime(
        &ledger,
        Policy {
            no_reservation: true,
            ..Default::default()
        },
        Some(false),
    );
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        request(false),
    )
    .await;
    assert_eq!(response.status(), 200);
    accounting.shutdown().await;
    let state = ledger.lock().unwrap();
    assert_eq!(state.effects, [Effect::RecordSpend]);
    assert!(state.released.is_empty());
    assert_eq!(
        state.spend[0].usage(),
        &ReportedUsage::Known(completed["usage"].clone())
    );
    assert_eq!(state.spend[0].charges().total(), Ok(cost(4)));
    assert_eq!(state.reports[0].status, SettlementStatus::Committed);
    assert!(matches!(
        state.reports[0]
            .progress
            .effect(Effect::ReleaseBudgetReservation),
        EffectState::NotRequired
    ));
}

#[rstest]
#[tokio::test]
async fn core_validation_failure_releases_admission_without_provider_execution(
    ledger: Arc<Mutex<Ledger>>,
) {
    let (server, _, _) = upstream(false).await;
    let accounting = runtime(&ledger, Policy::default(), Some(false));
    let response = support::post(
        accounted_app("openai/test-model", &server.uri(), Some(accounting.clone())),
        "/responses",
        json!({"model": "public/model"}),
    )
    .await;
    assert_eq!(response.status(), 400);
    accounting.shutdown().await;
    assert!(server.received_requests().await.unwrap().is_empty());
    let state = ledger.lock().unwrap();
    assert_eq!(state.admitted.len(), 1);
    assert_eq!(state.effects, EFFECTS);
    assert_eq!(state.released, ["reservation"]);
    assert_eq!(state.reports[0].terminal.outcome(), Outcome::Failed);
    assert_eq!(state.reports[0].terminal.usage(), &ReportedUsage::Unknown);
}
