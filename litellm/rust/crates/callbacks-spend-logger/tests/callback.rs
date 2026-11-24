//! Drives the spend log callback through its hooks with a scripted host, over a writer whose
//! ledger store records what it is asked to insert. Row inputs come from the row builder's
//! goldens, which `callbacks-spend-log/scripts/record_goldens.py` recorded from Python.

use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_callbacks::{
    BudgetReservation, Callback, Field, FieldSet, FieldValues, Hook, HookInput, HostRequest,
    PatchOp, ReservationEntry, Verdict, hosted,
};
use litellm_callbacks_budgets::Budgets;
use litellm_callbacks_spend_logger::{SpendLogCallback, SpendLogParts};
use litellm_coroutine::CoroutineState;
use litellm_redaction::TokenRedactor;
use litellm_redis_client::{
    Credentials, Namespace, RedisClient, RedisConfig, RedisUrl, Target, Timeouts,
};
use litellm_redis_store::{
    Number, Remote, RequestRedis,
    spend_counters::{SpendCounter, SpendCounters},
};
use litellm_redis_testing::RedisServer;
use litellm_repositories::SpendLogWrite;
use litellm_spend_log_writer::{
    LedgerStore, QueuedSpendLog, SpendLogFlusher, SpendLogWriter, WriterSettings,
};
use rstest::{fixture, rstest};
use serde_json::{Value as Json, json};
use tokio::runtime::Handle;

const KEY_COUNTER: &str = "spend:key:hashed";
const COMMITTED_KEY_COUNTER: &str = "spend:committed:key:hashed";
const REQUEST_ID: &str = "chatcmpl-golden";

#[derive(Clone, Default)]
struct Store {
    written: Arc<Mutex<Vec<SpendLogWrite>>>,
    accepts: Option<usize>,
}

impl LedgerStore for Store {
    async fn insert(&self, writes: &[SpendLogWrite]) -> Result<usize, litellm_db::Error> {
        self.written.lock().unwrap().extend_from_slice(writes);
        Ok(self.accepts.unwrap_or(writes.len()))
    }
}

struct Harness {
    callback: Arc<SpendLogCallback<Store>>,
    counters: Arc<SpendCounters>,
    store: Store,
    wakes: Arc<AtomicUsize>,
}

fn harness(store: Store) -> Harness {
    let counters = Arc::new(SpendCounters::new(None));
    counters
        .set_local(
            &SpendCounter::parse(KEY_COUNTER).unwrap(),
            Number::Float(5.0),
            None,
        )
        .unwrap();
    harness_over(store, counters)
}

fn harness_over(store: Store, counters: Arc<SpendCounters>) -> Harness {
    let wakes = Arc::new(AtomicUsize::new(0));
    let woken = Arc::clone(&wakes);
    let writer = Arc::new(SpendLogWriter::new(
        SpendLogFlusher::new(store.clone(), WriterSettings::default()),
        move || {
            woken.fetch_add(1, Ordering::Relaxed);
        },
    ));
    Harness {
        callback: Arc::new(SpendLogCallback::new(SpendLogParts {
            writer,
            budgets: Arc::new(Budgets::new(Arc::clone(&counters))),
            tokens: Arc::new(TokenRedactor::new(16, Vec::new())),
        })),
        counters,
        store,
        wakes,
    }
}

/// A recorded success row input for a completion, with its config, minus the request id the
/// callback takes from `spend_log_id`.
#[fixture]
fn golden() -> (Json, Json) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../callbacks-spend-log/tests/goldens/rows.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut record: Json = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Json>(&line.replace("#~#", "")).ok())
        .find(|record| {
            record["input"]["call_type"] == "acompletion"
                && record["input"]["standard_logging_object"].is_object()
                && record["input"]["kwargs"]["model"].is_string()
                && record["output"]
                    .as_str()
                    .is_some_and(|output| output.contains("\"status\": \"success\""))
                && record["config"]["turn_off_message_logging"] == false
        })
        .unwrap();
    record["input"]
        .as_object_mut()
        .unwrap()
        .remove("request_id");
    (record["config"].clone(), record["input"].clone())
}

fn settings(config: &Json) -> Json {
    json!({
        "disable_spend_updates": false,
        "disable_spend_logs": false,
        "disable_error_logs": false,
        "disable_responses_id_security": false,
        "row_config": config,
        "proxy_budget_name": "litellm-proxy-budget",
    })
}

fn identity() -> Json {
    json!({"token": "hashed", "user_id": "user-1", "team_id": "team-1", "end_user_id": null, "key_alias": "alias"})
}

fn reservation() -> BudgetReservation {
    BudgetReservation {
        reserved_cost: Some(2.0),
        entries: Some(vec![Some(ReservationEntry {
            counter_key: Some(KEY_COUNTER.to_owned()),
            entity_type: Some("key".to_owned()),
            entity_id: Some("hashed".to_owned()),
            reserved_cost: Some(2.0),
            applied_adjustment: Some(0.0),
            ..ReservationEntry::default()
        })]),
        ..BudgetReservation::default()
    }
}

fn reads() -> FieldSet {
    FieldSet::of(&[
        Field::SpendLogInput,
        Field::SpendLogId,
        Field::Cost,
        Field::BilledRetryAttempts,
        Field::BudgetReservation,
        Field::Outcome,
    ])
}

fn input(
    hook: Hook,
    spend_log_input: Json,
    cost: Json,
    reservation: Option<BudgetReservation>,
) -> HookInput {
    let mut fields = FieldValues::new(reads())
        .with(Field::SpendLogInput, spend_log_input)
        .with(Field::SpendLogId, json!({"id": REQUEST_ID}))
        .with(Field::Cost, cost);
    if let Some(reservation) = reservation {
        fields.insert(
            Field::BudgetReservation,
            serde_json::to_value(reservation).unwrap(),
        );
    }
    HookInput {
        hook,
        fields,
        call: None,
    }
}

fn priced(cost: f64) -> Json {
    json!({"response_cost": cost})
}

/// One `null` per batch item, the proxy's answer when nothing is cached or stored.
fn nulls(request: &Json, items: &str) -> Json {
    Json::Array(vec![
        Json::Null;
        request[items].as_array().map_or(0, Vec::len)
    ])
}

struct Ran {
    calls: Vec<(String, Json)>,
    written_back: Option<BudgetReservation>,
}

/// Runs one hook outside a request, with a host that answers every call and records them.
async fn run(harness: &Harness, input: HookInput, chains: bool) -> Ran {
    run_on(harness, input, chains, None).await
}

/// `run` with the counter step's increments declared on `redis`, the request's Redis batches.
async fn run_on(
    harness: &Harness,
    input: HookInput,
    chains: bool,
    redis: Option<RequestRedis>,
) -> Ran {
    let callback = Arc::clone(&harness.callback);
    let mut body =
        hosted(|host| async move { callback.handle_with(&input, host, redis.as_ref()).await });
    let mut calls = Vec::new();
    loop {
        match body.resume().await.unwrap() {
            CoroutineState::Complete(output) => {
                let written_back = match output.unwrap() {
                    Some(Verdict::Patch(patch)) => match &patch.ops[..] {
                        [
                            PatchOp::Set {
                                field: Field::BudgetReservation,
                                value,
                            },
                        ] => Some(serde_json::from_value(value.clone()).unwrap()),
                        other => panic!("unexpected patch {other:?}"),
                    },
                    None => None,
                    Some(other) => panic!("unexpected verdict {other:?}"),
                };
                return Ran {
                    calls,
                    written_back,
                };
            }
            CoroutineState::Yielded(HostRequest {
                call,
                request,
                reply,
            }) => {
                let answer = match call {
                    "accounting_context" => json!({
                        "windows": [],
                        "savings": request["built"].is_object().then(|| json!({"compression": 0.0, "prompt_caching": 0.0})),
                        "model_access_groups": [],
                        "customer_windows": [],
                        "committed_tracked": [],
                        "customer_entities": [],
                        "chains_through_db": chains,
                    }),
                    "increment_search_free_tier" => json!(SEARCH_COUNTER_TOTAL_AFTER),
                    "enrich_key_identity" => {
                        json!({"user_id": "enriched-user", "team_alias": "enriched-team"})
                    }
                    "budget_windows" => nulls(&request, "entities"),
                    "stored_spend" => nulls(&request, "counters"),
                    "cached_spend" => {
                        Json::Array(vec![
                            json!(0.0);
                            request["sources"].as_array().map_or(0, Vec::len)
                        ])
                    }
                    _ => Json::Null,
                };
                calls.push((call.to_owned(), request));
                reply.send(Ok(answer));
            }
        }
    }
}

const SEARCH_COUNTER_TOTAL_AFTER: f64 = 6.0;

fn success(config: &Json, row: Json) -> Json {
    json!({
        "kind": "success",
        "settings": settings(config),
        "identity": identity(),
        "call_type": "acompletion",
        "row": {"row": row},
    })
}

fn queued(harness: &Harness) -> Vec<QueuedSpendLog> {
    harness.callback.writer().take(100)
}

fn row_json(entry: &QueuedSpendLog) -> Json {
    serde_json::from_str(&entry.to_json()).unwrap()
}

fn counter(harness: &Harness, key: &str) -> Option<f64> {
    harness
        .counters
        .get_local(&SpendCounter::parse(key).unwrap())
        .unwrap()
        .map(Number::as_f64)
}

fn names(calls: &[(String, Json)]) -> Vec<&str> {
    calls.iter().map(|(name, _)| name.as_str()).collect()
}

fn call<'a>(calls: &'a [(String, Json)], name: &str) -> &'a Json {
    &calls
        .iter()
        .find(|(called, _)| called == name)
        .unwrap_or_else(|| panic!("{name} was not called"))
        .1
}

#[rstest]
#[tokio::test]
async fn a_priced_success_queues_one_ledger_row_under_the_typed_request_id(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            success(&config, row),
            priced(0.25),
            None,
        ),
        false,
    )
    .await;

    assert_eq!(
        names(&ran.calls),
        [
            "accounting_context",
            "after_write",
            "budget_windows",
            "stored_spend",
            "cached_spend",
            "budget_windows",
            "after_charge",
        ]
    );
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.25));
    let rows = queued(&harness);
    assert_eq!(rows.len(), 1);
    let stored = row_json(&rows[0]);
    assert_eq!(stored["request_id"], json!(REQUEST_ID));
    assert_eq!(stored["spend"], json!(0.25));
    assert_eq!(stored["accounting_log_disabled"], json!(false));
    assert!(stored["accounting_payload"].is_object());
    assert!(stored["accounting_delivery_id"].is_string());
    let after_write = call(&ran.calls, "after_write");
    assert_eq!(after_write["row"]["spend"], json!(0.25));
    assert_eq!(after_write["claimed"], json!(false));
    assert_eq!(after_write["spend_logs_kept"], json!(true));
    assert_eq!(
        call(&ran.calls, "after_charge")["response_cost"],
        json!(0.25)
    );
    assert_eq!(harness.wakes.load(Ordering::Relaxed), 0);
}

#[rstest]
#[tokio::test]
async fn spend_logs_off_queue_only_the_charge(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["settings"]["disable_spend_logs"] = json!(true);

    run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    let stored = row_json(&queued(&harness)[0]);
    assert_eq!(stored["accounting_log_disabled"], json!(true));
    assert!(stored.get("metadata").is_none());
}

#[rstest]
#[tokio::test]
async fn a_charged_reservation_is_settled_and_written_back(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            success(&config, row),
            priced(0.5),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.5));
    let settled = ran
        .written_back
        .expect("the settled reservation is written back");
    assert_eq!(settled.finalized, Some(true));
    let entry = settled.entries.unwrap()[0].clone().unwrap();
    assert_eq!(entry.applied_adjustment, Some(-1.5));
}

#[rstest]
#[tokio::test]
async fn disabled_spend_updates_skip_the_row_and_release_the_reservation(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["settings"]["disable_spend_updates"] = json!(true);

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            request,
            priced(0.25),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert!(ran.calls.is_empty());
    assert!(queued(&harness).is_empty());
    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.0));
    assert_eq!(ran.written_back.unwrap().finalized, Some(true));
}

#[rstest]
#[case::deferred_batch(priced(0.25), json!({"defers_batch_cost": true}), Some(3.0))]
#[case::unbilled_interaction(json!({}), json!({"unbilled_interaction": true}), Some(3.0))]
#[case::interaction_still_running(json!({}), json!({"unbilled_interaction": true, "interaction_in_progress": true}), Some(5.0))]
#[case::unpriced(json!({}), json!({}), Some(3.0))]
#[tokio::test]
async fn an_uncharged_success_releases_or_keeps_the_reservation(
    golden: (Json, Json),
    #[case] cost: Json,
    #[case] response: Json,
    #[case] counter_after: Option<f64>,
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["response"] = response;
    request["settings"]["background_interaction_cost_polling"] = json!(true);

    run(
        &harness,
        input(Hook::AttemptEnded, request, cost, Some(reservation())),
        false,
    )
    .await;

    assert!(queued(&harness).is_empty());
    assert_eq!(counter(&harness, KEY_COUNTER), counter_after);
}

#[rstest]
#[tokio::test]
async fn an_unpriced_success_raises_the_failed_tracking_alert_and_fails_the_batch(
    golden: (Json, Json),
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row.clone());
    request["batch_accounting"] = json!(true);

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            request,
            json!({"response_cost_failure_debug_info": "no price for the model"}),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert_eq!(
        names(&ran.calls),
        ["failed_tracking_alert", "batch_accounting"]
    );
    let alert = &ran.calls[0].1;
    let model = row["kwargs"]["model"].as_str().unwrap();
    assert_eq!(alert["failing_model"], json!(model));
    let message = alert["error_message"].as_str().unwrap();
    assert!(message.contains("no price for the model"), "{message}");
    assert!(message.contains("call_type: acompletion"), "{message}");
    assert_eq!(
        ran.calls[1].1,
        json!({"write_started": true, "failed": true})
    );
}

#[rstest]
#[tokio::test]
async fn a_charged_success_without_a_reservation_starts_the_batch_write(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["batch_accounting"] = json!(true);

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    assert_eq!(
        *call(&ran.calls, "batch_accounting"),
        json!({"write_started": true, "failed": false})
    );
}

fn grounded(row: &Json, provider: &str) -> Json {
    let mut row = row.clone();
    row["usage"]["prompt_tokens_details"] = json!({"web_search_requests": 3});
    row["standard_logging_object"]["custom_llm_provider"] = json!(provider);
    row["standard_logging_object"]["cost_breakdown"] = json!({"input_cost": 0.5});
    row["standard_logging_object"]["model_map_information"] = json!({"model_map_value": {
        "litellm_provider": provider,
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {"search_context_size_medium": 0.1},
        "search_free_requests": {"count": 5, "period": "day", "shared_group": "g3"},
    }});
    row
}

#[rstest]
#[tokio::test]
async fn grounding_under_the_free_allowance_is_not_charged(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, grounded(&row, "gemini"));
    request["settings"]["apply_search_free_tier"] = json!(true);

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(1.0), None),
        false,
    )
    .await;

    let counted = call(&ran.calls, "increment_search_free_tier");
    let day = counted["key"]
        .as_str()
        .unwrap()
        .strip_prefix("litellm:search_free_tier:gemini:g3:")
        .unwrap();
    assert!(
        chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d").is_ok(),
        "{day}"
    );
    assert_eq!(counted["units"], json!(3));
    let units = 3.0;
    let freed = (5.0 - (SEARCH_COUNTER_TOTAL_AFTER - units)).min(units);
    let charged = 1.0 - 0.1 * units * freed / units;
    let user = counter(&harness, "spend:user:user-1").unwrap();
    assert!((user - charged).abs() < 1e-12, "{user} != {charged}");
    assert_eq!(row_json(&queued(&harness)[0])["spend"], json!(charged));
    assert_eq!(
        call(&ran.calls, "after_charge")["response_cost"],
        json!(charged)
    );
    let logged = call(&ran.calls, "logged_cost");
    assert_eq!(logged["cost"], json!(1.0));
    assert!((logged["discount"].as_f64().unwrap() - (1.0 - charged)).abs() < 1e-12);
    assert!(logged["customer_margin"].is_null());
}

#[rstest]
#[tokio::test]
async fn a_finalized_cost_does_not_draw_from_the_allowance_again(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, grounded(&row, "gemini"));
    request["settings"]["apply_search_free_tier"] = json!(true);

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            request,
            json!({"response_cost": 1.0, "cost_finalized": true}),
            None,
        ),
        false,
    )
    .await;

    assert!(!names(&ran.calls).contains(&"increment_search_free_tier"));
    assert_eq!(row_json(&queued(&harness)[0])["spend"], json!(1.0));
}

#[rstest]
#[tokio::test]
async fn a_responses_turn_the_next_turn_reads_back_requests_a_flush(golden: (Json, Json)) {
    let (config, mut row) = golden;
    row["call_type"] = json!("aresponses");
    let harness = harness(Store::default());

    run(
        &harness,
        input(
            Hook::AttemptEnded,
            success(&config, row),
            priced(0.25),
            None,
        ),
        true,
    )
    .await;

    assert_eq!(harness.wakes.load(Ordering::Relaxed), 1);
    assert_eq!(queued(&harness).len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_request_with_no_identity_is_not_charged(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["identity"] = json!({});

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    assert!(ran.calls.is_empty());
    assert!(queued(&harness).is_empty());
}

#[rstest]
#[tokio::test]
async fn a_failed_attempt_is_left_to_the_fallback(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut hook = input(
        Hook::AttemptEnded,
        success(&config, row),
        priced(0.25),
        None,
    );
    hook.fields.insert(Field::Outcome, json!("failure"));

    run(&harness, hook, false).await;

    assert!(queued(&harness).is_empty());
}

#[rstest]
#[tokio::test]
async fn enrichment_fills_the_row_metadata_the_key_left_empty(golden: (Json, Json)) {
    let (config, mut row) = golden;
    row["metadata"]["user_api_key_user_id"] = Json::Null;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["identity"]["user_id"] = Json::Null;

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    assert_eq!(names(&ran.calls)[0], "enrich_key_identity");
    assert_eq!(ran.calls[1].1["user_id"], json!("enriched-user"));
    let metadata = row_json(&queued(&harness)[0])["metadata"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(metadata.contains("enriched-user"), "{metadata}");
}

#[rstest]
#[tokio::test]
async fn a_row_that_fails_to_build_keeps_its_counters(golden: (Json, Json)) {
    let (config, _) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, json!({"unbuildable": "boom"}));
    request["row"]["committed_tracked"] = json!(["spend:team:team-1"]);
    request["row"]["model_access_groups"] = json!(["group-a"]);

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    assert!(queued(&harness).is_empty());
    assert!(call(&ran.calls, "accounting_context")["built"].is_null());
    assert!(!names(&ran.calls).contains(&"after_write"));
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.25));
    assert_eq!(
        counter(&harness, "spend:model_access_group:group-a"),
        Some(0.25)
    );
}

fn job_row(row: &Json) -> Json {
    let mut job = row.clone();
    job["call_type"] = json!("aretrieve_batch");
    job
}

#[rstest]
#[case::first_poll_owns_the_charge(Some(1), 1, true)]
#[case::a_later_poll_charges_nothing(Some(0), 0, false)]
#[tokio::test]
async fn a_settle_once_job_row_is_claimed_through_the_ledger_store(
    golden: (Json, Json),
    #[case] accepts: Option<usize>,
    #[case] charges: usize,
    #[case] claimed: bool,
) {
    let (config, row) = golden;
    let harness = harness(Store {
        accepts,
        ..Store::default()
    });
    let mut request = success(&config, job_row(&row));
    request["call_type"] = json!("aretrieve_batch");
    let mut hook = input(Hook::AttemptEnded, request, priced(0.5), None);
    hook.fields
        .insert(Field::SpendLogId, json!({"id": "batch_abc_batch_cost"}));

    let ran = run(&harness, hook, false).await;

    assert_eq!(harness.store.written.lock().unwrap().len(), 1);
    assert!(queued(&harness).is_empty());
    assert_eq!(
        names(&ran.calls)
            .iter()
            .filter(|name| **name == "after_charge")
            .count(),
        charges
    );
    let after_write = ran
        .calls
        .iter()
        .find(|(name, _)| name == "after_write")
        .map(|(_, request)| {
            (
                request["claimed"].clone(),
                request["accounting_payload"]["request_id"].clone(),
            )
        });
    assert_eq!(
        after_write,
        claimed.then(|| (json!(true), json!("batch_abc_batch_cost")))
    );
}

#[rstest]
#[tokio::test]
async fn a_repeat_video_poll_charges_its_guardrails_under_its_own_id(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store {
        accepts: Some(0),
        ..Store::default()
    });
    let mut video = row.clone();
    video["call_type"] = json!("avideo_status");
    video["kwargs"]["litellm_call_id"] = json!("poll-2");
    video["standard_logging_object"]["guardrail_information"] = json!([{"guardrail_cost": 0.25}]);
    let mut request = success(&config, video);
    request["call_type"] = json!("avideo_status");
    let mut hook = input(Hook::AttemptEnded, request, priced(1.0), None);
    hook.fields
        .insert(Field::SpendLogId, json!({"id": "video_abc_video_cost"}));

    let ran = run(&harness, hook, false).await;

    assert_eq!(harness.store.written.lock().unwrap().len(), 1);
    let queued = queued(&harness);
    assert_eq!(queued.len(), 1);
    let stored = row_json(&queued[0]);
    assert_eq!(stored["request_id"], json!("poll-2_guardrail_cost"));
    assert_eq!(stored["spend"], json!(0.25));
    assert_eq!(
        call(&ran.calls, "after_charge")["response_cost"],
        json!(0.25)
    );
    assert!(call(&ran.calls, "after_write")["accounting_payload"].is_null());
}

#[rstest]
#[case::every_field(None, false)]
#[case::only_the_always_kept(Some(json!({"include": []})), true)]
#[tokio::test]
async fn a_claimed_job_row_keeps_the_metadata_fields_of_its_request_settings(
    golden: (Json, Json),
    #[case] fields: Option<Json>,
    #[case] cut: bool,
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, job_row(&row));
    request["call_type"] = json!("aretrieve_batch");
    if let Some(fields) = fields {
        request["settings"]["spend_logs_metadata_fields"] = fields;
    }
    let mut hook = input(Hook::AttemptEnded, request, priced(0.5), None);
    hook.fields
        .insert(Field::SpendLogId, json!({"id": "batch_abc_batch_cost"}));

    run(&harness, hook, false).await;

    let written = harness.store.written.lock().unwrap();
    let metadata: Json = serde_json::from_str(
        written[0]
            .log
            .as_ref()
            .and_then(|log| log.metadata.as_ref())
            .unwrap()
            .get(),
    )
    .unwrap();
    let names: Vec<&str> = metadata
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        names.iter().all(|name| [
            "cold_storage_object_key",
            "litellm_previous_response_ids",
            "status"
        ]
        .contains(name)),
        cut,
        "{names:?}"
    );
}

fn failure(config: &Json, rows: Vec<Json>) -> Json {
    json!({
        "kind": "failure",
        "settings": settings(config),
        "identity": identity(),
        "enrich": false,
        "track": true,
        "rows": rows.into_iter().map(|row| json!({"row": row})).collect::<Vec<_>>(),
    })
}

fn failure_input(
    request: Json,
    cost: Json,
    attempts: Json,
    reservation: Option<BudgetReservation>,
) -> HookInput {
    let mut hook = input(Hook::PostCallFailure, request, cost, reservation);
    if !attempts.is_null() {
        hook.fields.insert(Field::BilledRetryAttempts, attempts);
    }
    hook
}

#[rstest]
#[tokio::test]
async fn a_failure_releases_the_reservation_and_writes_every_billed_row(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        failure_input(
            failure(&config, vec![row.clone(), row]),
            json!({"failure_row_response_cost": 0.0}),
            json!([{"position": 0, "spend_log_id": {"id": "chatcmpl-attempt-0"}, "response_cost": 0.0}]),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.0));
    let ids: Vec<Json> = queued(&harness)
        .iter()
        .map(|entry| row_json(entry)["request_id"].clone())
        .collect();
    assert_eq!(ids, [json!(REQUEST_ID), json!("chatcmpl-attempt-0")]);
    assert_eq!(
        names(&ran.calls),
        [
            "accounting_context",
            "after_write",
            "accounting_context",
            "after_write"
        ]
    );
    assert_eq!(ran.written_back.unwrap().finalized, Some(true));
}

#[rstest]
#[tokio::test]
async fn a_failure_row_stores_the_released_reservation_in_its_request_body(golden: (Json, Json)) {
    let (config, mut row) = golden;
    row["stores_prompts"] = json!(true);
    row["litellm_params"]["proxy_server_request"] = json!({"body": {
        "model": "gpt-4o",
        "metadata": {"user_api_key_budget_reservation": serde_json::to_value(reservation()).unwrap()},
    }});
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        failure_input(
            failure(&config, vec![row]),
            json!({"failure_row_response_cost": 0.0}),
            Json::Null,
            Some(reservation()),
        ),
        false,
    )
    .await;

    let settled = ran.written_back.unwrap();
    let stored: Json = serde_json::from_str(
        row_json(&queued(&harness)[0])["proxy_server_request"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    let held = &stored["metadata"]["user_api_key_budget_reservation"];
    assert_eq!(held["finalized"], json!(true));
    assert_eq!(
        held["entries"][0]["applied_adjustment"],
        json!(
            settled.entries.unwrap()[0]
                .as_ref()
                .unwrap()
                .applied_adjustment
                .unwrap()
        )
    );
}

#[rstest]
#[tokio::test]
async fn a_failure_with_a_cost_charges_the_counters_of_each_costed_row(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    run(
        &harness,
        failure_input(
            failure(&config, vec![row.clone(), row]),
            json!({"failure_row_response_cost": 0.25}),
            json!([{"position": 0, "spend_log_id": {"id": "chatcmpl-attempt-0"}, "response_cost": 0.5}]),
            None,
        ),
        false,
    )
    .await;

    let spends: Vec<Json> = queued(&harness)
        .iter()
        .map(|entry| row_json(entry)["spend"].clone())
        .collect();
    assert_eq!(spends, [json!(0.25), json!(0.5)]);
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.75));
}

#[rstest]
#[tokio::test]
async fn a_completed_failure_cost_settles_the_reservation_like_a_success(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        failure_input(
            failure(&config, vec![row]),
            json!({"cost_finalized": true, "failure_row_response_cost": 0.5}),
            json!(null),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.5));
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.5));
    assert_eq!(ran.written_back.unwrap().finalized, Some(true));
}

#[rstest]
#[tokio::test]
async fn an_untracked_failure_only_releases(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = failure(&config, vec![row]);
    request["track"] = json!(false);

    run(
        &harness,
        failure_input(request, json!({}), json!(null), Some(reservation())),
        false,
    )
    .await;

    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.0));
    assert!(queued(&harness).is_empty());
}

#[tokio::test]
async fn a_run_without_the_host_is_refused() {
    let harness = harness(Store::default());
    let hook = input(Hook::AttemptEnded, json!({}), json!({}), None);

    assert!(harness.callback.handle(&hook, None).await.is_err());
}

/// The row's recorded `cost_breakdown`, from the metadata the row stores as JSON text.
fn stored_breakdown(stored: &Json) -> Json {
    serde_json::from_str::<Json>(stored["metadata"].as_str().unwrap()).unwrap()["cost_breakdown"]
        .clone()
}

fn surcharge_of(stored: &Json) -> Json {
    stored["accounting_payload"]["metadata"]["accounting_customer_margin_spend"].clone()
}

#[rstest]
#[case::ten_percent(json!(0.25), Some(json!(0.1)), 0.25, 0.1)]
#[case::integer_cost_and_margin(json!(2), Some(json!(1)), 2.0, 1.0)]
#[case::no_customer(json!(0.25), None, 0.25, 0.0)]
#[tokio::test]
async fn a_customer_success_is_charged_with_its_surcharge(
    golden: (Json, Json),
    #[case] cost: Json,
    #[case] margin: Option<Json>,
    #[case] priced_at: f64,
    #[case] rate: f64,
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    if let Some(margin) = margin {
        request["row"]["customer_margin"] = margin;
    }

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            request,
            json!({"response_cost": cost}),
            Some(reservation()),
        ),
        false,
    )
    .await;

    let surcharge = priced_at * rate;
    let charged = priced_at + surcharge;
    let stored = row_json(&queued(&harness)[0]);
    assert_eq!(stored["spend"], json!(charged));
    assert_eq!(stored["accounting_payload"]["spend"], json!(charged));
    assert_eq!(surcharge_of(&stored), json!(surcharge));
    if rate == 0.0 {
        assert!(stored_breakdown(&stored).is_null());
    } else {
        assert_eq!(
            stored_breakdown(&stored),
            json!({"customer_margin_percent": rate, "customer_margin_amount": surcharge, "total_cost": charged})
        );
    }
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(charged));
    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.0 + charged));
    assert_eq!(
        call(&ran.calls, "after_write")["billable_cost"],
        json!(charged)
    );
    assert_eq!(
        call(&ran.calls, "after_charge")["response_cost"],
        json!(charged)
    );
    let logged = ran.calls.iter().find(|(name, _)| name == "logged_cost");
    assert_eq!(
        logged.map(|(_, request)| request.clone()),
        (rate != 0.0)
            .then(|| json!({"cost": priced_at, "discount": null, "customer_margin": rate}))
    );
}

#[rstest]
#[tokio::test]
async fn a_finalized_success_cost_is_not_surcharged_again(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["row"]["customer_margin"] = json!(0.1);
    request["row"]["cost_finalized"] = json!(true);

    run(
        &harness,
        input(
            Hook::AttemptEnded,
            request,
            json!({"response_cost": 0.25, "cost_finalized": true}),
            None,
        ),
        false,
    )
    .await;

    let stored = row_json(&queued(&harness)[0]);
    assert_eq!(stored["spend"], json!(0.25));
    assert_eq!(surcharge_of(&stored), json!(0.0));
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.25));
}

#[rstest]
#[case::ten_percent(0.25, Some(json!(0.1)), 2.0, 0.1)]
#[case::integer_margin(0.25, Some(json!(1)), 2.0, 1.0)]
#[case::no_customer(0.25, None, 2.0, 0.0)]
#[tokio::test]
async fn a_customer_failure_charges_each_billed_row_with_its_surcharge(
    golden: (Json, Json),
    #[case] final_cost: f64,
    #[case] margin: Option<Json>,
    #[case] attempt_priced_at: f64,
    #[case] rate: f64,
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut snapshotless = row.clone();
    snapshotless["standard_logging_object"] = Json::Null;
    let mut request = failure(&config, vec![row, snapshotless]);
    if let Some(margin) = margin {
        for row in request["rows"].as_array_mut().unwrap() {
            row["customer_margin"] = margin.clone();
        }
    }

    let ran = run(
        &harness,
        failure_input(
            request,
            json!({"failure_row_response_cost": final_cost}),
            json!([{"position": 0, "spend_log_id": {"id": "chatcmpl-attempt-0"}, "response_cost": attempt_priced_at}]),
            None,
        ),
        false,
    )
    .await;

    let charged: Vec<f64> = [final_cost, attempt_priced_at]
        .iter()
        .map(|cost| cost + cost * rate)
        .collect();
    let stored: Vec<Json> = queued(&harness).iter().map(row_json).collect();
    assert_eq!(stored.len(), 2);
    for ((stored, charged), priced_at) in stored
        .iter()
        .zip(&charged)
        .zip([final_cost, attempt_priced_at])
    {
        let surcharge = priced_at * rate;
        assert_eq!(stored["spend"], json!(charged));
        assert_eq!(surcharge_of(stored), json!(surcharge));
        if rate == 0.0 {
            assert!(stored_breakdown(stored).is_null());
        } else {
            assert_eq!(
                stored_breakdown(stored),
                json!({"customer_margin_percent": rate, "customer_margin_amount": surcharge, "total_cost": charged})
            );
        }
    }
    let billed: Vec<Json> = ran
        .calls
        .iter()
        .filter(|(name, _)| name == "after_write")
        .map(|(_, request)| request["billable_cost"].clone())
        .collect();
    assert_eq!(billed, [json!(charged[0]), json!(charged[1])]);
    assert_eq!(
        counter(&harness, "spend:user:user-1"),
        Some(charged[0] + charged[1])
    );
}

#[rstest]
#[tokio::test]
async fn an_unbilled_failure_row_takes_no_surcharge(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = failure(&config, vec![row]);
    request["rows"][0]["customer_margin"] = json!(0.1);

    run(
        &harness,
        failure_input(
            request,
            json!({"failure_row_response_cost": 0.0}),
            Json::Null,
            None,
        ),
        false,
    )
    .await;

    let stored = row_json(&queued(&harness)[0]);
    assert_eq!(stored["spend"], json!(0.0));
    assert!(stored_breakdown(&stored).is_null());
    assert_eq!(counter(&harness, "spend:user:user-1"), None);
}

fn redis_counters(server: &RedisServer) -> Arc<SpendCounters> {
    let client = RedisClient::new(RedisConfig {
        target: Target::Url(RedisUrl {
            url: litellm_secret_value::SecretValue::new(server.url().to_owned()),
            tls: None,
            db: None,
            username: None,
            password: None,
        }),
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(500), None),
        breaker: None,
        events: None,
        health_check: None,
        keepalive: true,
    })
    .unwrap();
    Arc::new(SpendCounters::new(Some(Remote {
        client,
        default_ttl: Duration::from_secs(60),
    })))
}

/// A harness over Redis whose key counter and committed twin are warm at 5.0, and a request.
async fn in_a_request(server: &RedisServer) -> (Harness, RequestRedis) {
    let counters = redis_counters(server);
    for key in [KEY_COUNTER, COMMITTED_KEY_COUNTER] {
        counters
            .set(&SpendCounter::parse(key).unwrap(), Number::Float(5.0), None)
            .await
            .unwrap();
    }
    (
        harness_over(Store::default(), counters),
        RequestRedis::new(Handle::current(), Duration::from_secs(60)),
    )
}

#[rstest]
#[tokio::test]
async fn a_success_charge_waits_on_the_requests_post_call_batch_which_tracks_it(
    golden: (Json, Json),
) {
    let (config, row) = golden;
    let server = RedisServer::start().await.unwrap();
    let (harness, request) = in_a_request(&server).await;
    let mut charged = success(&config, row);
    charged["row"]["committed_tracked"] = json!([KEY_COUNTER]);

    run_on(
        &harness,
        input(Hook::AttemptEnded, charged, priced(0.25), None),
        false,
        Some(request.clone()),
    )
    .await;

    assert_eq!(counter(&harness, COMMITTED_KEY_COUNTER), Some(5.0));
    assert!(!request.tracked_writes_settled());
    request.flush_post().await;
    litellm_redis_testing::eventually(async || request.tracked_writes_settled().then_some(()))
        .await;
    assert_eq!(counter(&harness, COMMITTED_KEY_COUNTER), Some(5.25));
}

#[rstest]
#[tokio::test]
async fn a_failure_row_charge_is_sent_on_the_requests_post_call_batch_by_the_hook_itself(
    golden: (Json, Json),
) {
    let (config, row) = golden;
    let server = RedisServer::start().await.unwrap();
    let (harness, request) = in_a_request(&server).await;
    let mut charged = failure(&config, vec![row]);
    charged["rows"][0]["committed_tracked"] = json!([KEY_COUNTER]);

    run_on(
        &harness,
        failure_input(
            charged,
            json!({"failure_row_response_cost": 0.25}),
            Json::Null,
            None,
        ),
        false,
        Some(request.clone()),
    )
    .await;

    litellm_redis_testing::eventually(async || request.tracked_writes_settled().then_some(()))
        .await;
    assert_eq!(counter(&harness, COMMITTED_KEY_COUNTER), Some(5.25));
}
