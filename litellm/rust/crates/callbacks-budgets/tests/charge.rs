//! The post-call counter step against a real spend counter store, with a scripted proxy answering
//! its host calls.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_callbacks::{
    BudgetReservation, HostException, HostFault, HostRequest, ReservationEntry, hosted,
};
use litellm_callbacks_budgets::{
    CounterError, CounterOutcome, CounterStep, CustomerScopeSpend, CustomerWindow, charge_counters,
};
use litellm_circuit_breaker::{State, Transition};
use litellm_coroutine::CoroutineState;
use litellm_redis_client::{
    CallEvent, Credentials, Namespace, Operation, RedisClient, RedisConfig, RedisEvents, RedisUrl,
    Target, Timeouts,
};
use litellm_redis_store::{
    Number, Remote, RequestRedis,
    auth_cache::AuthObject,
    spend_counters::{SpendCounter, SpendCounters},
};
use litellm_redis_testing::RedisServer;
use rstest::rstest;
use serde_json::{Value, json};
use tokio::runtime::Handle;

const START: &str = "2026-11-01T00:00:00+00:00";

/// What the proxy answers the step with: stored spend by counter (a window by
/// `label:entity:duration`), cached auth spend by cache key, budget windows by cache key, and the
/// host calls it raises on.
#[derive(Clone, Default)]
struct Proxy {
    stored: HashMap<String, f64>,
    cached: HashMap<String, f64>,
    windows: HashMap<String, Vec<(String, Option<String>)>>,
    stored_windows: HashMap<String, Vec<(String, Option<String>)>>,
    failing: HashSet<&'static str>,
}

impl Proxy {
    fn answer(&self, call: &str, request: &Value) -> Result<Value, String> {
        if self.failing.contains(call) {
            return Err(format!("{call} is down"));
        }
        Ok(match call {
            "stored_spend" => Value::Array(
                request["counters"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|counter| {
                        let key = match counter["source"].as_str().unwrap() {
                            "entity" => counter["counter_key"].as_str().unwrap().to_owned(),
                            _ => format!(
                                "{}:{}:{}",
                                counter["entity_type"].as_str().unwrap(),
                                counter["entity_id"].as_str().unwrap(),
                                counter["window_duration"].as_str().unwrap()
                            ),
                        };
                        json!(self.stored.get(&key))
                    })
                    .collect(),
            ),
            "cached_spend" => Value::Array(
                request["sources"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|sources| {
                        json!(
                            sources
                                .as_array()
                                .unwrap()
                                .iter()
                                .find_map(|source| {
                                    self.cached.get(
                                        &serde_json::from_value::<AuthObject>(source.clone())
                                            .unwrap()
                                            .key(),
                                    )
                                })
                                .copied()
                                .unwrap_or(0.0)
                        )
                    })
                    .collect(),
            ),
            "budget_windows" => {
                let stored = request["stored"].as_bool().unwrap();
                Value::Array(
                    request["entities"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|entity| {
                            let key = entity["cache_key"].as_str().unwrap();
                            match (stored, self.windows.get(key), self.stored_windows.get(key)) {
                                (false, Some(windows), _) | (true, _, Some(windows)) => {
                                    json!(windows)
                                }
                                (false, None, _) => Value::Null,
                                (true, _, None) => json!([]),
                            }
                        })
                        .collect(),
                )
            }
            other => panic!("unexpected host call {other}"),
        })
    }
}

struct Run {
    outcome: CounterOutcome,
    calls: Vec<String>,
}

async fn run(
    store: &Arc<SpendCounters>,
    proxy: &Proxy,
    step: CounterStep,
    reservation: Option<BudgetReservation>,
    request: Option<RequestRedis>,
) -> Run {
    let store = Arc::clone(store);
    let mut body = hosted(move |host| async move {
        charge_counters(&store, &host, request.as_ref(), &step, reservation.as_ref()).await
    });
    let mut calls = Vec::new();
    loop {
        match body.resume().await.unwrap() {
            CoroutineState::Yielded(HostRequest {
                call,
                request,
                reply,
            }) => {
                calls.push(call.to_owned());
                // Let a concurrent step run while this one waits on the proxy.
                tokio::task::yield_now().await;
                reply.send(
                    proxy
                        .answer(call, &request)
                        .map_err(|message| HostFault::Raised(HostException::new(message, ()))),
                );
            }
            CoroutineState::Complete(outcome) => return Run { outcome, calls },
        }
    }
}

fn counter(key: &str) -> SpendCounter {
    SpendCounter::parse(key).unwrap()
}

fn memory(seed: &[(&str, f64)]) -> Arc<SpendCounters> {
    let store = Arc::new(SpendCounters::new(None));
    for (key, value) in seed {
        store
            .set_local(&counter(key), Number::Float(*value), None)
            .unwrap();
    }
    store
}

fn snapshot(store: &SpendCounters) -> HashMap<String, f64> {
    store
        .snapshot()
        .unwrap()
        .into_iter()
        .map(|(key, value)| (key, value.as_f64()))
        .collect()
}

fn values(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), *value))
        .collect()
}

fn step(cost: f64) -> CounterStep {
    CounterStep {
        response_cost: Some(cost),
        ..CounterStep::default()
    }
}

fn reservation(entries: &[(&str, f64)], reserved_cost: f64) -> BudgetReservation {
    BudgetReservation {
        reserved_cost: Some(reserved_cost),
        entries: Some(
            entries
                .iter()
                .map(|(key, applied)| {
                    Some(ReservationEntry {
                        counter_key: Some((*key).to_owned()),
                        entity_type: Some("Key".to_owned()),
                        entity_id: Some("k".to_owned()),
                        reserved_cost: Some(reserved_cost),
                        applied_adjustment: Some(*applied),
                        ..ReservationEntry::default()
                    })
                })
                .collect(),
        ),
        finalized: Some(false),
        ..BudgetReservation::default()
    }
}

fn host_error(outcome: &CounterOutcome) -> Option<String> {
    outcome.error.as_deref().map(|error| match error {
        CounterError::Host(error) => error.exception().unwrap().message().to_owned(),
        other => panic!("expected a host failure, got {other}"),
    })
}

#[tokio::test]
async fn every_cold_counter_seeds_from_its_row_or_the_auth_cache_then_takes_the_charge() {
    let store = memory(&[]);
    let proxy = Proxy {
        stored: values(&[
            ("spend:key:k", 10.0),
            ("spend:team:t", 20.0),
            ("spend:team_member:u:t", 1.0),
            ("spend:user:u", 30.0),
            ("spend:org:o", 40.0),
            ("spend:project:p", 50.0),
        ]),
        cached: values(&[("end_user_id:e", 3.0), ("tag:a", 4.0)]),
        ..Proxy::default()
    };
    let charged = CounterStep {
        token: Some("k".into()),
        team_id: Some("t".into()),
        user_id: Some("u".into()),
        org_id: Some("o".into()),
        project_id: Some("p".into()),
        end_user_id: Some("e".into()),
        tags: Some(vec![Some("a".into()), Some("b".into()), Some("a".into())]),
        model_access_groups: Some(vec![Some("g".into())]),
        ..step(2.0)
    };

    let Run { outcome, calls } = run(&store, &proxy, charged, None, None).await;

    assert!(outcome.error.is_none());
    assert_eq!(
        snapshot(&store),
        values(&[
            ("spend:key:k", 12.0),
            ("spend:team:t", 22.0),
            ("spend:team_member:u:t", 3.0),
            ("spend:user:u", 32.0),
            ("spend:org:o", 42.0),
            ("spend:project:p", 52.0),
            ("spend:end_user:e", 5.0),
            ("spend:tag:a", 6.0),
            ("spend:tag:b", 2.0),
            ("spend:model_access_group:g", 2.0),
        ])
    );
    assert_eq!(
        calls,
        [
            "budget_windows",
            "stored_spend",
            "cached_spend",
            "budget_windows"
        ]
    );
}

#[tokio::test]
async fn a_warm_counter_takes_the_charge_without_asking_the_proxy_for_spend() {
    let store = memory(&[("spend:key:k", 1.0), ("spend:org:o", 2.0)]);
    let charged = CounterStep {
        token: Some("k".into()),
        org_id: Some("o".into()),
        ..step(0.5)
    };

    let Run { outcome, calls } = run(&store, &Proxy::default(), charged, None, None).await;

    assert!(outcome.error.is_none() && !outcome.finalized);
    assert_eq!(
        snapshot(&store),
        values(&[("spend:key:k", 1.5), ("spend:org:o", 2.5)])
    );
    assert_eq!(calls, ["budget_windows", "budget_windows"]);
}

#[tokio::test]
async fn a_raw_key_is_charged_on_its_hashed_counter() {
    let store = memory(&[]);
    let charged = CounterStep {
        token: Some("sk-raw-key".into()),
        ..step(1.0)
    };

    let _ = run(&store, &Proxy::default(), charged, None, None).await;

    assert_eq!(
        snapshot(&store),
        values(&[(
            "spend:key:7e829077433c1b2fa414ae023e40914235063fcb5cf3cb0797e835c509a9b02a",
            1.0
        )])
    );
}

#[tokio::test]
async fn a_reserved_counter_settles_through_the_reconcile_and_the_reservation_is_finalized() {
    let store = memory(&[("spend:key:k", 5.0), ("spend:team:t", 1.0)]);
    let charged = CounterStep {
        token: Some("k".into()),
        team_id: Some("t".into()),
        ..step(2.0)
    };

    let Run { outcome, .. } = run(
        &store,
        &Proxy::default(),
        charged,
        Some(reservation(&[("spend:key:k", 0.0)], 5.0)),
        None,
    )
    .await;

    assert!(outcome.error.is_none());
    assert!(outcome.finalized);
    assert_eq!(outcome.stamps, [Some(-3.0)]);
    assert_eq!(
        snapshot(&store),
        values(&[("spend:key:k", 2.0), ("spend:team:t", 3.0)])
    );
}

#[tokio::test]
async fn a_reserved_team_member_counter_whose_team_id_holds_a_colon_is_charged_once() {
    let member = "spend:team_member:alice:org:engineering";
    let store = memory(&[(member, 5.0)]);
    let charged = CounterStep {
        user_id: Some("alice".into()),
        team_id: Some("org:engineering".into()),
        ..step(2.0)
    };

    let Run { outcome, .. } = run(
        &store,
        &Proxy::default(),
        charged,
        Some(reservation(&[(member, 0.0)], 5.0)),
        None,
    )
    .await;

    assert!(outcome.error.is_none());
    assert_eq!(snapshot(&store).get(member), Some(&2.0));
}

#[tokio::test]
async fn a_zero_cost_releases_the_reservation_without_asking_the_proxy() {
    let store = memory(&[("spend:key:k", 5.0)]);
    let charged = CounterStep {
        token: Some("k".into()),
        ..step(0.0)
    };

    let Run { outcome, calls } = run(
        &store,
        &Proxy::default(),
        charged,
        Some(reservation(&[("spend:key:k", 0.0)], 5.0)),
        None,
    )
    .await;

    assert!(outcome.finalized);
    assert_eq!(outcome.stamps, [Some(-5.0)]);
    assert_eq!(snapshot(&store), values(&[("spend:key:k", 0.0)]));
    assert!(calls.is_empty());
}

#[tokio::test]
async fn a_reserved_counter_that_expired_is_reseeded_from_its_row_and_charged_the_cost() {
    let store = memory(&[]);
    let proxy = Proxy {
        stored: values(&[("spend:key:k", 10.0)]),
        ..Proxy::default()
    };
    let charged = CounterStep {
        token: Some("k".into()),
        ..step(2.0)
    };

    let Run { outcome, .. } = run(
        &store,
        &proxy,
        charged,
        Some(reservation(&[("spend:key:k", 0.0)], 5.0)),
        None,
    )
    .await;

    assert!(outcome.finalized);
    assert_eq!(snapshot(&store), values(&[("spend:key:k", 12.0)]));
}

#[tokio::test]
async fn a_failed_reconcile_reseeds_its_reserved_counter_from_the_row_before_charging_it() {
    let store = memory(&[("spend:key:k", 5.0)]);
    let proxy = Proxy {
        stored: values(&[("spend:key:k", 9.0)]),
        ..Proxy::default()
    };
    let mut held = reservation(&[("spend:key:k", 0.0)], 5.0);
    held.entries.as_mut().unwrap().push(None);
    let charged = CounterStep {
        token: Some("k".into()),
        ..step(2.0)
    };

    let Run { outcome, .. } = run(&store, &proxy, charged, Some(held), None).await;

    assert!(outcome.error.is_none());
    assert!(outcome.finalized);
    assert_eq!(snapshot(&store), values(&[("spend:key:k", 11.0)]));
}

#[tokio::test]
async fn an_entry_whose_window_reset_before_settlement_is_stamped_at_nothing_spent() {
    let store = memory(&[("spend:key:k", 5.0)]);
    let charged = CounterStep {
        token: Some("k".into()),
        window_resets: vec![Some("2020-01-01T00:00:00+00:00".into())],
        ..step(2.0)
    };

    let Run { outcome, .. } = run(
        &store,
        &Proxy::default(),
        charged,
        Some(reservation(&[("spend:key:k", 0.0)], 5.0)),
        None,
    )
    .await;

    assert!(outcome.finalized);
    assert_eq!(outcome.stamps, [Some(-5.0)]);
}

#[tokio::test]
async fn an_already_finalized_reservation_charges_its_counters_directly() {
    let store = memory(&[("spend:key:k", 5.0)]);
    let mut finalized = reservation(&[("spend:key:k", 0.0)], 5.0);
    finalized.finalized = Some(true);
    let charged = CounterStep {
        token: Some("k".into()),
        ..step(2.0)
    };

    let Run { outcome, .. } = run(&store, &Proxy::default(), charged, Some(finalized), None).await;

    assert!(outcome.finalized);
    assert_eq!(snapshot(&store), values(&[("spend:key:k", 7.0)]));
}

#[tokio::test]
async fn a_failed_sequential_scope_drops_its_charges_while_every_other_scope_is_charged() {
    let store = memory(&[
        ("spend:key:k", 1.0),
        ("spend:user:u", 1.0),
        ("spend:tag:a", 1.0),
        ("spend:team_member:u:t", 1.0),
    ]);
    let proxy = Proxy {
        failing: HashSet::from(["stored_spend"]),
        ..Proxy::default()
    };
    let charged = CounterStep {
        token: Some("k".into()),
        team_id: Some("t".into()),
        user_id: Some("u".into()),
        end_user_id: Some("e".into()),
        tags: Some(vec![Some("a".into())]),
        ..step(2.0)
    };

    let Run { outcome, .. } = run(&store, &proxy, charged, Some(reservation(&[], 0.0)), None).await;

    assert_eq!(
        host_error(&outcome).as_deref(),
        Some("stored_spend is down")
    );
    assert!(!outcome.finalized);
    assert_eq!(
        snapshot(&store),
        values(&[
            ("spend:key:k", 3.0),
            ("spend:user:u", 3.0),
            ("spend:tag:a", 3.0),
            ("spend:team_member:u:t", 3.0),
            ("spend:end_user:e", 2.0),
        ])
    );
}

#[tokio::test]
async fn a_failed_concurrent_task_keeps_its_siblings_charges() {
    let store = memory(&[("spend:end_user:e", 1.0)]);
    let proxy = Proxy {
        failing: HashSet::from(["cached_spend"]),
        ..Proxy::default()
    };
    let charged = CounterStep {
        end_user_id: Some("e".into()),
        tags: Some(vec![Some("a".into())]),
        ..step(2.0)
    };

    let Run { outcome, .. } = run(&store, &proxy, charged, None, None).await;

    assert_eq!(
        host_error(&outcome).as_deref(),
        Some("cached_spend is down")
    );
    assert_eq!(snapshot(&store), values(&[("spend:end_user:e", 3.0)]));
}

#[tokio::test]
async fn budget_windows_are_charged_from_the_auth_cache_then_from_the_stored_row() {
    let store = memory(&[("spend:key:k", 1.0), ("spend:user:u", 1.0)]);
    let proxy = Proxy {
        windows: HashMap::from([(
            "k".to_owned(),
            vec![
                ("1d".to_owned(), Some(START.to_owned())),
                ("7d".to_owned(), None),
            ],
        )]),
        stored_windows: HashMap::from([(
            "u".to_owned(),
            vec![("30d".to_owned(), Some(START.to_owned()))],
        )]),
        stored: values(&[("Key:k:1d", 4.0), ("User:u:30d", 1.0)]),
        ..Proxy::default()
    };
    let charged = CounterStep {
        token: Some("k".into()),
        user_id: Some("u".into()),
        ..step(2.0)
    };

    let Run { outcome, calls } = run(&store, &proxy, charged, None, None).await;

    assert!(outcome.error.is_none());
    assert_eq!(
        snapshot(&store),
        values(&[
            ("spend:key:k", 3.0),
            ("spend:user:u", 3.0),
            ("spend:key:k:window:1d", 6.0),
            ("spend:user_window:u:30d", 3.0),
        ])
    );
    assert_eq!(
        calls,
        [
            "budget_windows",
            "stored_spend",
            "budget_windows",
            "stored_spend"
        ]
    );
}

#[tokio::test]
async fn a_window_with_no_stored_spend_or_a_reservation_is_not_charged() {
    let store = memory(&[("spend:team:t", 1.0), ("spend:team:t:window:7d", 1.0)]);
    let proxy = Proxy {
        windows: HashMap::from([(
            "team_id:t".to_owned(),
            vec![
                ("1d".to_owned(), Some(START.to_owned())),
                ("7d".to_owned(), Some(START.to_owned())),
            ],
        )]),
        ..Proxy::default()
    };
    let mut held = reservation(&[("spend:team:t:window:7d", 0.0)], 1.0);
    held.finalized = Some(false);
    let charged = CounterStep {
        team_id: Some("t".into()),
        ..step(1.0)
    };

    let Run { outcome, .. } = run(&store, &proxy, charged, Some(held), None).await;

    assert!(outcome.finalized);
    assert_eq!(
        snapshot(&store),
        values(&[("spend:team:t", 2.0), ("spend:team:t:window:7d", 1.0)])
    );
}

#[rstest]
#[case::key("spend:key:k", Some(9.0))]
#[case::end_user("spend:end_user:e", Some(9.0))]
#[case::tag("spend:tag:x", None)]
#[case::window("spend:team:t:window:1d", None)]
#[tokio::test]
async fn only_a_tracked_lifetime_counter_charges_its_committed_twin(
    #[case] tracked: &str,
    #[case] twin: Option<f64>,
) {
    let store = memory(&[(tracked, 1.0)]);
    let proxy = Proxy {
        stored: values(&[("spend:key:k", 7.0), ("spend:end_user:e", 7.0)]),
        ..Proxy::default()
    };
    let charged = CounterStep {
        committed_tracked: vec![counter(tracked)],
        ..step(2.0)
    };

    let Run { outcome, .. } = run(&store, &proxy, charged, None, None).await;

    assert!(outcome.error.is_none());
    let twin_key = format!("spend:committed:{}", &tracked["spend:".len()..]);
    assert_eq!(snapshot(&store).get(&twin_key).copied(), twin);
}

#[tokio::test]
async fn customer_scopes_and_windows_are_charged_seeding_the_cold_ones() {
    let store = memory(&[("spend:customer_team:ct:lifetime", 2.0)]);
    let proxy = Proxy {
        stored: values(&[("customer_org:c:1mo", 1.0)]),
        ..Proxy::default()
    };
    let charged = CounterStep {
        customer_scopes: vec![
            CustomerScopeSpend {
                counter_key: counter("spend:customer_org:c:lifetime"),
                spend: 5.0,
            },
            CustomerScopeSpend {
                counter_key: counter("spend:customer_team:ct:lifetime"),
                spend: 1.0,
            },
        ],
        customer_windows: vec![CustomerWindow {
            counter_key: counter("spend:customer_org:c:window:1mo"),
            entity_type: "customer_org".into(),
            entity_id: "c".into(),
            window_duration: Some("1mo".into()),
            window_start: Some(START.into()),
        }],
        ..step(2.0)
    };

    let Run { outcome, .. } = run(&store, &proxy, charged, None, None).await;

    assert!(outcome.error.is_none());
    assert_eq!(
        snapshot(&store),
        values(&[
            ("spend:customer_org:c:lifetime", 7.0),
            ("spend:customer_team:ct:lifetime", 4.0),
            ("spend:customer_org:c:window:1mo", 3.0),
        ])
    );
}

#[derive(Default)]
struct RoundTrips(Mutex<Vec<Operation>>);

impl RedisEvents for RoundTrips {
    fn call(&self, event: &CallEvent) {
        self.0.lock().unwrap().push(event.operation);
    }

    fn breaker(&self, _transition: &Transition<State>) {}
}

fn redis(server: &RedisServer, trips: Option<Arc<RoundTrips>>) -> Arc<SpendCounters> {
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
        events: trips.map(|trips| trips as Arc<dyn RedisEvents>),
        health_check: None,
        keepalive: true,
    })
    .unwrap();
    Arc::new(SpendCounters::new(Some(Remote {
        client,
        default_ttl: Duration::from_secs(60),
    })))
}

#[tokio::test]
async fn on_redis_the_warm_checks_share_one_mget_and_the_charges_one_pipeline() {
    let server = RedisServer::start().await.unwrap();
    let seeding = redis(&server, None);
    for key in [
        "spend:key:k",
        "spend:team:t",
        "spend:user:u",
        "spend:customer_org:c:lifetime",
    ] {
        seeding
            .set(&counter(key), Number::Float(1.0), None)
            .await
            .unwrap();
    }
    let trips = Arc::new(RoundTrips::default());
    let store = redis(&server, Some(Arc::clone(&trips)));
    let charged = CounterStep {
        token: Some("k".into()),
        team_id: Some("t".into()),
        user_id: Some("u".into()),
        customer_scopes: vec![CustomerScopeSpend {
            counter_key: counter("spend:customer_org:c:lifetime"),
            spend: 5.0,
        }],
        ..step(2.0)
    };
    let proxy = Proxy {
        cached: values(&[("team_membership:u:t", 0.0)]),
        ..Proxy::default()
    };

    let Run { outcome, .. } = run(&store, &proxy, charged, None, None).await;

    assert!(outcome.error.is_none());
    assert_eq!(
        *trips.0.lock().unwrap(),
        [Operation::Command("MGET"), Operation::Pipeline { ops: 5 }]
    );
    assert_eq!(
        snapshot(&store),
        values(&[
            ("spend:customer_org:c:lifetime", 3.0),
            ("spend:key:k", 3.0),
            ("spend:team:t", 3.0),
            ("spend:user:u", 3.0),
            ("spend:team_member:u:t", 2.0),
        ])
    );
}

#[tokio::test]
async fn a_charge_on_the_request_rides_its_post_call_pipeline_and_reports_committed_twins() {
    let server = RedisServer::start().await.unwrap();
    let store = redis(&server, None);
    for key in ["spend:key:k", "spend:committed:key:k"] {
        store
            .set(&counter(key), Number::Float(1.0), None)
            .await
            .unwrap();
    }
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));
    let charged = CounterStep {
        token: Some("k".into()),
        committed_tracked: vec![counter("spend:key:k")],
        ..step(2.0)
    };

    let Run { outcome, .. } = run(
        &store,
        &Proxy::default(),
        charged,
        None,
        Some(request.clone()),
    )
    .await;
    assert!(outcome.error.is_none());
    assert!(!request.tracked_writes_settled());
    request.flush_post().await;
    litellm_redis_testing::eventually(async || request.tracked_writes_settled().then_some(()))
        .await;

    assert_eq!(
        snapshot(&store),
        values(&[("spend:key:k", 3.0), ("spend:committed:key:k", 3.0)])
    );
}

#[tokio::test]
async fn a_pipeline_that_fails_drops_every_counter_it_carried_and_reports_the_failure() {
    let server = RedisServer::start().await.unwrap();
    let store = redis(&server, None);
    store
        .set(&counter("spend:key:k"), Number::Float(1.0), None)
        .await
        .unwrap();
    server.stop().await.unwrap();
    let charged = CounterStep {
        token: Some("k".into()),
        ..step(2.0)
    };

    let Run { outcome, .. } = run(
        &store,
        &Proxy::default(),
        charged,
        Some(reservation(&[], 0.0)),
        None,
    )
    .await;

    assert!(matches!(
        outcome.error.as_deref(),
        Some(CounterError::Store(_))
    ));
    assert!(!outcome.finalized);
    assert!(snapshot(&store).is_empty());
}

#[tokio::test]
async fn a_customer_counter_that_cannot_be_seeded_fails_the_step_with_the_rest() {
    let server = RedisServer::start().await.unwrap();
    let store = redis(&server, None);
    server.stop().await.unwrap();
    let charged = CounterStep {
        token: Some("k".into()),
        customer_scopes: vec![CustomerScopeSpend {
            counter_key: counter("spend:customer_org:c:lifetime"),
            spend: 5.0,
        }],
        ..step(2.0)
    };

    let Run { outcome, .. } = run(
        &store,
        &Proxy::default(),
        charged,
        Some(reservation(&[("spend:key:k", 0.0)], 5.0)),
        None,
    )
    .await;

    assert!(matches!(
        outcome.error.as_deref(),
        Some(CounterError::Store(_))
    ));
    assert!(!outcome.finalized);
    assert_eq!(outcome.stamps, [None]);
    assert!(snapshot(&store).is_empty());
}

#[tokio::test]
async fn concurrent_charges_of_one_cold_counter_read_its_stored_spend_once() {
    let store = memory(&[]);
    let proxy = Proxy {
        stored: values(&[("spend:key:k", 10.0)]),
        ..Proxy::default()
    };
    let charged = || CounterStep {
        token: Some("k".into()),
        ..step(2.0)
    };

    let (first, second) = tokio::join!(
        run(&store, &proxy, charged(), None, None),
        run(&store, &proxy, charged(), None, None),
    );

    assert!(first.outcome.error.is_none() && second.outcome.error.is_none());
    let stored_reads = [first.calls, second.calls]
        .iter()
        .flatten()
        .filter(|call| *call == "stored_spend")
        .count();
    assert_eq!(stored_reads, 1);
    assert_eq!(snapshot(&store).get("spend:key:k").copied(), Some(14.0));
}
