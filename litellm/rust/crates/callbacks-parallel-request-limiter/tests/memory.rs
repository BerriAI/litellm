use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_callbacks::Abort;
use litellm_callbacks_parallel_request_limiter::{
    Candidate, Code, Ctx, Descriptor, Engine, FailureFacts, LimiterState, LoggedCall, ModelFacts,
    ParallelRequestLimiter, Policy, PostCallFailureFacts, PreCallFacts, PriorResponse,
    RateLimitedModel, Resolved, Settings, SocketClose, SuccessFacts, TargetFacts,
    TokenRateLimitType,
};
use litellm_redis_client::{
    Credentials, Namespace, RedisClient, RedisConfig, RedisUrl, Script, Target, Timeouts, commands,
};
use litellm_redis_store::{
    Number, Remote,
    rate_limits::{Counter, Entry, Increment, Kind, Local, RateLimits, Scope},
};
use litellm_redis_testing::RedisServer;
use litellm_secret_value::SecretValue;
use rstest::rstest;
use serde_json::json;

struct Clock(AtomicU64);

impl Clock {
    fn at(seconds: f64) -> Arc<Self> {
        Arc::new(Self(AtomicU64::new(seconds.to_bits())))
    }

    fn now(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::SeqCst))
    }
}

fn limiter(reservation: bool) -> (ParallelRequestLimiter, Arc<RateLimits>, Arc<Clock>) {
    limiter_over(reservation, |_| RateLimits::new(None, None))
}

fn limiter_whose_memory_shares_its_clock(
    reservation: bool,
) -> (ParallelRequestLimiter, Arc<RateLimits>, Arc<Clock>) {
    limiter_over(reservation, |clock| {
        RateLimits::with_clock(
            None,
            None,
            Arc::new(move || Duration::from_secs_f64(clock.now())),
        )
    })
}

fn limiter_over(
    reservation: bool,
    store: impl FnOnce(Arc<Clock>) -> RateLimits,
) -> (ParallelRequestLimiter, Arc<RateLimits>, Arc<Clock>) {
    let clock = Clock::at(1_700_000_000.5);
    let store = Arc::new(store(Arc::clone(&clock)));
    let now = Arc::clone(&clock);
    let engine = Engine::new(
        Arc::clone(&store),
        Settings {
            window_size: 60,
            tpm_reservation_enabled: reservation,
        },
        Arc::new(move || now.now()),
    );
    (ParallelRequestLimiter::new(engine), store, clock)
}

fn descriptors(value: serde_json::Value) -> Vec<Descriptor> {
    serde_json::from_value(value).unwrap()
}

fn facts(descriptors: Vec<Descriptor>, estimated: Option<i64>) -> PreCallFacts {
    PreCallFacts {
        call_id: Some("call".into()),
        descriptors,
        requested_model: Some("gpt".into()),
        estimated_tokens: estimated,
        reserved_model: None,
        ..PreCallFacts::default()
    }
}

fn count(store: &RateLimits, key: &str, value: &str, kind: Kind) -> Option<Local> {
    store
        .get_local(&Entry::Counter(Counter {
            scope: Scope {
                descriptor: key.into(),
                value: value.into(),
            },
            kind,
        }))
        .unwrap()
}

#[rstest]
#[tokio::test]
async fn the_rpm_limit_admits_up_to_the_limit_then_refuses_with_the_python_detail() {
    let (limiter, _, _) = limiter(true);
    let rpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"requests_per_unit": 2}}]),
    );
    for _ in 0..2 {
        let state = LimiterState::default();
        assert_eq!(
            limiter
                .pre_call(&state, facts(rpm.clone(), None), None)
                .await,
            None
        );
    }
    let refused: Abort = limiter
        .pre_call(&LimiterState::default(), facts(rpm, None), None)
        .await
        .unwrap();
    assert_eq!(refused.status, 429);
    assert_eq!(
        refused.message,
        "Rate limit exceeded for api_key: k. Limit type: requests. Current limit: 2, Remaining: 0. Limit resets at: 2023-11-14 22:14:20 UTC"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn taking_the_slot_waits_for_a_release_in_flight() {
    let (limiter, store, _) = limiter(true);
    let limiter = Arc::new(limiter);
    let parallel = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"max_parallel_requests": 2}}]),
    );
    let state = Arc::new(LimiterState::default());
    assert_eq!(
        limiter.pre_call(&state, facts(parallel, None), None).await,
        None
    );
    let guard = store.memory_guard();
    let releasing = tokio::spawn({
        let (limiter, state) = (Arc::clone(&limiter), Arc::clone(&state));
        async move { limiter.release_on_disconnect(&state).await }
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let taking = tokio::spawn({
        let (limiter, state) = (Arc::clone(&limiter), Arc::clone(&state));
        async move { limiter.take_parallel_slot(&state).await }
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    drop(guard);
    releasing.await.unwrap();
    assert_eq!(taking.await.unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn a_parallel_slot_is_held_until_post_call_success_releases_it() {
    let (limiter, _, _) = limiter(true);
    let parallel = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"max_parallel_requests": 1}}]),
    );
    let first = LimiterState::default();
    assert_eq!(
        limiter
            .pre_call(&first, facts(parallel.clone(), None), None)
            .await,
        None
    );
    let refused = limiter
        .pre_call(
            &LimiterState::default(),
            facts(parallel.clone(), None),
            None,
        )
        .await
        .unwrap();
    assert_eq!(refused.headers["rate_limit_type"], "max_parallel_requests");
    limiter.post_call_success(&first, Default::default()).await;
    assert_eq!(
        limiter
            .pre_call(&LimiterState::default(), facts(parallel, None), None)
            .await,
        None
    );
}

#[rstest]
#[tokio::test]
async fn a_reservation_is_settled_to_actual_usage_and_refunded_on_failure() {
    let (limiter, store, _) = limiter(true);
    let tpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 1000}}]),
    );
    let state = LimiterState::default();
    assert_eq!(
        limiter
            .pre_call(&state, facts(tpm.clone(), Some(300)), None)
            .await,
        None
    );
    assert_eq!(
        count(&store, "api_key", "k", Kind::Tokens),
        Some(Local::Number(Number::Int(300)))
    );
    limiter
        .settle_success(
            &state,
            SuccessFacts {
                call_id: Some("call".into()),
                total_tokens: 120,
                targets: TargetFacts {
                    user_api_key_hash: Some("k".into()),
                    ..TargetFacts::default()
                },
                ..SuccessFacts::default()
            },
            None,
        )
        .await;
    assert_eq!(
        count(&store, "api_key", "k", Kind::Tokens),
        Some(Local::Number(Number::Int(120)))
    );

    let failed = LimiterState::default();
    assert_eq!(
        limiter.pre_call(&failed, facts(tpm, Some(500)), None).await,
        None
    );
    limiter
        .post_call_failure(&failed, PostCallFailureFacts::default())
        .await;
    assert_eq!(
        count(&store, "api_key", "k", Kind::Tokens),
        Some(Local::Number(Number::Int(120)))
    );
}

fn key_usage(tokens: i64) -> SuccessFacts {
    SuccessFacts {
        call_id: Some("call".into()),
        total_tokens: tokens,
        targets: TargetFacts {
            user_api_key_hash: Some("k".into()),
            ..TargetFacts::default()
        },
        ..SuccessFacts::default()
    }
}

#[rstest]
#[case::outage_then_refusal_reserved(true, false, 0)]
#[case::refusal_then_refusal_reserved(true, true, 777)]
#[case::outage_then_refusal_unreserved(false, false, 0)]
#[case::refusal_then_refusal_unreserved(false, true, 777)]
#[tokio::test]
async fn every_billed_refusal_of_a_fallback_chain_counts_toward_key_tpm(
    #[case] reservation: bool,
    #[case] first_hop_billed: bool,
    #[case] first_hop_charged: i64,
) {
    let (limiter, store, _) = limiter(reservation);
    let tpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 10_000}}]),
    );
    let state = LimiterState::default();
    assert_eq!(
        limiter.pre_call(&state, facts(tpm, Some(300)), None).await,
        None
    );
    let counted_at_admission = i64::from(!reservation);

    if first_hop_billed {
        limiter
            .settle_billed_failure(&state, key_usage(777), None)
            .await;
    } else {
        limiter
            .settle_failure(
                &state,
                FailureFacts {
                    call_id: Some("call".into()),
                    ..FailureFacts::default()
                },
                None,
            )
            .await;
    }
    limiter
        .settle_billed_failure(&state, key_usage(900), None)
        .await;
    limiter.settle_success(&state, key_usage(1000), None).await;

    assert_eq!(
        count(&store, "api_key", "k", Kind::Tokens),
        Some(Local::Number(Number::Int(
            counted_at_admission + first_hop_charged + 900 + 1000
        )))
    );
}

#[rstest]
#[tokio::test]
async fn a_refused_fallback_hop_counts_toward_its_own_model_limit_on_the_key() {
    let (limiter, store, _) = limiter(true);
    let tpm = descriptors(json!([
        {"key": "model_per_key", "value": "k:gpt", "rate_limit": {"tokens_per_unit": 10_000}},
    ]));
    let state = LimiterState::default();
    let reserved = PreCallFacts {
        reserved_model: Some(RateLimitedModel {
            requested: "gpt".into(),
            group: "gpt".into(),
        }),
        ..facts(tpm, Some(300))
    };
    assert_eq!(limiter.pre_call(&state, reserved, None).await, None);
    state.stash().fallback_models.insert("hop".into());
    limiter
        .settle_failure(
            &state,
            FailureFacts {
                call_id: Some("call".into()),
                ..FailureFacts::default()
            },
            None,
        )
        .await;
    limiter
        .settle_billed_failure(
            &state,
            SuccessFacts {
                served_model: Some(RateLimitedModel {
                    requested: "hop".into(),
                    group: "hop".into(),
                }),
                models: [(
                    "hop".to_owned(),
                    ModelFacts {
                        candidates: vec![Candidate {
                            group: "hop".into(),
                            key_owns: false,
                            counts_cached_tokens: false,
                        }],
                        ptu: None,
                    },
                )]
                .into(),
                ..key_usage(900)
            },
            None,
        )
        .await;

    assert_eq!(
        count(&store, "model_per_key", "k:hop", Kind::Tokens),
        Some(Local::Number(Number::Int(900)))
    );
    assert_eq!(
        count(&store, "model_per_key", "k:gpt", Kind::Tokens),
        Some(Local::Number(Number::Int(0)))
    );
}

fn billed_refusal(prompt_tokens: i64) -> LoggedCall {
    serde_json::from_value(json!({
        "call_id": "call",
        "provider_billed": true,
        "combined": {"kind": "usage", "usage": {"kind": "chat", "members": {
            "prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens,
        }}},
    }))
    .unwrap()
}

#[rstest]
#[case::reserved(true)]
#[case::unreserved(false)]
#[tokio::test]
async fn every_billed_refusal_of_a_fallback_chain_counts_toward_project_itpm_under_an_output_limit(
    #[case] reservation: bool,
) {
    let (limiter, store, _) = limiter(reservation);
    let admitted: PreCallFacts = serde_json::from_value(json!({
        "call_id": "call",
        "requested_model": "gpt",
        "descriptors": [{"key": "model_per_project_itpm", "value": "p:gpt",
                         "rate_limit": {"tokens_per_unit": 10_000}}],
        "io_estimate": {"input_tokens": 300, "output_tokens": 50},
    }))
    .unwrap();
    let state = LimiterState::default();
    assert_eq!(limiter.pre_call(&state, admitted, None).await, None);

    for prompt_tokens in [777, 900] {
        limiter
            .log_failure(
                Some(&state),
                false,
                billed_refusal(prompt_tokens),
                &Resolved::default(),
                TokenRateLimitType::Output,
                None,
            )
            .await;
    }

    assert_eq!(
        count(&store, "model_per_project_itpm", "p:gpt", Kind::Tokens),
        Some(Local::Number(Number::Float(777.0 + 900.0)))
    );
}

#[rstest]
#[case::no_window_open(None, 1092)]
#[case::window_open(Some(600), 500)]
#[tokio::test]
async fn an_unreserved_charge_counts_toward_the_next_reservation(
    #[case] admitted_first: Option<i64>,
    #[case] charged: i64,
) {
    let (limiter, _, _) = limiter(true);
    let tpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 1000}}]),
    );
    if let Some(reserved) = admitted_first {
        assert_eq!(
            limiter
                .pre_call(
                    &LimiterState::default(),
                    facts(tpm.clone(), Some(reserved)),
                    None
                )
                .await,
            None
        );
    }
    limiter
        .settle_turn(
            &LimiterState::default(),
            SuccessFacts {
                total_tokens: charged,
                targets: TargetFacts {
                    user_api_key_hash: Some("k".into()),
                    ..TargetFacts::default()
                },
                ..SuccessFacts::default()
            },
            None,
        )
        .await;

    let refused = limiter
        .pre_call(&LimiterState::default(), facts(tpm, Some(10)), None)
        .await
        .expect("admitted past tpm_limit");
    assert_eq!(refused.status, 429);
    assert_eq!(refused.headers["rate_limit_type"], "tokens");
}

#[rstest]
#[tokio::test]
async fn a_refund_into_no_window_cannot_offset_a_later_charge() {
    let (limiter, _, _) = limiter(true);
    limiter
        .engine()
        .refund_reserved(
            &[("api_key".into(), "k".into())],
            50,
            &[],
            Ctx {
                policy: Policy::default(),
                request: None,
            },
        )
        .await;
    limiter
        .settle_turn(
            &LimiterState::default(),
            SuccessFacts {
                total_tokens: 1092,
                targets: TargetFacts {
                    user_api_key_hash: Some("k".into()),
                    ..TargetFacts::default()
                },
                ..SuccessFacts::default()
            },
            None,
        )
        .await;

    let tpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 1080}}]),
    );
    assert!(
        limiter
            .pre_call(&LimiterState::default(), facts(tpm, Some(10)), None)
            .await
            .is_some()
    );
}

#[rstest]
#[tokio::test]
async fn a_failure_settled_after_its_window_expired_counts_toward_the_next_reservation() {
    let (limiter, _, clock) = limiter_whose_memory_shares_its_clock(true);
    let tpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 1000}}]),
    );
    let failed = LimiterState::default();
    assert_eq!(
        limiter
            .pre_call(&failed, facts(tpm.clone(), Some(10)), None)
            .await,
        None
    );
    clock
        .0
        .store((1_700_000_090.5_f64).to_bits(), Ordering::SeqCst);
    limiter
        .post_call_failure(
            &failed,
            PostCallFailureFacts {
                tpm_actual: 1092,
                ..PostCallFailureFacts::default()
            },
        )
        .await;

    let refused = limiter
        .pre_call(&LimiterState::default(), facts(tpm, Some(10)), None)
        .await
        .expect("admitted past tpm_limit");
    assert_eq!(refused.headers["rate_limit_type"], "tokens");
}

#[rstest]
#[tokio::test]
async fn a_charge_stays_counted_while_a_hundred_other_keys_are_charged_in_its_window() {
    let (limiter, _, clock) = limiter_whose_memory_shares_its_clock(true);
    let charge = |key: String| SuccessFacts {
        total_tokens: 1092,
        targets: TargetFacts {
            user_api_key_hash: Some(key),
            ..TargetFacts::default()
        },
        ..SuccessFacts::default()
    };
    limiter
        .settle_turn(&LimiterState::default(), charge("k".into()), None)
        .await;
    clock
        .0
        .store((1_700_000_001.5_f64).to_bits(), Ordering::SeqCst);
    for other in 0..100 {
        limiter
            .settle_turn(
                &LimiterState::default(),
                charge(format!("other-{other}")),
                None,
            )
            .await;
    }

    let tpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 1000}}]),
    );
    let refused = limiter
        .pre_call(&LimiterState::default(), facts(tpm, Some(10)), None)
        .await
        .expect("admitted past tpm_limit");
    assert_eq!(refused.headers["rate_limit_type"], "tokens");
}

#[rstest]
#[tokio::test]
async fn a_window_opened_over_a_leftover_counter_holds_the_charge_for_the_whole_window() {
    let (limiter, store, clock) = limiter_whose_memory_shares_its_clock(true);
    store
        .set_local(
            &Entry::Counter(Counter {
                scope: Scope {
                    descriptor: "api_key".into(),
                    value: "k".into(),
                },
                kind: Kind::Tokens,
            }),
            Local::Number(Number::Int(5)),
            Some(Duration::from_secs(60)),
        )
        .unwrap();
    clock
        .0
        .store((1_700_000_030.5_f64).to_bits(), Ordering::SeqCst);
    limiter
        .settle_turn(
            &LimiterState::default(),
            SuccessFacts {
                total_tokens: 1092,
                targets: TargetFacts {
                    user_api_key_hash: Some("k".into()),
                    ..TargetFacts::default()
                },
                ..SuccessFacts::default()
            },
            None,
        )
        .await;
    clock
        .0
        .store((1_700_000_065.5_f64).to_bits(), Ordering::SeqCst);

    let tpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 1000}}]),
    );
    let refused = limiter
        .pre_call(&LimiterState::default(), facts(tpm, Some(10)), None)
        .await
        .expect("admitted past tpm_limit");
    assert_eq!(refused.headers["rate_limit_type"], "tokens");
}

#[rstest]
#[case::cached_tokens_left_out(false, 120)]
#[case::cached_tokens_counted(true, 920)]
#[tokio::test]
async fn a_team_model_limit_counts_cached_tokens_only_when_the_team_asks(
    #[case] counts_cached_tokens: bool,
    #[case] team_model_tokens: i64,
) {
    let (limiter, store, _) = limiter(true);
    let tpm = descriptors(json!([
        {"key": "team", "value": "t", "rate_limit": {"tokens_per_unit": 5000}},
        {"key": "model_per_team", "value": "t:gpt", "rate_limit": {"tokens_per_unit": 5000}},
    ]));
    let state = LimiterState::default();
    assert_eq!(
        limiter.pre_call(&state, facts(tpm, Some(300)), None).await,
        None
    );
    let model = RateLimitedModel {
        requested: "gpt".into(),
        group: "gpt".into(),
    };
    limiter
        .settle_success(
            &state,
            SuccessFacts {
                call_id: Some("call".into()),
                total_tokens: 120,
                cached_tokens_left_out: 800,
                team_label: "t".into(),
                targets: TargetFacts {
                    user_api_key_team_id: Some("t".into()),
                    ..TargetFacts::default()
                },
                served_model: Some(model),
                models: [(
                    "gpt".to_owned(),
                    ModelFacts {
                        candidates: vec![Candidate {
                            group: "gpt".into(),
                            key_owns: false,
                            counts_cached_tokens,
                        }],
                        ptu: None,
                    },
                )]
                .into(),
                ..SuccessFacts::default()
            },
            None,
        )
        .await;
    assert_eq!(
        count(&store, "team", "t", Kind::Tokens),
        Some(Local::Number(Number::Int(120)))
    );
    assert_eq!(
        count(&store, "model_per_team", "t:gpt", Kind::Tokens),
        Some(Local::Number(Number::Int(team_model_tokens)))
    );
}

#[rstest]
#[tokio::test]
async fn a_fallback_sharing_the_reserved_team_model_scope_settles_cached_tokens_once() {
    let (limiter, store, _) = limiter(true);
    let tpm = descriptors(json!([
        {"key": "model_per_team", "value": "t:gpt", "rate_limit": {"tokens_per_unit": 5000}},
    ]));
    let state = LimiterState::default();
    let reserved = PreCallFacts {
        reserved_model: Some(RateLimitedModel {
            requested: "alias".into(),
            group: "gpt".into(),
        }),
        ..facts(tpm, Some(1000))
    };
    assert_eq!(limiter.pre_call(&state, reserved, None).await, None);
    state.stash().fallback_models.insert("target".into());
    limiter
        .settle_success(
            &state,
            SuccessFacts {
                call_id: Some("call".into()),
                total_tokens: 120,
                cached_tokens_left_out: 800,
                team_label: "t".into(),
                targets: TargetFacts {
                    user_api_key_team_id: Some("t".into()),
                    ..TargetFacts::default()
                },
                served_model: Some(RateLimitedModel {
                    requested: "target".into(),
                    group: "gpt".into(),
                }),
                models: [(
                    "target".to_owned(),
                    ModelFacts {
                        candidates: vec![Candidate {
                            group: "gpt".into(),
                            key_owns: false,
                            counts_cached_tokens: true,
                        }],
                        ptu: None,
                    },
                )]
                .into(),
                ..SuccessFacts::default()
            },
            None,
        )
        .await;
    assert_eq!(
        count(&store, "model_per_team", "t:gpt", Kind::Tokens),
        Some(Local::Number(Number::Int(920)))
    );
}

#[rstest]
#[tokio::test]
async fn a_tpm_refusal_releases_the_slot_and_names_tokens() {
    let (limiter, _, _) = limiter(true);
    let both = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 100, "max_parallel_requests": 1}}]),
    );
    let state = LimiterState::default();
    let refused = limiter
        .pre_call(&state, facts(both.clone(), Some(500)), None)
        .await
        .unwrap();
    assert_eq!(refused.headers["rate_limit_type"], "tokens");
    assert!(state.snapshot().parallel_slot.is_none());
    assert!(state.snapshot().first_pass_admitted);
    assert_eq!(
        limiter
            .pre_call(&LimiterState::default(), facts(both, Some(50)), None)
            .await,
        None
    );
}

#[rstest]
#[tokio::test]
async fn the_memory_precheck_refuses_after_the_window_until_the_memory_entry_expires() {
    let (limiter, _, clock) = limiter(true);
    let rpm =
        descriptors(json!([{"key": "user", "value": "u", "rate_limit": {"requests_per_unit": 1}}]));
    assert_eq!(
        limiter
            .pre_call(&LimiterState::default(), facts(rpm.clone(), None), None)
            .await,
        None
    );
    assert!(
        limiter
            .pre_call(&LimiterState::default(), facts(rpm.clone(), None), None)
            .await
            .is_some()
    );
    clock
        .0
        .store((1_700_000_061.0_f64).to_bits(), Ordering::SeqCst);
    assert!(
        limiter
            .pre_call(&LimiterState::default(), facts(rpm, None), None)
            .await
            .is_some()
    );
}

#[rstest]
#[tokio::test]
async fn the_statuses_reach_the_logging_payload_headers() {
    let (limiter, _, _) = limiter(true);
    let rpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"requests_per_unit": 5}}]),
    );
    let state = Arc::new(LimiterState::default());
    assert_eq!(limiter.pre_call(&state, facts(rpm, None), None).await, None);
    let response = state.snapshot().rate_limit_response.unwrap();
    assert_eq!(response.overall, Code::Ok);
    let headers = litellm_callbacks_parallel_request_limiter::header_map(&response.statuses);
    assert_eq!(headers["x-ratelimit-api_key-remaining-requests"], json!(4));
}

#[rstest]
#[case::no_descriptors_keep_it(json!([]), false, vec!["dynamic"])]
#[case::a_reused_slot_keeps_it(json!([{"key": "api_key", "value": "k", "rate_limit": {"requests_per_unit": 5}}]), true, vec!["dynamic"])]
#[case::a_reused_slot_extends_it_with_tokens(json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 500}}]), true, vec!["dynamic", "api_key"])]
#[case::a_fresh_first_pass_replaces_it(json!([{"key": "api_key", "value": "k", "rate_limit": {"requests_per_unit": 5}}]), false, vec!["api_key"])]
#[tokio::test]
async fn the_response_another_limiter_stored_is_kept_like_the_python_stash(
    #[case] limits: serde_json::Value,
    #[case] reuses_request_slot: bool,
    #[case] keys: Vec<&str>,
) {
    let (limiter, _, _) = limiter(true);
    let prior: PriorResponse = serde_json::from_value(json!({
        "overall_code": "OK",
        "statuses": [{"code": "OK", "current_limit": 9, "limit_remaining": 8,
                      "rate_limit_type": "requests", "descriptor_key": "dynamic"}],
    }))
    .unwrap();
    let state = Arc::new(LimiterState::default());
    let admitted = PreCallFacts {
        reuses_request_slot,
        prior_response: Some(prior),
        ..facts(descriptors(limits), Some(10))
    };
    assert_eq!(limiter.pre_call(&state, admitted, None).await, None);
    let statuses = state.snapshot().rate_limit_response.unwrap().statuses;
    assert_eq!(
        statuses
            .iter()
            .map(|status| status.descriptor_key.as_str())
            .collect::<Vec<_>>(),
        keys
    );
}

fn key_tokens(store: &RateLimits) -> Option<f64> {
    match count(store, "api_key", "k", Kind::Tokens)? {
        Local::Number(number) => Some(number.as_f64()),
        _ => None,
    }
}

#[rstest]
#[case::turns_finished(SocketClose::TurnsFinished, 0.0)]
#[case::relay_failed(SocketClose::RelayFailed, 0.0)]
#[case::refused_at_open(SocketClose::RefusedAtOpen, 0.0)]
#[case::turn_unfinished(SocketClose::TurnUnfinished, 300.0)]
#[case::settled_by_its_log(SocketClose::SettledByItsLog, 300.0)]
#[tokio::test]
async fn a_socket_left_to_its_close_holds_its_tpm_through_its_failure_callbacks_until_the_close_settles_it(
    #[case] close: SocketClose,
    #[case] settled: f64,
) {
    let (limiter, store, _) = limiter(true);
    let socket = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 1000, "max_parallel_requests": 1}}]),
    );
    let state = LimiterState::default();
    assert_eq!(
        limiter
            .pre_call(&state, facts(socket.clone(), Some(300)), None)
            .await,
        None
    );
    limiter.leave_reservation_to_socket_close(&state);
    limiter
        .post_call_failure(&state, PostCallFailureFacts::default())
        .await;
    assert!(state.snapshot().parallel_slot.is_none());
    assert_eq!(key_tokens(&store), Some(300.0));
    limiter.close_socket(&state, close).await;
    limiter.close_socket(&state, close).await;
    assert_eq!(key_tokens(&store), Some(settled));
}

#[rstest]
#[tokio::test]
async fn a_socket_refund_after_its_window_rolled_over_leaves_the_next_window_alone() {
    let (limiter, store, clock) = limiter(true);
    let tpm = descriptors(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 1000}}]),
    );
    let socket = LimiterState::default();
    assert_eq!(
        limiter
            .pre_call(&socket, facts(tpm.clone(), Some(300)), None)
            .await,
        None
    );
    clock
        .0
        .store((1_700_000_061.0_f64).to_bits(), Ordering::SeqCst);
    let next = LimiterState::default();
    assert_eq!(
        limiter.pre_call(&next, facts(tpm, Some(100)), None).await,
        None
    );
    let before = key_tokens(&store);
    limiter
        .close_socket(&socket, SocketClose::RelayFailed)
        .await;
    assert_eq!(key_tokens(&store), before);
}

fn client_of(server: &RedisServer) -> RedisClient {
    RedisClient::new(RedisConfig {
        target: Target::Url(RedisUrl {
            url: SecretValue::new(server.url().to_owned()),
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
    .unwrap()
}

fn tokens_of(descriptor: &str, value: &str, amount: i64) -> Increment {
    Increment {
        counter: Counter {
            scope: Scope {
                descriptor: descriptor.into(),
                value: value.into(),
            },
            kind: Kind::Tokens,
        },
        amount: Number::Int(amount),
        ttl: Some(Duration::from_secs(60)),
    }
}

#[rstest]
#[tokio::test]
async fn a_slot_group_whose_script_fails_does_not_charge_the_groups_before_it_again() {
    let server = RedisServer::start().await.unwrap();
    let client = client_of(&server);
    let sixty = Duration::from_secs(60);
    client
        .send(commands::set("{user:poisoned}:window", "1", sixty))
        .await
        .unwrap();
    client
        .send(commands::eval::<i64>(
            &Script::new("return redis.call('RPUSH', KEYS[1], 'not a number')"),
            &["{user:poisoned}:tokens"],
            Vec::<String>::new(),
        ))
        .await
        .unwrap();
    let remote = Remote {
        client: client.clone(),
        default_ttl: sixty,
    };
    let (limiter, store, _) = limiter_over(true, |_| RateLimits::new(Some(remote), None));
    let charges = [
        tokens_of("api_key", "charged", 100),
        tokens_of("user", "poisoned", 100),
    ];
    let scopes: Vec<Scope> = charges
        .iter()
        .map(|charge| charge.counter.scope.clone())
        .collect();
    assert_eq!(store.slot_groups(&scopes, true).len(), 2);

    limiter
        .engine()
        .increment_with_ttl_preservation(
            &charges,
            Ctx {
                policy: Policy {
                    force_grouping: true,
                    ..Policy::default()
                },
                request: None,
            },
        )
        .await;

    assert_eq!(
        client
            .send(commands::get::<String>("{api_key:charged}:tokens"))
            .await
            .unwrap(),
        Some("100".to_owned())
    );
}
