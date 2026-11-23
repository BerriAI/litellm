use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use litellm_callbacks::Abort;
use litellm_callbacks_parallel_request_limiter::{
    Code, Descriptor, Engine, LimiterState, ParallelRequestLimiter, PostCallFailureFacts,
    PreCallFacts, PriorResponse, Settings, SuccessFacts, TargetFacts,
};
use litellm_redis_store::{
    Number,
    rate_limits::{Counter, Entry, Kind, Local, RateLimits, Scope},
};
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
    let store = Arc::new(RateLimits::new(None, None));
    let clock = Clock::at(1_700_000_000.5);
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
