mod support;

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_redis_client::commands;
use litellm_redis_client::{Amount, Expiry, Operation, Ttl};
use litellm_redis_store::{
    Number, RequestRedis, Send as RedisSend, TierPolicy,
    router_usage::{COUNTER_TTL, Counter, Increment, Kind, RouterUsage, Scope},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{USAGE_POLICY as POLICY, connect, deployment_usage, remote, usage, within};
use tokio::runtime::Handle;

const DAY: Duration = Duration::from_secs(86_400);

fn tpm_and_rpm(model: Option<&str>) -> [Counter; 2] {
    [Kind::Tpm, Kind::Rpm].map(|kind| Counter {
        scope: Scope::GlobalRouter,
        deployment_id: "dep".to_owned(),
        model: model.map(str::to_owned),
        window: "12-34".to_owned(),
        kind,
    })
}

fn increment(counter: &Counter, amount: Number, ttl: Duration) -> Increment {
    Increment {
        counter: counter.clone(),
        amount,
        ttl: Some(ttl),
    }
}

async fn ttl_of(raw: &litellm_redis_client::RedisClient, key: &str) -> Duration {
    let Ttl::Expires(remaining) = raw.send(commands::ttl(key)).await.unwrap() else {
        panic!("{key} has no TTL");
    };
    remaining
}

#[rstest]
#[case::tpm(Kind::Tpm, "tpm")]
#[case::rpm(Kind::Rpm, "rpm")]
#[case::itpm(Kind::Itpm, "itpm")]
#[case::otpm(Kind::Otpm, "otpm")]
#[case::tpd(Kind::Tpd, "tpd")]
#[case::rpd(Kind::Rpd, "rpd")]
fn a_counter_kind_round_trips_through_its_key_segment(#[case] kind: Kind, #[case] segment: &str) {
    assert_eq!(<&str>::from(kind), segment);
    assert_eq!(segment.parse::<Kind>().unwrap(), kind);
}

#[rstest]
#[case::global_router(Scope::GlobalRouter, "global_router")]
#[case::deployment(Scope::Deployment, "deployment")]
fn a_counter_scope_round_trips_through_its_spelling(#[case] scope: Scope, #[case] spelling: &str) {
    assert_eq!(<&str>::from(scope), spelling);
    assert_eq!(spelling.parse::<Scope>().unwrap(), scope);
}

#[rstest]
#[tokio::test]
async fn reads_deployment_counters_python_incremented_under_mains_keys(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let cases = deployment_usage();
    for (case, delta) in cases.iter().zip([4.0, 1200.5]) {
        raw.send(commands::incr_by_float(
            case.key,
            Amount::Float(delta),
            Some(Expiry::Refresh(DAY)),
        ))
        .await
        .unwrap();
    }

    assert_eq!(
        store
            .get_many(&cases.map(|case| case.counter))
            .await
            .unwrap(),
        vec![Some(Number::Int(4)), Some(Number::Float(1200.5))]
    );
}

#[rstest]
#[tokio::test]
async fn a_declared_usage_read_rides_the_request_batch_and_backfills_memory(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = support::connect_observed(&deployment);
    let store = Arc::new(RouterUsage::new(Some(remote(client)), POLICY));
    let cases = deployment_usage();
    for (case, delta) in cases.iter().zip([2.0, 30.0]) {
        raw.send(commands::incr_by_float(
            case.key,
            Amount::Float(delta),
            Some(Expiry::Refresh(DAY)),
        ))
        .await
        .unwrap();
    }
    let counters = cases.map(|case| case.counter);
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    let declared = store
        .declare_get_many(RedisSend::PreCall(&request), &counters)
        .unwrap();
    assert!(trips.seen().is_empty());
    request.flush_pre().await;
    let read = declared.await.unwrap();

    let one_mget = match topology {
        Topology::Standalone => Operation::Pipeline { ops: 1 },
        Topology::Cluster => Operation::Command("MGET"),
    };
    assert_eq!(trips.seen(), [one_mget]);
    assert_eq!(read, vec![Some(Number::Int(2)), Some(Number::Int(30))]);
    assert_eq!(
        store.get_local_many(&counters).unwrap(),
        vec![Some(Number::Int(2)), Some(Number::Int(30))]
    );
}

// main's ArmedRead settles on the batch's reply: a prefetch the pick never takes still backfills
// memory, and the throttle then keeps later reads off Redis
#[rstest]
#[tokio::test]
async fn an_abandoned_usage_prefetch_still_backfills_memory_and_holds_the_throttle(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = support::connect_observed(&deployment);
    let store = Arc::new(RouterUsage::new(Some(remote(client)), POLICY));
    let [rpm, tpd] = deployment_usage();
    raw.send(commands::incr_by_float(
        rpm.key,
        Amount::Float(6.0),
        Some(Expiry::Refresh(COUNTER_TTL)),
    ))
    .await
    .unwrap();
    let counters = [rpm.counter.clone(), tpd.counter.clone()];
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    drop(
        store
            .declare_get_many(RedisSend::PreCall(&request), &counters)
            .unwrap(),
    );
    request.flush_pre().await;
    litellm_redis_testing::eventually(async || store.get_local(&rpm.counter).unwrap()).await;
    let trips_after_prefetch = trips.seen().len();
    let read = store.get_many(&counters).await.unwrap();

    assert_eq!(read, vec![Some(Number::Int(6)), None]);
    assert_eq!(trips.seen().len(), trips_after_prefetch);
}

// The model rate limit check charges a request's rpm and rpd in one async_increment_cache_pipeline,
// whose RedisCache pipeline refreshes each window with EXPIRE
#[rstest]
#[tokio::test]
async fn a_pipeline_increment_is_one_round_trip_refreshing_each_window(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = support::connect_observed(&deployment);
    let store = RouterUsage::new(Some(remote(client)), POLICY);
    let [rpm, tpd] = deployment_usage();
    raw.send(commands::incr_by_float(
        rpm.key,
        Amount::Float(3.0),
        Some(Expiry::Refresh(Duration::from_secs(10))),
    ))
    .await
    .unwrap();

    let totals = store
        .increment_pipeline(&[
            increment(&rpm.counter, Number::Int(1), COUNTER_TTL),
            increment(&tpd.counter, Number::Int(1), DAY),
        ])
        .await;

    assert_eq!(trips.seen(), [Operation::Pipeline { ops: 2 }]);
    assert_eq!(totals, Some(vec![Number::Float(4.0), Number::Float(1.0)]));
    assert!(within(ttl_of(&raw, rpm.key).await, 60));
    assert!(within(ttl_of(&raw, tpd.key).await, 86_400));
}

// LowestTPMLoggingHandler_v2.async_log_success_event's async_increment_cache_post_call
#[rstest]
#[tokio::test]
async fn a_post_call_increment_counts_in_memory_now_and_in_redis_at_flush_post(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let [rpm, _] = deployment_usage();
    raw.send(commands::incr_by_float(
        rpm.key,
        Amount::Float(5.0),
        Some(Expiry::Refresh(COUNTER_TTL)),
    ))
    .await
    .unwrap();
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    store.increment_post_call(
        &request,
        &[increment(&rpm.counter, Number::Int(7), COUNTER_TTL)],
    );
    let before_flush: Option<String> = raw.send(commands::get(rpm.key)).await.unwrap();
    request.flush_post().await;
    let after_flush: Option<String> = raw.send(commands::get(rpm.key)).await.unwrap();

    assert_eq!(store.get_local(&rpm.counter).unwrap(), Some(Number::Int(7)));
    assert_eq!(before_flush, Some("5".to_owned()));
    assert_eq!(after_flush, Some("12".to_owned()));
}

// BaseRoutingStrategy's periodic sync counts in memory first, then sends the queued increments
// through redis_cache.async_increment_pipeline alone
#[rstest]
#[tokio::test]
async fn a_remote_increment_answers_the_redis_totals_and_leaves_memory_alone(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let [rpm, _] = deployment_usage();
    let queued = increment(&rpm.counter, Number::Int(2), COUNTER_TTL);
    raw.send(commands::incr_by_float(
        rpm.key,
        Amount::Float(10.0),
        Some(Expiry::Refresh(COUNTER_TTL)),
    ))
    .await
    .unwrap();

    let counted = store.increment_local(&queued).unwrap();
    let totals = store
        .increment_remote(std::slice::from_ref(&queued))
        .await
        .unwrap();

    assert_eq!(counted, Number::Int(2));
    assert_eq!(totals, Some(vec![Number::Float(12.0)]));
    assert_eq!(store.get_local(&rpm.counter).unwrap(), Some(Number::Int(2)));
}

#[rstest]
#[tokio::test]
async fn a_remote_increment_is_none_without_redis_and_an_error_when_redis_fails() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let detached = RouterUsage::new(None, POLICY);
    let attached = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let [rpm, _] = deployment_usage();
    let queued = [increment(&rpm.counter, Number::Int(1), COUNTER_TTL)];
    deployment.stop().await.unwrap();

    assert_eq!(detached.increment_remote(&queued).await.unwrap(), None);
    assert!(attached.increment_remote(&queued).await.is_err());
}

#[rstest]
#[tokio::test]
async fn a_local_merge_replaces_the_memory_count_without_reaching_redis() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let (client, trips) = support::connect_observed(&deployment);
    let store = RouterUsage::new(Some(remote(client)), POLICY);
    let [rpm, tpd] = deployment_usage();

    store
        .increment_local(&increment(&rpm.counter, Number::Int(1), COUNTER_TTL))
        .unwrap();
    store.set_local(&rpm.counter, Number::Float(9.0)).unwrap();

    assert_eq!(
        store.get_local_many(&[rpm.counter, tpd.counter]).unwrap(),
        vec![Some(Number::Float(9.0)), None]
    );
    assert!(trips.seen().is_empty());
}

#[rstest]
#[tokio::test]
async fn current_usage_skips_redis_once_memory_reaches_the_limit() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let (client, trips) = support::connect_observed(&deployment);
    let store = RouterUsage::new(Some(remote(client)), POLICY);
    let [rpm, _] = deployment_usage();
    store
        .increment_local(&increment(&rpm.counter, Number::Int(5), COUNTER_TTL))
        .unwrap();

    let at_limit = store.current(&rpm.counter, Number::Int(5)).await.unwrap();
    let skipped = trips.seen();
    let below_limit = store.current(&rpm.counter, Number::Int(6)).await.unwrap();

    assert_eq!(at_limit, Some(Number::Int(5)));
    assert!(skipped.is_empty());
    assert_eq!(below_limit, None, "Redis holds no count, and Redis wins");
    assert_eq!(trips.seen(), [Operation::Command("GET")]);
}

#[rstest]
#[tokio::test]
async fn current_usage_reads_none_on_a_failure_and_memory_once_the_breaker_opens() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let breaker = litellm_circuit_breaker::Breaker::new(
        litellm_circuit_breaker::Config {
            failure_threshold: 1,
            recovery_timeout: Duration::from_secs(60),
            ..litellm_circuit_breaker::Config::default()
        },
        Box::new(|_| {}),
    );
    let store = RouterUsage::new(
        Some(remote(support::connect_guarded(&deployment, breaker))),
        POLICY,
    );
    let [rpm, _] = deployment_usage();
    store
        .increment_local(&increment(&rpm.counter, Number::Int(2), COUNTER_TTL))
        .unwrap();
    deployment.stop().await.unwrap();

    let failed = store.current(&rpm.counter, Number::Int(10)).await.unwrap();
    let refused = store.current(&rpm.counter, Number::Int(10)).await.unwrap();

    assert_eq!(failed, None);
    assert_eq!(refused, Some(Number::Int(2)));
}

#[rstest]
#[tokio::test]
async fn current_usage_without_redis_is_the_memory_count() {
    let store = RouterUsage::new(None, POLICY);
    let [rpm, _] = deployment_usage();
    store
        .increment_local(&increment(&rpm.counter, Number::Int(3), COUNTER_TTL))
        .unwrap();

    assert_eq!(
        store.current(&rpm.counter, Number::Int(10)).await.unwrap(),
        Some(Number::Int(3))
    );
}

#[rstest]
#[case::unknown("tokens")]
#[case::wrong_case("TPM")]
fn other_text_is_not_a_counter_kind(#[case] segment: &str) {
    assert!(segment.parse::<Kind>().is_err());
}

// DualCache._backfill_kwargs: a router's batch read keeps a Redis value in memory for
// litellm.default_in_memory_ttl when one is set, else for InMemoryCache's own 600 s
#[rstest]
#[case::configured(Some(Duration::from_secs(5)), None)]
#[case::unset(None, Some(Number::Int(3)))]
#[tokio::test]
async fn a_batch_read_backfills_memory_for_default_in_memory_ttl(
    #[case] default_in_memory_ttl: Option<Duration>,
    #[case] after_six_seconds: Option<Number>,
) {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let seconds = Arc::new(AtomicU64::new(1_000));
    let now = {
        let seconds = seconds.clone();
        Arc::new(move || Duration::from_secs(seconds.load(Ordering::SeqCst)))
    };
    let policy = TierPolicy {
        refresh_after: Duration::from_secs(3600),
        default_in_memory_ttl,
        ..POLICY
    };
    let store = RouterUsage::with_clock(Some(remote(connect(&deployment))), policy, now);
    let [case, _] = usage();
    raw.send(commands::incr_by_float(
        case.key,
        Amount::Float(3.0),
        Some(Expiry::Refresh(COUNTER_TTL)),
    ))
    .await
    .unwrap();
    store
        .get_many(std::slice::from_ref(&case.counter))
        .await
        .unwrap();
    raw.send(commands::del(&[case.key])).await.unwrap();
    seconds.fetch_add(6, Ordering::SeqCst);

    let read = store.get_many(&[case.counter]).await.unwrap();

    assert_eq!(read, vec![after_six_seconds]);
}

#[rstest]
#[tokio::test]
async fn reads_counters_python_incremented_under_mains_keys(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let cases = usage();
    for (case, delta) in cases.iter().zip([7.75, 3.0]) {
        raw.send(commands::incr_by_float(
            case.key,
            Amount::Float(delta),
            Some(Expiry::Refresh(COUNTER_TTL)),
        ))
        .await
        .unwrap();
    }

    assert_eq!(
        store
            .get_many(&cases.map(|case| case.counter))
            .await
            .unwrap(),
        vec![Some(Number::Float(7.75)), Some(Number::Int(3))]
    );
}

#[rstest]
#[tokio::test]
async fn deployment_increments_write_incrbyfloat_totals_with_a_refreshed_minute_window(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let writer = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let reader = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let [case, _] = usage();
    let model = case.counter.model.as_deref();
    let [tpm_key, rpm_key] =
        ["tpm", "rpm"].map(|kind| case.key.replace(":tpm:", &format!(":{kind}:")));

    writer
        .increment_deployment("dep", model, "12-34", Number::Float(5.5), 1)
        .await
        .unwrap();
    raw.send(commands::expire(&tpm_key, Duration::from_secs(5)))
        .await
        .unwrap();
    writer
        .increment_deployment("dep", model, "12-34", Number::Int(2), 1)
        .await
        .unwrap();

    let written: [Option<String>; 2] = [
        raw.send(commands::get(&tpm_key)).await.unwrap(),
        raw.send(commands::get(&rpm_key)).await.unwrap(),
    ];
    let Ttl::Expires(remaining) = raw.send(commands::ttl(&tpm_key)).await.unwrap() else {
        panic!("{tpm_key} has no TTL");
    };
    assert_eq!(written, [Some("7.5".to_owned()), Some("2".to_owned())]);
    assert!(within(remaining, 60), "{remaining:?}");
    assert_eq!(
        reader.get_many(&tpm_and_rpm(model)).await.unwrap(),
        vec![Some(Number::Float(7.5)), Some(Number::Int(2))]
    );
    assert_eq!(
        writer.get_many(&tpm_and_rpm(model)).await.unwrap(),
        vec![Some(Number::Int(7)), Some(Number::Int(2))]
    );
}

#[rstest]
#[tokio::test]
async fn a_request_increment_keeps_the_window_python_started(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let [_, case] = usage();
    let key = case.key;
    raw.send(commands::incr_by_float(
        key,
        Amount::Float(1.0),
        Some(Expiry::KeepExisting(Duration::from_secs(20))),
    ))
    .await
    .unwrap();

    let total = store
        .increment(&case.counter, Number::Int(1), COUNTER_TTL)
        .await
        .unwrap();
    let Ttl::Expires(remaining) = raw.send(commands::ttl(key)).await.unwrap() else {
        panic!("{key} has no TTL");
    };

    assert_eq!(total, Number::Float(2.0));
    assert!(within(remaining, 20), "{remaining:?}");
}

#[rstest]
#[tokio::test]
async fn without_redis_counters_accumulate_in_memory_until_flushed() {
    let store = RouterUsage::new(None, POLICY);
    let counters = tpm_and_rpm(Some("model"));

    store
        .increment_deployment("dep", Some("model"), "12-34", Number::Int(2), 1)
        .await
        .unwrap();
    store
        .increment_deployment("dep", Some("model"), "12-34", Number::Float(5.5), 1)
        .await
        .unwrap();
    let counted = store.get_many(&counters).await.unwrap();
    store.flush_local_counters().unwrap();

    assert_eq!(
        counted,
        vec![Some(Number::Float(7.5)), Some(Number::Int(2))]
    );
    assert_eq!(store.get_many(&counters).await.unwrap(), vec![None, None]);
}

#[rstest]
#[tokio::test]
async fn a_late_attach_keeps_memory_and_later_attaches_are_ignored(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let attached = RedisDeployment::start(topology).await.unwrap();
    let ignored = RedisDeployment::start(topology).await.unwrap();
    let store = RouterUsage::new(None, POLICY);
    let [case, _] = usage();
    store
        .increment(&case.counter, Number::Int(2), COUNTER_TTL)
        .await
        .unwrap();

    assert!(store.attach(remote(connect(&attached))));
    assert!(!store.attach(remote(connect(&ignored))));
    let total = store
        .increment(&case.counter, Number::Int(3), COUNTER_TTL)
        .await
        .unwrap();

    assert_eq!(total, Number::Float(3.0));
    assert_eq!(
        store
            .get_many(std::slice::from_ref(&case.counter))
            .await
            .unwrap(),
        vec![Some(Number::Int(5))]
    );
    let ignored_total: Option<String> = connect(&ignored)
        .send(commands::get(case.key))
        .await
        .unwrap();
    assert_eq!(ignored_total, None);
}

#[rstest]
#[tokio::test]
async fn a_failed_pipeline_keeps_the_memory_counts_as_integers() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    deployment.stop().await.unwrap();

    store
        .increment_deployment("dep", Some("model"), "12-34", Number::Float(5.75), 1)
        .await
        .unwrap();

    assert_eq!(
        store.get_many(&tpm_and_rpm(Some("model"))).await.unwrap(),
        vec![Some(Number::Int(5)), Some(Number::Int(1))]
    );
}

// main's _increment_deployment sends async_increment_pipeline on its own round trip, never on the
// request's post-call pipeline
#[rstest]
#[tokio::test]
async fn a_deployment_increment_is_its_own_pipeline_and_mirrors_the_redis_totals(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = support::connect_observed(&deployment);
    let store = RouterUsage::new(Some(remote(client)), POLICY);
    let counters = tpm_and_rpm(Some("model"));
    let [tpm_key, rpm_key] =
        ["tpm", "rpm"].map(|kind| format!("global_router:dep:model:{kind}:12-34"));
    raw.send(commands::incr_by_float(
        &tpm_key,
        Amount::Float(10.0),
        Some(Expiry::Refresh(COUNTER_TTL)),
    ))
    .await
    .unwrap();

    store
        .increment_deployment("dep", Some("model"), "12-34", Number::Float(5.5), 1)
        .await
        .unwrap();

    assert_eq!(trips.seen(), [Operation::Pipeline { ops: 2 }]);
    assert_eq!(
        store.get_many(&counters).await.unwrap(),
        vec![Some(Number::Int(15)), Some(Number::Int(1))]
    );
    let written: [Option<String>; 2] = [
        raw.send(commands::get(&tpm_key)).await.unwrap(),
        raw.send(commands::get(&rpm_key)).await.unwrap(),
    ];
    assert_eq!(written, [Some("15.5".to_owned()), Some("1".to_owned())]);
    let Ttl::Expires(remaining) = raw.send(commands::ttl(&rpm_key)).await.unwrap() else {
        panic!("{rpm_key} has no TTL");
    };
    assert!(within(remaining, 60), "{remaining:?}");
}

#[rstest]
#[tokio::test]
async fn a_failed_deployment_pipeline_keeps_the_memory_counts_as_integers() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let counters = tpm_and_rpm(Some("model"));
    deployment.stop().await.unwrap();

    store
        .increment_deployment("dep", Some("model"), "12-34", Number::Float(5.75), 1)
        .await
        .unwrap();

    assert_eq!(
        store.get_many(&counters).await.unwrap(),
        [Some(Number::Int(5)), Some(Number::Int(1))]
    );
}

// The io-token check stashes a counter's fields and the store spells the key, so the key must stay
// main's RouterCacheEnum.ITPM / OTPM spelling
#[rstest]
#[case::itpm(Kind::Itpm, "global_router:dep-1:bedrock/claude:itpm:12-34")]
#[case::otpm(Kind::Otpm, "global_router:dep-1:bedrock/claude:otpm:12-34")]
#[tokio::test]
async fn io_token_counters_increment_under_mains_keys(#[case] kind: Kind, #[case] key: &str) {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let counter = Counter {
        scope: Scope::GlobalRouter,
        deployment_id: "dep-1".to_owned(),
        model: Some("bedrock/claude".to_owned()),
        window: "12-34".to_owned(),
        kind,
    };

    store
        .increment(&counter, Number::Int(8), COUNTER_TTL)
        .await
        .unwrap();

    let written: Option<String> = raw.send(commands::get(key)).await.unwrap();
    assert_eq!(written, Some("8".to_owned()));
}

#[rstest]
#[tokio::test]
async fn shared_usage_prefers_what_every_pod_counted_and_falls_back_to_memory() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterUsage::new(Some(remote(connect(&deployment))), POLICY);
    let [rpm, tpd] = deployment_usage();
    for case in [&rpm, &tpd] {
        store
            .increment_local(&increment(&case.counter, Number::Int(2), COUNTER_TTL))
            .unwrap();
    }
    raw.send(commands::incr_by_float(
        rpm.key,
        Amount::Float(9.0),
        Some(Expiry::Refresh(COUNTER_TTL)),
    ))
    .await
    .unwrap();

    let counted = store.shared_or_local(&rpm.counter).await.unwrap();
    let absent = store.shared_or_local(&tpd.counter).await.unwrap();
    deployment.stop().await.unwrap();
    let failed = store.shared_or_local(&rpm.counter).await.unwrap();

    assert_eq!(counted, Some(Number::Int(9)));
    assert_eq!(
        absent,
        Some(Number::Int(2)),
        "Redis has no count, so memory answers"
    );
    assert_eq!(failed, Some(Number::Int(2)));
}
